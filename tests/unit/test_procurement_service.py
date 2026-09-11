"""Unit tests for Phase 2: Authoritative Procurement Service.

Verifies:
1. Valid supported offer evaluation
2. Unsupported produce detection & rejection
3. Missing quantity handling
4. Missing pickup location handling
5. Invalid/unsupported unit rejection
6. Below minimum quantity constraint violation
7. Above maximum quantity constraint violation
8. Past pickup timing rejection
9. Scheduling horizon ceiling exceeded
10. Valid offer receives exact authoritative rate from PricingService
11. Bill calculation accurately delegated to BillingService
12. Pricing discrepancy detected without altering authoritative rate
13. Custom regional policy evaluation
14. Custom commodity policy evaluation
15. Partial/future harvest representation without side effects
16. Zero database mutations during evaluation
17. Zero order creation during evaluation
18. Zero state machine transitions or mutations
19. Grounded pricing rationale generation
"""

import pytest
from datetime import datetime, timezone, timedelta

from orca.domain.schemas import ExtractedOffer
from orca.domain.models import Rate
from orca.domain.procurement import (
    ProcurementConstraints,
    ProcurementEvaluation,
    register_procurement_policy,
)
from orca.services.procurement import procurement_service, ProcurementService
from orca.services.pricing import pricing_service
from orca.services.billing import billing_service
from orca.services.order import order_service
from orca.services.payment import payment_service
from orca.services.collection import collection_service


@pytest.fixture
def service():
    """ProcurementService instance with standard authoritative dependencies."""
    return ProcurementService(pricing=pricing_service, billing=billing_service)


# =============================================================================
# 1. Valid Supported Offer
# =============================================================================
def test_valid_supported_offer(service: ProcurementService):
    """Verify a complete, valid offer produces an acceptable evaluation with bill."""
    offer = ExtractedOffer(
        produce_type="potato",
        quantity=50.0,
        unit="kg",
        pickup_location="Springfield",
        availability_window="tomorrow at 10:00 AM",
    )
    result = service.evaluate_offer(offer, region_code="GLOBAL_DEFAULT")

    assert result.is_acceptable is True
    assert result.produce_supported is True
    assert result.rate is not None
    assert result.rate.rate_per_unit == 0.40
    assert result.bill_summary is not None
    assert result.bill_summary.total_amount == 20.0
    assert result.missing_fields == []
    assert result.rejection_reasons == []
    assert result.pricing_discrepancy is False
    assert result.quantity_within_limits is True
    assert result.timing_within_limits is True


# =============================================================================
# 2. Unsupported Produce
# =============================================================================
def test_unsupported_produce(service: ProcurementService):
    """Verify unsupported crops are explicitly rejected without rate or bill."""
    offer = ExtractedOffer(
        produce_type="mangoes",
        quantity=50.0,
        unit="kg",
        pickup_location="Springfield",
        availability_window="tomorrow at 10:00 AM",
    )
    result = service.evaluate_offer(offer)

    assert result.is_acceptable is False
    assert result.produce_supported is False
    assert any("UNSUPPORTED_PRODUCE:mangoes" in r for r in result.rejection_reasons)
    assert result.rate is None
    assert result.bill_summary is None
    assert "not procured" in result.guidance_message.lower()


# =============================================================================
# 3. Missing Quantity
# =============================================================================
def test_missing_quantity(service: ProcurementService):
    """Verify missing quantity prevents acceptance and is tracked in missing_fields."""
    offer = ExtractedOffer(
        produce_type="potato",
        quantity=None,
        unit="kg",
        pickup_location="Springfield",
        availability_window="tomorrow at 10:00 AM",
    )
    result = service.evaluate_offer(offer)

    assert result.is_acceptable is False
    assert "quantity" in result.missing_fields
    assert result.bill_summary is None


# =============================================================================
# 4. Missing Location
# =============================================================================
def test_missing_location(service: ProcurementService):
    """Verify missing pickup location is tracked in missing_fields."""
    offer = ExtractedOffer(
        produce_type="potato",
        quantity=50.0,
        unit="kg",
        pickup_location="",
        availability_window="tomorrow at 10:00 AM",
    )
    result = service.evaluate_offer(offer)

    assert result.is_acceptable is False
    assert "pickup_location" in result.missing_fields
    assert result.bill_summary is None


# =============================================================================
# 5. Invalid Unit
# =============================================================================
def test_invalid_unit(service: ProcurementService):
    """Verify unsupported unit produces rejection reason and fails acceptance."""
    offer = ExtractedOffer(
        produce_type="potato",
        quantity=50.0,
        unit="bushels",
        pickup_location="Springfield",
        availability_window="tomorrow at 10:00 AM",
    )
    result = service.evaluate_offer(offer)

    assert result.is_acceptable is False
    assert any("UNSUPPORTED_UNIT:bushels" in r for r in result.rejection_reasons)
    assert result.bill_summary is None


# =============================================================================
# 6. Below Minimum Quantity
# =============================================================================
def test_below_minimum_quantity(service: ProcurementService):
    """Verify quantity below configured minimum is rejected."""
    offer = ExtractedOffer(
        produce_type="potato",
        quantity=2.0,  # Min is 5.0
        unit="kg",
        pickup_location="Springfield",
        availability_window="tomorrow at 10:00 AM",
    )
    result = service.evaluate_offer(offer)

    assert result.is_acceptable is False
    assert result.quantity_within_limits is False
    assert any("QUANTITY_BELOW_MINIMUM:5.0" in r for r in result.rejection_reasons)
    assert result.bill_summary is None


# =============================================================================
# 7. Above Maximum Quantity
# =============================================================================
def test_above_maximum_quantity(service: ProcurementService):
    """Verify bulk volume above maximum threshold is rejected."""
    offer = ExtractedOffer(
        produce_type="potato",
        quantity=15000.0,  # Max is 10000.0
        unit="kg",
        pickup_location="Springfield",
        availability_window="tomorrow at 10:00 AM",
    )
    result = service.evaluate_offer(offer)

    assert result.is_acceptable is False
    assert result.quantity_within_limits is False
    assert any("QUANTITY_EXCEEDS_MAXIMUM:10000.0" in r for r in result.rejection_reasons)
    assert result.bill_summary is None


# =============================================================================
# 8. Past Pickup Time
# =============================================================================
def test_past_pickup_time(service: ProcurementService):
    """Verify historical pickup dates are rejected."""
    now = datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)
    yesterday_str = (now - timedelta(days=1)).strftime("%Y-%m-%d 10:00 AM")

    # Pass an explicit past datetime in pickup_datetime
    past_dt = now - timedelta(days=1)
    offer = ExtractedOffer(
        produce_type="potato",
        quantity=50.0,
        unit="kg",
        pickup_location="Springfield",
        availability_window=yesterday_str,
    )
    # Directly test constraints validation with past datetime
    violations = service.validate_constraints(
        quantity=50.0, unit="kg", pickup_datetime=past_dt, now=now
    )
    assert "SCHEDULE_IN_PAST" in violations


# =============================================================================
# 9. Scheduling Horizon Exceeded
# =============================================================================
def test_scheduling_horizon_exceeded(service: ProcurementService):
    """Verify pickup scheduled beyond maximum advance days is rejected."""
    now = datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)
    future_dt = now + timedelta(days=25)  # Max is 14 days

    violations = service.validate_constraints(
        quantity=50.0, unit="kg", pickup_datetime=future_dt, now=now
    )
    assert any("SCHEDULE_EXCEEDS_MAX_ADVANCE:14_DAYS" in v for v in violations)


# =============================================================================
# 10. Valid Offer with Authoritative Rate
# =============================================================================
def test_valid_offer_with_authoritative_rate(service: ProcurementService):
    """Verify authoritative rates for each supported crop are correctly resolved."""
    crops_expected = {
        "potato": 0.40,
        "tomato": 0.60,
        "onion": 0.35,
    }
    for crop, expected_rate in crops_expected.items():
        offer = ExtractedOffer(
            produce_type=crop,
            quantity=100.0,
            unit="kg",
            pickup_location="Springfield",
            availability_window="tomorrow at 10:00 AM",
        )
        res = service.evaluate_offer(offer)
        assert res.is_acceptable is True
        assert res.rate.rate_per_unit == expected_rate
        assert res.authoritative_rate == expected_rate


# =============================================================================
# 11. Bill Calculation Delegated to BillingService
# =============================================================================
def test_bill_calculation_delegated_to_billing_service(service: ProcurementService):
    """Verify bill summary arithmetic matches BillingService output exactly."""
    offer = ExtractedOffer(
        produce_type="tomato",
        quantity=75.0,
        unit="kg",
        pickup_location="Springfield",
        availability_window="tomorrow at 10:00 AM",
    )
    res = service.evaluate_offer(offer)
    expected_bill = billing_service.calculate_bill("tomato", 75.0, "kg", 0.60)

    assert res.bill_summary is not None
    assert res.bill_summary.subtotal == expected_bill.subtotal
    assert res.bill_summary.tax_amount == expected_bill.tax_amount
    assert res.bill_summary.total_amount == expected_bill.total_amount
    assert res.bill_summary.total_amount == 45.0


# =============================================================================
# 12. Pricing Discrepancy (Authoritative Rate Never Modified)
# =============================================================================
def test_pricing_discrepancy_detected_without_changing_authoritative_rate(service: ProcurementService):
    """Verify farmer counter-rate triggers discrepancy flag but does NOT mutate authoritative rate."""
    offer = ExtractedOffer(
        produce_type="potato",
        quantity=50.0,
        unit="kg",
        offered_rate=0.55,  # Farmer wants $0.55; authoritative is $0.40
        pickup_location="Springfield",
        availability_window="tomorrow at 10:00 AM",
    )
    res = service.evaluate_offer(offer)

    assert res.pricing_discrepancy is True
    assert res.farmer_offered_rate == 0.55
    assert res.authoritative_rate == 0.40
    # Authoritative rate in evaluation remains 0.40!
    assert res.rate.rate_per_unit == 0.40
    assert res.bill_summary.rate_per_unit == 0.40
    assert res.bill_summary.total_amount == 20.0  # 50 * 0.40, NOT 50 * 0.55!


# =============================================================================
# 13. Custom Regional Policy
# =============================================================================
def test_custom_regional_policy(service: ProcurementService):
    """Verify evaluation respects region-scoped policy configuration."""
    # IN_MH regional policy has min_quantity = 10.0, max_advance_days = 10
    offer = ExtractedOffer(
        produce_type="potato",
        quantity=8.0,  # Below IN_MH min of 10.0, but above GLOBAL min of 5.0
        unit="kg",
        pickup_location="Nashik",
        availability_window="tomorrow at 10:00 AM",
    )
    res = service.evaluate_offer(offer, region_code="IN_MH")

    assert res.is_acceptable is False
    assert any("QUANTITY_BELOW_MINIMUM:10.0" in r for r in res.rejection_reasons)
    assert res.policy_id == "IN_MH_POLICY"


# =============================================================================
# 14. Custom Commodity Policy
# =============================================================================
def test_custom_commodity_policy(service: ProcurementService):
    """Verify evaluation respects commodity-scoped policy overrides."""
    # Register strict onion policy: min 100 kg
    onion_policy = ProcurementConstraints(
        policy_id="BULK_ONION_POLICY",
        region_code="GLOBAL_DEFAULT",
        produce_type="onion",
        min_quantity=100.0,
        max_quantity=5000.0,
    )
    register_procurement_policy(onion_policy)

    # 40 kg onion fails under bulk onion policy
    onion_offer = ExtractedOffer(
        produce_type="onion",
        quantity=40.0,
        unit="kg",
        pickup_location="Springfield",
        availability_window="tomorrow at 10:00 AM",
    )
    res_onion = service.evaluate_offer(onion_offer)
    assert res_onion.is_acceptable is False
    assert any("QUANTITY_BELOW_MINIMUM:100.0" in r for r in res_onion.rejection_reasons)
    assert res_onion.policy_id == "BULK_ONION_POLICY"

    # 40 kg potato passes under standard policy
    potato_offer = ExtractedOffer(
        produce_type="potato",
        quantity=40.0,
        unit="kg",
        pickup_location="Springfield",
        availability_window="tomorrow at 10:00 AM",
    )
    res_potato = service.evaluate_offer(potato_offer)
    assert res_potato.is_acceptable is True


# =============================================================================
# 15. Partial Availability Representation
# =============================================================================
def test_partial_availability_representation(service: ProcurementService):
    """Verify offer with staged future availability evaluates ready batch cleanly."""
    offer = ExtractedOffer(
        produce_type="potato",
        quantity=50.0,
        unit="kg",
        pickup_location="Springfield",
        availability_window="tomorrow at 10:00 AM",
        future_availability="another 150 kg in 2 weeks",
    )
    res = service.evaluate_offer(offer)

    # The ready 50 kg lot is accepted
    assert res.is_acceptable is True
    assert res.bill_summary.quantity == 50.0
    assert res.bill_summary.total_amount == 20.0


# =============================================================================
# 16, 17, 18. Zero Side Effects (No DB, No Order, No State Machine Mutation)
# =============================================================================
def test_evaluation_has_zero_side_effects(service: ProcurementService):
    """Verify evaluate_offer is purely observational and mutates zero backend entities."""
    # Capture state before
    orders_count_before = len(order_service._orders)
    payments_count_before = len(payment_service._payments)
    tasks_count_before = len(collection_service._tasks)

    # Run multiple evaluations (valid, invalid, discrepancy)
    offer_valid = ExtractedOffer(produce_type="potato", quantity=50.0, unit="kg", pickup_location="Springfield", availability_window="tomorrow at 10:00 AM")
    offer_invalid = ExtractedOffer(produce_type="mangoes", quantity=5.0, unit="kg", pickup_location="Springfield", availability_window="tomorrow at 10:00 AM")
    offer_counter = ExtractedOffer(produce_type="potato", quantity=50.0, unit="kg", offered_rate=0.99, pickup_location="Springfield", availability_window="tomorrow at 10:00 AM")

    service.evaluate_offer(offer_valid)
    service.evaluate_offer(offer_invalid)
    service.evaluate_offer(offer_counter)

    # Assert completely unchanged backend state
    assert len(order_service._orders) == orders_count_before
    assert len(payment_service._payments) == payments_count_before
    assert len(collection_service._tasks) == tasks_count_before


# =============================================================================
# 19. Grounded Pricing Rationale
# =============================================================================
def test_pricing_rationale_grounded(service: ProcurementService):
    """Verify get_pricing_rationale returns grounded authoritative details."""
    rationale = service.get_pricing_rationale("potato", "GLOBAL_DEFAULT")

    assert rationale["produce_type"] == "potato"
    assert rationale["supported"] is True
    assert rationale["rate_per_unit"] == 0.40
    assert rationale["currency"] == "USD"
    assert rationale["fixed_rate_model"] is True
    assert "ORCA procurement operates on authoritative fixed market" in rationale["policy_message"]
    assert len(rationale["benefits"]) >= 4
    assert any("runner" in b.lower() for b in rationale["benefits"])
