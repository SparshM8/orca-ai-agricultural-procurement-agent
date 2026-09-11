"""Unit tests for Phase 1: Procurement Domain Models, Policies, and Intents.

Verifies:
1. ProcurementConstraints configuration and policy-driven defaults
2. Regional and commodity-specific policy resolution with fallbacks
3. Quantity limits validation (positive, min, max)
4. Unit compatibility validation
5. Advance scheduling horizon validation (past date rejection, max advance days)
6. ProcurementEvaluation schema invariants
7. FutureHarvestDeclaration conversational representation
8. OBJECTION_PRICING and AMEND_OFFER IntentType definitions
9. ExtractedEntities new procurement fields
10. ExtractedOffer future_availability field
"""

import pytest
from datetime import datetime, timezone, timedelta

from orca.domain.intents import IntentType, ExtractedEntities
from orca.domain.schemas import ExtractedOffer, BillSummary
from orca.domain.models import Rate
from orca.domain.procurement import (
    ProcurementConstraints,
    ProcurementEvaluation,
    FutureHarvestDeclaration,
    get_procurement_policy,
    register_procurement_policy,
)


# =============================================================================
# 1. Policy Resolution & Hierarchy Tests
# =============================================================================
def test_default_procurement_policy_loaded():
    """Verify GLOBAL_DEFAULT policy provides sensible configuration-driven defaults."""
    policy = get_procurement_policy("GLOBAL_DEFAULT")
    assert policy.policy_id == "GLOBAL_DEFAULT_POLICY"
    assert policy.min_quantity == 5.0
    assert policy.max_quantity == 10000.0
    assert policy.max_advance_days == 14
    assert "kg" in policy.allowed_units
    assert "quintal" in policy.allowed_units


def test_regional_procurement_policy_resolution():
    """Verify regional policy lookup and unknown region fallback."""
    # Configured regional profile
    policy_us = get_procurement_policy("US_STANDARD")
    assert policy_us.min_quantity == 10.0
    assert policy_us.max_quantity == 20000.0
    assert "lb" in policy_us.allowed_units

    policy_in = get_procurement_policy("IN_MH")
    assert policy_in.max_advance_days == 10

    # Unknown region fallback to GLOBAL_DEFAULT
    policy_unknown = get_procurement_policy("UNKNOWN_REGION_XYZ")
    assert policy_unknown.region_code == "GLOBAL_DEFAULT"
    assert policy_unknown.policy_id == "GLOBAL_DEFAULT_POLICY"


def test_commodity_specific_policy_registration():
    """Verify registration and resolution of commodity-scoped policies."""
    avocado_policy = ProcurementConstraints(
        policy_id="GLOBAL_AVOCADO_POLICY",
        region_code="GLOBAL_DEFAULT",
        produce_type="avocado",
        min_quantity=20.0,
        max_quantity=500.0,
        allowed_units=["kg", "crate"],
        max_advance_days=7,
    )
    register_procurement_policy(avocado_policy)

    # Commodity-specific lookup
    resolved = get_procurement_policy("GLOBAL_DEFAULT", "avocado")
    assert resolved.policy_id == "GLOBAL_AVOCADO_POLICY"
    assert resolved.min_quantity == 20.0
    assert resolved.max_quantity == 500.0
    assert resolved.max_advance_days == 7

    # General commodity falls back to regional default
    resolved_potato = get_procurement_policy("GLOBAL_DEFAULT", "potato")
    assert resolved_potato.policy_id == "GLOBAL_DEFAULT_POLICY"
    assert resolved_potato.min_quantity == 5.0


# =============================================================================
# 2. Constraint Validation Tests (Quantity, Unit, Scheduling)
# =============================================================================
def test_validate_constraints_quantity():
    """Verify quantity boundary validation against policy thresholds."""
    policy = ProcurementConstraints(min_quantity=10.0, max_quantity=500.0)

    # Valid
    assert policy.validate_constraints(quantity=50.0, unit="kg") == []

    # Non-positive
    assert "QUANTITY_NON_POSITIVE" in policy.validate_constraints(quantity=0.0, unit="kg")
    assert "QUANTITY_NON_POSITIVE" in policy.validate_constraints(quantity=-5.0, unit="kg")

    # Below minimum
    violations_below = policy.validate_constraints(quantity=8.0, unit="kg")
    assert any("QUANTITY_BELOW_MINIMUM" in v for v in violations_below)

    # Above maximum
    violations_above = policy.validate_constraints(quantity=600.0, unit="kg")
    assert any("QUANTITY_EXCEEDS_MAXIMUM" in v for v in violations_above)


def test_validate_constraints_units():
    """Verify unit compatibility validation against policy allowed units."""
    policy = ProcurementConstraints(allowed_units=["kg", "ton"])

    # Valid units (case-insensitive)
    assert policy.validate_constraints(quantity=50.0, unit="kg") == []
    assert policy.validate_constraints(quantity=50.0, unit="KG") == []
    assert policy.validate_constraints(quantity=20.0, unit="ton") == []

    # Unsupported unit
    violations = policy.validate_constraints(quantity=50.0, unit="bushels")
    assert any("UNSUPPORTED_UNIT:bushels" in v for v in violations)


def test_validate_constraints_scheduling():
    """Verify scheduling horizon validation against max advance days and past dates."""
    now = datetime(2026, 9, 10, 10, 0, 0, tzinfo=timezone.utc)
    policy = ProcurementConstraints(max_advance_days=14)

    # Valid date (3 days in advance)
    valid_dt = now + timedelta(days=3)
    assert policy.validate_constraints(quantity=50.0, unit="kg", pickup_datetime=valid_dt, now=now) == []

    # Past date (1 day ago)
    past_dt = now - timedelta(days=1)
    violations_past = policy.validate_constraints(quantity=50.0, unit="kg", pickup_datetime=past_dt, now=now)
    assert "SCHEDULE_IN_PAST" in violations_past

    # Too far in future (20 days when max is 14)
    distant_dt = now + timedelta(days=20)
    violations_distant = policy.validate_constraints(quantity=50.0, unit="kg", pickup_datetime=distant_dt, now=now)
    assert any("SCHEDULE_EXCEEDS_MAX_ADVANCE" in v for v in violations_distant)


# =============================================================================
# 3. Evaluation and Future Availability Schemas
# =============================================================================
def test_procurement_evaluation_schema():
    """Verify ProcurementEvaluation model instantiates properly with all attributes."""
    eval_res = ProcurementEvaluation(
        is_acceptable=True,
        produce_supported=True,
        rate=Rate(
            id="r1",
            produce_type="potato",
            region_code="GLOBAL_DEFAULT",
            rate_per_unit=0.40,
            unit="kg",
        ),
        bill_summary=BillSummary(
            produce_type="potato",
            quantity=50.0,
            unit="kg",
            rate_per_unit=0.40,
            currency="USD",
            subtotal=20.0,
            tax_amount=0.0,
            total_amount=20.0,
        ),
        missing_fields=[],
        rejection_reasons=[],
        pricing_discrepancy=False,
        authoritative_rate=0.40,
    )
    assert eval_res.is_acceptable is True
    assert eval_res.produce_supported is True
    assert eval_res.bill_summary.total_amount == 20.0
    assert eval_res.policy_id == "GLOBAL_DEFAULT_POLICY"


def test_future_harvest_declaration_schema():
    """Verify FutureHarvestDeclaration observational model."""
    decl = FutureHarvestDeclaration(
        produce_type="potato",
        future_quantity=150.0,
        unit="kg",
        estimated_window="in 2 weeks",
        notes="Remaining field harvest pending dry weather",
    )
    assert decl.produce_type == "potato"
    assert decl.future_quantity == 150.0
    assert decl.unit == "kg"
    assert decl.notes is not None


# =============================================================================
# 4. Intent & Entity Extensions
# =============================================================================
def test_objection_and_amendment_intent_types():
    """Verify OBJECTION_PRICING and AMEND_OFFER are registered in IntentType enum."""
    assert IntentType.OBJECTION_PRICING.value == "OBJECTION_PRICING"
    assert IntentType.AMEND_OFFER.value == "AMEND_OFFER"
    # Ensure total intent types
    all_intents = [i.value for i in IntentType]
    assert "OBJECTION_PRICING" in all_intents
    assert "AMEND_OFFER" in all_intents
    assert "OFFER_PRODUCE" in all_intents
    assert "CONFIRM_ORDER" in all_intents
    assert "AUTHORIZE_PAYMENT" in all_intents


def test_extracted_entities_procurement_fields():
    """Verify new entity fields for negotiation and staged harvest."""
    entities = ExtractedEntities(
        counter_rate=0.55,
        future_quantity=100.0,
        future_timing="next Monday",
        amended_field="quantity",
    )
    dumped = entities.model_dump()
    assert dumped["counter_rate"] == 0.55
    assert dumped["future_quantity"] == 100.0
    assert dumped["future_timing"] == "next Monday"
    assert dumped["amended_field"] == "quantity"


def test_extracted_offer_future_availability_field():
    """Verify ExtractedOffer can store future_availability notes."""
    offer = ExtractedOffer(
        produce_type="tomato",
        quantity=50.0,
        unit="kg",
        future_availability="another 200 kg next week",
    )
    assert offer.future_availability == "another 200 kg next week"
