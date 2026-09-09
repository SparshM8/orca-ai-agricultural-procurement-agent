"""Unit tests for authoritative pricing and deterministic billing."""

from orca.services.pricing import pricing_service
from orca.services.billing import billing_service


def test_authoritative_rate_lookup():
    """Verify system retrieves configured baseline rates."""
    rate = pricing_service.get_applicable_rate("potato", "GLOBAL_DEFAULT")
    assert rate is not None
    assert rate.produce_type == "potato"
    assert rate.rate_per_unit == 0.40
    assert rate.currency == "USD"


def test_unknown_produce_returns_none():
    """Verify unconfigured produce does not return invented rate."""
    rate = pricing_service.get_applicable_rate("dragonfruit", "GLOBAL_DEFAULT")
    assert rate is None


def test_deterministic_bill_calculation():
    """Verify billing arithmetic: subtotal = quantity * rate."""
    bill = billing_service.calculate_bill(
        produce_type="potato",
        quantity=10.0,
        unit="kg",
        rate_per_unit=0.40,
        region_code="GLOBAL_DEFAULT",
    )
    assert bill.subtotal == 4.00
    assert bill.tax_amount == 0.00
    assert bill.total_amount == 4.00
    assert bill.currency == "USD"
