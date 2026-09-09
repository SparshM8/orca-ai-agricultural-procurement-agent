"""Unit tests verifying that backend validation cannot be bypassed by the AI layer."""

import pytest
from orca.services.order import order_service
from orca.agent.tools import tool_registry


def test_ai_cannot_fabricate_purchase_rate():
    """Verify order_service rejects arbitrary rate supplied by AI layer."""
    # Authoritative rate for potato in GLOBAL_DEFAULT is 0.40
    # Attempting to force a rate of 0.05 must raise ValueError
    with pytest.raises(ValueError, match="Rate mismatch"):
        order_service.create_order(
            farmer_id="farmer_123",
            produce_type="potato",
            quantity=10.0,
            unit="kg",
            pickup_location="Farm 1",
            validated_rate=0.05,  # Fraudulent / hallucinated rate
            conversation_id="conv_fraud_001",
        )


def test_unsupported_produce_rejected():
    """Verify order creation fails for unsupported produce items."""
    with pytest.raises(ValueError, match="Validation failed"):
        order_service.create_order(
            farmer_id="farmer_123",
            produce_type="papaya",  # Not in supported produce table
            quantity=5.0,
            unit="kg",
            pickup_location="Farm 1",
            conversation_id="conv_unsupported_001",
        )


def test_invalid_quantity_rejected():
    """Verify zero or negative quantity fails validation."""
    with pytest.raises(ValueError, match="Validation failed"):
        order_service.create_order(
            farmer_id="farmer_123",
            produce_type="potato",
            quantity=-10.0,
            unit="kg",
            pickup_location="Farm 1",
            conversation_id="conv_negative_qty",
        )


def test_order_creation_idempotency():
    """Verify FR-019: Prevent duplicate order creation from repeated confirmation."""
    order1 = order_service.create_order(
        farmer_id="farmer_123",
        produce_type="potato",
        quantity=10.0,
        unit="kg",
        pickup_location="Farm 1",
        conversation_id="conv_idem_001",
    )
    # Re-send same confirmation with same conversation_id
    order2 = order_service.create_order(
        farmer_id="farmer_123",
        produce_type="potato",
        quantity=10.0,
        unit="kg",
        pickup_location="Farm 1",
        conversation_id="conv_idem_001",
    )
    assert order1.id == order2.id
    assert order1.total_amount == order2.total_amount


def test_agent_tool_catches_validation_error():
    """Verify AgentToolRegistry.create_order safely returns structured error dict on rejection."""
    res = tool_registry.create_order(
        farmer_id="farmer_123",
        produce="potato",
        quantity=10.0,
        unit="kg",
        pickup_location="Farm 1",
        validated_rate=0.99,  # Mismatches authoritative 0.40
        conversation_id="conv_tool_err",
    )
    assert res["success"] is False
    assert "error" in res
    assert "Rate mismatch" in res["error"]
