"""Unit tests for the order state machine."""

import pytest
from orca.domain.state_machine import (
    OrderState,
    can_transition,
    validate_transition,
    InvalidStateTransitionError,
)


def test_standard_happy_path_transitions():
    """Verify standard procurement state sequence from OFFER_RECEIVED to COMPLETED."""
    path = [
        (OrderState.OFFER_RECEIVED, OrderState.RATE_VALIDATED),
        (OrderState.RATE_VALIDATED, OrderState.AWAITING_FARMER_CONFIRMATION),
        (OrderState.AWAITING_FARMER_CONFIRMATION, OrderState.ORDER_CONFIRMED),
        (OrderState.ORDER_CONFIRMED, OrderState.PAYMENT_PENDING),
        (OrderState.PAYMENT_PENDING, OrderState.PAYMENT_CONFIRMED),
        (OrderState.PAYMENT_CONFIRMED, OrderState.COLLECTION_PENDING),
        (OrderState.COLLECTION_PENDING, OrderState.COLLECTION_ASSIGNED),
        (OrderState.COLLECTION_ASSIGNED, OrderState.PICKED_UP),
        (OrderState.PICKED_UP, OrderState.COMPLETED),
    ]

    for current, target in path:
        assert can_transition(current, target) is True
        validate_transition(current, target)  # Should not raise


def test_clarification_path():
    """Verify transition through DETAILS_PENDING."""
    assert can_transition(OrderState.OFFER_RECEIVED, OrderState.DETAILS_PENDING) is True
    assert can_transition(OrderState.DETAILS_PENDING, OrderState.RATE_VALIDATED) is True


def test_invalid_transitions_raise_error():
    """Verify illegal transitions trigger InvalidStateTransitionError."""
    # Cannot jump directly from OFFER_RECEIVED to COMPLETED
    assert can_transition(OrderState.OFFER_RECEIVED, OrderState.COMPLETED) is False
    with pytest.raises(InvalidStateTransitionError):
        validate_transition(OrderState.OFFER_RECEIVED, OrderState.COMPLETED)

    # Cannot jump from OFFER_RECEIVED to PAYMENT_CONFIRMED without confirmation
    assert can_transition(OrderState.OFFER_RECEIVED, OrderState.PAYMENT_CONFIRMED) is False
    with pytest.raises(InvalidStateTransitionError):
        validate_transition(OrderState.OFFER_RECEIVED, OrderState.PAYMENT_CONFIRMED)

    # Terminal states have no transitions
    assert can_transition(OrderState.COMPLETED, OrderState.OFFER_RECEIVED) is False
    with pytest.raises(InvalidStateTransitionError):
        validate_transition(OrderState.COMPLETED, OrderState.OFFER_RECEIVED)
