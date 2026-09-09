"""Order state machine according to SRS Section 10.

Defines the 13 canonical states and valid transitions for the procurement workflow.
"""

from enum import Enum
from typing import Dict, Set


class OrderState(str, Enum):
    """The 13 canonical transaction and order states from SRS Section 10."""

    OFFER_RECEIVED = "OFFER_RECEIVED"
    DETAILS_PENDING = "DETAILS_PENDING"
    RATE_VALIDATED = "RATE_VALIDATED"
    AWAITING_FARMER_CONFIRMATION = "AWAITING_FARMER_CONFIRMATION"
    ORDER_CONFIRMED = "ORDER_CONFIRMED"
    PAYMENT_PENDING = "PAYMENT_PENDING"
    PAYMENT_CONFIRMED = "PAYMENT_CONFIRMED"
    COLLECTION_PENDING = "COLLECTION_PENDING"
    COLLECTION_ASSIGNED = "COLLECTION_ASSIGNED"
    PICKED_UP = "PICKED_UP"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    EXCEPTION = "EXCEPTION"


class InvalidStateTransitionError(Exception):
    """Raised when an illegal order state transition is attempted."""

    def __init__(self, current_state: OrderState, target_state: OrderState):
        super().__init__(
            f"Invalid state transition: Cannot transition from {current_state.value} to {target_state.value}"
        )
        self.current_state = current_state
        self.target_state = target_state


# Valid transition graph enforcing deterministic state progression
VALID_TRANSITIONS: Dict[OrderState, Set[OrderState]] = {
    OrderState.OFFER_RECEIVED: {
        OrderState.DETAILS_PENDING,
        OrderState.RATE_VALIDATED,
        OrderState.CANCELLED,
    },
    OrderState.DETAILS_PENDING: {
        OrderState.DETAILS_PENDING,
        OrderState.RATE_VALIDATED,
        OrderState.CANCELLED,
    },
    OrderState.RATE_VALIDATED: {
        OrderState.AWAITING_FARMER_CONFIRMATION,
        OrderState.DETAILS_PENDING,
        OrderState.CANCELLED,
    },
    OrderState.AWAITING_FARMER_CONFIRMATION: {
        OrderState.ORDER_CONFIRMED,
        OrderState.DETAILS_PENDING,
        OrderState.CANCELLED,
    },
    OrderState.ORDER_CONFIRMED: {
        OrderState.PAYMENT_PENDING,
        OrderState.EXCEPTION,
        OrderState.CANCELLED,
    },
    OrderState.PAYMENT_PENDING: {
        OrderState.PAYMENT_CONFIRMED,
        OrderState.EXCEPTION,
        OrderState.CANCELLED,
    },
    OrderState.PAYMENT_CONFIRMED: {
        OrderState.COLLECTION_PENDING,
        OrderState.EXCEPTION,
    },
    OrderState.COLLECTION_PENDING: {
        OrderState.COLLECTION_ASSIGNED,
        OrderState.EXCEPTION,
        OrderState.CANCELLED,
    },
    OrderState.COLLECTION_ASSIGNED: {
        OrderState.PICKED_UP,
        OrderState.EXCEPTION,
        OrderState.CANCELLED,
    },
    OrderState.PICKED_UP: {
        OrderState.COMPLETED,
        OrderState.EXCEPTION,
    },
    OrderState.COMPLETED: set(),  # Terminal state
    OrderState.CANCELLED: set(),  # Terminal state
    OrderState.EXCEPTION: {
        OrderState.CANCELLED,
        OrderState.PAYMENT_PENDING,
        OrderState.COLLECTION_PENDING,
        OrderState.COLLECTION_ASSIGNED,
    },
}


def can_transition(current: OrderState, target: OrderState) -> bool:
    """Check whether a transition between two states is valid."""
    return target in VALID_TRANSITIONS.get(current, set())


def validate_transition(current: OrderState, target: OrderState) -> None:
    """Enforce valid transition or raise InvalidStateTransitionError."""
    if not can_transition(current, target):
        raise InvalidStateTransitionError(current, target)
