"""Order lifecycle and state management service (FR-016 to FR-020)."""

import uuid
from typing import Dict, Optional
from orca.domain.models import Order
from orca.domain.state_machine import OrderState, validate_transition
from orca.services.billing import billing_service


class OrderService:
    """Manages order creation, duplicate prevention, and state transitions."""

    def __init__(self):
        # In-memory store for orders during initial development
        self._orders: Dict[str, Order] = {}
        # Idempotency tracker: conversation_id -> order_id
        self._confirmed_conversations: Dict[str, str] = {}

    def create_order(
        self,
        farmer_id: str,
        produce_type: str,
        quantity: float,
        unit: str,
        validated_rate: float,
        pickup_location: str,
        region_code: str = "GLOBAL_DEFAULT",
        conversation_id: Optional[str] = None,
    ) -> Order:
        """Create a validated order idempotently.

        FR-016: Unique order ID.
        FR-019: Prevent duplicate order creation from repeated confirmation messages.
        """
        # Idempotency check
        if conversation_id and conversation_id in self._confirmed_conversations:
            existing_order_id = self._confirmed_conversations[conversation_id]
            return self._orders[existing_order_id]

        bill = billing_service.calculate_bill(
            produce_type=produce_type,
            quantity=quantity,
            unit=unit,
            rate_per_unit=validated_rate,
            region_code=region_code,
        )

        order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"
        order = Order(
            id=order_id,
            farmer_id=farmer_id,
            produce_type=produce_type,
            quantity=quantity,
            unit=unit,
            validated_rate=validated_rate,
            currency=bill.currency,
            total_amount=bill.total_amount,
            pickup_location=pickup_location,
            status=OrderState.ORDER_CONFIRMED,
        )

        self._orders[order_id] = order
        if conversation_id:
            self._confirmed_conversations[conversation_id] = order_id

        return order

    def get_order(self, order_id: str) -> Optional[Order]:
        """Retrieve order by ID."""
        return self._orders.get(order_id)

    def transition_state(self, order_id: str, new_state: OrderState) -> Order:
        """Enforce valid state transition on order."""
        order = self.get_order(order_id)
        if not order:
            raise ValueError(f"Order not found: {order_id}")

        validate_transition(order.status, new_state)
        order.status = new_state
        return order


order_service = OrderService()
