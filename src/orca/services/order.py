"""Order lifecycle and state management service (FR-016 to FR-020)."""

import uuid
from typing import Dict, Optional
from orca.domain.models import Order
from orca.domain.state_machine import OrderState, validate_transition
from orca.services.billing import billing_service
from orca.services.pricing import pricing_service


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
        pickup_location: str,
        validated_rate: Optional[float] = None,
        region_code: str = "GLOBAL_DEFAULT",
        conversation_id: Optional[str] = None,
    ) -> Order:
        """Create a validated order idempotently.

        FR-011 to FR-015: Strict authoritative rate enforcement (cannot be bypassed by AI layer).
        FR-016: Unique order ID.
        FR-019: Prevent duplicate order creation from repeated confirmation messages.
        """
        # Idempotency check (FR-019)
        if conversation_id and conversation_id in self._confirmed_conversations:
            existing_order_id = self._confirmed_conversations[conversation_id]
            return self._orders[existing_order_id]

        # 1. Validate produce and quantity against business rules
        if not pricing_service.validate_produce_offer(produce_type, quantity, unit, region_code):
            raise ValueError(
                f"Validation failed: Unsupported produce '{produce_type}', invalid quantity ({quantity}), or unsupported unit '{unit}'."
            )

        # 2. Authoritative rate lookup (FR-012, FR-013)
        authoritative_rate_obj = pricing_service.get_applicable_rate(produce_type, region_code)
        if not authoritative_rate_obj:
            raise ValueError(f"No authoritative rate configured for '{produce_type}' in region '{region_code}'.")

        rate_to_apply = authoritative_rate_obj.rate_per_unit

        # If caller passed a validated_rate, verify it matches authoritative rate
        if validated_rate is not None and abs(validated_rate - rate_to_apply) > 1e-4:
            raise ValueError(
                f"Rate mismatch: AI-supplied rate ({validated_rate}) does not match authoritative backend rate ({rate_to_apply})."
            )

        # 3. Deterministic calculation (FR-014)
        bill = billing_service.calculate_bill(
            produce_type=produce_type,
            quantity=quantity,
            unit=unit,
            rate_per_unit=rate_to_apply,
            region_code=region_code,
        )

        order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"
        order = Order(
            id=order_id,
            farmer_id=farmer_id,
            produce_type=produce_type,
            quantity=quantity,
            unit=unit,
            validated_rate=rate_to_apply,
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
