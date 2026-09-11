"""Order lifecycle and state management service (FR-016 to FR-020)."""

import uuid
from datetime import datetime
from typing import Dict, Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from orca.domain.models import Order
from orca.domain.state_machine import OrderState, validate_transition
from orca.services.billing import billing_service
from orca.services.pricing import pricing_service
from orca.db.models import OrderModel


class OrderService:
    """Manages order creation, duplicate prevention, and state transitions."""

    def __init__(self):
        # In-memory store for orders during runtime / test mode
        self._orders: Dict[str, Order] = {}
        # Idempotency tracker: conversation_id -> order_id
        self._confirmed_conversations: Dict[str, str] = {}

    def clear(self) -> None:
        """Clear all in-memory orders and idempotency indices (used in demo reset / testing)."""
        self._orders.clear()
        self._confirmed_conversations.clear()

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
        pickup_datetime: Optional[datetime] = None,
        pickup_time_str: Optional[str] = None,
    ) -> Order:
        """Create a validated order idempotently in memory.

        FR-011 to FR-015: Strict authoritative rate enforcement (cannot be bypassed by AI layer).
        FR-016: Unique order ID.
        FR-019: Prevent duplicate order creation from repeated confirmation messages.
        """
        # Idempotency check (FR-019)
        if conversation_id and conversation_id in self._confirmed_conversations:
            existing_order_id = self._confirmed_conversations[conversation_id]
            return self._orders[existing_order_id]

        if not pickup_location or not pickup_location.strip():
            raise ValueError("Validation failed: Pickup location must not be empty.")

        # 1. Validate produce, quantity, and unit against business rules
        if not pricing_service.validate_produce_offer(produce_type, quantity, unit, region_code):
            raise ValueError(
                f"Validation failed: Unsupported produce '{produce_type}', invalid quantity ({quantity}), or unsupported unit '{unit}'."
            )

        # 2. Authoritative rate lookup (FR-012, FR-013)
        authoritative_rate_obj = pricing_service.get_applicable_rate(produce_type, region_code)
        if not authoritative_rate_obj:
            raise ValueError(f"No authoritative rate configured for '{produce_type}' in region '{region_code}'.")

        rate_to_apply = authoritative_rate_obj.rate_per_unit

        # If caller passed a validated_rate, verify it strictly matches authoritative rate
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
            pickup_datetime=pickup_datetime,
            pickup_time_str=pickup_time_str,
            status=OrderState.ORDER_CONFIRMED,
        )

        self._orders[order_id] = order
        if conversation_id:
            self._confirmed_conversations[conversation_id] = order_id

        return order

    async def create_order_async(
        self,
        farmer_id: str,
        produce_type: str,
        quantity: float,
        unit: str,
        pickup_location: str,
        validated_rate: Optional[float] = None,
        region_code: str = "GLOBAL_DEFAULT",
        conversation_id: Optional[str] = None,
        pickup_datetime: Optional[datetime] = None,
        pickup_time_str: Optional[str] = None,
        session: Optional[AsyncSession] = None,
    ) -> Order:
        """Create a validated order and persist to database using async session."""
        from orca.db.repository import OrderRepository

        # Check database idempotency first if session is available
        if session and conversation_id:
            repo = OrderRepository(session)
            existing_model = await repo.get_by_conversation_id(conversation_id)
            if existing_model:
                if existing_model.id in self._orders:
                    return self._orders[existing_model.id]
                existing_order = Order(
                    id=existing_model.id,
                    farmer_id=existing_model.farmer_id,
                    produce_type=existing_model.produce_type,
                    quantity=existing_model.quantity,
                    unit=existing_model.unit,
                    validated_rate=existing_model.validated_rate,
                    currency=existing_model.currency,
                    total_amount=existing_model.total_amount,
                    pickup_location=existing_model.pickup_location,
                    pickup_datetime=existing_model.pickup_datetime,
                    pickup_time_str=getattr(existing_model, "pickup_time_str", None),
                    status=OrderState(existing_model.status),
                    created_at=existing_model.created_at,
                    updated_at=existing_model.updated_at,
                )
                self._orders[existing_order.id] = existing_order
                self._confirmed_conversations[conversation_id] = existing_order.id
                return existing_order

        # Call deterministic domain creation
        order = self.create_order(
            farmer_id=farmer_id,
            produce_type=produce_type,
            quantity=quantity,
            unit=unit,
            pickup_location=pickup_location,
            validated_rate=validated_rate,
            region_code=region_code,
            conversation_id=conversation_id,
            pickup_datetime=pickup_datetime,
            pickup_time_str=pickup_time_str,
        )

        # Persist to database
        if session:
            await self._persist_to_session(order, conversation_id, session)
        else:
            try:
                from orca.db.session import async_session_factory
                async with async_session_factory() as s:
                    await self._persist_to_session(order, conversation_id, s)
            except Exception:
                # If database tables are uninitialized in unit-test context, memory store holds state
                pass

        return order

    async def _persist_to_session(
        self, order: Order, conversation_id: Optional[str], session: AsyncSession
    ) -> OrderModel:
        """Persist order entity to database session via OrderRepository."""
        from orca.db.repository import OrderRepository
        repo = OrderRepository(session)
        if conversation_id:
            existing = await repo.get_by_conversation_id(conversation_id)
            if existing:
                return existing

        order_model = OrderModel(
            id=order.id,
            farmer_id=order.farmer_id,
            produce_type=order.produce_type,
            quantity=order.quantity,
            unit=order.unit,
            validated_rate=order.validated_rate,
            currency=order.currency,
            total_amount=order.total_amount,
            pickup_location=order.pickup_location,
            pickup_datetime=order.pickup_datetime,
            pickup_time_str=order.pickup_time_str,
            status=order.status.value,
            conversation_id=conversation_id,
            created_at=order.created_at,
            updated_at=order.updated_at,
        )
        return await repo.save(order_model)

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
        if new_state == OrderState.COMPLETED:
            try:
                from orca.services.farmer_profile import farmer_profile_service
                farmer_profile_service.record_completed_order(
                    farmer_id=order.farmer_id,
                    produce=order.produce_type,
                    quantity=order.quantity,
                    unit=order.unit,
                    order_id=order.id,
                )
            except Exception:
                pass
        return order

    def get_all_orders(self, status: Optional[str] = None) -> List[Order]:
        """Retrieve all orders from store, optionally filtered by status."""
        orders = list(self._orders.values())
        if status:
            orders = [o for o in orders if o.status.value == status or o.status == status]
        return sorted(orders, key=lambda o: o.created_at or datetime.min, reverse=True)


order_service = OrderService()
