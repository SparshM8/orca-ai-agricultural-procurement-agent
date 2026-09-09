"""Payment workflow and status tracking service (FR-021 to FR-025).

Enforces:
- Payment amount strictly derived from persisted order total.
- Moving order to PAYMENT_PENDING upon initiation.
- Moving order to PAYMENT_CONFIRMED on success.
- Keeping order in PAYMENT_PENDING with retry message on failure.
- Idempotency preventing duplicate payment records.
- Provider-independent adapter interface.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, Any
from sqlalchemy.ext.asyncio import AsyncSession

from orca.domain.models import Payment
from orca.domain.state_machine import OrderState
from orca.services.order import order_service
from orca.adapters.payment.base import BasePaymentAdapter
from orca.adapters.payment.demo import DemoPaymentAdapter
from orca.db.models import PaymentModel


class PaymentService:
    """Manages payment initiation, status updates, and adapter orchestration."""

    def __init__(self, adapter: Optional[BasePaymentAdapter] = None):
        self.adapter = adapter or DemoPaymentAdapter()
        # In-memory store: payment_id -> Payment
        self._payments: Dict[str, Payment] = {}
        # Idempotency index: order_id -> payment_id
        self._order_payments: Dict[str, str] = {}

    def get_payment(self, payment_id: str) -> Optional[Payment]:
        """Retrieve payment by ID."""
        return self._payments.get(payment_id)

    def get_payment_for_order(self, order_id: str) -> Optional[Payment]:
        """Find payment associated with order (idempotency lookup)."""
        payment_id = self._order_payments.get(order_id)
        if payment_id:
            return self._payments.get(payment_id)
        for p in self._payments.values():
            if p.order_id == order_id:
                return p
        return None

    async def initiate_order_payment(
        self,
        order_id: str,
        recipient_reference: Optional[str] = None,
        session: Optional[AsyncSession] = None,
    ) -> Tuple[Payment, bool, str]:
        """Initiate payment for a confirmed order.

        FR-021: Payment status associated with order.
        FR-022: Payment amount derived from validated order total, not caller.
        FR-023: Timestamp and status recorded.
        FR-025: Sandbox / demo provider supported.
        """
        order = order_service.get_order(order_id)
        if not order:
            raise ValueError(f"Order not found: {order_id}")

        # Idempotency: If payment already completed successfully for this order, return it
        existing_payment = self.get_payment_for_order(order_id)
        if existing_payment and existing_payment.status == "SUCCESS":
            if order.status != OrderState.PAYMENT_CONFIRMED:
                order_service.transition_state(order_id, OrderState.PAYMENT_CONFIRMED)
            return existing_payment, True, "Payment already completed successfully."

        # FR-021: Payment cannot be created for an invalid order state
        if order.status not in [OrderState.ORDER_CONFIRMED, OrderState.PAYMENT_PENDING]:
            raise ValueError(
                f"Invalid order state for payment: Order {order_id} is in state {order.status.value}, "
                f"must be in ORDER_CONFIRMED or PAYMENT_PENDING."
            )

        # Advance order to PAYMENT_PENDING if in ORDER_CONFIRMED
        if order.status == OrderState.ORDER_CONFIRMED:
            order_service.transition_state(order_id, OrderState.PAYMENT_PENDING)

        # Amount is strictly derived from the persisted order total
        amount = order.total_amount
        currency = order.currency
        ref = recipient_reference or f"FARMER_{order.farmer_id}"

        # Reuse existing payment record if retrying, or create new unique record
        if existing_payment:
            payment = existing_payment
        else:
            payment_id = f"PAY-{uuid.uuid4().hex[:8].upper()}"
            payment = Payment(
                id=payment_id,
                order_id=order_id,
                amount=amount,
                currency=currency,
                status="PENDING",
                provider_reference=None,
                timestamp=datetime.now(timezone.utc),
            )
            self._payments[payment_id] = payment
            self._order_payments[order_id] = payment_id

        # Invoke payment adapter
        provider_res = await self.adapter.create_payment(
            order_id=order_id,
            amount=amount,
            currency=currency,
            recipient_reference=ref,
        )

        # Process outcome
        if provider_res.get("status") == "SUCCESS":
            payment.status = "SUCCESS"
            payment.provider_reference = provider_res.get("provider_tx_id")
            payment.timestamp = datetime.now(timezone.utc)
            # Advance order state to PAYMENT_CONFIRMED
            order_service.transition_state(order_id, OrderState.PAYMENT_CONFIRMED)
            is_success = True
            msg = f"Payment of {currency} {amount:.2f} confirmed successfully (Ref: {payment.provider_reference})."
        else:
            payment.status = "FAILED"
            payment.provider_reference = provider_res.get("provider_tx_id")
            payment.timestamp = datetime.now(timezone.utc)
            # On failure: order remains in PAYMENT_PENDING
            is_success = False
            error_detail = provider_res.get("error_message", "Payment processing failed.")
            msg = f"Payment failed: {error_detail} Reply 'Retry payment' to attempt processing again."

        # Persist payment to database
        await self._persist_payment(payment, session)

        return payment, is_success, msg

    async def _persist_payment(
        self, payment: Payment, session: Optional[AsyncSession] = None
    ) -> Optional[PaymentModel]:
        """Persist payment record to database."""
        from orca.db.repository import PaymentRepository

        async def _do_persist(s: AsyncSession):
            repo = PaymentRepository(s)
            existing = await repo.get_by_id(payment.id)
            if existing:
                return await repo.update_status(
                    payment.id, payment.status, payment.provider_reference
                )
            model = PaymentModel(
                id=payment.id,
                order_id=payment.order_id,
                amount=payment.amount,
                currency=payment.currency,
                status=payment.status,
                provider_reference=payment.provider_reference,
                timestamp=payment.timestamp,
            )
            return await repo.save(model)

        if session:
            return await _do_persist(session)
        else:
            try:
                from orca.db.session import async_session_factory
                async with async_session_factory() as s:
                    return await _do_persist(s)
            except Exception:
                return None

    def confirm_payment(self, payment_id: str) -> Payment:
        """Mark payment confirmed synchronously for backward compatibility."""
        payment = self._payments.get(payment_id)
        if not payment:
            raise ValueError(f"Payment not found: {payment_id}")

        payment.status = "SUCCESS"
        order_service.transition_state(payment.order_id, OrderState.PAYMENT_CONFIRMED)
        return payment


payment_service = PaymentService()
