"""Payment workflow and status tracking service (FR-021 to FR-025)."""

import uuid
from typing import Dict, Optional
from orca.domain.models import Payment
from orca.domain.state_machine import OrderState
from orca.services.order import order_service


class PaymentService:
    """Manages payment initiation and status updates."""

    def __init__(self):
        self._payments: Dict[str, Payment] = {}

    def create_payment_request(
        self,
        order_id: str,
        amount: float,
        currency: str = "USD",
        provider_reference: Optional[str] = None,
    ) -> Payment:
        """Create payment record and transition order to PAYMENT_PENDING.

        FR-022: Payment amount derived from validated order.
        FR-025: Sandbox simulation supported.
        """
        payment_id = f"PAY-{uuid.uuid4().hex[:8].upper()}"
        payment = Payment(
            id=payment_id,
            order_id=order_id,
            amount=amount,
            currency=currency,
            status="PENDING",
            provider_reference=provider_reference or f"SIM_TX_{payment_id}",
        )
        self._payments[payment_id] = payment

        # Transition order to PAYMENT_PENDING
        order_service.transition_state(order_id, OrderState.PAYMENT_PENDING)
        return payment

    def confirm_payment(self, payment_id: str) -> Payment:
        """Mark payment confirmed and advance order state to PAYMENT_CONFIRMED."""
        payment = self._payments.get(payment_id)
        if not payment:
            raise ValueError(f"Payment not found: {payment_id}")

        payment.status = "SUCCESS"
        order_service.transition_state(payment.order_id, OrderState.PAYMENT_CONFIRMED)
        return payment

    def get_payment_for_order(self, order_id: str) -> Optional[Payment]:
        """Find payment associated with order."""
        for p in self._payments.values():
            if p.order_id == order_id:
                return p
        return None


payment_service = PaymentService()
