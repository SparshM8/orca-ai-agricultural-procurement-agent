"""Local demo payment provider adapter.

Replaceable adapter implementing simulated payment processing with configurable outcomes.
"""

import uuid
from typing import Dict, Any, Optional
from orca.adapters.payment.base import BasePaymentAdapter


class DemoPaymentAdapter(BasePaymentAdapter):
    """Local demo payment provider supporting configurable simulated outcomes.

    Enables testing both successful and failed payment paths without real external gateways.
    """

    def __init__(self, default_should_succeed: bool = True):
        self.default_should_succeed = default_should_succeed
        self._transactions: Dict[str, Dict[str, Any]] = {}
        # Allows configuring per-order outcome for testing
        self._order_outcomes: Dict[str, bool] = {}

    def set_order_outcome(self, order_id: str, should_succeed: bool) -> None:
        """Explicitly set payment outcome for a specific order (useful for testing)."""
        self._order_outcomes[order_id] = should_succeed

    async def create_payment(
        self,
        order_id: str,
        amount: float,
        currency: str,
        recipient_reference: str,
    ) -> Dict[str, Any]:
        """Simulate creating a payment transaction with the demo provider."""
        should_succeed = self._order_outcomes.get(order_id, self.default_should_succeed)
        tx_id = f"DEMO_TX_{order_id}_{uuid.uuid4().hex[:6].upper()}"

        tx_record = {
            "provider_tx_id": tx_id,
            "order_id": order_id,
            "amount": amount,
            "currency": currency,
            "recipient_reference": recipient_reference,
            "status": "SUCCESS" if should_succeed else "FAILED",
            "error_message": None if should_succeed else "Demo payment failed: Insufficient funds or provider timeout",
        }
        self._transactions[tx_id] = tx_record
        return tx_record

    async def verify_payment(self, provider_tx_id: str) -> Dict[str, Any]:
        """Verify payment status from demo provider."""
        return self._transactions.get(provider_tx_id, {"status": "NOT_FOUND"})
