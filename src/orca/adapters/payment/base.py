"""Abstract payment provider adapter interface (FR-021 to FR-025)."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BasePaymentAdapter(ABC):
    """Abstract payment provider adapter."""

    @abstractmethod
    async def create_payment(
        self,
        order_id: str,
        amount: float,
        currency: str,
        recipient_reference: str,
    ) -> Dict[str, Any]:
        """Initiate payment transaction with provider."""
        pass

    @abstractmethod
    async def verify_payment(self, provider_tx_id: str) -> Dict[str, Any]:
        """Verify payment status from provider."""
        pass
