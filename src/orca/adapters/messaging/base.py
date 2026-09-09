"""Abstract messaging adapter interface (FR-032, FR-033).

Decouples agent and business logic from specific channels (WhatsApp, Webhook, SMS).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from orca.domain.schemas import InboundMessage, OutboundMessage


class BaseMessagingAdapter(ABC):
    """Abstract messaging adapter."""

    @abstractmethod
    async def parse_inbound(self, payload: Dict[str, Any]) -> InboundMessage:
        """Parse channel-specific webhook payload into normalized InboundMessage."""
        pass

    @abstractmethod
    async def send_outbound(self, message: OutboundMessage) -> bool:
        """Deliver normalized OutboundMessage through channel provider."""
        pass
