"""Abstract messaging and channel provider interface (FR-032, FR-033).

Decouples conversational agent, pricing, and fulfillment state machines
from specific channels (Web Chat, Webhook Simulator, WhatsApp, SMS).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from orca.domain.schemas import InboundMessage, OutboundMessage


class ChannelProvider(ABC):
    """Abstract channel provider interface for multi-channel messaging."""

    @property
    @abstractmethod
    def channel_name(self) -> str:
        """Normalized unique name of this channel (e.g., 'web_chat', 'demo', 'whatsapp')."""
        pass

    @abstractmethod
    async def parse_inbound(
        self, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None
    ) -> InboundMessage:
        """Parse and normalize channel-specific webhook/request payload into InboundMessage.

        Raises:
            ValueError: If payload is malformed, missing mandatory fields, or invalid.
        """
        pass

    @abstractmethod
    async def send_outbound(self, message: OutboundMessage) -> Dict[str, Any]:
        """Deliver normalized OutboundMessage through channel transport.

        Returns:
            Dict containing delivery receipt status and channel transmission metadata.
        """
        pass


# Backward compatibility alias
BaseMessagingAdapter = ChannelProvider
