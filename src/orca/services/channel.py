"""Channel Service coordinating inbound normalization, deduplication, and outbound delivery."""

import logging
from typing import Any, Dict, Optional

from orca.domain.schemas import InboundMessage, OutboundMessage
from orca.adapters.messaging.factory import get_channel_provider
from orca.agent.orchestrator import orchestrator

logger = logging.getLogger(__name__)


class ChannelService:
    """Decoupled channel service managing webhook intake, deduplication, and conversational agent execution."""

    def __init__(self):
        # In-memory idempotency cache for duplicate webhook/message detection
        # Maps message_id -> cached reply response dict
        self._processed_messages: Dict[str, Dict[str, Any]] = {}

    def clear(self) -> None:
        """Clear message deduplication cache and provider sent message queues."""
        self._processed_messages.clear()

    async def handle_inbound(
        self,
        channel_name: str,
        raw_payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Handle raw inbound channel webhook or API message end-to-end.
        
        1. Normalizes raw payload using the designated ChannelProvider.
        2. Detects and prevents duplicate message processing (idempotency).
        3. Preserves conversation identity across multi-turn exchanges.
        4. Invokes AgentOrchestrator to advance transaction dialogue.
        5. Dispatches normalized outbound reply via ChannelProvider.
        """
        provider = get_channel_provider(channel_name)

        # 1. Parse and normalize payload
        inbound: InboundMessage = await provider.parse_inbound(raw_payload, headers=headers)

        # 2. Duplicate detection / Idempotency check
        if inbound.message_id and inbound.message_id in self._processed_messages:
            logger.info("Duplicate event detected for message_id: %s. Returning cached reply.", inbound.message_id)
            cached = dict(self._processed_messages[inbound.message_id])
            cached["duplicate"] = True
            cached["status"] = "duplicate"
            return cached

        # 3. Conversation Identity Preservation
        conversation_id = f"conv_{inbound.sender_id}"

        # 4. Invoke Agent Orchestrator
        outbound: OutboundMessage = await orchestrator.process_message(
            conversation_id=conversation_id,
            message_text=inbound.text,
            farmer_id=inbound.sender_id,
        )

        # 5. Dispatch outbound message via ChannelProvider
        delivery_receipt = await provider.send_outbound(outbound)

        # 6. Formulate response
        result = {
            "status": "processed",
            "duplicate": False,
            "message_id": inbound.message_id,
            "sender_id": inbound.sender_id,
            "channel": provider.channel_name,
            "reply": outbound.text,
            "metadata": outbound.metadata,
            "delivery_receipt": delivery_receipt,
        }

        # Cache result if message_id is provided
        if inbound.message_id:
            self._processed_messages[inbound.message_id] = result

        return result

    def is_message_processed(self, message_id: str) -> bool:
        """Check whether a given message_id was previously processed."""
        return message_id in self._processed_messages

    def clear_cache(self) -> None:
        """Clear idempotency cache (used for unit testing)."""
        self._processed_messages.clear()


channel_service = ChannelService()
