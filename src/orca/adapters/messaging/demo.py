"""Demo / Local channel adapter simulating web chat, webhook events, and console interactions."""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from orca.domain.schemas import InboundMessage, OutboundMessage
from orca.adapters.messaging.base import ChannelProvider


class DemoChannelAdapter(ChannelProvider):
    """Local, provider-independent channel adapter for testing, development, and Farmer Chat.
    
    Supports:
    - Direct JSON payloads ({"sender_id": "...", "text": "...", "message_id": "..."})
    - Webhook envelopes ({"event": "message_received", "data": {"from": "...", "body": "..."}})
    - Meta/WhatsApp style mock envelope ({"entry": [{"changes": [{"value": {"messages": [...]}}]}]})
    - In-memory outbound message queue for verification and observability.
    """

    def __init__(self, name: str = "web_chat"):
        self._name = name
        self.sent_messages: List[OutboundMessage] = []
        self.delivery_receipts: List[Dict[str, Any]] = []

    @property
    def channel_name(self) -> str:
        return self._name

    async def parse_inbound(
        self, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None
    ) -> InboundMessage:
        """Parse, validate, and normalize incoming webhook/request payload."""
        if not isinstance(payload, dict):
            raise ValueError("Webhook payload must be a JSON object.")

        sender_id = None
        text = None
        message_id = None
        channel_name = payload.get("channel") or self._name

        # Format A: Meta / WhatsApp Cloud API style envelope
        if "entry" in payload and isinstance(payload["entry"], list) and payload["entry"]:
            try:
                change = payload["entry"][0]["changes"][0]["value"]
                messages = change.get("messages", [])
                if messages:
                    msg = messages[0]
                    sender_id = msg.get("from")
                    message_id = msg.get("id")
                    if "text" in msg and isinstance(msg["text"], dict):
                        text = msg["text"].get("body")
                    elif isinstance(msg.get("text"), str):
                        text = msg.get("text")
            except (IndexError, KeyError, TypeError):
                pass

        # Format B: Generic webhook envelope {"event": ..., "data": {"from": ..., "body": ...}}
        if not sender_id and "data" in payload and isinstance(payload["data"], dict):
            data = payload["data"]
            sender_id = data.get("from") or data.get("sender_id") or data.get("sender")
            text = data.get("body") or data.get("text") or data.get("message")
            message_id = data.get("id") or data.get("message_id")

        # Format C: Direct payload {"sender_id": ..., "text": ..., "message_id": ...}
        if not sender_id:
            sender_id = payload.get("sender_id") or payload.get("from") or payload.get("farmer_id")
        if not text:
            text = payload.get("text") or payload.get("body") or payload.get("message")
        if not message_id:
            message_id = payload.get("message_id") or payload.get("id")

        # Validation
        if not sender_id or not str(sender_id).strip():
            raise ValueError("Payload missing required sender identifier ('sender_id' or 'from').")

        if text is None or not str(text).strip():
            raise ValueError("Payload message content ('text' or 'body') cannot be empty.")

        norm_sender = str(sender_id).strip()
        norm_text = str(text).strip()
        norm_msg_id = str(message_id).strip() if message_id else None

        return InboundMessage(
            message_id=norm_msg_id,
            sender_id=norm_sender,
            channel=channel_name,
            text=norm_text,
            timestamp=datetime.now(timezone.utc),
            raw_payload=payload,
            metadata={"headers": headers or {}},
        )

    async def send_outbound(self, message: OutboundMessage) -> Dict[str, Any]:
        """Deliver normalized outbound message into the in-memory log."""
        outbound_copy = message.model_copy()
        if not outbound_copy.message_id:
            outbound_copy.message_id = f"out_{uuid.uuid4().hex[:10]}"

        self.sent_messages.append(outbound_copy)

        receipt = {
            "status": "delivered",
            "message_id": outbound_copy.message_id,
            "recipient_id": outbound_copy.recipient_id,
            "channel": self._name,
            "timestamp": outbound_copy.timestamp.isoformat(),
            "preview": outbound_copy.text[:50] + ("..." if len(outbound_copy.text) > 50 else ""),
        }
        self.delivery_receipts.append(receipt)
        return receipt

    def get_sent_messages(self, recipient_id: Optional[str] = None) -> List[OutboundMessage]:
        """Retrieve sent outbound messages, optionally filtered by recipient."""
        if recipient_id:
            return [m for m in self.sent_messages if m.recipient_id == recipient_id]
        return list(self.sent_messages)

    def clear(self) -> None:
        """Clear transmission queues (used in unit testing)."""
        self.sent_messages.clear()
        self.delivery_receipts.clear()
