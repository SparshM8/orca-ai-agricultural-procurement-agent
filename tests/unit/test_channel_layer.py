"""Unit and integration tests for provider-independent Channel Layer and Webhook Simulator.

Verifies:
1. Inbound message normalization (direct JSON, webhook event, WhatsApp Cloud API envelope)
2. Outbound message normalization & transmission receipts
3. Channel -> Orchestrator flow
4. Conversation identity preservation across multi-turn exchanges
5. Invalid webhook payload rejection (missing sender, empty text, invalid type)
6. Duplicate message / event handling (idempotency cache)
7. HTTP webhook endpoints (/webhook/{channel}, /api/channels/{channel}/messages)
"""

import pytest
from httpx import AsyncClient, ASGITransport

from orca.domain.schemas import InboundMessage, OutboundMessage
from orca.adapters.messaging.demo import DemoChannelAdapter
from orca.adapters.messaging.factory import get_channel_provider, register_channel_provider
from orca.services.channel import ChannelService, channel_service
from orca.api.app import app


# -----------------------------------------------------------------------------
# 1. Inbound Message Normalization
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_inbound_message_normalization_direct_payload():
    """Verify DemoChannelAdapter normalizes standard direct payloads."""
    adapter = DemoChannelAdapter(name="test_direct")
    payload = {
        "sender_id": "+254712345678",
        "text": "I have 50 kg of potatoes in Nairobi",
        "message_id": "msg_direct_001",
    }

    inbound = await adapter.parse_inbound(payload)

    assert isinstance(inbound, InboundMessage)
    assert inbound.sender_id == "+254712345678"
    assert inbound.text == "I have 50 kg of potatoes in Nairobi"
    assert inbound.message_id == "msg_direct_001"
    assert inbound.channel == "test_direct"
    assert inbound.timestamp is not None


@pytest.mark.asyncio
async def test_inbound_message_normalization_webhook_envelope():
    """Verify DemoChannelAdapter normalizes generic webhook event envelopes."""
    adapter = DemoChannelAdapter(name="test_envelope")
    payload = {
        "event": "message_received",
        "data": {
            "from": "farmer_nakuru_02",
            "body": "100 kg of tomatoes ready Friday at 2 PM",
            "id": "evt_webhook_777",
        }
    }

    inbound = await adapter.parse_inbound(payload)

    assert inbound.sender_id == "farmer_nakuru_02"
    assert inbound.text == "100 kg of tomatoes ready Friday at 2 PM"
    assert inbound.message_id == "evt_webhook_777"
    assert inbound.channel == "test_envelope"


@pytest.mark.asyncio
async def test_inbound_message_normalization_whatsapp_cloud_api_format():
    """Verify DemoChannelAdapter normalizes Meta / WhatsApp Cloud API style envelopes."""
    adapter = DemoChannelAdapter(name="whatsapp")
    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "WHATSAPP_BUSINESS_ACCOUNT_ID",
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "messages": [
                                {
                                    "from": "+254700000000",
                                    "id": "wamid.HBgLMjU0NzAwMDAwMDAwFQIAEhggQ0RFRjEy",
                                    "text": {
                                        "body": "I want to sell 75 kg onions in Springfield"
                                    },
                                    "type": "text"
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }

    inbound = await adapter.parse_inbound(payload)

    assert inbound.sender_id == "+254700000000"
    assert inbound.text == "I want to sell 75 kg onions in Springfield"
    assert inbound.message_id == "wamid.HBgLMjU0NzAwMDAwMDAwFQIAEhggQ0RFRjEy"
    assert inbound.channel == "whatsapp"


# -----------------------------------------------------------------------------
# 2. Outbound Message Normalization
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_outbound_message_normalization_and_queue():
    """Verify DemoChannelAdapter delivers OutboundMessage and generates transmission receipts."""
    adapter = DemoChannelAdapter(name="test_outbound")
    adapter.clear()

    outbound = OutboundMessage(
        recipient_id="+254712345678",
        channel="test_outbound",
        text="Thank you! Authoritative rate: USD 0.40 per kg.",
    )

    receipt = await adapter.send_outbound(outbound)

    assert receipt["status"] == "delivered"
    assert receipt["recipient_id"] == "+254712345678"
    assert receipt["channel"] == "test_outbound"
    assert "Authoritative rate" in receipt["preview"]

    # In-memory queue assertions
    sent = adapter.get_sent_messages(recipient_id="+254712345678")
    assert len(sent) == 1
    assert sent[0].text == "Thank you! Authoritative rate: USD 0.40 per kg."
    assert sent[0].message_id is not None


# -----------------------------------------------------------------------------
# 3. Channel -> Orchestrator Flow
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_channel_to_orchestrator_flow():
    """Verify ChannelService normalizes inbound message and executes AgentOrchestrator."""
    service = ChannelService()
    payload = {
        "sender_id": "farmer_flow_test_01",
        "text": "I have 40 kg of potatoes in Nairobi available tomorrow at 10 AM",
        "message_id": "msg_flow_001",
    }

    result = await service.handle_inbound("simulator", payload)

    assert result["status"] == "processed"
    assert result["duplicate"] is False
    assert result["sender_id"] == "farmer_flow_test_01"
    assert "potato" in result["reply"].lower()
    assert "USD 0.40" in result["reply"]
    assert "USD 16.00" in result["reply"]
    assert "tomorrow at 10:00 AM" in result["reply"]
    assert result["delivery_receipt"]["status"] == "delivered"


# -----------------------------------------------------------------------------
# 4. Conversation Identity Preservation
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_conversation_identity_preservation():
    """Verify conversation context and history are preserved across multi-turn channel exchanges."""
    service = ChannelService()
    farmer_id = "farmer_identity_test_09"

    # Turn 1: Partial offer (missing pickup location)
    t1_payload = {
        "sender_id": farmer_id,
        "text": "I have 50 kg of potatoes available tomorrow at 10 AM",
        "message_id": "turn_1_msg",
    }
    r1 = await service.handle_inbound("web_chat", t1_payload)
    assert r1["status"] == "processed"
    # Agent should ask for clarification regarding pickup location
    assert "location" in r1["reply"].lower() or "where" in r1["reply"].lower()

    # Turn 2: Provide pickup location
    t2_payload = {
        "sender_id": farmer_id,
        "text": "In Nairobi",
        "message_id": "turn_2_msg",
    }
    r2 = await service.handle_inbound("web_chat", t2_payload)
    assert r2["status"] == "processed"
    # Identity preserved: potatoes, 50 kg, and Nairobi are combined into quote summary
    assert "potato" in r2["reply"].lower()
    assert "50" in r2["reply"]
    assert "Nairobi" in r2["reply"]
    assert "tomorrow at 10:00 AM" in r2["reply"]
    assert "Confirm" in r2["reply"]

    # Turn 3: Confirm order
    t3_payload = {
        "sender_id": farmer_id,
        "text": "Confirm",
        "message_id": "turn_3_msg",
    }
    r3 = await service.handle_inbound("web_chat", t3_payload)
    assert r3["status"] == "processed"
    assert "confirmed" in r3["reply"].lower() or "order id" in r3["reply"].lower()


# -----------------------------------------------------------------------------
# 5. Invalid Webhook Payload Rejection
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_invalid_webhook_payload_rejection():
    """Verify channel adapter rejects invalid payloads with descriptive ValueErrors."""
    adapter = DemoChannelAdapter(name="validation_test")

    # A: Missing sender identifier
    with pytest.raises(ValueError, match="missing required sender identifier"):
        await adapter.parse_inbound({"text": "50 kg potatoes"})

    # B: Empty / whitespace-only text
    with pytest.raises(ValueError, match="cannot be empty"):
        await adapter.parse_inbound({"sender_id": "+254700000000", "text": "   "})

    # C: None text
    with pytest.raises(ValueError, match="cannot be empty"):
        await adapter.parse_inbound({"sender_id": "+254700000000", "text": None})

    # D: Non-dict payload
    with pytest.raises(ValueError, match="must be a JSON object"):
        await adapter.parse_inbound(["not", "a", "dict"])


# -----------------------------------------------------------------------------
# 6. Duplicate Message / Event Handling (Idempotency)
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_duplicate_message_handling():
    """Verify duplicate webhook events with same message_id return cached reply without re-executing orchestrator."""
    service = ChannelService()
    service.clear_cache()

    duplicate_msg_id = "evt_dedup_unique_9999"
    payload = {
        "sender_id": "farmer_dedup_01",
        "text": "I have 30 kg of potatoes in Nairobi",
        "message_id": duplicate_msg_id,
    }

    # First delivery: processed normally
    res1 = await service.handle_inbound("simulator", payload)
    assert res1["status"] == "processed"
    assert res1["duplicate"] is False
    assert "potato" in res1["reply"].lower()

    # Second delivery with SAME message_id: must be recognized as duplicate
    res2 = await service.handle_inbound("simulator", payload)
    assert res2["status"] == "duplicate"
    assert res2["duplicate"] is True
    assert res2["reply"] == res1["reply"]
    assert res2["message_id"] == duplicate_msg_id


# -----------------------------------------------------------------------------
# 7. HTTP Webhook Endpoints Integration
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_webhook_endpoints_integration():
    """Verify /webhook/{channel_name} and /api/channels/{channel_name}/webhook routes."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Post to /webhook/simulator
        payload = {
            "sender_id": "+254788112233",
            "text": "I have 25 kg of potatoes in Nairobi available tomorrow at 10 AM",
            "message_id": "http_msg_001",
        }
        res = await client.post("/webhook/simulator", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "processed"
        assert data["channel"] == "simulator"
        assert "potato" in data["reply"].lower()

        # 2. Verify duplicate via HTTP
        res_dup = await client.post("/webhook/simulator", json=payload)
        assert res_dup.status_code == 200
        dup_data = res_dup.json()
        assert dup_data["status"] == "duplicate"
        assert dup_data["duplicate"] is True

        # 3. Inspect outbound messages on channel
        msg_res = await client.get("/api/channels/simulator/messages?recipient_id=%2B254788112233")
        assert msg_res.status_code == 200
        msg_data = msg_res.json()
        assert msg_data["channel"] == "simulator"
        assert msg_data["count"] >= 1

        # 4. Invalid payload -> 400 Bad Request
        bad_res = await client.post("/webhook/simulator", json={"sender_id": "", "text": "no sender"})
        assert bad_res.status_code == 400
        assert "missing required sender" in bad_res.json()["detail"].lower()
