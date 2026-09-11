"""Unit tests for the Agent Trace / Decision Audit observability layer.

Verifies:
1. Trace creation on successful produce offer with state progression
2. Trace creation on UNKNOWN intent with zero tool execution
3. Trace captures Gemini fallback and sanitized fallback reasons
4. Trace captures read-only tool execution (e.g. order status inquiry)
5. Trace captures payment authorization with state transition
6. Trace captures tool execution failures
7. Multi-turn sequential state_before and state_after correctness
8. Sensitive field and secret sanitization across text, payloads, and traces
9. Complete trace is NOT exposed in farmer-facing outbound metadata
10. Only trace_id is present in outbound metadata
11. Trace is finalized on early returns (unsupported produce, details pending, missing order ID)
12. Trace history in DialogueContext is strictly bounded to max 50
13. Observational purity: trace mutation/clearing never affects domain state
"""

import re
import pytest
from unittest.mock import AsyncMock, patch

from orca.domain.intents import (
    IntentType,
    ConversationalIntentResult,
    ExtractedEntities,
)
from orca.domain.state_machine import OrderState
from orca.domain.trace import (
    AgentTrace,
    sanitize_text,
    sanitize_payload,
)
from orca.agent.intent.rule_based_classifier import RuleBasedIntentClassifier
from orca.agent.intent.resilient_classifier import ResilientIntentClassifier
from orca.agent.orchestrator import AgentOrchestrator, DialogueContext
from orca.services.payment import payment_service


@pytest.fixture
def orchestrator():
    """AgentOrchestrator configured for deterministic testing."""
    return AgentOrchestrator(auto_create_collection_task=True)


# =============================================================================
# 1. Successful Offer Trace
# =============================================================================
@pytest.mark.asyncio
async def test_trace_successful_offer(orchestrator: AgentOrchestrator):
    """Verify trace created on complete offer with evaluate_produce_offer tool execution."""
    conv_id = "test_trace_offer_01"
    msg = "I have 50 kg of potatoes in Springfield tomorrow at 10 AM"

    outbound = await orchestrator.process_message(
        conversation_id=conv_id,
        message_text=msg,
        farmer_id="farmer_trace_1",
    )

    ctx = orchestrator.get_context(conv_id)
    assert ctx.last_trace is not None
    assert isinstance(ctx.last_trace, AgentTrace)
    trace = ctx.last_trace

    assert trace.conversation_id == conv_id
    assert trace.sender_id == "farmer_trace_1"
    assert trace.detected_intent == IntentType.OFFER_PRODUCE.value
    assert trace.selected_tool == "evaluate_produce_offer"
    assert trace.tool_success is True
    assert trace.state_before == OrderState.OFFER_RECEIVED.value
    assert trace.state_after == OrderState.AWAITING_FARMER_CONFIRMATION.value
    offer_entity = trace.extracted_entities.get("offer", {})
    assert offer_entity.get("produce_type") == "potato"
    assert offer_entity.get("quantity") == 50.0
    assert offer_entity.get("unit") == "kg"
    assert offer_entity.get("pickup_location") == "Springfield"
    assert trace.error is None
    assert trace.trace_id.startswith("trc_")


# =============================================================================
# 2. Unknown Intent Trace
# =============================================================================
@pytest.mark.asyncio
async def test_trace_unknown_intent(orchestrator: AgentOrchestrator):
    """Verify UNKNOWN intent turn creates trace with zero tool execution."""
    conv_id = "test_trace_unknown_01"
    msg = "What is the airspeed velocity of an unladen swallow?"

    outbound = await orchestrator.process_message(
        conversation_id=conv_id,
        message_text=msg,
        farmer_id="farmer_trace_2",
    )

    ctx = orchestrator.get_context(conv_id)
    assert ctx.last_trace is not None
    trace = ctx.last_trace

    assert trace.detected_intent == IntentType.UNKNOWN.value
    assert trace.selected_tool is None
    assert not trace.tool_arguments
    assert trace.tool_success is None
    assert trace.state_before == trace.state_after
    assert outbound.metadata.get("intent") == "UNKNOWN"
    assert "ORCA" in outbound.text


# =============================================================================
# 3. Gemini Fallback Trace
# =============================================================================
@pytest.mark.asyncio
async def test_trace_gemini_fallback():
    """Verify trace records fallback_occurred=True, classifier_used, and sanitized reason."""
    mock_primary = AsyncMock()
    mock_primary.classify.side_effect = RuntimeError(
        "Gemini 429 quota exhausted with api_key=AIzaSyD_SECRET_KEY_12345"
    )

    fallback_classifier = RuleBasedIntentClassifier()
    resilient_classifier = ResilientIntentClassifier(
        primary=mock_primary,
        fallback=fallback_classifier,
    )
    orch = AgentOrchestrator(
        intent_classifier=resilient_classifier,
        auto_create_collection_task=True,
    )

    conv_id = "test_trace_fallback_01"
    msg = "I have 50 kg of potatoes in Springfield tomorrow at 10 AM"

    outbound = await orch.process_message(
        conversation_id=conv_id,
        message_text=msg,
        farmer_id="farmer_trace_3",
    )

    ctx = orch.get_context(conv_id)
    assert ctx.last_trace is not None
    trace = ctx.last_trace

    assert trace.fallback_occurred is True
    assert trace.classifier_used == "RuleBasedIntentClassifier"
    assert trace.fallback_reason is not None
    # Verify the secret in the exception was sanitized
    assert "AIzaSyD_SECRET_KEY_12345" not in trace.fallback_reason
    assert "[REDACTED" in trace.fallback_reason
    assert trace.detected_intent == IntentType.OFFER_PRODUCE.value
    assert trace.tool_success is True


# =============================================================================
# 4. Read-Only Status Inquiry Trace
# =============================================================================
@pytest.mark.asyncio
async def test_trace_read_only_status_inquiry(orchestrator: AgentOrchestrator):
    """Verify read-only query executes get_order_status and preserves state before/after."""
    conv_id = "test_trace_status_01"

    # Set up confirmed order
    await orchestrator.process_message(
        conv_id, "I have 50 kg of potatoes in Springfield tomorrow at 10 AM", "farmer_trace_4"
    )
    await orchestrator.process_message(conv_id, "Confirm", "farmer_trace_4")

    ctx = orchestrator.get_context(conv_id)
    state_prior_to_inquiry = ctx.state.value

    # Inquire about order status
    outbound = await orchestrator.process_message(
        conv_id, "What's happening with my order?", "farmer_trace_4"
    )

    trace = ctx.last_trace
    assert trace is not None
    assert trace.detected_intent == IntentType.INQUIRE_ORDER_STATUS.value
    assert trace.selected_tool == "get_order_status"
    assert trace.tool_success is True
    assert trace.state_before == state_prior_to_inquiry
    assert trace.state_after == state_prior_to_inquiry
    assert trace.tool_arguments.get("order_id") == ctx.current_order_id


# =============================================================================
# 5. Successful Payment Trace
# =============================================================================
@pytest.mark.asyncio
async def test_trace_successful_payment(orchestrator: AgentOrchestrator):
    """Verify payment authorization records initiate_order_payment and state progression."""
    conv_id = "test_trace_payment_01"

    await orchestrator.process_message(
        conv_id, "I have 50 kg of potatoes in Springfield tomorrow at 10 AM", "farmer_trace_5"
    )
    await orchestrator.process_message(conv_id, "Confirm", "farmer_trace_5")

    ctx = orchestrator.get_context(conv_id)
    assert ctx.state == OrderState.PAYMENT_PENDING

    # Authorize payment
    outbound = await orchestrator.process_message(conv_id, "Okay, pay for it.", "farmer_trace_5")

    trace = ctx.last_trace
    assert trace is not None
    assert trace.detected_intent == IntentType.AUTHORIZE_PAYMENT.value
    assert trace.tool_success is True
    assert trace.state_before == OrderState.PAYMENT_PENDING.value
    assert trace.state_after == OrderState.COLLECTION_PENDING.value


# =============================================================================
# 6. Tool Failure Trace
# =============================================================================
@pytest.mark.asyncio
async def test_trace_tool_failure(orchestrator: AgentOrchestrator):
    """Verify tool failure records tool_success=False and captures tool_error."""
    conv_id = "test_trace_tool_fail_01"

    await orchestrator.process_message(
        conv_id, "I have 50 kg of potatoes in Springfield tomorrow at 10 AM", "farmer_trace_6"
    )
    await orchestrator.process_message(conv_id, "Confirm", "farmer_trace_6")

    ctx = orchestrator.get_context(conv_id)
    order_id = ctx.current_order_id

    # Mock payment adapter to simulate payment failure
    mock_payment_rec = payment_service.get_payment_for_order(order_id)
    with patch.object(
        payment_service,
        "initiate_order_payment",
        new_callable=AsyncMock,
        return_value=(mock_payment_rec, False, "Bank settlement timeout"),
    ):
        outbound = await orchestrator.process_message(conv_id, "Pay", "farmer_trace_6")

    trace = ctx.last_trace
    assert trace is not None
    assert trace.tool_success is False
    assert trace.tool_error == "Bank settlement timeout"
    assert trace.state_after == OrderState.PAYMENT_PENDING.value


# =============================================================================
# 7. Multi-Turn State Before / After Progression
# =============================================================================
@pytest.mark.asyncio
async def test_trace_state_before_after_correctness(orchestrator: AgentOrchestrator):
    """Verify sequential multi-turn state_before and state_after across the lifecycle."""
    conv_id = "test_trace_lifecycle_01"

    # Turn 1: Offer
    await orchestrator.process_message(
        conv_id, "I have 50 kg of potatoes in Springfield tomorrow at 10 AM", "farmer_seq"
    )
    ctx = orchestrator.get_context(conv_id)
    assert len(ctx.traces) == 1
    assert ctx.traces[0].state_before == OrderState.OFFER_RECEIVED.value
    assert ctx.traces[0].state_after == OrderState.AWAITING_FARMER_CONFIRMATION.value

    # Turn 2: Confirm
    await orchestrator.process_message(conv_id, "Confirm", "farmer_seq")
    assert len(ctx.traces) == 2
    assert ctx.traces[1].state_before == OrderState.AWAITING_FARMER_CONFIRMATION.value
    assert ctx.traces[1].state_after == OrderState.PAYMENT_PENDING.value

    # Turn 3: Inquire (Read-only)
    await orchestrator.process_message(conv_id, "What's happening with my order?", "farmer_seq")
    assert len(ctx.traces) == 3
    assert ctx.traces[2].state_before == OrderState.PAYMENT_PENDING.value
    assert ctx.traces[2].state_after == OrderState.PAYMENT_PENDING.value

    # Turn 4: Pay
    await orchestrator.process_message(conv_id, "Pay now", "farmer_seq")
    assert len(ctx.traces) == 4
    assert ctx.traces[3].state_before == OrderState.PAYMENT_PENDING.value
    assert ctx.traces[3].state_after == OrderState.COLLECTION_PENDING.value


# =============================================================================
# 8. Secret & Sensitive Field Sanitization
# =============================================================================
def test_trace_secret_sanitization_helpers():
    """Verify recursive sanitization of credentials, keys, and long text."""
    # Pattern sanitization
    raw_text = (
        "Check this: api_key='sk-test-1234567890abcdef' and "
        "AIzaSyD_test1234567890abcdefghijklmnopqrst and "
        "AQ.xyz1234567890abcdefghijklmnopqrstuvwxyz and "
        "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    )
    sanitized = sanitize_text(raw_text)
    assert "sk-test-1234567890abcdef" not in sanitized
    assert "AIzaSyD" not in sanitized
    assert "AQ.xyz" not in sanitized
    assert "eyJhbGciOi" not in sanitized
    assert "[REDACTED" in sanitized

    # Length bounding
    long_str = "x" * 500
    bounded = sanitize_text(long_str, max_len=200)
    assert len(bounded) < 250
    assert "... [truncated]" in bounded

    # Recursive payload sanitization
    payload = {
        "produce": "potatoes",
        "quantity": 50,
        "api_key": "SUPER_SECRET_KEY",
        "authorization": "Bearer secret_token",
        "nested": {
            "client_secret": "password123",
            "safe_field": "visible_value",
            "token": "token_abc",
        },
        "items": [{"password": "secret", "name": "item1"}],
    }
    sanitized_payload = sanitize_payload(payload)
    assert sanitized_payload["api_key"] == "[REDACTED]"
    assert sanitized_payload["authorization"] == "[REDACTED]"
    assert sanitized_payload["nested"]["client_secret"] == "[REDACTED]"
    assert sanitized_payload["nested"]["token"] == "[REDACTED]"
    assert sanitized_payload["nested"]["safe_field"] == "visible_value"
    assert sanitized_payload["items"][0]["password"] == "[REDACTED]"
    assert sanitized_payload["items"][0]["name"] == "item1"


@pytest.mark.asyncio
async def test_trace_inbound_input_sanitization(orchestrator: AgentOrchestrator):
    """Verify sensitive input in farmer message is redacted in trace."""
    conv_id = "test_trace_secret_input_01"
    msg = "I have 50 kg of potatoes in Springfield with api_key=AIzaSyD1234567890123456789012345678901 tomorrow at 10 AM"

    await orchestrator.process_message(
        conversation_id=conv_id,
        message_text=msg,
        farmer_id="farmer_secret",
    )

    ctx = orchestrator.get_context(conv_id)
    assert ctx.last_trace is not None
    assert "AIzaSyD1234567890123456789012345678901" not in ctx.last_trace.input_text
    assert "[REDACTED" in ctx.last_trace.input_text


# =============================================================================
# 9 & 10. Outbound Metadata Privacy (trace_id only)
# =============================================================================
@pytest.mark.asyncio
async def test_trace_not_exposed_in_outbound_metadata(orchestrator: AgentOrchestrator):
    """Verify internal trace diagnostics are NOT leaked into outbound metadata."""
    conv_id = "test_trace_privacy_01"
    outbound = await orchestrator.process_message(
        conv_id, "I have 50 kg of potatoes in Springfield tomorrow at 10 AM", "farmer_priv"
    )

    ctx = orchestrator.get_context(conv_id)
    assert outbound.metadata is not None

    # trace_id must be present
    assert "trace_id" in outbound.metadata
    assert outbound.metadata["trace_id"] == ctx.last_trace.trace_id

    # Internal observability fields must NOT be in outbound metadata
    forbidden_keys = [
        "classifier_used",
        "fallback_occurred",
        "fallback_reason",
        "tool_arguments",
        "tool_error",
        "tool_success",
        "selected_tool",
        "input_text",
        "response_path",
        "detected_intent",
    ]
    for key in forbidden_keys:
        assert key not in outbound.metadata, f"Forbidden internal key '{key}' found in outbound metadata"


# =============================================================================
# 11. Trace Finalization on Early Returns
# =============================================================================
@pytest.mark.asyncio
async def test_trace_finalized_on_early_returns(orchestrator: AgentOrchestrator):
    """Verify traces are finalized even when turn returns early (unsupported crop, missing info)."""
    # 1. Unsupported crop
    conv_id_1 = "test_trace_early_crop"
    await orchestrator.process_message(conv_id_1, "I have 50 kg of pineapples in Springfield tomorrow at 10 AM", "farmer_e1")
    ctx_1 = orchestrator.get_context(conv_id_1)
    assert ctx_1.last_trace is not None
    assert ctx_1.last_trace.detected_intent == IntentType.OFFER_PRODUCE.value
    assert ctx_1.last_trace.state_before == OrderState.OFFER_RECEIVED.value
    assert ctx_1.last_trace.state_after == OrderState.DETAILS_PENDING.value

    # 2. Incomplete offer (missing location and timing)
    conv_id_2 = "test_trace_early_incomplete"
    await orchestrator.process_message(conv_id_2, "I have 50 kg of potatoes", "farmer_e2")
    ctx_2 = orchestrator.get_context(conv_id_2)
    assert ctx_2.last_trace is not None
    assert ctx_2.last_trace.state_after == OrderState.DETAILS_PENDING.value

    # 3. Status inquiry without order in context
    conv_id_3 = "test_trace_early_no_order"
    await orchestrator.process_message(conv_id_3, "What's happening with my order?", "farmer_e3")
    ctx_3 = orchestrator.get_context(conv_id_3)
    assert ctx_3.last_trace is not None
    assert ctx_3.last_trace.detected_intent == IntentType.INQUIRE_ORDER_STATUS.value


# =============================================================================
# 12. Bounded Trace History in DialogueContext
# =============================================================================
@pytest.mark.asyncio
async def test_trace_history_bounded(orchestrator: AgentOrchestrator):
    """Verify DialogueContext retains at most 50 traces and prunes older ones cleanly."""
    conv_id = "test_trace_bounded_01"

    # Send 55 turns
    for i in range(55):
        await orchestrator.process_message(
            conv_id, f"Inquiry number {i}", "farmer_bound"
        )

    ctx = orchestrator.get_context(conv_id)
    assert len(ctx.traces) == 50
    assert ctx.last_trace is not None
    assert "Inquiry number 54" in ctx.last_trace.input_text
    assert ctx.traces[-1] == ctx.last_trace

    # Test get_recent_traces helper
    recent_5 = ctx.get_recent_traces(limit=5)
    assert len(recent_5) == 5
    assert recent_5[-1] == ctx.last_trace


# =============================================================================
# 13. Observational Purity
# =============================================================================
@pytest.mark.asyncio
async def test_trace_observational_purity(orchestrator: AgentOrchestrator):
    """Verify clearing or corrupting trace history has zero effect on business state or order creation."""
    conv_id = "test_trace_purity_01"

    # Turn 1: Offer
    await orchestrator.process_message(
        conv_id, "I have 50 kg of potatoes in Springfield tomorrow at 10 AM", "farmer_purity"
    )

    ctx = orchestrator.get_context(conv_id)
    assert ctx.state == OrderState.AWAITING_FARMER_CONFIRMATION

    # Deliberately clear/corrupt observational traces
    ctx.traces.clear()
    ctx.last_trace = None

    # Turn 2: Confirm order
    outbound = await orchestrator.process_message(conv_id, "Confirm", "farmer_purity")

    # Verify authoritative domain state and order creation were completely unaffected
    assert ctx.state == OrderState.PAYMENT_PENDING
    assert ctx.current_order_id is not None
    assert "Order Confirmed!" in outbound.text
    # And a new trace was generated for this turn
    assert ctx.last_trace is not None
    assert ctx.last_trace.state_before == OrderState.AWAITING_FARMER_CONFIRMATION.value
    assert ctx.last_trace.state_after == OrderState.PAYMENT_PENDING.value
