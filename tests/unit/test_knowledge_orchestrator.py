"""Unit tests for Phase 4: Business Knowledge Orchestrator Integration.

Verifies:
1. Business knowledge queries ("What is ORCA?", "How does ORCA work?", "What is your procurement model?")
2. Operational FAQ queries ("How does pickup work?", "How do payments work?", "What happens if the runner is late?", "How is total billing calculated?")
3. Ambiguous knowledge queries ("What are your operational policies?") return concise clarification with category overview
4. No-match safety: unsupported questions clearly state absence of grounded info with zero hallucinations
5. Transaction protection: order status, payment status, runner status, pickup reschedule, confirm, pay, and produce offers are never intercepted
6. State immutability: knowledge requests never advance/regress state or mutate orders/payments/collections
7. AgentTrace records knowledge requests safely with selected tool and zero secrets
8. Multi-turn transaction flow continues seamlessly when interspersed with knowledge inquiries
"""

import pytest
import uuid
import re

from orca.domain.state_machine import OrderState
from orca.domain.intents import IntentType
from orca.agent.orchestrator import AgentOrchestrator
from orca.services.order import order_service
from orca.services.payment import payment_service
from orca.services.collection import collection_service


@pytest.fixture
def orchestrator():
    """Create fresh AgentOrchestrator instance for testing."""
    return AgentOrchestrator(auto_process_payment=False)


# =============================================================================
# A. Business Knowledge Inquiries
# =============================================================================

@pytest.mark.asyncio
async def test_business_knowledge_inquiries(orchestrator: AgentOrchestrator):
    """A. Verify grounded answers for core business info queries."""
    queries = [
        "What is ORCA?",
        "How does ORCA work?",
        "What is your procurement model?",
    ]

    for q in queries:
        conv_id = f"conv_bk_{uuid.uuid4().hex[:8]}"
        res = await orchestrator.process_message(conv_id, q, farmer_id="farmer_test")

        assert res.metadata["matched"] is True
        assert res.metadata["answerable"] is True
        assert res.metadata["intent"] == IntentType.INQUIRE_BUSINESS_INFO.value
        assert "direct digital agricultural procurement platform" in res.text.lower() or "procurement model" in res.text.lower()
        assert res.metadata["state"] == OrderState.OFFER_RECEIVED.value


# =============================================================================
# B. Operational FAQ Inquiries
# =============================================================================

@pytest.mark.asyncio
async def test_operational_faq_pickup_work(orchestrator: AgentOrchestrator):
    """B1. Verify 'How does pickup work?' returns grounded logistics knowledge."""
    conv_id = f"conv_faq_{uuid.uuid4().hex[:8]}"
    res = await orchestrator.process_message(conv_id, "How does pickup work?", farmer_id="farmer_test")

    assert res.metadata["matched"] is True
    assert res.metadata["answerable"] is True
    assert res.metadata["intent"] == IntentType.INQUIRE_OPERATIONAL_FAQ.value
    assert "runner" in res.text.lower()
    assert "farm-gate" in res.text.lower()


@pytest.mark.asyncio
async def test_operational_faq_payments_work(orchestrator: AgentOrchestrator):
    """B2. Verify 'How do payments work?' returns grounded payment policy knowledge."""
    conv_id = f"conv_faq_{uuid.uuid4().hex[:8]}"
    res = await orchestrator.process_message(conv_id, "How do payments work?", farmer_id="farmer_test")

    assert res.metadata["matched"] is True
    assert res.metadata["answerable"] is True
    assert res.metadata["intent"] == IntentType.INQUIRE_OPERATIONAL_FAQ.value
    assert "payment" in res.text.lower()
    assert "electronic" in res.text.lower() or "payout" in res.text.lower()


@pytest.mark.asyncio
async def test_operational_faq_runner_late(orchestrator: AgentOrchestrator):
    """B3. Verify 'What happens if the runner is late?' returns dispute/exception policy."""
    conv_id = f"conv_faq_{uuid.uuid4().hex[:8]}"
    res = await orchestrator.process_message(conv_id, "What happens if the runner is late?", farmer_id="farmer_test")

    assert res.metadata["matched"] is True
    assert res.metadata["answerable"] is True
    assert "rescheduling" in res.text.lower() or "exception" in res.text.lower() or "failed" in res.text.lower()


@pytest.mark.asyncio
async def test_operational_faq_billing_calculation(orchestrator: AgentOrchestrator):
    """B4. Verify 'How is total billing calculated?' returns billing calculation knowledge."""
    conv_id = f"conv_faq_{uuid.uuid4().hex[:8]}"
    res = await orchestrator.process_message(conv_id, "How is total billing calculated?", farmer_id="farmer_test")

    assert res.metadata["matched"] is True
    assert res.metadata["answerable"] is True
    assert res.metadata["intent"] == IntentType.INQUIRE_OPERATIONAL_FAQ.value
    assert "billingservice" in res.text.lower() or "total payout" in res.text.lower() or "authoritative" in res.text.lower()


# =============================================================================
# C. Ambiguous Knowledge Inquiries
# =============================================================================

@pytest.mark.asyncio
async def test_ambiguous_knowledge_query(orchestrator: AgentOrchestrator):
    """C. Verify broad/ambiguous query asks for clarification with category overview."""
    conv_id = f"conv_amb_{uuid.uuid4().hex[:8]}"
    res = await orchestrator.process_message(conv_id, "What are your operational policies?", farmer_id="farmer_test")

    assert res.metadata["matched"] is False
    assert res.metadata["answerable"] is False
    assert res.metadata["clarification_required"] is True
    # Verify concise clarification question and category overview presented
    assert "clarify" in res.text.lower() or "specify" in res.text.lower()
    assert "procurement" in res.text.lower()
    assert "pricing" in res.text.lower()
    assert "logistics" in res.text.lower()
    assert "payment" in res.text.lower()
    assert "dispute" in res.text.lower()


# =============================================================================
# D. No-Match Safety
# =============================================================================

@pytest.mark.asyncio
async def test_no_match_safety(orchestrator: AgentOrchestrator):
    """D. Verify unsupported queries do not hallucinate facts or rates."""
    conv_id = f"conv_nomatch_{uuid.uuid4().hex[:8]}"
    res = await orchestrator.process_message(conv_id, "Who won the World Cup in 1998?", farmer_id="farmer_test")

    # UNKNOWN intent or non-matched knowledge returns standard assistant guidance, never fake facts
    price_pattern = re.compile(r"\$\s*\d+(?:\.\d+)?|\b\d+(?:\.\d+)?\s*(?:usd|kes|inr|cents?)\b", re.IGNORECASE)
    assert not price_pattern.findall(res.text)


# =============================================================================
# E. Transaction Protection
# =============================================================================

@pytest.mark.asyncio
async def test_transaction_protection_order_status(orchestrator: AgentOrchestrator):
    """E1. Verify 'Where is my order?' routes to order status inquiry, not knowledge."""
    conv_id = f"conv_tx_{uuid.uuid4().hex[:8]}"
    res = await orchestrator.process_message(conv_id, "Where is my order?", farmer_id="farmer_test")
    assert res.metadata.get("missing_entity") == "order_id"
    ctx = orchestrator.get_context(conv_id)
    assert ctx.last_intent == IntentType.INQUIRE_ORDER_STATUS.value


@pytest.mark.asyncio
async def test_transaction_protection_runner_status(orchestrator: AgentOrchestrator):
    """E2. Verify 'Where is my runner?' routes to collection status inquiry."""
    conv_id = f"conv_tx_{uuid.uuid4().hex[:8]}"
    res = await orchestrator.process_message(conv_id, "Where is my runner?", farmer_id="farmer_test")
    assert res.metadata.get("missing_entity") == "order_id"
    ctx = orchestrator.get_context(conv_id)
    assert ctx.last_intent == IntentType.INQUIRE_COLLECTION_STATUS.value


@pytest.mark.asyncio
async def test_transaction_protection_payment_status(orchestrator: AgentOrchestrator):
    """E3. Verify 'Where is my payment?' routes to payment status inquiry."""
    conv_id = f"conv_tx_{uuid.uuid4().hex[:8]}"
    res = await orchestrator.process_message(conv_id, "Where is my payment?", farmer_id="farmer_test")
    assert res.metadata.get("missing_entity") == "order_id"
    ctx = orchestrator.get_context(conv_id)
    assert ctx.last_intent == IntentType.INQUIRE_PAYMENT_STATUS.value


@pytest.mark.asyncio
async def test_transaction_protection_reschedule(orchestrator: AgentOrchestrator):
    """E4. Verify 'Can we reschedule pickup to tomorrow at 10 AM?' routes to pickup change."""
    conv_id = f"conv_tx_{uuid.uuid4().hex[:8]}"
    res = await orchestrator.process_message(conv_id, "Can we reschedule pickup to tomorrow at 10 AM?", farmer_id="farmer_test")
    assert res.metadata.get("missing_entity") == "order_id"
    ctx = orchestrator.get_context(conv_id)
    assert ctx.last_intent == IntentType.REQUEST_PICKUP_CHANGE.value


@pytest.mark.asyncio
async def test_transaction_protection_confirm_pay_offer(orchestrator: AgentOrchestrator):
    """E5. Verify Confirm, Pay now, and normal offer continue their transaction flows."""
    conv_id = f"conv_tx_{uuid.uuid4().hex[:8]}"
    
    # Offer
    r1 = await orchestrator.process_message(
        conv_id, "I have 50 kg of potatoes in Springfield tomorrow at 10 AM.", farmer_id="farmer_test"
    )
    assert r1.metadata["state"] == OrderState.AWAITING_FARMER_CONFIRMATION.value
    
    # Confirm
    r2 = await orchestrator.process_message(conv_id, "Confirm", farmer_id="farmer_test")
    assert r2.metadata["state"] in [OrderState.ORDER_CONFIRMED.value, OrderState.PAYMENT_PENDING.value]
    
    # Pay now
    r3 = await orchestrator.process_message(conv_id, "Pay now", farmer_id="farmer_test")
    assert r3.metadata["state"] in [OrderState.PAYMENT_CONFIRMED.value, OrderState.COLLECTION_PENDING.value]


# =============================================================================
# F. State Immutability
# =============================================================================

@pytest.mark.asyncio
async def test_state_immutability_for_knowledge(orchestrator: AgentOrchestrator):
    """F. Verify knowledge requests never mutate state, active offers, orders, payments, or tasks."""
    conv_id = f"conv_mut_{uuid.uuid4().hex[:8]}"

    # Initial state: OFFER_RECEIVED
    ctx = orchestrator.get_or_create_context(conv_id, farmer_id="farmer_test")
    assert ctx.state == OrderState.OFFER_RECEIVED

    initial_orders_count = len(order_service._orders)
    initial_payments_count = len(payment_service._payments)
    initial_tasks_count = len(collection_service._tasks)

    # Ask knowledge query
    res = await orchestrator.process_message(conv_id, "What is ORCA?", farmer_id="farmer_test")
    assert ctx.state == OrderState.OFFER_RECEIVED
    assert len(order_service._orders) == initial_orders_count
    assert len(payment_service._payments) == initial_payments_count
    assert len(collection_service._tasks) == initial_tasks_count

    # Move context to AWAITING_FARMER_CONFIRMATION via an offer
    await orchestrator.process_message(
        conv_id, "I have 50 kg of potatoes in Springfield tomorrow at 10 AM.", farmer_id="farmer_test"
    )
    assert ctx.state == OrderState.AWAITING_FARMER_CONFIRMATION
    offer_snapshot = ctx.offer.model_dump()

    # Interleave operational FAQ
    res2 = await orchestrator.process_message(conv_id, "How does pickup work?", farmer_id="farmer_test")
    # State and offer MUST remain identical
    assert ctx.state == OrderState.AWAITING_FARMER_CONFIRMATION
    assert ctx.offer.model_dump() == offer_snapshot
    assert len(order_service._orders) == initial_orders_count


# =============================================================================
# G. AgentTrace Verification
# =============================================================================

@pytest.mark.asyncio
async def test_agent_trace_recorded_for_knowledge(orchestrator: AgentOrchestrator):
    """G. Verify knowledge execution creates a safe AgentTrace entry with correct tool and zero secrets."""
    conv_id = f"conv_trc_{uuid.uuid4().hex[:8]}"
    await orchestrator.process_message(conv_id, "What is ORCA?", farmer_id="farmer_test")

    ctx = orchestrator.get_context(conv_id)
    assert ctx is not None
    assert ctx.last_trace is not None

    trace = ctx.last_trace
    assert trace.detected_intent == IntentType.INQUIRE_BUSINESS_INFO.value
    assert trace.confidence >= 0.90
    assert trace.selected_tool == "get_business_knowledge"
    assert trace.tool_success is True
    assert trace.state_before == OrderState.OFFER_RECEIVED.value
    assert trace.state_after == OrderState.OFFER_RECEIVED.value
    assert trace.input_text == "What is ORCA?"

    # Ensure no secrets in trace dump
    trace_json = trace.model_dump_json()
    assert "api_key" not in trace_json.lower()
    assert "secret" not in trace_json.lower()


# =============================================================================
# H. Seamless Multi-Turn Flow
# =============================================================================

@pytest.mark.asyncio
async def test_multi_turn_transaction_flow_with_interleaved_knowledge(orchestrator: AgentOrchestrator):
    """H. Farmer makes offer -> asks Q&A -> confirms -> asks payment FAQ -> pays."""
    conv_id = f"conv_flow_{uuid.uuid4().hex[:8]}"

    # 1. Offer
    t1 = await orchestrator.process_message(
        conv_id, "I have 50 kg of potatoes in Springfield tomorrow at 10 AM.", farmer_id="farmer_test"
    )
    assert t1.metadata["state"] == OrderState.AWAITING_FARMER_CONFIRMATION.value

    # 2. Logistics FAQ in the middle of negotiation
    t2 = await orchestrator.process_message(conv_id, "How does pickup work?", farmer_id="farmer_test")
    assert t2.metadata["state"] == OrderState.AWAITING_FARMER_CONFIRMATION.value
    assert "runner" in t2.text.lower()

    # 3. Farmer proceeds to confirm
    t3 = await orchestrator.process_message(conv_id, "Confirm", farmer_id="farmer_test")
    assert t3.metadata["state"] in [OrderState.ORDER_CONFIRMED.value, OrderState.PAYMENT_PENDING.value]
    order_id = t3.metadata["order_id"]
    assert order_id is not None

    # 4. Payment FAQ while in payment pending
    t4 = await orchestrator.process_message(conv_id, "When do I get paid?", farmer_id="farmer_test")
    assert t4.metadata["state"] == OrderState.PAYMENT_PENDING.value
    assert "payment" in t4.text.lower()

    # 5. Farmer authorizes payment
    t5 = await orchestrator.process_message(conv_id, "Pay now", farmer_id="farmer_test")
    assert t5.metadata["state"] in [OrderState.PAYMENT_CONFIRMED.value, OrderState.COLLECTION_PENDING.value]
