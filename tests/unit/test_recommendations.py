"""Unit tests for Phase 5: Grounded Procurement Recommendations.

Verifies:
1. Recommendation classification (RECOMMEND_PRODUCE)
2. Transaction protection (OFFER_PRODUCE, CONFIRM_ORDER, AUTHORIZE_PAYMENT, INQUIRE_ORDER_STATUS, INQUIRE_OPERATIONAL_FAQ)
3. RecommendationService unit tests (supported, unsupported, catalog, clarification)
4. No hallucination: no invented prices, profit estimates, or demand predictions
5. State immutability: zero mutation to state, active offer, orders, payments, collection tasks
6. AgentTrace: safe observational trace recording
7. Multi-turn protection: active offer preserved during recommendation Q&A
"""

import pytest
import uuid
import re

from orca.domain.state_machine import OrderState
from orca.domain.intents import IntentType, ExtractedEntities
from orca.agent.intent.rule_based_classifier import RuleBasedIntentClassifier
from orca.services.recommendation import recommendation_service
from orca.agent.orchestrator import AgentOrchestrator
from orca.services.order import order_service
from orca.services.payment import payment_service
from orca.services.collection import collection_service
from orca.agent.tools import tool_registry
from orca.agent.intent.mapping import IntentToToolMapper


@pytest.fixture
def orchestrator():
    """Create fresh AgentOrchestrator instance for testing."""
    return AgentOrchestrator(auto_process_payment=False)


# =============================================================================
# A. Recommendation Classification
# =============================================================================

@pytest.mark.asyncio
async def test_recommendation_classification_queries():
    """A. Verify rule-based classification for recommendation questions."""
    classifier = RuleBasedIntentClassifier()
    queries = [
        ("What can I sell through ORCA?", IntentType.RECOMMEND_PRODUCE, None),
        ("Which crops can I sell?", IntentType.RECOMMEND_PRODUCE, None),
        ("What produce do you accept?", IntentType.RECOMMEND_PRODUCE, None),
        ("What can I offer?", IntentType.RECOMMEND_PRODUCE, None),
        ("What else can I sell?", IntentType.RECOMMEND_PRODUCE, None),
        ("Can I sell potatoes through ORCA?", IntentType.RECOMMEND_PRODUCE, "potatoes"),
        ("Can I sell tomatoes?", IntentType.RECOMMEND_PRODUCE, "tomatoes"),
        ("Can I sell papaya through ORCA?", IntentType.RECOMMEND_PRODUCE, "papaya"),
    ]

    for q, expected_intent, expected_prod in queries:
        res = await classifier.classify(q)
        assert res.intent == expected_intent, f"Failed intent for: {q}"
        if expected_prod:
            assert res.entities.recommendation_produce == expected_prod, f"Failed produce for: {q}"


# =============================================================================
# B. Transaction Protection
# =============================================================================

@pytest.mark.asyncio
async def test_transaction_intents_not_intercepted_by_recommendation():
    """B. Verify existing transactional intents are strictly preserved."""
    classifier = RuleBasedIntentClassifier()

    # 1. Offer produce
    r1 = await classifier.classify("I have 50 kg of potatoes in Springfield tomorrow at 10 AM")
    assert r1.intent == IntentType.OFFER_PRODUCE

    r1b = await classifier.classify("I have potatoes to sell")
    assert r1b.intent == IntentType.OFFER_PRODUCE

    # 2. Confirm order
    r2 = await classifier.classify("Confirm")
    assert r2.intent == IntentType.CONFIRM_ORDER

    # 3. Authorize payment
    r3 = await classifier.classify("Pay now")
    assert r3.intent == IntentType.AUTHORIZE_PAYMENT

    # 4. Inquire order status
    r4 = await classifier.classify("Where is my order?")
    assert r4.intent == IntentType.INQUIRE_ORDER_STATUS

    # 5. Operational FAQ
    r5 = await classifier.classify("How does pickup work?")
    assert r5.intent == IntentType.INQUIRE_OPERATIONAL_FAQ

    # 6. Business info
    r6 = await classifier.classify("What is ORCA?")
    assert r6.intent == IntentType.INQUIRE_BUSINESS_INFO


# =============================================================================
# C. RecommendationService Unit Tests
# =============================================================================

def test_recommendation_service_known_supported_produce():
    """C1. Verify supported crop returns positive recommendation with grounded reasons."""
    res = recommendation_service.get_recommendations(
        query="Can I sell potatoes through ORCA?",
        produce="potatoes",
    )
    assert res.matched is True
    assert len(res.recommendations) == 1
    rec = res.recommendations[0]
    assert rec.produce == "Potato"
    assert "actively procured" in rec.reason.lower()
    assert "runner" in rec.reason.lower()
    assert rec.confidence == 1.0


def test_recommendation_service_unsupported_produce():
    """C2. Verify unsupported crop returns matched=False and lists supported crops."""
    res = recommendation_service.get_recommendations(
        query="Can I sell papaya through ORCA?",
        produce="papaya",
    )
    assert res.matched is False
    assert len(res.recommendations) == 0
    assert "not currently procured" in res.message.lower()
    assert "potato" in res.message.lower()


def test_recommendation_service_catalog_recommendation():
    """C3. Verify general query returns full grounded supported catalog."""
    res = recommendation_service.get_recommendations(query="What can I sell through ORCA?")
    assert res.matched is True
    assert len(res.recommendations) == 3
    prods = {r.produce for r in res.recommendations}
    assert {"Potato", "Onion", "Tomato"} == prods


def test_recommendation_service_ambiguous_empty():
    """C4. Verify empty query prompts for clarification."""
    res = recommendation_service.get_recommendations(query="   ")
    assert res.matched is False
    assert res.clarification_required is True


# =============================================================================
# D. No Hallucination Safety
# =============================================================================

@pytest.mark.asyncio
async def test_no_hallucination_in_recommendations(orchestrator: AgentOrchestrator):
    """D. Recommendation output must not contain profit estimates or speculative demand."""
    conv_id = f"conv_rec_{uuid.uuid4().hex[:8]}"
    res = await orchestrator.process_message(conv_id, "What can I sell through ORCA?", farmer_id="farmer_test")

    speculative_terms = [
        "profit", "profitable", "forecast", "demand will rise", "best price",
        "market prediction", "roi", "yield forecast", "earn up to"
    ]
    for term in speculative_terms:
        assert term not in res.text.lower(), f"Speculative term '{term}' found in response"


# =============================================================================
# E. State Immutability
# =============================================================================

@pytest.mark.asyncio
async def test_state_immutability_on_recommendation(orchestrator: AgentOrchestrator):
    """E. Recommendation calls do not mutate state, orders, payments, or tasks."""
    conv_id = f"conv_rec_mut_{uuid.uuid4().hex[:8]}"
    ctx = orchestrator.get_or_create_context(conv_id, farmer_id="farmer_test")
    assert ctx.state == OrderState.OFFER_RECEIVED

    init_orders = len(order_service._orders)
    init_payments = len(payment_service._payments)
    init_tasks = len(collection_service._tasks)

    # Turn 1: Recommendation query
    res = await orchestrator.process_message(conv_id, "What produce do you accept?", farmer_id="farmer_test")
    assert ctx.state == OrderState.OFFER_RECEIVED
    assert len(order_service._orders) == init_orders
    assert len(payment_service._payments) == init_payments
    assert len(collection_service._tasks) == init_tasks

    # Turn 2: Specific crop recommendation
    res2 = await orchestrator.process_message(conv_id, "Can I sell potatoes through ORCA?", farmer_id="farmer_test")
    assert ctx.state == OrderState.OFFER_RECEIVED
    assert len(order_service._orders) == init_orders


# =============================================================================
# F. AgentTrace Recording
# =============================================================================

@pytest.mark.asyncio
async def test_agent_trace_recorded_for_recommendation(orchestrator: AgentOrchestrator):
    """F. Verify recommendation execution creates a safe observational AgentTrace entry."""
    conv_id = f"conv_rec_trc_{uuid.uuid4().hex[:8]}"
    await orchestrator.process_message(conv_id, "What can I sell through ORCA?", farmer_id="farmer_test")

    ctx = orchestrator.get_context(conv_id)
    assert ctx is not None
    assert ctx.last_trace is not None

    trace = ctx.last_trace
    assert trace.detected_intent == IntentType.RECOMMEND_PRODUCE.value
    assert trace.selected_tool == "get_procurement_recommendations"
    assert trace.tool_success is True
    assert trace.state_before == OrderState.OFFER_RECEIVED.value
    assert trace.state_after == OrderState.OFFER_RECEIVED.value

    # Zero secrets in trace
    dump = trace.model_dump_json()
    assert "api_key" not in dump.lower()
    assert "secret" not in dump.lower()


# =============================================================================
# G. Multi-Turn Protection
# =============================================================================

@pytest.mark.asyncio
async def test_multi_turn_offer_preserved_during_recommendation(orchestrator: AgentOrchestrator):
    """G. Active offer must NOT be overwritten when farmer asks for recommendations."""
    conv_id = f"conv_rec_multiturn_{uuid.uuid4().hex[:8]}"

    # Turn 1: Farmer makes potato offer
    t1 = await orchestrator.process_message(
        conv_id, "I have 50 kg of potatoes in Springfield tomorrow at 10 AM.", farmer_id="farmer_test"
    )
    assert t1.metadata["state"] == OrderState.AWAITING_FARMER_CONFIRMATION.value

    ctx = orchestrator.get_context(conv_id)
    assert ctx.offer.produce_type == "potato"
    assert ctx.offer.quantity == 50.0
    assert ctx.offer.pickup_location == "Springfield"

    # Turn 2: Farmer asks recommendation Q&A mid-flow
    t2 = await orchestrator.process_message(conv_id, "What else can I sell?", farmer_id="farmer_test")
    assert t2.metadata["state"] == OrderState.AWAITING_FARMER_CONFIRMATION.value
    assert t2.metadata["intent"] == IntentType.RECOMMEND_PRODUCE.value

    # Invariant: Active offer is completely intact!
    assert ctx.offer.produce_type == "potato"
    assert ctx.offer.quantity == 50.0
    assert ctx.offer.pickup_location == "Springfield"

    # Turn 3: Farmer confirms the original offer
    t3 = await orchestrator.process_message(conv_id, "Confirm", farmer_id="farmer_test")
    assert t3.metadata["state"] in [OrderState.ORDER_CONFIRMED.value, OrderState.PAYMENT_PENDING.value]
    assert ctx.current_order_id is not None
    order = order_service.get_order(ctx.current_order_id)
    assert order.produce_type == "potato"
    assert order.quantity == 50.0
