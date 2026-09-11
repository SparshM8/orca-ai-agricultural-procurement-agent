"""Unit tests for Factual Farmer Profile and Grounded Personalization (Phase 6).

Covers:
A. Profile creation/retrieval
B. Observed offer recording
C. Completed order history
D. Explicit preference recording
E. Personalized classification
F. Generic recommendation vs personalized recommendation separation
G. Personalization with sufficient history
H. Personalization with no history
I. Unsupported historical produce
J. Explanation contains factual grounding
K. Transaction protection
L. State immutability
M. Multi-turn active-offer preservation
N. AgentTrace
O. Gemini fallback
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from orca.domain.farmer_profile import (
    FarmerProfile,
    FactualOfferObservation,
    FactualCompletedOrder,
    PersonalizedRecommendationResult,
)
from orca.services.farmer_profile import farmer_profile_service
from orca.services.personalization import personalization_service
from orca.services.pricing import pricing_service
from orca.services.order import order_service
from orca.domain.intents import IntentType, ConversationalIntentResult, ExtractedEntities
from orca.domain.state_machine import OrderState
from orca.agent.intent.rule_based_classifier import RuleBasedIntentClassifier
from orca.agent.intent.resilient_classifier import ResilientIntentClassifier
from orca.agent.orchestrator import AgentOrchestrator


@pytest.fixture(autouse=True)
def clean_services():
    """Ensure clean state across tests."""
    farmer_profile_service.clear()
    order_service.clear()
    pricing_service.reset_baseline_rates()
    yield
    farmer_profile_service.clear()
    order_service.clear()


# =============================================================================
# A. Profile Creation & Retrieval
# =============================================================================
def test_profile_creation_retrieval():
    profile = farmer_profile_service.get_profile("farmer_alpha")
    assert profile.farmer_id == "farmer_alpha"
    assert profile.completed_order_count == 0
    assert profile.observed_produce_history == []
    assert profile.explicit_preferences == []
    assert profile.last_offered_produce is None
    assert profile.last_completed_produce is None

    # Retrieve again should yield the exact same record
    same_profile = farmer_profile_service.get_profile("farmer_alpha")
    assert same_profile.farmer_id == "farmer_alpha"


# =============================================================================
# B. Observed Offer Recording
# =============================================================================
def test_observed_offer_recording():
    farmer_profile_service.record_observed_offer(
        farmer_id="farmer_beta",
        produce="Potatoes",
        quantity=50.0,
        unit="kg",
        location="Springfield",
    )
    profile = farmer_profile_service.get_profile("farmer_beta")
    assert profile.last_offered_produce == "potato"
    assert "potato" in profile.observed_produce_history
    assert "kg" in profile.observed_units
    assert profile.location == "Springfield"
    assert len(profile.observed_offers) == 1
    assert profile.observed_offers[0].produce == "potato"
    assert profile.observed_offers[0].quantity == 50.0


# =============================================================================
# C. Completed Order History
# =============================================================================
def test_completed_order_history():
    farmer_profile_service.record_completed_order(
        farmer_id="farmer_gamma",
        produce="Tomato",
        quantity=100.0,
        unit="kg",
        order_id="ORD-112233",
    )
    profile = farmer_profile_service.get_profile("farmer_gamma")
    assert profile.completed_order_count == 1
    assert profile.last_completed_produce == "tomato"
    assert "tomato" in profile.observed_produce_history
    assert len(profile.completed_orders) == 1
    assert profile.completed_orders[0].order_id == "ORD-112233"


# =============================================================================
# D. Explicit Preference Recording
# =============================================================================
def test_explicit_preference_recording():
    farmer_profile_service.record_explicit_preference(
        farmer_id="farmer_delta",
        preference="Onions",
    )
    # Recording the same preference again should not duplicate
    farmer_profile_service.record_explicit_preference(
        farmer_id="farmer_delta",
        preference="onion",
    )
    profile = farmer_profile_service.get_profile("farmer_delta")
    assert profile.explicit_preferences == ["onion"]


# =============================================================================
# E. Personalized Classification
# =============================================================================
@pytest.mark.asyncio
async def test_personalized_classification():
    classifier = RuleBasedIntentClassifier()

    queries = [
        "What would you recommend for me?",
        "What should I sell?",
        "What would be good for me?",
        "Based on my previous orders, what can I sell?",
        "What do I usually sell?",
        "What did I sell before?",
        "Recommend something for me",
        "Check my sales history",
    ]

    for q in queries:
        res = await classifier.classify(q)
        assert res.intent == IntentType.PERSONALIZE_RECOMMENDATION, f"Failed for query: {q}"
        assert res.confidence >= 0.9

    # Stated explicit preference
    pref_res = await classifier.classify("I usually sell onions.")
    assert pref_res.intent == IntentType.PERSONALIZE_RECOMMENDATION
    assert pref_res.entities.explicit_preference == "onion"


# =============================================================================
# F. Generic vs Personalized Separation
# =============================================================================
@pytest.mark.asyncio
async def test_generic_vs_personalized_separation():
    classifier = RuleBasedIntentClassifier()

    # Generic catalog questions -> RECOMMEND_PRODUCE
    generic_queries = [
        "What crops can I sell?",
        "Which crops can I sell?",
        "What produce do you accept?",
        "What can I sell through ORCA?",
        "Can I sell potatoes through ORCA?",
        "Can I sell tomatoes?",
    ]
    for q in generic_queries:
        res = await classifier.classify(q)
        assert res.intent == IntentType.RECOMMEND_PRODUCE, f"Failed for generic query: {q}"

    # Personalized queries -> PERSONALIZE_RECOMMENDATION
    personal_queries = [
        "What would you recommend for me?",
        "What should I sell?",
        "What did I sell before?",
    ]
    for q in personal_queries:
        res = await classifier.classify(q)
        assert res.intent == IntentType.PERSONALIZE_RECOMMENDATION, f"Failed for personal query: {q}"


# =============================================================================
# G. Personalization with Sufficient History
# =============================================================================
def test_personalization_with_sufficient_history():
    farmer_profile_service.record_completed_order(
        farmer_id="farmer_hist", produce="potato", quantity=50.0, unit="kg"
    )
    farmer_profile_service.record_completed_order(
        farmer_id="farmer_hist", produce="onion", quantity=80.0, unit="kg"
    )

    res = personalization_service.get_personalized_recommendations("farmer_hist")
    assert res.matched is True
    assert res.grounding_source == "profile_history"
    assert len(res.recommendations) == 2
    prods = [r.produce for r in res.recommendations]
    assert "Potato" in prods
    assert "Onion" in prods


# =============================================================================
# H. Personalization with No History
# =============================================================================
def test_personalization_with_no_history():
    res = personalization_service.get_personalized_recommendations("farmer_fresh")
    assert res.matched is False
    assert res.grounding_source == "insufficient_history"
    assert res.clarification_required is True
    assert "don't have enough history" in res.message.lower()
    assert "supported produce" in res.message.lower()


# =============================================================================
# I. Unsupported Historical Produce
# =============================================================================
def test_unsupported_historical_produce():
    # Farmer previously completed papaya (not in ORCA catalog)
    farmer_profile_service.record_completed_order(
        farmer_id="farmer_papaya", produce="papaya", quantity=30.0, unit="kg"
    )
    res = personalization_service.get_personalized_recommendations("farmer_papaya")
    # Must NOT recommend papaya
    for r in res.recommendations:
        assert r.produce.lower() != "papaya"
    assert "not currently supported" in res.message.lower()
    assert "papaya" in res.message.lower()


# =============================================================================
# J. Explanation Contains Factual Grounding
# =============================================================================
def test_explanation_contains_factual_grounding():
    farmer_profile_service.record_completed_order(
        farmer_id="farmer_fact", produce="potato", quantity=40.0, unit="kg"
    )
    res = personalization_service.get_personalized_recommendations("farmer_fact")
    assert res.matched is True
    rec = res.recommendations[0]
    assert "relevant because you previously completed an order for potato" in rec.factual_reason
    assert "authoritative rate" in rec.factual_reason
    assert "looks like a good choice" not in rec.factual_reason


# =============================================================================
# K. Transaction Protection
# =============================================================================
@pytest.mark.asyncio
async def test_transaction_protection():
    classifier = RuleBasedIntentClassifier()

    tx_queries = [
        ("I have 50 kg of potatoes in Springfield available tomorrow", IntentType.OFFER_PRODUCE),
        ("Where is my order ORD-1234?", IntentType.INQUIRE_ORDER_STATUS),
        ("How does pickup work?", IntentType.INQUIRE_OPERATIONAL_FAQ),
        ("Confirm", IntentType.CONFIRM_ORDER),
        ("Pay now", IntentType.AUTHORIZE_PAYMENT),
    ]

    for text, expected in tx_queries:
        res = await classifier.classify(text)
        assert res.intent == expected, f"Transaction query '{text}' incorrectly routed to {res.intent}"


# =============================================================================
# L. State Immutability
# =============================================================================
@pytest.mark.asyncio
async def test_state_immutability():
    orchestrator = AgentOrchestrator(intent_classifier=RuleBasedIntentClassifier())
    conv_id = "conv_immut_test"

    # Pre-populate history
    farmer_profile_service.record_completed_order(
        farmer_id="farmer_immut", produce="potato", quantity=50.0, unit="kg"
    )

    ctx = orchestrator.get_or_create_context(conv_id, farmer_id="farmer_immut")
    initial_state = ctx.state

    outbound = await orchestrator.process_message(
        conversation_id=conv_id,
        message_text="What would you recommend for me?",
        farmer_id="farmer_immut",
    )

    assert ctx.state == initial_state
    assert outbound.metadata["state"] == initial_state.value
    assert outbound.metadata["matched"] is True
    assert outbound.metadata["grounding_source"] == "profile_history"


# =============================================================================
# M. Multi-Turn Active-Offer Preservation
# =============================================================================
@pytest.mark.asyncio
async def test_multi_turn_active_offer_preservation():
    orchestrator = AgentOrchestrator(intent_classifier=RuleBasedIntentClassifier())
    conv_id = "conv_multiturn_personal"

    # Turn 1: Farmer makes an offer
    res1 = await orchestrator.process_message(
        conversation_id=conv_id,
        message_text="I have 50 kg of potatoes in Springfield available tomorrow at 10 AM.",
        farmer_id="farmer_multi",
    )
    ctx = orchestrator.get_context(conv_id)
    assert ctx.state == OrderState.AWAITING_FARMER_CONFIRMATION
    assert ctx.offer.produce_type == "potato"
    assert ctx.offer.quantity == 50.0

    # Turn 2: Farmer asks for personalized recommendation in the middle of offer review
    res2 = await orchestrator.process_message(
        conversation_id=conv_id,
        message_text="What would you recommend for me?",
        farmer_id="farmer_multi",
    )
    # State and pending offer must remain intact!
    assert ctx.state == OrderState.AWAITING_FARMER_CONFIRMATION
    assert ctx.offer.produce_type == "potato"
    assert ctx.offer.quantity == 50.0

    # Turn 3: Farmer confirms the original offer
    res3 = await orchestrator.process_message(
        conversation_id=conv_id,
        message_text="Confirm",
        farmer_id="farmer_multi",
    )
    assert ctx.state in [OrderState.ORDER_CONFIRMED, OrderState.PAYMENT_PENDING]
    assert ctx.current_order_id is not None
    confirmed_order = order_service.get_order(ctx.current_order_id)
    assert confirmed_order is not None
    assert confirmed_order.produce_type == "potato"
    assert confirmed_order.quantity == 50.0


# =============================================================================
# N. AgentTrace Recording
# =============================================================================
@pytest.mark.asyncio
async def test_agent_trace_for_personalization():
    orchestrator = AgentOrchestrator(intent_classifier=RuleBasedIntentClassifier())
    conv_id = "conv_trace_personal"

    farmer_profile_service.record_explicit_preference("farmer_trc", "onions")

    await orchestrator.process_message(
        conversation_id=conv_id,
        message_text="What would you recommend for me?",
        farmer_id="farmer_trc",
    )

    ctx = orchestrator.get_context(conv_id)
    assert len(ctx.traces) >= 1
    last_trace = ctx.traces[-1]
    assert last_trace.detected_intent == "PERSONALIZE_RECOMMENDATION"
    assert last_trace.selected_tool == "get_personalized_recommendations"
    assert last_trace.tool_success is True
    assert last_trace.state_before == last_trace.state_after
    assert last_trace.tool_arguments["grounding_source"] == "explicit_preference"


# =============================================================================
# O. Gemini Fallback
# =============================================================================
@pytest.mark.asyncio
async def test_gemini_fallback():
    mock_gemini = AsyncMock()
    # Simulate Gemini raising a transient 429 quota exhaustion
    mock_gemini.classify.side_effect = Exception("ResourceExhausted: 429 quota exceeded")

    rule_classifier = RuleBasedIntentClassifier()
    resilient_classifier = ResilientIntentClassifier(primary=mock_gemini, fallback=rule_classifier)

    result = await resilient_classifier.classify("What should I sell?")
    assert result.intent == IntentType.PERSONALIZE_RECOMMENDATION
    assert result.fallback_occurred is True
    assert "429" in (result.fallback_reason or "")
