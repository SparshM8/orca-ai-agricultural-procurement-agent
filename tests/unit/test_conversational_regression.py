"""Regression unit tests for conversational classification bug fixes and orchestrator safety guards.

Verifies:
1. "What's happening with my order?" -> INQUIRE_ORDER_STATUS
2. "Any update on my order?" -> INQUIRE_ORDER_STATUS
3. "When is the pickup?" -> INQUIRE_COLLECTION_STATUS
4. "Okay, pay for it." -> AUTHORIZE_PAYMENT
5. "Okay, confirm my order." -> CONFIRM_ORDER
6. Missing-order-ID status inquiry prompts for Order ID without generic greeting or tools
7. Status inquiry while PAYMENT_PENDING does NOT regress state or re-calculate totals
8. Status inquiry while COLLECTION_PENDING returns runner info without state regression
9. UNKNOWN still executes zero tools
10. Existing multi-turn offer flow still works
11. State regression protection across PAYMENT_CONFIRMED, COLLECTION_ASSIGNED, PICKED_UP, COMPLETED
"""

import pytest
from unittest.mock import AsyncMock, patch

from orca.domain.intents import IntentType, ConversationalIntentResult, ExtractedEntities
from orca.domain.state_machine import OrderState
from orca.agent.intent.rule_based_classifier import RuleBasedIntentClassifier
from orca.agent.orchestrator import AgentOrchestrator, DialogueContext
from orca.services.order import order_service


@pytest.fixture
def classifier():
    return RuleBasedIntentClassifier()


@pytest.fixture
def orchestrator():
    return AgentOrchestrator(auto_create_collection_task=True)


# =============================================================================
# 1. Natural Language Intent Classification Tests
# =============================================================================
@pytest.mark.asyncio
async def test_classify_order_status_variations(classifier: RuleBasedIntentClassifier):
    """Test recognition of colloquial order status inquiries."""
    phrases = [
        "What's happening with my order?",
        "what is happening with my order",
        "Any update on my order?",
        "How is my order?",
        "What's the status of my order?",
        "Can you check my order?",
        "Where is my order?",
    ]
    for p in phrases:
        res = await classifier.classify(p)
        assert res.intent == IntentType.INQUIRE_ORDER_STATUS, f"Failed for phrase: '{p}'"


@pytest.mark.asyncio
async def test_classify_collection_status_variations(classifier: RuleBasedIntentClassifier):
    """Test recognition of colloquial collection/pickup status inquiries."""
    phrases = [
        "When is the pickup?",
        "When is my pickup?",
        "Pickup time?",
        "When will you collect it?",
        "When will my produce be picked up?",
        "Who is coming to pick it up?",
    ]
    for p in phrases:
        res = await classifier.classify(p)
        assert res.intent == IntentType.INQUIRE_COLLECTION_STATUS, f"Failed for phrase: '{p}'"


@pytest.mark.asyncio
async def test_classify_payment_authorization_variations(classifier: RuleBasedIntentClassifier):
    """Test recognition of payment authorization with conversational prefixes/fillers."""
    phrases = [
        "Okay, pay for it.",
        "Please pay.",
        "Go ahead and pay.",
        "Yes, pay now.",
        "Can you process the payment?",
        "Pay for it",
        "Make payment",
    ]
    for p in phrases:
        res = await classifier.classify(p)
        assert res.intent == IntentType.AUTHORIZE_PAYMENT, f"Failed for phrase: '{p}'"


@pytest.mark.asyncio
async def test_classify_order_confirmation_variations(classifier: RuleBasedIntentClassifier):
    """Test recognition of order confirmation with conversational prefixes/fillers."""
    phrases = [
        "Okay, confirm my order.",
        "Please confirm.",
        "Go ahead and confirm.",
        "Yes, I accept.",
        "Sure, let's proceed.",
        "Okay, confirm my carrot order.",
        "Confirm",
        "Deal",
    ]
    for p in phrases:
        res = await classifier.classify(p)
        assert res.intent == IntentType.CONFIRM_ORDER, f"Failed for phrase: '{p}'"


# =============================================================================
# 2. End-to-End Orchestrator Inquiries and Missing Entity Tests
# =============================================================================
@pytest.mark.asyncio
async def test_missing_order_id_status_inquiry(orchestrator: AgentOrchestrator):
    """When no order ID is known, ask the farmer for Order ID (not generic UNKNOWN greeting)."""
    conv_id = "test_regr_missing_order_id"
    outbound = await orchestrator.process_message(
        conv_id, "What's happening with my order?", farmer_id="farmer_regr_1"
    )

    ctx = orchestrator.get_context(conv_id)
    assert ctx.last_intent == IntentType.INQUIRE_ORDER_STATUS.value
    assert ctx.last_tool is None
    assert ctx.active_clarification == "order_id"
    assert "Order ID" in outbound.text
    assert outbound.metadata.get("missing_entity") == "order_id"
    # Ensure it did NOT return generic UNKNOWN greeting
    assert "I am ORCA" not in outbound.text


@pytest.mark.asyncio
async def test_missing_order_id_collection_inquiry(orchestrator: AgentOrchestrator):
    """When no order ID is known, ask for Order ID for collection inquiry."""
    conv_id = "test_regr_missing_coll_order_id"
    outbound = await orchestrator.process_message(
        conv_id, "When is the pickup?", farmer_id="farmer_regr_2"
    )

    ctx = orchestrator.get_context(conv_id)
    assert ctx.last_intent == IntentType.INQUIRE_COLLECTION_STATUS.value
    assert ctx.last_tool is None
    assert ctx.active_clarification == "order_id"
    assert "Order ID" in outbound.text
    assert outbound.metadata.get("missing_entity") == "order_id"


@pytest.mark.asyncio
async def test_unknown_intent_executes_zero_tools(orchestrator: AgentOrchestrator):
    """UNKNOWN intent must execute zero tools, zero database entries, and return capability greeting."""
    conv_id = "test_regr_unknown"
    outbound = await orchestrator.process_message(
        conv_id, "Can you help me with something completely unrelated?", farmer_id="farmer_regr_3"
    )

    ctx = orchestrator.get_context(conv_id)
    assert ctx.last_intent == IntentType.UNKNOWN.value
    assert ctx.last_tool is None
    assert ctx.current_order_id is None
    assert ctx.state == OrderState.OFFER_RECEIVED
    assert "I am ORCA" in outbound.text
    assert "We currently procure:" in outbound.text


# =============================================================================
# 3. State Preservation & Protection Against Regression (Fix 1 & Orchestrator Guard)
# =============================================================================
@pytest.mark.asyncio
async def test_order_status_while_payment_pending_does_not_regress_state(orchestrator: AgentOrchestrator):
    """CRITICAL: 'What's happening with my order?' during PAYMENT_PENDING must NOT regress state to AWAITING_FARMER_CONFIRMATION."""
    conv_id = "test_regr_pay_pending_status"

    # Turn 1: Offer produce
    await orchestrator.process_message(
        conv_id, "I have 50 kg of potatoes in Springfield tomorrow at 10 AM", farmer_id="farmer_regr_4"
    )
    # Turn 2: Confirm order
    await orchestrator.process_message(conv_id, "Confirm", farmer_id="farmer_regr_4")

    ctx = orchestrator.get_context(conv_id)
    assert ctx.state == OrderState.PAYMENT_PENDING
    active_order_id = ctx.current_order_id
    assert active_order_id is not None

    # Turn 3: Ask about order status
    outbound3 = await orchestrator.process_message(
        conv_id, "What's happening with my order?", farmer_id="farmer_regr_4"
    )

    # State MUST remain PAYMENT_PENDING, not regress to AWAITING_FARMER_CONFIRMATION
    assert ctx.state == OrderState.PAYMENT_PENDING
    assert ctx.current_order_id == active_order_id
    assert active_order_id in outbound3.text
    assert "Collection & Order Status:" in outbound3.text
    assert "PAYMENT_PENDING" in outbound3.text
    # Must NOT have re-evaluated produce offer
    assert "Please reply with 'Confirm'" not in outbound3.text
    assert ctx.last_tool == "get_order_status"


@pytest.mark.asyncio
async def test_collection_status_while_collection_pending(orchestrator: AgentOrchestrator):
    """'When is the pickup?' during COLLECTION_PENDING returns runner details without state regression."""
    conv_id = "test_regr_coll_pending_status"

    # Offer -> Confirm -> Pay
    await orchestrator.process_message(
        conv_id, "I have 50 kg of potatoes in Springfield tomorrow at 10 AM", farmer_id="farmer_regr_5"
    )
    await orchestrator.process_message(conv_id, "Okay, confirm my order.", farmer_id="farmer_regr_5")
    await orchestrator.process_message(conv_id, "Okay, pay for it.", farmer_id="farmer_regr_5")

    ctx = orchestrator.get_context(conv_id)
    assert ctx.state == OrderState.COLLECTION_PENDING
    active_order_id = ctx.current_order_id

    # Inquire collection status
    outbound = await orchestrator.process_message(
        conv_id, "When is the pickup?", farmer_id="farmer_regr_5"
    )

    assert ctx.state == OrderState.COLLECTION_PENDING
    assert ctx.last_tool == "get_collection_status"
    assert active_order_id in outbound.text
    assert "Collection & Order Status:" in outbound.text
    assert "PENDING" in outbound.text


@pytest.mark.asyncio
async def test_safety_guard_blocks_regression_across_all_active_states():
    """Defensive Orchestrator Guard test: Even if intent classifier erroneously emits OFFER_PRODUCE,
    the orchestrator MUST NOT regress state from PAYMENT_PENDING, PAYMENT_CONFIRMED, COLLECTION_PENDING,
    COLLECTION_ASSIGNED, PICKED_UP, or COMPLETED.
    """
    class ErrantOfferIntentClassifier(RuleBasedIntentClassifier):
        async def classify(self, text, context=None):
            # Artificially simulate classifier mistake emitting OFFER_PRODUCE
            res = await super().classify(text, context)
            res.intent = IntentType.OFFER_PRODUCE
            return res

    guarded_orchestrator = AgentOrchestrator(
        intent_classifier=ErrantOfferIntentClassifier(),
        auto_create_collection_task=True,
    )

    test_states = [
        OrderState.PAYMENT_PENDING,
        OrderState.PAYMENT_CONFIRMED,
        OrderState.COLLECTION_PENDING,
        OrderState.COLLECTION_ASSIGNED,
        OrderState.PICKED_UP,
        OrderState.COMPLETED,
    ]

    for idx, target_state in enumerate(test_states):
        conv_id = f"test_guard_state_{idx}"
        # Create real backend order in DB
        order = await order_service.create_order_async(
            farmer_id=f"farmer_guard_{idx}",
            produce_type="potato",
            quantity=50.0,
            unit="kg",
            pickup_location="Springfield",
        )
        order.status = target_state

        ctx = guarded_orchestrator.get_or_create_context(conv_id, farmer_id=f"farmer_guard_{idx}")
        ctx.current_order_id = order.id
        ctx.state = target_state

        # Farmer sends non-offer message
        outbound = await guarded_orchestrator.process_message(
            conv_id, "What's happening with my order?", farmer_id=f"farmer_guard_{idx}"
        )

        # Assert guard prevented regression!
        assert ctx.state == target_state, f"State regressed from {target_state} to {ctx.state}!"
        assert outbound.metadata.get("guard_triggered") is True
        assert "Please reply with 'Confirm'" not in outbound.text


# =============================================================================
# 4. Multi-Turn Offer Completion Tests
# =============================================================================
@pytest.mark.asyncio
async def test_multiturn_offer_flow_preservation(orchestrator: AgentOrchestrator):
    """Legitimate multi-turn offer completion in DETAILS_PENDING works without loss of parameters."""
    conv_id = "test_regr_multiturn_flow"

    # Turn 1: Farmer only provides produce and quantity
    t1 = await orchestrator.process_message(conv_id, "I have 75 kg of onions", farmer_id="farmer_multi")
    ctx = orchestrator.get_context(conv_id)
    assert ctx.state == OrderState.DETAILS_PENDING
    assert ctx.offer.produce_type == "onion"
    assert ctx.offer.quantity == 75.0
    assert "location" in t1.text.lower() or "where" in t1.text.lower()

    # Turn 2: Farmer supplies pickup location and timing
    t2 = await orchestrator.process_message(
        conv_id, "In Springfield tomorrow morning at 9 AM", farmer_id="farmer_multi"
    )
    assert ctx.state == OrderState.AWAITING_FARMER_CONFIRMATION
    assert ctx.offer.produce_type == "onion"
    assert ctx.offer.quantity == 75.0
    assert ctx.offer.pickup_location == "Springfield"
    assert "Total Payout: USD 26.25" in t2.text  # 75 * 0.35
    assert "Confirm" in t2.text

    # Turn 3: Farmer confirms using conversational filler
    t3 = await orchestrator.process_message(conv_id, "Okay, confirm my order.", farmer_id="farmer_multi")
    assert ctx.state == OrderState.PAYMENT_PENDING
    assert ctx.current_order_id is not None
    assert "Order Confirmed!" in t3.text


@pytest.mark.asyncio
async def test_unsupported_crop_confirmation_safely_rejected(orchestrator: AgentOrchestrator):
    """Farmer offering carrots cannot confirm transaction."""
    conv_id = "test_regr_carrot_rejection"

    # Turn 1: Offer carrots
    t1 = await orchestrator.process_message(
        conv_id, "I have 50 kg of carrots in Delhi tomorrow at 10 AM.", farmer_id="farmer_carrot"
    )
    ctx = orchestrator.get_context(conv_id)
    assert ctx.state == OrderState.DETAILS_PENDING
    assert ctx.current_order_id is None
    assert "onion, potato, tomato" in t1.text

    # Turn 2: Try to confirm carrot order
    t2 = await orchestrator.process_message(
        conv_id, "Okay, confirm my carrot order.", farmer_id="farmer_carrot"
    )
    assert ctx.current_order_id is None
    assert ctx.state == OrderState.DETAILS_PENDING
    assert "cannot confirm" in t2.text.lower()
    assert "carrot" in t2.text.lower()
