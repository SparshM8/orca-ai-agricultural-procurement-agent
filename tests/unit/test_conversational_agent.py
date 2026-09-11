"""Focused unit tests for the Conversational Agent Layer and Orchestrator Integration.

Covers:
1. Core intent routing through AgentOrchestrator (all 10 intents + UNKNOWN)
2. Deterministic tool execution via AgentToolRegistry
3. First-class UNKNOWN intent (zero tools executed, zero state mutation, helpful capability greeting)
4. Missing entity clarification (e.g. asking for Order ID, multi-turn entity supply)
5. Multi-turn conversational shifts (inquiries mid-negotiation without losing offer context)
6. Grounded deterministic responses (no hallucinated rates or totals)
7. Resilient fallback when primary AI classifier encounters transient errors
"""

import pytest
from unittest.mock import AsyncMock, patch
from orca.agent.orchestrator import AgentOrchestrator
from orca.domain.state_machine import OrderState
from orca.domain.intents import IntentType, ConversationalIntentResult, ExtractedEntities
from orca.agent.intent.resilient_classifier import ResilientIntentClassifier
from orca.agent.intent.rule_based_classifier import RuleBasedIntentClassifier
from orca.agent.intent.base import BaseIntentClassifier
from orca.services.order import order_service
from orca.services.payment import payment_service
from orca.services.collection import collection_service


@pytest.fixture
def orchestrator():
    """Fresh orchestrator instance for testing conversational agent flows."""
    return AgentOrchestrator(auto_create_collection_task=True)


# -----------------------------------------------------------------------------
# 1. UNKNOWN Intent (First-Class Path)
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_unknown_intent_path_executes_zero_tools_and_does_not_mutate_state(orchestrator: AgentOrchestrator):
    """UNKNOWN intent must not execute tools, must not mutate state, and must return a helpful capability greeting."""
    conv_id = "test_conv_unknown_01"
    msg = "Tell me a joke about tractors"

    outbound = await orchestrator.process_message(conv_id, msg, farmer_id="farmer_unknown")

    ctx = orchestrator.get_context(conv_id)
    assert ctx is not None
    assert ctx.last_intent == IntentType.UNKNOWN.value
    assert ctx.last_tool is None
    assert ctx.current_order_id is None
    assert ctx.state == OrderState.OFFER_RECEIVED
    assert outbound.metadata["intent"] == "UNKNOWN"
    assert "I am ORCA" in outbound.text
    assert "We currently procure:" in outbound.text
    assert "onion, potato, tomato" in outbound.text


# -----------------------------------------------------------------------------
# 2. Inquire Supported Produce (INQUIRE_SUPPORTED_PRODUCE)
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_inquire_supported_produce_returns_authoritative_catalog(orchestrator: AgentOrchestrator):
    """Inquiring about supported produce calls get_supported_produce and formats grounded rates."""
    conv_id = "test_conv_inquire_produce_02"
    msg = "What crops do you buy?"

    outbound = await orchestrator.process_message(conv_id, msg, farmer_id="farmer_crops")

    ctx = orchestrator.get_context(conv_id)
    assert ctx.last_intent == IntentType.INQUIRE_SUPPORTED_PRODUCE.value
    assert ctx.last_tool == "get_supported_produce"
    assert "Potato" in outbound.text
    assert "Onion" in outbound.text
    assert "Tomato" in outbound.text
    assert "USD 0.40 per kg" in outbound.text or "USD 0.40" in outbound.text
    assert outbound.metadata["intent"] == "INQUIRE_SUPPORTED_PRODUCE"


# -----------------------------------------------------------------------------
# 3. Request Clarification (REQUEST_CLARIFICATION)
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_request_clarification_pricing_and_availability(orchestrator: AgentOrchestrator):
    """Clarification requests explain rates or terms without modifying dialogue state."""
    conv_id = "test_conv_clarify_03"

    # Price explanation
    out1 = await orchestrator.process_message(conv_id, "Why is the price fixed?", farmer_id="farmer_clarify")
    ctx = orchestrator.get_context(conv_id)
    assert ctx.last_intent == IntentType.REQUEST_CLARIFICATION.value
    assert ctx.last_tool == "explain_requirement"
    assert "authoritative" in out1.text.lower() or "fair market" in out1.text.lower()
    assert out1.metadata["topic"] == "pricing"

    # Window explanation
    out2 = await orchestrator.process_message(conv_id, "What is an availability window?", farmer_id="farmer_clarify")
    assert ctx.last_intent == IntentType.REQUEST_CLARIFICATION.value
    assert "availability window" in out2.text.lower()
    assert out2.metadata["topic"] == "availability_window"


# -----------------------------------------------------------------------------
# 4. Inquire Order Status (INQUIRE_ORDER_STATUS)
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_inquire_order_status_with_and_without_order_id(orchestrator: AgentOrchestrator):
    """If order_id is missing, prompts for order_id. When provided, returns grounded status."""
    conv_id = "test_conv_order_status_04"

    # Turn 1: No order ID in context or text -> targeted clarification
    out1 = await orchestrator.process_message(conv_id, "Check my order status", farmer_id="farmer_stat")
    ctx = orchestrator.get_context(conv_id)
    assert ctx.last_intent == IntentType.INQUIRE_ORDER_STATUS.value
    assert ctx.active_clarification == "order_id"
    assert "Order ID" in out1.text
    outbound_missing = out1.metadata.get("missing_entity")
    assert outbound_missing == "order_id"

    # Create an actual backend order to query
    order = await order_service.create_order_async(
        farmer_id="farmer_stat",
        produce_type="potato",
        quantity=50.0,
        unit="kg",
        pickup_location="Warehouse North",
    )

    # Turn 2: Farmer supplies Order ID
    out2 = await orchestrator.process_message(conv_id, order.id, farmer_id="farmer_stat")
    assert ctx.last_tool == "get_order_status"
    assert ctx.active_clarification is None
    assert order.id in out2.text
    assert "50 kg of potato" in out2.text
    assert "ORDER_CONFIRMED" in out2.text


# -----------------------------------------------------------------------------
# 5. Inquire Payment Status (INQUIRE_PAYMENT_STATUS)
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_inquire_payment_status_flow(orchestrator: AgentOrchestrator):
    """Payment status inquiry checks payment record and returns authoritative payout details."""
    conv_id = "test_conv_pay_status_05"

    order = await order_service.create_order_async(
        farmer_id="farmer_payout",
        produce_type="onion",
        quantity=100.0,
        unit="kg",
        pickup_location="Shed 4",
    )
    payment, _ = await payment_service.create_payment_request(order.id)

    msg = f"Check payment status for {order.id}"
    outbound = await orchestrator.process_message(conv_id, msg, farmer_id="farmer_payout")

    ctx = orchestrator.get_context(conv_id)
    assert ctx.last_intent == IntentType.INQUIRE_PAYMENT_STATUS.value
    assert ctx.last_tool == "get_payment_status"
    assert order.id in outbound.text
    assert payment.id in outbound.text
    assert "PAYMENT_PENDING" in outbound.text
    assert "35.00" in outbound.text  # 100 kg * $0.35/kg = $35.00


# -----------------------------------------------------------------------------
# 6. Inquire Collection Status (INQUIRE_COLLECTION_STATUS)
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_inquire_collection_status_flow(orchestrator: AgentOrchestrator):
    """Collection status inquiry queries runner logistics task and reports runner assignment."""
    conv_id = "test_conv_coll_status_06"

    order = await order_service.create_order_async(
        farmer_id="farmer_runner",
        produce_type="tomato",
        quantity=30.0,
        unit="kg",
        pickup_location="Greenhouse 2",
    )
    order_service.transition_state(order.id, OrderState.PAYMENT_PENDING)
    order_service.transition_state(order.id, OrderState.PAYMENT_CONFIRMED)
    task = await collection_service.create_collection_task(order.id)
    await collection_service.accept_task(task.id, runner_id="runner_sarah_09")

    msg = f"Where is the runner for {order.id}?"
    outbound = await orchestrator.process_message(conv_id, msg, farmer_id="farmer_runner")

    ctx = orchestrator.get_context(conv_id)
    assert ctx.last_intent == IntentType.INQUIRE_COLLECTION_STATUS.value
    assert ctx.last_tool == "get_collection_status"
    assert "runner_sarah_09" in outbound.text
    assert task.id in outbound.text
    assert "COLLECTION_ASSIGNED" in outbound.text


# -----------------------------------------------------------------------------
# 7. Request Pickup Reschedule (REQUEST_PICKUP_CHANGE)
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_request_pickup_reschedule_flow(orchestrator: AgentOrchestrator):
    """Farmer requesting pickup change updates scheduled timing and reflects in collection task."""
    conv_id = "test_conv_reschedule_07"

    order = await order_service.create_order_async(
        farmer_id="farmer_resched",
        produce_type="potato",
        quantity=50.0,
        unit="kg",
        pickup_location="Barn 1",
    )
    order_service.transition_state(order.id, OrderState.PAYMENT_PENDING)
    order_service.transition_state(order.id, OrderState.PAYMENT_CONFIRMED)
    task = await collection_service.create_collection_task(order.id)

    msg = f"Please change pickup for {order.id} to tomorrow at 4 PM"
    outbound = await orchestrator.process_message(conv_id, msg, farmer_id="farmer_resched")

    ctx = orchestrator.get_context(conv_id)
    assert ctx.last_intent == IntentType.REQUEST_PICKUP_CHANGE.value
    assert ctx.last_tool == "request_pickup_reschedule"
    assert outbound.metadata["rescheduled"] is True
    assert "tomorrow at 4:00 PM" in outbound.text

    # Verify task in collection service was updated
    updated_task = collection_service.get_task(task.id)
    assert updated_task.scheduled_time_str == "tomorrow at 4:00 PM"


# -----------------------------------------------------------------------------
# 8. Report Collection Problem (REPORT_COLLECTION_PROBLEM)
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_report_collection_problem_flow(orchestrator: AgentOrchestrator):
    """Reporting a problem with collection marks the issue and flags the task."""
    conv_id = "test_conv_problem_08"

    order = await order_service.create_order_async(
        farmer_id="farmer_prob",
        produce_type="potato",
        quantity=50.0,
        unit="kg",
        pickup_location="Farm Gate",
    )
    order_service.transition_state(order.id, OrderState.PAYMENT_PENDING)
    order_service.transition_state(order.id, OrderState.PAYMENT_CONFIRMED)
    task = await collection_service.create_collection_task(order.id)
    await collection_service.accept_task(task.id, runner_id="runner_flaky")

    msg = f"Runner did not show up for {order.id}"
    outbound = await orchestrator.process_message(conv_id, msg, farmer_id="farmer_prob")

    ctx = orchestrator.get_context(conv_id)
    assert ctx.last_intent == IntentType.REPORT_COLLECTION_PROBLEM.value
    assert ctx.last_tool == "report_collection_problem"
    assert outbound.metadata["problem_reported"] is True
    assert "FAILED" in outbound.text or "exception" in outbound.text.lower()

    # Verify order transitioned to EXCEPTION
    refreshed_order = order_service.get_order(order.id)
    assert refreshed_order.status == OrderState.EXCEPTION


# -----------------------------------------------------------------------------
# 9. Multi-Turn Conversational Shifts (Offer -> Question -> Offer Continuation)
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_multiturn_conversational_shift_preserves_offer_context(orchestrator: AgentOrchestrator):
    """Farmer mid-offer asks about crops, then continues offer without losing earlier parameters."""
    conv_id = "test_conv_shift_09"

    # Turn 1: Farmer mentions produce and quantity
    t1 = await orchestrator.process_message(conv_id, "I have 60 kg of potatoes", farmer_id="farmer_shift")
    ctx = orchestrator.get_context(conv_id)
    assert ctx.offer.produce_type == "potato"
    assert ctx.offer.quantity == 60.0
    assert ctx.state == OrderState.DETAILS_PENDING

    # Turn 2: Farmer shifts to question: "What other produce do you buy?"
    t2 = await orchestrator.process_message(conv_id, "What other produce do you buy?", farmer_id="farmer_shift")
    assert ctx.last_intent == IntentType.INQUIRE_SUPPORTED_PRODUCE.value
    # Crucial: Offer context must NOT be destroyed by inquiry
    assert ctx.offer.produce_type == "potato"
    assert ctx.offer.quantity == 60.0
    assert "Tomato" in t2.text

    # Turn 3: Farmer resumes offer with location and timing
    t3 = await orchestrator.process_message(
        conv_id,
        "Pickup at Springfield Farm tomorrow at 10 AM",
        farmer_id="farmer_shift",
    )
    assert ctx.last_intent == IntentType.OFFER_PRODUCE.value
    assert ctx.state == OrderState.AWAITING_FARMER_CONFIRMATION
    assert ctx.offer.produce_type == "potato"
    assert ctx.offer.quantity == 60.0
    assert ctx.offer.pickup_location == "Springfield Farm"
    assert ctx.offer.availability_window == "tomorrow at 10:00 AM"
    assert "Total Payout: USD 24.00" in t3.text  # 60 * 0.40
    assert "Confirm" in t3.text

    # Turn 4: Farmer confirms -> Order created
    t4 = await orchestrator.process_message(conv_id, "Confirm", farmer_id="farmer_shift")
    assert ctx.state == OrderState.PAYMENT_PENDING
    assert ctx.current_order_id is not None
    assert "Order Confirmed!" in t4.text


# -----------------------------------------------------------------------------
# 10. Resilient Fallback Under Primary AI Transient Failure
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_resilient_fallback_on_primary_transient_error():
    """When primary AI intent classifier raises a transient error, fallback seamlessly classifies and responds."""
    class FlakyAIClassifier(BaseIntentClassifier):
        async def classify(self, text, context=None):
            raise ConnectionError("503 Service Unavailable: High server load")

    fallback_classifier = RuleBasedIntentClassifier()
    resilient_classifier = ResilientIntentClassifier(
        primary=FlakyAIClassifier(),
        fallback=fallback_classifier,
    )

    orchestrator = AgentOrchestrator(
        intent_classifier=resilient_classifier,
        auto_create_collection_task=False,
    )

    conv_id = "test_conv_fallback_10"
    msg = "I have 40 kg of tomatoes in Nairobi available tomorrow at 9 AM"

    # Must NOT crash or raise 503; must fallback to rule-based classification cleanly
    outbound = await orchestrator.process_message(conv_id, msg, farmer_id="farmer_resilience")

    assert resilient_classifier.last_fallback_occurred is True
    assert "503" in str(resilient_classifier.last_error)

    ctx = orchestrator.get_context(conv_id)
    assert ctx.offer.produce_type == "tomato"
    assert ctx.offer.quantity == 40.0
    assert ctx.state == OrderState.AWAITING_FARMER_CONFIRMATION
    # Tomato authoritative rate: 0.60 USD/kg -> total = 24.00 USD
    assert outbound.metadata["authoritative_rate"] == 0.60
    assert outbound.metadata["total_amount"] == 24.00
    assert "Total Payout: USD 24.00" in outbound.text
