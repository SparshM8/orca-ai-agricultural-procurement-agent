"""Unit tests for Human Handoff and Deterministic Exception Routing (Phase 7).

Covers:
A. Human assistance classification
B. Handoff case creation
C. Handoff case retrieval
D. Case assignment
E. Case resolution
F. Farmer-requested handoff
G. Handoff with active offer
H. Handoff with existing order
I. Transaction state immutability
J. No-match question does NOT automatically create a handoff
K. Collection-problem escalation
L. Payment-dispute escalation
M. AgentTrace
N. Demo reset clears handoff cases
O. Gemini fallback
P. Multi-turn flow (Offer -> Handoff -> Confirm -> Pay -> Collection)
"""

import pytest
from unittest.mock import AsyncMock

from orca.domain.handoff import (
    HumanHandoffCase,
    HandoffStatus,
    HandoffReason,
)
from orca.services.handoff import handoff_service
from orca.services.order import order_service
from orca.services.payment import payment_service
from orca.services.collection import collection_service
from orca.services.pricing import pricing_service
from orca.domain.intents import IntentType
from orca.domain.state_machine import OrderState
from orca.agent.intent.rule_based_classifier import RuleBasedIntentClassifier
from orca.agent.intent.resilient_classifier import ResilientIntentClassifier
from orca.agent.orchestrator import AgentOrchestrator


@pytest.fixture(autouse=True)
def clean_services():
    """Ensure clean state across tests."""
    handoff_service.clear()
    order_service.clear()
    payment_service.clear()
    collection_service.clear()
    pricing_service.reset_baseline_rates()
    yield
    handoff_service.clear()
    order_service.clear()
    payment_service.clear()
    collection_service.clear()


# =============================================================================
# A. Human Assistance Classification
# =============================================================================
@pytest.mark.asyncio
async def test_human_assistance_classification():
    classifier = RuleBasedIntentClassifier()

    queries = [
        "I need help",
        "Can someone call me?",
        "I want to talk to a human",
        "Please connect me to an agent",
        "I need a person to help with this",
        "Can I speak to someone?",
        "Can someone help me with this?",
    ]

    for q in queries:
        res = await classifier.classify(q)
        assert res.intent == IntentType.REQUEST_HUMAN_ASSISTANCE, f"Failed for query: {q}"
        assert res.confidence >= 0.9

    # Payment dispute query
    pay_res = await classifier.classify("My payment is wrong")
    assert pay_res.intent == IntentType.REQUEST_HUMAN_ASSISTANCE
    assert pay_res.entities.handoff_reason == "PAYMENT_DISPUTE"


# =============================================================================
# B. Handoff Case Creation
# =============================================================================
def test_handoff_case_creation():
    case = handoff_service.create_case(
        farmer_id="farmer_101",
        reason=HandoffReason.FARMER_REQUEST,
        summary="Farmer requested assistance with app navigation",
    )
    assert case.case_id.startswith("HC-")
    assert case.farmer_id == "farmer_101"
    assert case.reason == HandoffReason.FARMER_REQUEST
    assert case.status == HandoffStatus.OPEN
    assert case.created_at is not None


# =============================================================================
# C. Handoff Case Retrieval
# =============================================================================
def test_handoff_case_retrieval():
    case = handoff_service.create_case(
        farmer_id="farmer_102",
        reason=HandoffReason.OPERATIONAL_EXCEPTION,
        summary="Pickup location not found on GPS",
    )
    retrieved = handoff_service.get_case(case.case_id)
    assert retrieved is not None
    assert retrieved.case_id == case.case_id

    # Unknown ID returns None
    assert handoff_service.get_case("HC-NONEXISTENT") is None


# =============================================================================
# D. Case Assignment
# =============================================================================
def test_case_assignment():
    case = handoff_service.create_case(
        farmer_id="farmer_103",
        reason=HandoffReason.FARMER_REQUEST,
        summary="Requested callback",
    )
    assigned = handoff_service.assign_case(case.case_id, "agent_alice")
    assert assigned.status == HandoffStatus.ASSIGNED
    assert assigned.assigned_to == "agent_alice"


# =============================================================================
# E. Case Resolution & Cancellation
# =============================================================================
def test_case_resolution_and_cancellation():
    case1 = handoff_service.create_case(farmer_id="farmer_104", summary="Test resolution")
    resolved = handoff_service.resolve_case(case1.case_id, "Contacted farmer via phone")
    assert resolved.status == HandoffStatus.RESOLVED
    assert resolved.resolution_notes == "Contacted farmer via phone"

    case2 = handoff_service.create_case(farmer_id="farmer_105", summary="Test cancellation")
    cancelled = handoff_service.cancel_case(case2.case_id)
    assert cancelled.status == HandoffStatus.CANCELLED


# =============================================================================
# F. Farmer-Requested Handoff via Orchestrator
# =============================================================================
@pytest.mark.asyncio
async def test_farmer_requested_handoff():
    orchestrator = AgentOrchestrator(intent_classifier=RuleBasedIntentClassifier())
    conv_id = "conv_handoff_f"

    outbound = await orchestrator.process_message(
        conversation_id=conv_id,
        message_text="Please connect me to an agent",
        farmer_id="farmer_req",
    )
    assert outbound.metadata["intent"] == "REQUEST_HUMAN_ASSISTANCE"
    assert "handoff_case_id" in outbound.metadata
    case_id = outbound.metadata["handoff_case_id"]
    assert case_id.startswith("HC-")
    assert case_id in outbound.text

    # Verify case exists in service
    case = handoff_service.get_case(case_id)
    assert case is not None
    assert case.farmer_id == "farmer_req"
    assert case.reason == HandoffReason.FARMER_REQUEST


# =============================================================================
# G. Handoff with Active Offer Preserved
# =============================================================================
@pytest.mark.asyncio
async def test_handoff_with_active_offer():
    orchestrator = AgentOrchestrator(intent_classifier=RuleBasedIntentClassifier())
    conv_id = "conv_handoff_offer"

    # Turn 1: Offer 50 kg potatoes
    await orchestrator.process_message(
        conversation_id=conv_id,
        message_text="I have 50 kg of potatoes in Springfield available tomorrow at 10 AM.",
        farmer_id="farmer_act",
    )
    ctx = orchestrator.get_context(conv_id)
    assert ctx.state == OrderState.AWAITING_FARMER_CONFIRMATION
    assert ctx.offer.produce_type == "potato"
    assert ctx.offer.quantity == 50.0

    # Turn 2: Farmer requests human assistance
    outbound = await orchestrator.process_message(
        conversation_id=conv_id,
        message_text="Can someone call me?",
        farmer_id="farmer_act",
    )
    # State and active offer must remain untouched!
    assert ctx.state == OrderState.AWAITING_FARMER_CONFIRMATION
    assert ctx.offer.produce_type == "potato"
    assert ctx.offer.quantity == 50.0
    assert "HC-" in outbound.text

    # Turn 3: Confirm original offer succeeds
    confirm_out = await orchestrator.process_message(
        conversation_id=conv_id,
        message_text="Confirm",
        farmer_id="farmer_act",
    )
    assert ctx.state in [OrderState.ORDER_CONFIRMED, OrderState.PAYMENT_PENDING]
    assert ctx.current_order_id is not None


# =============================================================================
# H. Handoff with Existing Order
# =============================================================================
@pytest.mark.asyncio
async def test_handoff_with_existing_order():
    orchestrator = AgentOrchestrator(intent_classifier=RuleBasedIntentClassifier())
    conv_id = "conv_handoff_ord"

    # Create an order
    order = order_service.create_order(
        farmer_id="farmer_ord",
        produce_type="potato",
        quantity=50.0,
        unit="kg",
        pickup_location="Springfield",
        validated_rate=0.40,
        conversation_id=conv_id,
    )
    ctx = orchestrator.get_or_create_context(conv_id, farmer_id="farmer_ord")
    ctx.current_order_id = order.id
    ctx.state = OrderState.ORDER_CONFIRMED

    outbound = await orchestrator.process_message(
        conversation_id=conv_id,
        message_text="I want to talk to a human about my order",
        farmer_id="farmer_ord",
    )
    assert outbound.metadata["order_id"] == order.id
    assert order.id in outbound.text
    case = handoff_service.get_case(outbound.metadata["handoff_case_id"])
    assert case.order_id == order.id


# =============================================================================
# I. Transaction State Immutability
# =============================================================================
@pytest.mark.asyncio
async def test_transaction_state_immutability():
    orchestrator = AgentOrchestrator(intent_classifier=RuleBasedIntentClassifier())
    conv_id = "conv_handoff_immut"

    ctx = orchestrator.get_or_create_context(conv_id, farmer_id="farmer_imm")
    initial_state = ctx.state

    await orchestrator.process_message(
        conversation_id=conv_id,
        message_text="I need a person to help with this",
        farmer_id="farmer_imm",
    )
    assert ctx.state == initial_state


# =============================================================================
# J. No-Match Question Does NOT Create Handoff
# =============================================================================
@pytest.mark.asyncio
async def test_no_match_question_does_not_create_handoff():
    orchestrator = AgentOrchestrator(intent_classifier=RuleBasedIntentClassifier())
    conv_id = "conv_offtopic_test"

    initial_cases_count = len(handoff_service.list_all_cases())

    outbound = await orchestrator.process_message(
        conversation_id=conv_id,
        message_text="Who won the World Cup in 1998?",
        farmer_id="farmer_offtopic",
    )
    # Must NOT create a handoff case
    assert len(handoff_service.list_all_cases()) == initial_cases_count
    assert "HC-" not in outbound.text


# =============================================================================
# K. Collection-Problem Escalation
# =============================================================================
@pytest.mark.asyncio
async def test_collection_problem_escalation():
    orchestrator = AgentOrchestrator(intent_classifier=RuleBasedIntentClassifier())
    conv_id = "conv_coll_prob"

    # Pre-create order and task
    order = order_service.create_order(
        farmer_id="farmer_coll",
        produce_type="potato",
        quantity=50.0,
        unit="kg",
        pickup_location="Springfield",
        validated_rate=0.40,
        conversation_id=conv_id,
    )
    await payment_service.initiate_order_payment(order.id)
    await collection_service.create_collection_task(order.id)

    ctx = orchestrator.get_or_create_context(conv_id, farmer_id="farmer_coll")
    ctx.current_order_id = order.id
    ctx.state = OrderState.COLLECTION_PENDING

    outbound = await orchestrator.process_message(
        conversation_id=conv_id,
        message_text="Runner never arrived for pickup",
        farmer_id="farmer_coll",
    )
    assert outbound.metadata["problem_reported"] is True
    # Verify handoff case was created for collection failure
    cases = handoff_service.list_all_cases()
    coll_cases = [c for c in cases if c.reason == HandoffReason.COLLECTION_FAILURE]
    assert len(coll_cases) >= 1
    assert coll_cases[0].order_id == order.id


# =============================================================================
# L. Payment-Dispute Escalation
# =============================================================================
@pytest.mark.asyncio
async def test_payment_dispute_escalation():
    orchestrator = AgentOrchestrator(intent_classifier=RuleBasedIntentClassifier())
    conv_id = "conv_pay_dispute"

    order = order_service.create_order(
        farmer_id="farmer_dispute",
        produce_type="potato",
        quantity=50.0,
        unit="kg",
        pickup_location="Springfield",
        validated_rate=0.40,
        conversation_id=conv_id,
    )
    ctx = orchestrator.get_or_create_context(conv_id, farmer_id="farmer_dispute")
    ctx.current_order_id = order.id
    ctx.state = OrderState.ORDER_CONFIRMED

    outbound = await orchestrator.process_message(
        conversation_id=conv_id,
        message_text="My payment is wrong for this order",
        farmer_id="farmer_dispute",
    )
    assert outbound.metadata["handoff_reason"] == "PAYMENT_DISPUTE"
    case = handoff_service.get_case(outbound.metadata["handoff_case_id"])
    assert case.reason == HandoffReason.PAYMENT_DISPUTE
    assert case.order_id == order.id


# =============================================================================
# M. AgentTrace Recording
# =============================================================================
@pytest.mark.asyncio
async def test_agent_trace_for_handoff():
    orchestrator = AgentOrchestrator(intent_classifier=RuleBasedIntentClassifier())
    conv_id = "conv_trace_handoff"

    await orchestrator.process_message(
        conversation_id=conv_id,
        message_text="Please connect me to an agent",
        farmer_id="farmer_trace",
    )
    ctx = orchestrator.get_context(conv_id)
    assert len(ctx.traces) >= 1
    last_trace = ctx.traces[-1]
    assert last_trace.detected_intent == "REQUEST_HUMAN_ASSISTANCE"
    assert last_trace.selected_tool == "create_human_handoff"
    assert last_trace.tool_success is True
    assert last_trace.state_before == last_trace.state_after
    assert "handoff_case_id" in last_trace.tool_arguments


# =============================================================================
# N. Demo Reset Clears Handoff Cases
# =============================================================================
def test_demo_reset_clears_handoff():
    handoff_service.create_case(farmer_id="farmer_reset", summary="Test reset")
    assert len(handoff_service.list_all_cases()) == 1

    handoff_service.clear()
    assert len(handoff_service.list_all_cases()) == 0


# =============================================================================
# O. Gemini Fallback
# =============================================================================
@pytest.mark.asyncio
async def test_gemini_fallback():
    mock_gemini = AsyncMock()
    mock_gemini.classify.side_effect = Exception("ServiceUnavailable: 503 backend error")

    rule_classifier = RuleBasedIntentClassifier()
    resilient_classifier = ResilientIntentClassifier(primary=mock_gemini, fallback=rule_classifier)

    result = await resilient_classifier.classify("I want to talk to a human")
    assert result.intent == IntentType.REQUEST_HUMAN_ASSISTANCE
    assert result.fallback_occurred is True
    assert "503" in (result.fallback_reason or "")


# =============================================================================
# P. Multi-Turn Lifecycle Flow
# =============================================================================
@pytest.mark.asyncio
async def test_multi_turn_flow_with_handoff():
    orchestrator = AgentOrchestrator(intent_classifier=RuleBasedIntentClassifier())
    conv_id = "conv_full_flow"

    # Turn 1: Farmer offers 50 kg potatoes
    res1 = await orchestrator.process_message(
        conversation_id=conv_id,
        message_text="I have 50 kg of potatoes in Springfield available tomorrow at 10 AM.",
        farmer_id="farmer_flow",
    )
    ctx = orchestrator.get_context(conv_id)
    assert ctx.state == OrderState.AWAITING_FARMER_CONFIRMATION

    # Turn 2: Farmer requests human assistance mid-dialogue
    res2 = await orchestrator.process_message(
        conversation_id=conv_id,
        message_text="Can someone help me with this?",
        farmer_id="farmer_flow",
    )
    assert ctx.state == OrderState.AWAITING_FARMER_CONFIRMATION
    assert ctx.offer.produce_type == "potato"
    assert "HC-" in res2.text

    # Turn 3: Farmer confirms
    res3 = await orchestrator.process_message(
        conversation_id=conv_id,
        message_text="Confirm",
        farmer_id="farmer_flow",
    )
    assert ctx.state in [OrderState.ORDER_CONFIRMED, OrderState.PAYMENT_PENDING]
    order_id = ctx.current_order_id
    assert order_id is not None

    # Turn 4: Farmer pays
    res4 = await orchestrator.process_message(
        conversation_id=conv_id,
        message_text="Pay",
        farmer_id="farmer_flow",
    )
    assert ctx.state in [OrderState.PAYMENT_CONFIRMED, OrderState.COLLECTION_PENDING]
    task = collection_service.get_task_by_order(order_id)
    assert task is not None

    # Runner accepts & confirms pickup
    accepted_task = await collection_service.accept_task(task.id, "runner-01")
    assert accepted_task.status == "ASSIGNED"
    completed_task = await collection_service.confirm_pickup(task.id, "runner-01")
    assert completed_task.status == "COMPLETED"

    # Final order verification
    final_order = order_service.get_order(order_id)
    assert final_order.status == OrderState.COMPLETED
