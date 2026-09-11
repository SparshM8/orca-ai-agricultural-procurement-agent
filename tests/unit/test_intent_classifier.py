"""Unit tests for conversational intent classification, entity extraction, and deterministic tool mapping."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from google.genai import errors

from orca.domain.intents import IntentType, ToolExecutionResult
from orca.domain.schemas import ExtractedOffer
from orca.agent.intent import (
    RuleBasedIntentClassifier,
    GeminiIntentClassifier,
    ResilientIntentClassifier,
    IntentToToolMapper,
)
from orca.agent.tools import tool_registry


# -----------------------------------------------------------------------------
# 1. Rule-Based Intent Classifier: Comprehensive Intent Recognition
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_rule_based_classify_confirm_order():
    classifier = RuleBasedIntentClassifier()
    for phrase in ["confirm", "I accept the offer", "yes I agree", "deal", "sounds good"]:
        res = await classifier.classify(phrase)
        assert res.intent == IntentType.CONFIRM_ORDER, f"Failed for phrase: '{phrase}'"


@pytest.mark.asyncio
async def test_rule_based_classify_authorize_payment():
    classifier = RuleBasedIntentClassifier()
    for phrase in ["pay now", "make payment", "process payment", "retry payment", "please pay"]:
        res = await classifier.classify(phrase)
        assert res.intent == IntentType.AUTHORIZE_PAYMENT, f"Failed for phrase: '{phrase}'"


@pytest.mark.asyncio
async def test_rule_based_classify_supported_produce():
    classifier = RuleBasedIntentClassifier()
    for phrase in [
        "what produce do you buy?",
        "which crops do you purchase",
        "what can I sell?",
        "are you buying onions?",
        "do you purchase potatoes",
    ]:
        res = await classifier.classify(phrase)
        assert res.intent == IntentType.INQUIRE_SUPPORTED_PRODUCE, f"Failed for phrase: '{phrase}'"


@pytest.mark.asyncio
async def test_rule_based_classify_order_status():
    classifier = RuleBasedIntentClassifier()
    res = await classifier.classify("What is the status of my order ORD-ABC1234?")
    assert res.intent == IntentType.INQUIRE_ORDER_STATUS
    assert res.entities.order_id == "ORD-ABC1234"

    res2 = await classifier.classify("Where is my order?")
    assert res2.intent == IntentType.INQUIRE_ORDER_STATUS


@pytest.mark.asyncio
async def test_rule_based_classify_payment_status():
    classifier = RuleBasedIntentClassifier()
    for phrase in [
        "Did my payment go through?",
        "What is my payment status?",
        "Where is my payout?",
        "Has the payment been made?",
    ]:
        res = await classifier.classify(phrase)
        assert res.intent == IntentType.INQUIRE_PAYMENT_STATUS, f"Failed for phrase: '{phrase}'"


@pytest.mark.asyncio
async def test_rule_based_classify_collection_status():
    classifier = RuleBasedIntentClassifier()
    for phrase in [
        "What is the collection status?",
        "When will the runner arrive?",
        "Who is my runner?",
        "When will you collect the produce?",
    ]:
        res = await classifier.classify(phrase)
        assert res.intent == IntentType.INQUIRE_COLLECTION_STATUS, f"Failed for phrase: '{phrase}'"


@pytest.mark.asyncio
async def test_rule_based_classify_pickup_change():
    classifier = RuleBasedIntentClassifier()
    res = await classifier.classify("Can we reschedule pickup to tomorrow at 10 AM?")
    assert res.intent == IntentType.REQUEST_PICKUP_CHANGE
    assert res.entities.new_pickup_time == "tomorrow at 10:00 AM"


@pytest.mark.asyncio
async def test_rule_based_classify_collection_problem():
    classifier = RuleBasedIntentClassifier()
    res = await classifier.classify("The runner did not show up today!")
    assert res.intent == IntentType.REPORT_COLLECTION_PROBLEM
    assert res.entities.problem_reason is not None
    assert "runner did not show up" in res.entities.problem_reason


@pytest.mark.asyncio
async def test_rule_based_classify_clarification():
    classifier = RuleBasedIntentClassifier()
    res = await classifier.classify("Why is the price $0.40 per kg?")
    assert res.intent == IntentType.REQUEST_CLARIFICATION
    assert res.entities.clarification_subject is not None

    res2 = await classifier.classify("What is an availability window?")
    assert res2.intent == IntentType.REQUEST_CLARIFICATION


@pytest.mark.asyncio
async def test_rule_based_classify_offer_produce():
    classifier = RuleBasedIntentClassifier()
    res = await classifier.classify("I have 50 kg of potatoes in Nairobi tomorrow at 10 AM")
    assert res.intent == IntentType.OFFER_PRODUCE
    assert res.entities.offer is not None
    assert res.entities.offer.produce_type == "potato"
    assert res.entities.offer.quantity == 50.0
    assert res.entities.offer.unit == "kg"
    assert res.entities.offer.pickup_location == "Nairobi"


@pytest.mark.asyncio
async def test_rule_based_classify_unknown():
    classifier = RuleBasedIntentClassifier()
    for phrase in ["hello", "hi there", "good morning", "asldkfjqwpoieu", "what is the weather today"]:
        res = await classifier.classify(phrase)
        assert res.intent == IntentType.UNKNOWN, f"Expected UNKNOWN for '{phrase}'"


# -----------------------------------------------------------------------------
# 2. Gemini Intent Classifier (Mocked)
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gemini_intent_classifier_structured_result():
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "intent": "INQUIRE_ORDER_STATUS",
        "confidence": 0.98,
        "order_id": "ORD-XYZ9876",
        "runner_id": None,
        "new_pickup_location": None,
        "new_pickup_time": None,
        "problem_reason": None,
        "clarification_subject": None,
    })
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    classifier = GeminiIntentClassifier(api_key="mock-key", model_name="mock-model", client=mock_client)
    result = await classifier.classify("Check status for order ORD-XYZ9876")

    assert result.intent == IntentType.INQUIRE_ORDER_STATUS
    assert result.confidence == 0.98
    assert result.entities.order_id == "ORD-XYZ9876"


@pytest.mark.asyncio
async def test_gemini_intent_classifier_offer_produce_reuses_extractor():
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "intent": "OFFER_PRODUCE",
        "confidence": 0.95,
        "order_id": None,
        "runner_id": None,
        "new_pickup_location": None,
        "new_pickup_time": None,
        "problem_reason": None,
        "clarification_subject": None,
    })
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    mock_offer_extractor = MagicMock()
    mock_offer_extractor.extract = AsyncMock(return_value=ExtractedOffer(
        produce_type="onion",
        quantity=200.0,
        unit="kg",
        pickup_location="Pune",
    ))

    classifier = GeminiIntentClassifier(
        api_key="mock-key",
        model_name="mock-model",
        offer_extractor=mock_offer_extractor,
        client=mock_client,
    )
    result = await classifier.classify("I want to sell 200 kg onions in Pune")

    assert result.intent == IntentType.OFFER_PRODUCE
    mock_offer_extractor.extract.assert_awaited_once()
    assert result.entities.offer.produce_type == "onion"
    assert result.entities.offer.quantity == 200.0


# -----------------------------------------------------------------------------
# 3. Resilient Intent Classifier: Fallback on 503 / Error
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_resilient_intent_classifier_fallback_on_503():
    mock_client = MagicMock()
    server_error = errors.ServerError(503, {"error": {"message": "Service unavailable"}})
    mock_client.aio.models.generate_content = AsyncMock(side_effect=server_error)

    gemini_classifier = GeminiIntentClassifier(api_key="mock-key", model_name="mock-model", client=mock_client)
    resilient = ResilientIntentClassifier(
        primary=gemini_classifier,
        fallback=RuleBasedIntentClassifier(),
    )

    # Calling with a payment status query should fall back to rule-based smoothly
    result = await resilient.classify("Did my payment go through for ORD-1234?")

    assert resilient.last_fallback_occurred is True
    assert isinstance(resilient.last_error, errors.ServerError)
    assert result.intent == IntentType.INQUIRE_PAYMENT_STATUS
    assert result.entities.order_id == "ORD-1234"


# -----------------------------------------------------------------------------
# 4. Deterministic Intent-to-Tool Mapping
# -----------------------------------------------------------------------------
def test_intent_to_tool_mapping_unknown_executes_no_tool():
    """Verify UNKNOWN intent path plans zero tool execution (Constraint 4)."""
    from orca.domain.intents import ConversationalIntentResult

    intent_res = ConversationalIntentResult(intent=IntentType.UNKNOWN)
    plan = IntentToToolMapper.map_intent_to_tool(intent_res)

    assert plan.intent == IntentType.UNKNOWN
    assert plan.tool_name is None
    assert plan.arguments == {}
    assert plan.is_state_advancing is False


def test_intent_to_tool_mapping_supported_produce():
    from orca.domain.intents import ConversationalIntentResult

    intent_res = ConversationalIntentResult(intent=IntentType.INQUIRE_SUPPORTED_PRODUCE)
    plan = IntentToToolMapper.map_intent_to_tool(intent_res)

    assert plan.tool_name == "get_supported_produce"
    assert "region_code" in plan.arguments
    assert plan.is_state_advancing is False


def test_intent_to_tool_mapping_order_status_validation():
    from orca.domain.intents import ConversationalIntentResult, ExtractedEntities

    # Without order_id: validation_error is reported
    intent_no_id = ConversationalIntentResult(intent=IntentType.INQUIRE_ORDER_STATUS)
    plan_no_id = IntentToToolMapper.map_intent_to_tool(intent_no_id)
    assert plan_no_id.tool_name == "get_order_status"
    assert plan_no_id.validation_error is not None

    # With order_id: cleanly populated
    intent_with_id = ConversationalIntentResult(
        intent=IntentType.INQUIRE_ORDER_STATUS,
        entities=ExtractedEntities(order_id="ORD-1234"),
    )
    plan_with_id = IntentToToolMapper.map_intent_to_tool(intent_with_id)
    assert plan_with_id.tool_name == "get_order_status"
    assert plan_with_id.arguments["order_id"] == "ORD-1234"
    assert plan_with_id.validation_error is None


# -----------------------------------------------------------------------------
# 5. AgentToolRegistry: Execution Boundary
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_tool_registry_rejects_unauthorized_tools():
    """Verify registry rejects tools not on the approved whitelist."""
    result = await tool_registry.execute_tool("drop_database", {"table": "orders"})
    assert result.success is False
    assert "not an approved backend tool" in result.error_message


@pytest.mark.asyncio
async def test_tool_registry_executes_get_supported_produce():
    result = await tool_registry.execute_tool("get_supported_produce", {"region_code": "GLOBAL_DEFAULT"})
    assert result.success is True
    assert result.tool_name == "get_supported_produce"
    assert "items" in result.data
    assert "potato" in result.data["items"]


@pytest.mark.asyncio
async def test_tool_registry_executes_explain_requirement():
    result = await tool_registry.execute_tool("explain_requirement", {"subject": "availability window"})
    assert result.success is True
    assert result.data.get("topic") == "availability_window"
    assert "availability window is" in result.data.get("explanation", "").lower()
