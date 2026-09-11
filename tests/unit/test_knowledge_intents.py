"""Unit tests for business knowledge intent classification and resilient fallback.

Verifies:
1. Rule-based recognition of INQUIRE_BUSINESS_INFO
2. Rule-based recognition of INQUIRE_OPERATIONAL_FAQ
3. Knowledge topic extraction
4. Ambiguous knowledge queries
5. Case-insensitivity
6. Filler words resilience
7. Natural conversational phrasing
8. Order status not stolen
9. Payment status not stolen
10. Collection status not stolen
11. Pickup reschedule not stolen
12. Offer intent not stolen
13. Confirmation / payment commands not stolen
14. Gemini structured classification (mocked)
15. Gemini 429 fallback
16. Gemini 503 fallback
17. Gemini timeout fallback
18. UNKNOWN remains UNKNOWN for off-topic queries
19. Zero tool names produced by classifiers
20. Entity serialization
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from orca.domain.intents import (
    IntentType,
    ExtractedEntities,
    ConversationalIntentResult,
)
from orca.agent.intent.rule_based_classifier import RuleBasedIntentClassifier
from orca.agent.intent.gemini_classifier import GeminiIntentClassifier
from orca.agent.intent.resilient_classifier import ResilientIntentClassifier


# -----------------------------------------------------------------------------
# 1. Business Info Examples
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_business_info_examples():
    classifier = RuleBasedIntentClassifier()
    queries = [
        "How does ORCA work?",
        "What does ORCA do?",
        "Tell me about ORCA.",
        "What is your procurement model?",
        "How can I sell my produce through ORCA?",
        "How do I sell through ORCA?",
        "What is ORCA?",
    ]
    for q in queries:
        res = await classifier.classify(q)
        assert res.intent == IntentType.INQUIRE_BUSINESS_INFO, f"Failed for query: {q}"
        assert res.confidence >= 0.90


# -----------------------------------------------------------------------------
# 2. Operational FAQ Examples
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_operational_faq_examples():
    classifier = RuleBasedIntentClassifier()
    queries = [
        "How does pickup work?",
        "How do payments work?",
        "When do I get paid?",
        "What happens if the runner is late?",
        "What happens if there is a collection problem?",
        "How does an offer amendment work?",
        "How is total billing calculated?",
    ]
    for q in queries:
        res = await classifier.classify(q)
        assert res.intent == IntentType.INQUIRE_OPERATIONAL_FAQ, f"Failed for query: {q}"
        assert res.confidence >= 0.90


# -----------------------------------------------------------------------------
# 3. Knowledge Topic Extraction
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_knowledge_topic_extraction():
    classifier = RuleBasedIntentClassifier()

    # Pricing policy
    res1 = await classifier.classify("How does ORCA pricing work?")
    assert res1.intent == IntentType.INQUIRE_BUSINESS_INFO
    assert res1.entities.knowledge_topic == "PRICING_POLICY"

    res1b = await classifier.classify("How is total billing calculated?")
    assert res1b.intent == IntentType.INQUIRE_OPERATIONAL_FAQ
    assert res1b.entities.knowledge_topic == "PRICING_POLICY"

    # Logistics policy
    res2 = await classifier.classify("How does pickup work?")
    assert res2.intent == IntentType.INQUIRE_OPERATIONAL_FAQ
    assert res2.entities.knowledge_topic == "LOGISTICS_POLICY"

    # Payment policy
    res3 = await classifier.classify("How do payments work?")
    assert res3.intent == IntentType.INQUIRE_OPERATIONAL_FAQ
    assert res3.entities.knowledge_topic == "PAYMENT_POLICY"

    # Dispute resolution
    res4 = await classifier.classify("What happens if pickup fails?")
    assert res4.intent == IntentType.INQUIRE_OPERATIONAL_FAQ
    assert res4.entities.knowledge_topic == "DISPUTE_RESOLUTION"

    # Procurement process
    res5 = await classifier.classify("How does an offer amendment work?")
    assert res5.intent == IntentType.INQUIRE_OPERATIONAL_FAQ
    assert res5.entities.knowledge_topic == "PROCUREMENT_PROCESS"

    # System capabilities
    res6 = await classifier.classify("What are your limitations?")
    assert res6.intent == IntentType.INQUIRE_OPERATIONAL_FAQ
    assert res6.entities.knowledge_topic == "SYSTEM_CAPABILITIES"


# -----------------------------------------------------------------------------
# 4. Ambiguous Knowledge Query
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ambiguous_knowledge_query():
    classifier = RuleBasedIntentClassifier()
    res = await classifier.classify("What are your operational policies?")
    assert res.intent == IntentType.INQUIRE_OPERATIONAL_FAQ
    assert res.entities.knowledge_topic is None


# -----------------------------------------------------------------------------
# 5. Case-Insensitive Recognition
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_case_insensitive_recognition():
    classifier = RuleBasedIntentClassifier()
    queries = [
        ("HOW DOES ORCA WORK?", IntentType.INQUIRE_BUSINESS_INFO),
        ("hOw DoEs PiCkUp WoRk?", IntentType.INQUIRE_OPERATIONAL_FAQ),
        ("tElL mE aBoUt OrCa", IntentType.INQUIRE_BUSINESS_INFO),
        ("hOw Do PaYmEnTs WoRk?", IntentType.INQUIRE_OPERATIONAL_FAQ),
    ]
    for q, expected in queries:
        res = await classifier.classify(q)
        assert res.intent == expected, f"Failed for case variation: {q}"


# -----------------------------------------------------------------------------
# 6. Filler Words
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_filler_words():
    classifier = RuleBasedIntentClassifier()
    queries = [
        "Um, can you please tell me about ORCA?",
        "Hey there, how does pickup work?",
        "Well, honestly, what does ORCA do?",
    ]
    for q in queries:
        res = await classifier.classify(q)
        assert res.intent in [IntentType.INQUIRE_BUSINESS_INFO, IntentType.INQUIRE_OPERATIONAL_FAQ]


# -----------------------------------------------------------------------------
# 7. Natural Conversational Phrasing
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_natural_conversational_phrasing():
    classifier = RuleBasedIntentClassifier()
    phrases = [
        ("Can you explain how you guys work?", IntentType.INQUIRE_BUSINESS_INFO),
        ("How does your collection process work?", IntentType.INQUIRE_OPERATIONAL_FAQ),
        ("What's your payment policy?", IntentType.INQUIRE_OPERATIONAL_FAQ),
        ("What happens if a pickup fails?", IntentType.INQUIRE_OPERATIONAL_FAQ),
        ("Can you tell me about your procurement process?", IntentType.INQUIRE_BUSINESS_INFO),
    ]
    for q, expected in phrases:
        res = await classifier.classify(q)
        assert res.intent == expected, f"Failed for natural phrase: {q}"


# -----------------------------------------------------------------------------
# 8-13. Intent Priority & Non-Stealing Protection
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_order_status_not_stolen():
    classifier = RuleBasedIntentClassifier()
    queries = [
        "What is my order status?",
        "Check order ORD-1234",
        "Where is my order?",
        "Status of my order",
    ]
    for q in queries:
        res = await classifier.classify(q)
        assert res.intent == IntentType.INQUIRE_ORDER_STATUS, f"Stolen order status: {q}"


@pytest.mark.asyncio
async def test_payment_status_not_stolen():
    classifier = RuleBasedIntentClassifier()
    queries = [
        "Where is my payment?",
        "Did you pay?",
        "Payment status",
        "When do I get paid for ORD-1234?",
    ]
    for q in queries:
        res = await classifier.classify(q)
        assert res.intent == IntentType.INQUIRE_PAYMENT_STATUS, f"Stolen payment status: {q}"


@pytest.mark.asyncio
async def test_collection_status_not_stolen():
    classifier = RuleBasedIntentClassifier()
    queries = [
        "Where is my runner?",
        "When will the runner arrive?",
        "Who is my runner?",
        "Collection status",
    ]
    for q in queries:
        res = await classifier.classify(q)
        assert res.intent == IntentType.INQUIRE_COLLECTION_STATUS, f"Stolen collection status: {q}"


@pytest.mark.asyncio
async def test_pickup_reschedule_not_stolen():
    classifier = RuleBasedIntentClassifier()
    queries = [
        "Can you change my pickup to Friday?",
        "Reschedule pickup to tomorrow at 2 PM",
        "Can we reschedule pickup to North Farm?",
    ]
    for q in queries:
        res = await classifier.classify(q)
        assert res.intent == IntentType.REQUEST_PICKUP_CHANGE, f"Stolen pickup change: {q}"


@pytest.mark.asyncio
async def test_offer_intent_not_stolen():
    classifier = RuleBasedIntentClassifier()
    queries = [
        "I have 50 kg of potatoes in Springfield tomorrow at 10 AM",
        "Selling 100 kg tomatoes",
    ]
    for q in queries:
        res = await classifier.classify(q)
        assert res.intent == IntentType.OFFER_PRODUCE, f"Stolen offer: {q}"


@pytest.mark.asyncio
async def test_confirmation_payment_commands_not_stolen():
    classifier = RuleBasedIntentClassifier()
    confirm_res = await classifier.classify("Confirm")
    assert confirm_res.intent == IntentType.CONFIRM_ORDER

    pay_res = await classifier.classify("Pay now")
    assert pay_res.intent == IntentType.AUTHORIZE_PAYMENT


# -----------------------------------------------------------------------------
# 14. Gemini Structured Classification (Mocked)
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gemini_structured_classification_mocked():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"intent": "INQUIRE_BUSINESS_INFO", "confidence": 0.96, "knowledge_topic": "COMPANY_INFO"}'
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    classifier = GeminiIntentClassifier(api_key="test_key", client=mock_client)
    res = await classifier.classify("How does ORCA work?")
    assert res.intent == IntentType.INQUIRE_BUSINESS_INFO
    assert res.confidence == 0.96
    assert res.entities.knowledge_topic == "COMPANY_INFO"
    assert res.classifier_name == "GeminiIntentClassifier"
    assert res.fallback_occurred is False


# -----------------------------------------------------------------------------
# 15-17. Resilient Fallback Tests (429, 503, Timeout)
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gemini_429_fallback():
    mock_primary = AsyncMock(spec=GeminiIntentClassifier)
    mock_primary.classify.side_effect = Exception("429 Resource exhausted: Quota exceeded")

    fallback = RuleBasedIntentClassifier()
    resilient = ResilientIntentClassifier(primary=mock_primary, fallback=fallback)

    res = await resilient.classify("How does ORCA work?")
    assert res.intent == IntentType.INQUIRE_BUSINESS_INFO
    assert res.fallback_occurred is True
    assert "429" in res.fallback_reason


@pytest.mark.asyncio
async def test_gemini_503_fallback():
    mock_primary = AsyncMock(spec=GeminiIntentClassifier)
    mock_primary.classify.side_effect = Exception("503 Service Unavailable")

    fallback = RuleBasedIntentClassifier()
    resilient = ResilientIntentClassifier(primary=mock_primary, fallback=fallback)

    res = await resilient.classify("How does pickup work?")
    assert res.intent == IntentType.INQUIRE_OPERATIONAL_FAQ
    assert res.fallback_occurred is True
    assert "503" in res.fallback_reason


@pytest.mark.asyncio
async def test_gemini_timeout_fallback():
    mock_primary = AsyncMock(spec=GeminiIntentClassifier)
    mock_primary.classify.side_effect = asyncio.TimeoutError("Gemini API call timed out")

    fallback = RuleBasedIntentClassifier()
    resilient = ResilientIntentClassifier(primary=mock_primary, fallback=fallback)

    res = await resilient.classify("What does ORCA do?")
    assert res.intent == IntentType.INQUIRE_BUSINESS_INFO
    assert res.fallback_occurred is True
    assert "timed out" in res.fallback_reason


# -----------------------------------------------------------------------------
# 18. UNKNOWN Remains UNKNOWN
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_unknown_remains_unknown():
    classifier = RuleBasedIntentClassifier()
    queries = [
        "Who won the World Cup in 1998?",
        "Explain quantum computing to me",
        "xyz random blabla",
    ]
    for q in queries:
        res = await classifier.classify(q)
        assert res.intent == IntentType.UNKNOWN


# -----------------------------------------------------------------------------
# 19. Zero Executable Tool Names Produced
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_zero_executable_tool_names_produced():
    classifier = RuleBasedIntentClassifier()
    res = await classifier.classify("How does ORCA work?")
    # The intent classifier result contract only contains intent, entities, confidence, raw query
    assert not hasattr(res, "tool_name")
    assert not hasattr(res, "executable_tool")


# -----------------------------------------------------------------------------
# 20. Entity Serialization
# -----------------------------------------------------------------------------
def test_entity_serialization_includes_knowledge_topic():
    entities = ExtractedEntities(
        knowledge_topic="LOGISTICS_POLICY",
        order_id="ORD-1234",
    )
    d = entities.model_dump()
    assert d["knowledge_topic"] == "LOGISTICS_POLICY"
    assert d["order_id"] == "ORD-1234"

    json_str = entities.model_dump_json()
    assert '"knowledge_topic":"LOGISTICS_POLICY"' in json_str or '"knowledge_topic": "LOGISTICS_POLICY"' in json_str
