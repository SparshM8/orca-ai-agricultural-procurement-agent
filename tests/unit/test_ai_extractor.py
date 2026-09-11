"""Unit tests for modular AI offer extractors, resilience fallback, and factory."""

import json
import pytest
import httpx
from unittest.mock import AsyncMock

from orca.domain.schemas import ExtractedOffer
from orca.agent.extractors import (
    BaseOfferExtractor,
    RuleBasedPatternExtractor,
    OpenAICompatibleAIExtractor,
    ResilientOfferExtractor,
    get_offer_extractor,
)
from orca.agent.orchestrator import AgentOrchestrator


@pytest.mark.asyncio
async def test_openai_compatible_extractor_success():
    """Verify OpenAICompatibleAIExtractor parses valid LLM JSON completion."""
    mock_payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "produce_type": "potato",
                        "quantity": 75.0,
                        "unit": "kg",
                        "offered_rate": 0.65,
                        "pickup_location": "Kiambu Farm",
                        "availability_window": "tomorrow at 10 AM",
                        "pickup_datetime": "tomorrow at 10 AM",
                        "farmer_confirmed": None
                    })
                }
            }
        ]
    }

    mock_response = httpx.Response(
        status_code=200,
        json=mock_payload,
        request=httpx.Request("POST", "http://localhost:11434/v1/chat/completions"),
    )

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post = AsyncMock(return_value=mock_response)

    extractor = OpenAICompatibleAIExtractor(
        base_url="http://mock-ai:11434/v1",
        client=mock_client,
    )

    result = await extractor.extract("I have 75 kg of potatoes at Kiambu Farm available tomorrow at 10 AM")

    assert result.produce_type == "potato"
    assert result.quantity == 75.0
    assert result.unit == "kg"
    assert result.offered_rate == 0.65
    assert result.pickup_location == "Kiambu Farm"
    assert result.availability_window == "tomorrow at 10:00 AM"
    assert result.pickup_datetime == "tomorrow at 10:00 AM"


@pytest.mark.asyncio
async def test_openai_compatible_extractor_markdown_fences():
    """Verify extractor handles markdown-fenced JSON output."""
    raw_markdown = """```json
{
    "produce_type": "tomato",
    "quantity": 120.0,
    "unit": "kg",
    "pickup_location": "Nakuru",
    "availability_window": "Friday at 2 PM"
}
```"""

    mock_payload = {
        "choices": [
            {
                "message": {
                    "content": raw_markdown
                }
            }
        ]
    }

    mock_response = httpx.Response(
        status_code=200,
        json=mock_payload,
        request=httpx.Request("POST", "http://localhost:11434/v1/chat/completions"),
    )

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post = AsyncMock(return_value=mock_response)

    extractor = OpenAICompatibleAIExtractor(client=mock_client)
    result = await extractor.extract("120 kg tomatoes in Nakuru ready Friday at 2 PM")

    assert result.produce_type == "tomato"
    assert result.quantity == 120.0
    assert result.unit == "kg"
    assert result.pickup_location == "Nakuru"
    assert result.availability_window == "Friday at 2:00 PM"


@pytest.mark.asyncio
async def test_resilient_extractor_calls_primary_when_healthy():
    """Verify ResilientOfferExtractor uses primary AI output when it succeeds."""
    mock_primary = AsyncMock(spec=BaseOfferExtractor)
    mock_primary.extract = AsyncMock(return_value=ExtractedOffer(
        produce_type="potato",
        quantity=50.0,
        unit="kg",
        pickup_location="Eldoret"
    ))
    mock_fallback = AsyncMock(spec=BaseOfferExtractor)

    resilient = ResilientOfferExtractor(primary=mock_primary, fallback=mock_fallback)
    result = await resilient.extract("50 kg potato in Eldoret")

    assert result.produce_type == "potato"
    assert result.quantity == 50.0
    mock_primary.extract.assert_called_once()
    mock_fallback.extract.assert_not_called()


@pytest.mark.asyncio
async def test_resilient_extractor_falls_back_on_connection_error():
    """Verify ResilientOfferExtractor gracefully falls back to RuleBasedPatternExtractor on connection error."""
    mock_primary = AsyncMock(spec=BaseOfferExtractor)
    mock_primary.extract = AsyncMock(side_effect=httpx.ConnectError("Connection refused to Ollama / vLLM"))

    fallback = RuleBasedPatternExtractor()
    resilient = ResilientOfferExtractor(primary=mock_primary, fallback=fallback)

    # Fallback extractor should parse the text seamlessly
    result = await resilient.extract("I have 60 kg of potatoes in Nairobi available tomorrow at 10 AM")

    assert result.produce_type == "potato"
    assert result.quantity == 60.0
    assert result.unit == "kg"
    assert result.pickup_location == "Nairobi"
    assert result.availability_window == "tomorrow at 10:00 AM"
    assert result.pickup_datetime == "tomorrow at 10:00 AM"


@pytest.mark.asyncio
async def test_resilient_extractor_falls_back_on_timeout_and_malformed_json():
    """Verify fallback triggers on timeout or malformed model output."""
    # 1. Timeout
    mock_timeout_primary = AsyncMock(spec=BaseOfferExtractor)
    mock_timeout_primary.extract = AsyncMock(side_effect=httpx.TimeoutException("Model timeout after 15s"))

    fallback = RuleBasedPatternExtractor()
    resilient_timeout = ResilientOfferExtractor(primary=mock_timeout_primary, fallback=fallback)

    res1 = await resilient_timeout.extract("100 kg onions in Springfield")
    assert res1.produce_type == "onion"
    assert res1.quantity == 100.0
    assert res1.unit == "kg"
    assert res1.pickup_location == "Springfield"

    # 2. Malformed JSON
    mock_bad_json_primary = AsyncMock(spec=BaseOfferExtractor)
    mock_bad_json_primary.extract = AsyncMock(side_effect=json.JSONDecodeError("Expecting value", "bad string", 0))
    resilient_bad_json = ResilientOfferExtractor(primary=mock_bad_json_primary, fallback=fallback)

    res2 = await resilient_bad_json.extract("45 kg tomatoes in Greenfield")
    assert res2.produce_type == "tomato"
    assert res2.quantity == 45.0
    assert res2.pickup_location == "Greenfield"


def test_extractor_factory():
    """Verify get_offer_extractor factory creates correct extractor classes according to config."""
    # 1. Default rule_based
    ext_default = get_offer_extractor(provider="rule_based")
    assert isinstance(ext_default, RuleBasedPatternExtractor)

    # 2. openai_compatible with fallback
    ext_ai_resilient = get_offer_extractor(provider="openai_compatible", fallback_on_failure=True)
    assert isinstance(ext_ai_resilient, ResilientOfferExtractor)
    assert isinstance(ext_ai_resilient.primary, OpenAICompatibleAIExtractor)
    assert isinstance(ext_ai_resilient.fallback, RuleBasedPatternExtractor)

    # 3. openai_compatible without fallback
    ext_ai_direct = get_offer_extractor(provider="openai_compatible", fallback_on_failure=False)
    assert isinstance(ext_ai_direct, OpenAICompatibleAIExtractor)


@pytest.mark.asyncio
async def test_orchestrator_integration_with_ai_fallback():
    """Verify AgentOrchestrator works end-to-end even when AI provider endpoint is unreachable."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Cannot connect to http://localhost:11434"))

    ai_extractor = OpenAICompatibleAIExtractor(
        base_url="http://localhost:11434/v1",
        client=mock_client,
    )
    resilient_extractor = ResilientOfferExtractor(
        primary=ai_extractor,
        fallback=RuleBasedPatternExtractor(),
    )

    orchestrator = AgentOrchestrator(extractor=resilient_extractor)
    response = await orchestrator.process_message(
        conversation_id="conv_ai_fallback_test",
        message_text="I have 50 kg of potatoes in Nairobi ready tomorrow at 10 AM.",
    )

    # Even though AI raised ConnectError, fallback extracted everything and produced a valid quote summary!
    assert "potato" in response.text.lower()
    assert "50" in response.text
    assert "Nairobi" in response.text
    assert "tomorrow at 10:00 AM" in response.text
    assert "Quote Summary" in response.text or "Confirm" in response.text
