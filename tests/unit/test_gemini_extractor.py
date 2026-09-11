"""Unit tests verifying Google Gemini AI extraction, fallback resilience, and backend authority.

Covers all 10 required scenarios:
1. Gemini structured extraction
2. Unknown produce extraction (e.g., carrot in Delhi)
3. Different locations (Delhi, Pune, Nakuru, etc.)
4. Different quantity/unit combinations (tonnes, kilos, bags)
5. Date and time extraction ("tomorrow at 4 AM", "Friday morning", "next Wednesday afternoon")
6. Multi-turn context preservation and merging
7. Gemini connection/API error -> automatic fallback to RuleBasedPatternExtractor
8. Invalid / malformed Gemini output -> automatic fallback
9. Gemini attempting to provide a price does not bypass authoritative backend pricing
10. Missing procurement rate for extracted produce is safely handled by backend
"""
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from google.genai import errors

from orca.domain.schemas import ExtractedOffer
from orca.domain.state_machine import OrderState
from orca.agent.extractors.gemini import GeminiOfferExtractor
from orca.agent.extractors.rule_based import RuleBasedPatternExtractor
from orca.agent.extractors.resilient import ResilientOfferExtractor
from orca.agent.extractors.factory import get_offer_extractor
from orca.agent.orchestrator import AgentOrchestrator



def create_mock_gemini_client(response_dict_or_str):
    """Helper creating a mock Google GenAI client returning a given response."""
    mock_response = MagicMock()
    if isinstance(response_dict_or_str, dict):
        mock_response.text = json.dumps(response_dict_or_str)
    else:
        mock_response.text = str(response_dict_or_str)

    mock_aio_models = MagicMock()
    mock_aio_models.generate_content = AsyncMock(return_value=mock_response)

    mock_client = MagicMock()
    mock_client.aio = MagicMock()
    mock_client.aio.models = mock_aio_models
    return mock_client


# -----------------------------------------------------------------------------
# 1. Gemini structured extraction
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gemini_structured_extraction():
    """Verify GeminiOfferExtractor parses produce, quantity, unit, location, and timing."""
    payload = {
        "produce_type": "potato",
        "quantity": 50.0,
        "unit": "kg",
        "offered_rate": None,
        "pickup_location": "Nairobi",
        "availability_window": "tomorrow at 10 AM",
        "pickup_datetime": "tomorrow at 10 AM",
        "farmer_confirmed": None,
    }
    client = create_mock_gemini_client(payload)
    extractor = GeminiOfferExtractor(api_key="mock-key", model_name="mock-model", client=client)

    result = await extractor.extract("I have 50 kg of potatoes in Nairobi available tomorrow at 10 AM.")

    assert result.produce_type == "potato"
    assert result.quantity == 50.0
    assert result.unit == "kg"
    assert result.pickup_location == "Nairobi"
    assert result.availability_window == "tomorrow at 10:00 AM"
    assert result.pickup_datetime == "tomorrow at 10:00 AM"


# -----------------------------------------------------------------------------
# 2. Unknown produce extraction (e.g. carrot in Delhi)
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_unknown_produce_extraction_carrot():
    """Verify Gemini extracts open-vocabulary produce like 'carrot' not in rule-based dictionary."""
    payload = {
        "produce_type": "carrot",
        "quantity": 50.0,
        "unit": "kg",
        "offered_rate": None,
        "pickup_location": "Delhi",
        "availability_window": "tomorrow at 4 AM",
        "pickup_datetime": "tomorrow at 4 AM",
        "farmer_confirmed": None,
    }
    client = create_mock_gemini_client(payload)
    extractor = GeminiOfferExtractor(api_key="mock-key", model_name="mock-model", client=client)

    result = await extractor.extract("I have 50 kg of carrots in Delhi available tomorrow at 4 AM.")

    assert result.produce_type == "carrot"
    assert result.quantity == 50.0
    assert result.unit == "kg"
    assert result.pickup_location == "Delhi"
    assert result.availability_window == "tomorrow at 4:00 AM"
    assert result.pickup_datetime == "tomorrow at 4:00 AM"


# -----------------------------------------------------------------------------
# 3. Different locations (Delhi, Pune, Nakuru, etc.)
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_different_locations():
    """Verify extraction handles varying location formats (e.g. 'near Pune', 'Nakuru District')."""
    payload = {
        "produce_type": "onion",
        "quantity": 2.0,
        "unit": "ton",
        "pickup_location": "near Pune",
        "availability_window": "Friday morning",
        "pickup_datetime": None,
    }
    client = create_mock_gemini_client(payload)
    extractor = GeminiOfferExtractor(api_key="mock-key", model_name="mock-model", client=client)

    result = await extractor.extract("I have around 2 tonnes of onions near Pune and can hand them over Friday morning.")

    assert result.produce_type == "onion"
    assert result.pickup_location == "near Pune"
    assert result.availability_window == "Friday morning"


# -----------------------------------------------------------------------------
# 4. Different quantity/unit combinations (tonnes, kilos, bags)
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_different_quantity_unit_combinations():
    """Verify extraction handles diverse quantities and normalized units."""
    # Example: 300 kilos
    payload = {
        "produce_type": "tomato",
        "quantity": 300.0,
        "unit": "kg",
        "pickup_location": "Green Farm",
        "availability_window": "next Wednesday afternoon",
    }
    client = create_mock_gemini_client(payload)
    extractor = GeminiOfferExtractor(api_key="mock-key", model_name="mock-model", client=client)

    result = await extractor.extract("I've got 300 kilos of tomatoes ready at Green Farm next Wednesday afternoon.")

    assert result.produce_type == "tomato"
    assert result.quantity == 300.0
    assert result.unit == "kg"


# -----------------------------------------------------------------------------
# 5. Date and time extraction
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_date_and_time_extraction():
    """Verify pickup time expressions are extracted and normalized."""
    payload = {
        "produce_type": "potato",
        "quantity": 100.0,
        "unit": "kg",
        "pickup_location": "Eldoret",
        "availability_window": "tomorrow at 4 AM",
        "pickup_datetime": "tomorrow at 4 AM",
    }
    client = create_mock_gemini_client(payload)
    extractor = GeminiOfferExtractor(api_key="mock-key", model_name="mock-model", client=client)

    result = await extractor.extract("100 kg potatoes in Eldoret ready tomorrow at 4 AM")

    assert result.availability_window == "tomorrow at 4:00 AM"
    assert result.pickup_datetime == "tomorrow at 4:00 AM"


# -----------------------------------------------------------------------------
# 6. Multi-turn extraction
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_multi_turn_extraction():
    """Verify Gemini extractor merges context across multiple conversation turns."""
    # Turn 1: Produce and quantity
    t1_payload = {
        "produce_type": "potato",
        "quantity": 50.0,
        "unit": "kg",
        "pickup_location": None,
        "availability_window": None,
    }
    client1 = create_mock_gemini_client(t1_payload)
    extractor1 = GeminiOfferExtractor(api_key="mock-key", model_name="mock-model", client=client1)
    offer_t1 = await extractor1.extract("I have 50 kg of potatoes")

    assert offer_t1.produce_type == "potato"
    assert offer_t1.quantity == 50.0

    # Turn 2: Location and timing provided in second turn
    t2_payload = {
        "produce_type": "potato",
        "quantity": 50.0,
        "unit": "kg",
        "pickup_location": "Delhi",
        "availability_window": "tomorrow at 4 AM",
        "pickup_datetime": "tomorrow at 4 AM",
    }
    client2 = create_mock_gemini_client(t2_payload)
    extractor2 = GeminiOfferExtractor(api_key="mock-key", model_name="mock-model", client=client2)
    offer_t2 = await extractor2.extract("I am in Delhi, pickup tomorrow at 4 AM", current_offer=offer_t1)

    assert offer_t2.produce_type == "potato"
    assert offer_t2.quantity == 50.0
    assert offer_t2.pickup_location == "Delhi"
    assert offer_t2.availability_window == "tomorrow at 4:00 AM"


# -----------------------------------------------------------------------------
# 7. Gemini failure -> rule-based fallback
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gemini_failure_fallback():
    """Verify ResilientOfferExtractor gracefully falls back to rule-based extractor when Gemini fails."""
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("Gemini API connection error"))

    gemini_extractor = GeminiOfferExtractor(api_key="mock-key", model_name="mock-model", client=mock_client)
    resilient_extractor = ResilientOfferExtractor(
        primary=gemini_extractor,
        fallback=RuleBasedPatternExtractor(),
    )

    # Fallback should process the message seamlessly using RuleBasedPatternExtractor
    result = await resilient_extractor.extract("I have 60 kg of potatoes in Nairobi available tomorrow at 10 AM")

    assert result.produce_type == "potato"
    assert result.quantity == 60.0
    assert result.unit == "kg"
    assert result.pickup_location == "Nairobi"
    assert result.availability_window == "tomorrow at 10:00 AM"


# -----------------------------------------------------------------------------
# 8. Invalid Gemini output -> fallback
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_invalid_gemini_output_fallback():
    """Verify fallback triggers when Gemini returns malformed or non-JSON content."""
    client = create_mock_gemini_client("<html>502 Bad Gateway</html>")
    gemini_extractor = GeminiOfferExtractor(api_key="mock-key", model_name="mock-model", client=client)
    resilient_extractor = ResilientOfferExtractor(
        primary=gemini_extractor,
        fallback=RuleBasedPatternExtractor(),
    )

    result = await resilient_extractor.extract("40 kg onions in Springfield")

    assert result.produce_type == "onion"
    assert result.quantity == 40.0
    assert result.unit == "kg"
    assert result.pickup_location == "Springfield"


# -----------------------------------------------------------------------------
# 9. Gemini price does not bypass authoritative backend pricing
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gemini_price_does_not_bypass_authoritative_rate():
    """Verify backend enforces authoritative pricing even if Gemini extracts an offered_rate."""
    # Farmer asks for $0.99, but authoritative rate for potato is $0.40
    payload = {
        "produce_type": "potato",
        "quantity": 10.0,
        "unit": "kg",
        "offered_rate": 0.99,
        "pickup_location": "Springfield",
        "availability_window": "tomorrow at 10 AM",
        "pickup_datetime": "tomorrow at 10 AM",
    }
    client = create_mock_gemini_client(payload)
    gemini_extractor = GeminiOfferExtractor(api_key="mock-key", model_name="mock-model", client=client)
    orchestrator = AgentOrchestrator(extractor=gemini_extractor)

    response = await orchestrator.process_message(
        conversation_id="conv_gemini_price_check",
        message_text="I have 10 kg of potatoes in Springfield tomorrow at 10 AM. I want $0.99 per kg.",
        farmer_id="farmer_price_01",
    )

    # Authoritative rate enforced: 10 kg * $0.40 = $4.00, NOT 10 * $0.99 = $9.90
    assert response.metadata["authoritative_rate"] == 0.40
    assert response.metadata["total_amount"] == 4.00
    assert "USD 4.00" in response.text
    assert "suggested a price of USD 0.99" in response.text


# -----------------------------------------------------------------------------
# 10. Missing procurement rate handled by backend
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_missing_procurement_rate_handled_by_backend():
    """Verify backend safely rejects unconfigured produce (e.g. carrots) without crashing."""
    payload = {
        "produce_type": "carrot",
        "quantity": 50.0,
        "unit": "kg",
        "offered_rate": None,
        "pickup_location": "Delhi",
        "availability_window": "tomorrow at 4 AM",
        "pickup_datetime": "tomorrow at 4 AM",
    }
    client = create_mock_gemini_client(payload)
    gemini_extractor = GeminiOfferExtractor(api_key="mock-key", model_name="mock-model", client=client)
    orchestrator = AgentOrchestrator(extractor=gemini_extractor)

    response = await orchestrator.process_message(
        conversation_id="conv_carrot_unsupported",
        message_text="I have 50 kg of carrots in Delhi available tomorrow at 4 AM.",
        farmer_id="farmer_carrot_01",
    )

    # Backend must report that no active procurement rate exists for carrot
    assert "carrot" in response.text.lower()
    assert "do not currently" in response.text.lower()
    assert "procure" in response.text.lower()
    assert response.metadata["state"] == OrderState.DETAILS_PENDING.value


# -----------------------------------------------------------------------------
# 11. Gemini 503 ServerError -> Retries then gracefully falls back
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gemini_503_retry_and_fallback():
    """Verify Gemini 503 UNAVAILABLE / ServerError retries twice then triggers rule-based fallback."""
    mock_client = MagicMock()
    server_error = errors.ServerError(503, {"error": {"message": "The model is overloaded. Please try again later."}})
    mock_client.aio.models.generate_content = AsyncMock(side_effect=server_error)

    gemini_extractor = GeminiOfferExtractor(api_key="mock-key", model_name="mock-model", client=mock_client)
    resilient_extractor = ResilientOfferExtractor(
        primary=gemini_extractor,
        fallback=RuleBasedPatternExtractor(),
    )

    result = await resilient_extractor.extract("I have 40 kg of onions in Nairobi available tomorrow at 10 AM")

    # Initial call + 2 retries = 3 calls
    assert mock_client.aio.models.generate_content.call_count == 3
    assert resilient_extractor.last_fallback_occurred is True
    assert isinstance(resilient_extractor.last_error, errors.ServerError)

    # Fallback produced accurate offer attributes
    assert result.produce_type == "onion"
    assert result.quantity == 40.0
    assert result.unit == "kg"
    assert result.pickup_location == "Nairobi"


# -----------------------------------------------------------------------------
# 12. Transient 503 recovers on retry without triggering fallback
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gemini_transient_retry_success():
    """Verify that if Gemini encounters a transient error on attempt 1 but succeeds on retry 2, fallback is not triggered."""
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "produce_type": "potato",
        "quantity": 100.0,
        "unit": "kg",
        "pickup_location": "Delhi",
        "availability_window": "tomorrow at 4 AM",
        "pickup_datetime": "tomorrow at 4 AM",
    })

    server_error = errors.ServerError(503, {"error": {"message": "Temporary service glitch"}})
    mock_client = MagicMock()
    # 1st call fails with 503, 2nd call succeeds
    mock_client.aio.models.generate_content = AsyncMock(side_effect=[server_error, mock_response])

    gemini_extractor = GeminiOfferExtractor(api_key="mock-key", model_name="mock-model", client=mock_client)
    resilient_extractor = ResilientOfferExtractor(
        primary=gemini_extractor,
        fallback=RuleBasedPatternExtractor(),
    )

    result = await resilient_extractor.extract("I have 100 kg of potatoes in Delhi tomorrow at 4 AM")

    assert mock_client.aio.models.generate_content.call_count == 2
    assert resilient_extractor.last_fallback_occurred is False
    assert resilient_extractor.last_error is None
    assert result.produce_type == "potato"
    assert result.quantity == 100.0


# -----------------------------------------------------------------------------
# 13. TimeoutError -> Retries and gracefully falls back
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gemini_timeout_fallback():
    """Verify that timeout errors trigger transient retries and fall back to rule-based extractor."""
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=asyncio.TimeoutError("Call timed out"))

    gemini_extractor = GeminiOfferExtractor(api_key="mock-key", model_name="mock-model", client=mock_client)
    resilient_extractor = ResilientOfferExtractor(
        primary=gemini_extractor,
        fallback=RuleBasedPatternExtractor(),
    )

    result = await resilient_extractor.extract("I have 50 kg of potatoes in Nairobi available tomorrow at 10 AM")

    assert mock_client.aio.models.generate_content.call_count == 3
    assert resilient_extractor.last_fallback_occurred is True
    assert isinstance(resilient_extractor.last_error, (asyncio.TimeoutError, TimeoutError))
    assert result.produce_type == "potato"
    assert result.quantity == 50.0


# -----------------------------------------------------------------------------
# 14. Network ConnectionError -> Retries and falls back
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gemini_network_error_fallback():
    """Verify network connection failures trigger fallback without crashing."""
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=ConnectionResetError("Connection reset by peer"))

    gemini_extractor = GeminiOfferExtractor(api_key="mock-key", model_name="mock-model", client=mock_client)
    resilient_extractor = ResilientOfferExtractor(
        primary=gemini_extractor,
        fallback=RuleBasedPatternExtractor(),
    )

    result = await resilient_extractor.extract("I have 80 kg of onions in Mombasa")

    assert mock_client.aio.models.generate_content.call_count == 3
    assert resilient_extractor.last_fallback_occurred is True
    assert result.produce_type == "onion"
    assert result.quantity == 80.0


# -----------------------------------------------------------------------------
# 15. Permanent configuration error (Invalid API key) -> No retry, falls back cleanly
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_gemini_invalid_api_key_configuration():
    """Verify permanent configuration error (e.g. 400 ClientError) is reported, does not retry, and falls back."""
    mock_client = MagicMock()
    client_error = errors.ClientError(400, {"error": {"message": "API key not valid. Please pass a valid API key."}})
    mock_client.aio.models.generate_content = AsyncMock(side_effect=client_error)

    gemini_extractor = GeminiOfferExtractor(api_key="invalid-key", model_name="mock-model", client=mock_client)
    resilient_extractor = ResilientOfferExtractor(
        primary=gemini_extractor,
        fallback=RuleBasedPatternExtractor(),
    )

    result = await resilient_extractor.extract("I have 50 kg of potatoes in Nairobi available tomorrow at 10 AM")

    # Should NOT retry permanent error (only 1 call)
    assert mock_client.aio.models.generate_content.call_count == 1
    assert resilient_extractor.last_fallback_occurred is True
    assert isinstance(resilient_extractor.last_error, errors.ClientError)
    assert result.produce_type == "potato"
    assert result.quantity == 50.0


# -----------------------------------------------------------------------------
# 16. Orchestrator with resilient Gemini 503 never causes 500 error
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_orchestrator_with_resilient_gemini_503_never_500s():
    """Verify orchestrator successfully completes farmer negotiation even when Gemini throws 503."""
    mock_client = MagicMock()
    server_error = errors.ServerError(503, {"error": {"message": "Service unavailable"}})
    mock_client.aio.models.generate_content = AsyncMock(side_effect=server_error)

    gemini_extractor = GeminiOfferExtractor(api_key="mock-key", model_name="mock-model", client=mock_client)
    resilient_extractor = ResilientOfferExtractor(
        primary=gemini_extractor,
        fallback=RuleBasedPatternExtractor(),
    )
    orchestrator = AgentOrchestrator(extractor=resilient_extractor)

    # Message should be processed seamlessly via rule-based fallback without throwing 500
    response = await orchestrator.process_message(
        conversation_id="conv_live_503_safe",
        message_text="I have 50 kg of potatoes in Nairobi available tomorrow at 10 AM",
        farmer_id="farmer_503_safe",
    )

    assert response is not None
    assert response.metadata["state"] == OrderState.AWAITING_FARMER_CONFIRMATION.value
    assert "potato" in response.text.lower()
    assert "nairobi" in response.text.lower()
    assert resilient_extractor.last_fallback_occurred is True

