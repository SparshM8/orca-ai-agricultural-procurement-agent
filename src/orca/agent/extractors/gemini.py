"""Google Gemini AI offer extractor using official google-genai SDK."""

import os
import json
import logging
import re
import asyncio
from typing import Optional, Dict, Any

from orca.domain.schemas import ExtractedOffer
from orca.agent.extractors.base import BaseOfferExtractor, parse_pickup_timing
from orca.core.config import settings

logger = logging.getLogger(__name__)

GEMINI_SYSTEM_INSTRUCTION = """You are the Natural Language Extraction Engine for ORCA Agricultural Procurement.
Your role is to extract structured produce offer attributes from the farmer's natural-language message.

You must respond with a valid JSON object matching this schema:
{
  "produce_type": string or null (normalized singular name, e.g. "carrot", "onion", "potato", "tomato", "cabbage"),
  "quantity": number or null (e.g. 50.0, 2.0, 300.0),
  "unit": string or null (e.g. "kg", "ton", "bag", "quintal", "lb"),
  "offered_rate": number or null (farmer's proposed rate per unit ONLY if explicitly stated by the farmer, otherwise null),
  "pickup_location": string or null (city, village, town, district, farm address, e.g. "Delhi", "Pune", "Nairobi"),
  "availability_window": string or null (timing or date phrase, e.g. "tomorrow at 4 AM", "Friday morning", "next Wednesday afternoon"),
  "pickup_datetime": string or null (specific pickup timing if mentioned, e.g. "tomorrow at 4 AM", otherwise null),
  "farmer_confirmed": boolean or null (true if message explicitly confirms/accepts an offer, false if declines/cancels, otherwise null)
}

CRITICAL EXTRACTION RULES:
1. Normalize produce names to singular form (e.g., "carrots" -> "carrot", "onions" -> "onion", "tomatoes" -> "tomato", "potatoes" -> "potato").
2. Normalize units to standard terms: "tonnes" -> "ton", "kilos" or "kilograms" -> "kg", "pounds" -> "lb", "bags" -> "bag".
3. Extract open-vocabulary produce: do not limit yourself to specific produce types; extract whatever produce the farmer mentioned (e.g., carrot, onion, mango, tomato, wheat).
4. NEVER invent or fabricate purchase rates or totals. Only extract 'offered_rate' if the farmer explicitly asks for or suggests a specific price (e.g., "I want $0.50/kg"). If no price is mentioned by the farmer, set 'offered_rate' to null.
5. If any field is not mentioned or unknown, set its value to null.
6. Output valid JSON only. No prose, no markdown fences.
"""


class GeminiOfferExtractor(BaseOfferExtractor):
    """Real AI offer extractor powered by the official Google Gemini SDK (google-genai)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout_seconds: Optional[float] = None,
        client: Optional[Any] = None,
    ):
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or settings.GEMINI_API_KEY
            or settings.AI_API_KEY
        )
        self.model_name = model_name or settings.AI_MODEL_NAME
        self.temperature = temperature if temperature is not None else settings.AI_TEMPERATURE
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else settings.AI_TIMEOUT_SECONDS
        self._client = client

    def _get_client(self):
        """Lazily initialize Google GenAI client if not provided."""
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not configured. Set GEMINI_API_KEY in environment or .env file.")
        from google import genai
        self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _is_transient_error(self, exc: Exception) -> bool:
        """Determine if an exception represents a temporary, retryable condition."""
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            return True
        if isinstance(exc, (ConnectionError, OSError)):
            return True

        err_str = str(exc).lower()
        if (
            "quota" in err_str
            or "resource_exhausted" in err_str
            or "exceeded your current quota" in err_str
        ):
            return False

        try:
            from google.genai import errors
            if isinstance(exc, errors.ServerError):
                return True
        except ImportError:
            pass

        transient_keywords = [
            "503", "unavailable", "timeout", "timed out",
            "connection reset", "connection refused", "broken pipe",
            "overloaded",
        ]
        return any(k in err_str for k in transient_keywords)

    def _sanitize_message(self, msg: str) -> str:
        """Redact API keys or sensitive query params from logs/error strings."""
        if self.api_key and self.api_key in msg:
            msg = msg.replace(self.api_key, "[REDACTED_API_KEY]")
        return re.sub(r'([?&]key=)[^&\s]+', r'\1[REDACTED_API_KEY]', msg)

    async def extract(
        self, text: str, current_offer: Optional[ExtractedOffer] = None
    ) -> ExtractedOffer:
        """Call Google Gemini to extract structured ExtractedOffer with transient retries."""
        client = self._get_client()

        context_str = ""
        if current_offer:
            known = {k: v for k, v in current_offer.model_dump().items() if v is not None}
            if known:
                context_str = f"\nPreviously collected details from conversation:\n{json.dumps(known, indent=2)}\n"

        user_content = f"{context_str}\nFarmer message to extract:\n\"{text}\""

        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=GEMINI_SYSTEM_INSTRUCTION,
            temperature=self.temperature,
            response_mime_type="application/json",
            response_schema=ExtractedOffer,
        )

        max_retries = 2
        retry_delay = 0.5
        response = None

        for attempt in range(max_retries + 1):
            try:
                coro = client.aio.models.generate_content(
                    model=self.model_name,
                    contents=user_content,
                    config=config,
                )

                if self.timeout_seconds:
                    response = await asyncio.wait_for(coro, timeout=self.timeout_seconds)
                else:
                    response = await coro
                break
            except Exception as exc:
                sanitized = self._sanitize_message(str(exc))
                is_transient = self._is_transient_error(exc)
                if is_transient and attempt < max_retries:
                    logger.warning(
                        "Transient error contacting Gemini API (attempt %d/%d): %s. Retrying in %.1fs...",
                        attempt + 1,
                        max_retries,
                        sanitized,
                        retry_delay,
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5
                    continue

                if not is_transient:
                    logger.error(
                        "Permanent Gemini API / configuration error for model '%s': %s",
                        self.model_name,
                        sanitized,
                    )
                else:
                    logger.warning(
                        "Transient Gemini error persisted after %d retries for model '%s': %s",
                        max_retries,
                        self.model_name,
                        sanitized,
                    )
                raise exc

        raw_text = getattr(response, "text", "") or ""
        extracted_data = self._parse_json_response(raw_text)

        offer = current_offer.model_copy() if current_offer else ExtractedOffer()
        for field, value in extracted_data.items():
            if hasattr(offer, field) and value is not None:
                setattr(offer, field, value)

        # Timing normalization
        timing = offer.pickup_datetime or offer.availability_window
        if timing:
            _, normalized_str = parse_pickup_timing(timing)
            if normalized_str:
                offer.availability_window = normalized_str
                if "at " in normalized_str.lower():
                    offer.pickup_datetime = normalized_str

        return offer

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Parse raw model text into a dictionary, handling markdown blocks if present."""
        cleaned = text.strip()
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
        else:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1:
                cleaned = cleaned[start : end + 1]

        return json.loads(cleaned)
