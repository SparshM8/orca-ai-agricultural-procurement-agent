"""OpenAI-compatible AI offer extractor for local/free or remote LLMs."""

import json
import logging
import re
from typing import Optional, Dict, Any
import httpx

from orca.domain.schemas import ExtractedOffer
from orca.agent.extractors.base import BaseOfferExtractor, parse_pickup_timing
from orca.core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_EXTRACTION_PROMPT = """You are the Natural Language Extraction Engine for ORCA Agricultural Procurement.
Your role is to extract structured produce offer attributes from the farmer's natural-language message.

You must respond ONLY with a valid JSON object matching this schema:
{
  "produce_type": string or null (e.g., "potato", "tomato", "onion"),
  "quantity": number or null (e.g., 50.0),
  "unit": string or null (e.g., "kg", "ton", "bag", "lb"),
  "offered_rate": number or null (farmer's proposed rate per unit if explicitly mentioned, otherwise null),
  "pickup_location": string or null (farm address, village, town, or city name),
  "availability_window": string or null (timing or date phrase, e.g. "tomorrow at 10 AM", "Friday at 2 PM", "this week"),
  "pickup_datetime": string or null (specific date/time if mentioned, otherwise null),
  "farmer_confirmed": boolean or null (true if the message indicates acceptance/confirmation, false if cancellation, otherwise null)
}

CRITICAL RULES:
1. Normalize produce names to singular form: "potatoes" -> "potato", "tomatoes" -> "tomato", "onions" -> "onion".
2. Normalize units to standard symbols: "kilograms" -> "kg", "tons" -> "ton".
3. Extract only information explicitly present in the farmer's message or previously collected details.
4. If an attribute is missing or unknown, set its value to null.
5. Output valid JSON only. No prose, no explanations, no markdown blocks.
"""


class OpenAICompatibleAIExtractor(BaseOfferExtractor):
    """AI offer extractor using OpenAI-compatible REST API (Ollama, LM Studio, vLLM, LiteLLM, OpenAI)."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout_seconds: Optional[float] = None,
        client: Optional[httpx.AsyncClient] = None,
    ):
        self.base_url = (base_url or settings.AI_BASE_URL or "http://localhost:11434/v1").rstrip("/")
        self.api_key = api_key or settings.AI_API_KEY or "not-needed"
        self.model_name = model_name or settings.AI_MODEL_NAME or "llama3.2"
        self.temperature = temperature if temperature is not None else settings.AI_TEMPERATURE
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else settings.AI_TIMEOUT_SECONDS
        self._client = client

    async def extract(
        self, text: str, current_offer: Optional[ExtractedOffer] = None
    ) -> ExtractedOffer:
        """Call AI completion endpoint and parse structured ExtractedOffer."""
        context_str = ""
        if current_offer:
            known = {k: v for k, v in current_offer.model_dump().items() if v is not None}
            if known:
                context_str = f"\nPreviously collected details from conversation:\n{json.dumps(known, indent=2)}\n"

        user_content = f"{context_str}\nFarmer message to extract:\n\"{text}\""

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_EXTRACTION_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        endpoint = f"{self.base_url}/chat/completions"

        if self._client:
            response = await self._client.post(endpoint, json=payload, headers=headers, timeout=self.timeout_seconds)
        else:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(endpoint, json=payload, headers=headers)

        if response.status_code != 200:
            raise RuntimeError(f"AI provider returned HTTP {response.status_code}: {response.text}")

        data = response.json()
        raw_content = data["choices"][0]["message"]["content"]
        extracted_data = self._parse_json_response(raw_content)

        # Merge non-null fields with existing context
        offer = current_offer.model_copy() if current_offer else ExtractedOffer()
        for field, value in extracted_data.items():
            if hasattr(offer, field) and value is not None:
                setattr(offer, field, value)

        # Apply timing normalization
        timing = offer.pickup_datetime or offer.availability_window
        if timing:
            _, normalized_str = parse_pickup_timing(timing)
            if normalized_str:
                offer.availability_window = normalized_str
                if "at " in normalized_str.lower():
                    offer.pickup_datetime = normalized_str

        return offer

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Parse raw model text into a dictionary, stripping markdown fences if present."""
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
