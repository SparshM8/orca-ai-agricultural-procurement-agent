"""Gemini-powered conversational intent classifier and entity extractor."""

import os
import json
import logging
import re
import asyncio
from typing import Optional, Dict, Any

from orca.domain.intents import (
    IntentType,
    ExtractedEntities,
    ConversationalIntentResult,
)
from orca.agent.intent.base import BaseIntentClassifier
from orca.agent.extractors.base import BaseOfferExtractor
from orca.agent.extractors.gemini import GeminiOfferExtractor
from orca.core.config import settings

logger = logging.getLogger(__name__)

GEMINI_INTENT_SYSTEM_INSTRUCTION = """You are the Conversational Intent Understanding Engine for ORCA Agricultural Procurement.
Your role is to classify the farmer's intent and extract relevant identifiers or entities from their natural language message.

VALID INTENTS:
- OFFER_PRODUCE: Farmer wants to sell crops, states what produce they have, or supplies missing quantity/location/timing for an ongoing offer.
- INQUIRE_SUPPORTED_PRODUCE: Farmer asks what crops/produce ORCA buys or accepts.
- INQUIRE_ORDER_STATUS: Farmer asks about the status of an existing order or checks an order ID (e.g. ORD-1234).
- INQUIRE_PAYMENT_STATUS: Farmer asks if payout/payment was made or checks payment status.
- INQUIRE_COLLECTION_STATUS: Farmer asks about pickup, logistics, runner assignment, or collection status.
- REQUEST_CLARIFICATION: Farmer asks for clarification, why a rate is what it is, or what a term means.
- CONFIRM_ORDER: Farmer explicitly confirms/accepts the proposed transaction summary (e.g., 'Confirm', 'I agree', 'Deal').
- AUTHORIZE_PAYMENT: Farmer authorizes or requests payment processing (e.g., 'Pay now', 'Make payment', 'Retry payment').
- REQUEST_PICKUP_CHANGE: Farmer asks to reschedule, move, or change pickup time or location.
- REPORT_COLLECTION_PROBLEM: Farmer reports an issue with the runner, missed pickup, or damaged crates.
- OBJECTION_PRICING: Farmer pushes back on purchase price, offers a counter rate, or says the price is too low (e.g., 'Your price is too low', 'Can you pay 0.50?').
- AMEND_OFFER: Farmer modifies, updates, or corrects an offer before or after confirmation (e.g., 'Actually I have 80 kg', 'Change that to 75 kg', 'Update pickup to Farm 2').
- INQUIRE_BUSINESS_INFO: Farmer asks general questions about ORCA, company mission, direct procurement model, or how the platform works (e.g., 'How does ORCA work?', 'What does ORCA do?', 'Tell me about ORCA').
- INQUIRE_OPERATIONAL_FAQ: Farmer asks about operational processes or policies like pickup logistics, runner collection, payment schedules, amendment rules, or dispute resolution (e.g., 'How does pickup work?', 'How do payments work?', 'When do I get paid?', 'What happens if the runner is late?').
- RECOMMEND_PRODUCE: Farmer asks what crops they can sell, what produce ORCA accepts, asks for recommendations on what to sell, or asks if a specific crop is procured (e.g., 'What can I sell through ORCA?', 'Which crops can I sell?', 'What produce do you accept?', 'Can I sell potatoes through ORCA?').
- PERSONALIZE_RECOMMENDATION: Farmer asks for personalized recommendations, asks what they should sell, what is good for them, or what they usually or previously sold (e.g., 'What would you recommend for me?', 'What should I sell?', 'What would be good for me?', 'Based on my previous orders, what can I sell?', 'What did I sell before?', 'What do I usually sell?').
- REQUEST_HUMAN_ASSISTANCE: Farmer asks for human support, asks to talk to a person/agent, asks for a callback, or reports a payment discrepancy/dispute (e.g., 'I need help', 'Can someone call me?', 'I want to talk to a human', 'Please connect me to an agent', 'My payment is wrong').
- UNKNOWN: General greetings ('hello'), off-topic remarks, or ambiguous messages that do not match the above.

CRITICAL RULES:
1. Output valid JSON matching this schema:
{
  "intent": string (must be one of the VALID INTENTS),
  "confidence": number (between 0.0 and 1.0),
  "order_id": string or null (e.g. "ORD-1234"),
  "runner_id": string or null,
  "new_pickup_location": string or null,
  "new_pickup_time": string or null,
  "problem_reason": string or null,
  "clarification_subject": string or null,
  "counter_rate": number or null,
  "future_quantity": number or null,
  "future_timing": string or null,
  "amended_field": string or null,
  "knowledge_topic": string or null (one of "COMPANY_INFO", "PROCUREMENT_PROCESS", "PRICING_POLICY", "LOGISTICS_POLICY", "PAYMENT_POLICY", "DISPUTE_RESOLUTION", "SYSTEM_CAPABILITIES", or null),
  "recommendation_produce": string or null (e.g. "potatoes"),
  "explicit_preference": string or null (e.g. "onion"),
  "handoff_reason": string or null (e.g. "PAYMENT_DISPUTE", "FARMER_REQUEST")
}
2. You must NEVER choose or output tool names. Only classify the intent and extract entities.
3. If the intent is unclear or is a general greeting, choose "UNKNOWN".
4. Output valid JSON only. No markdown formatting.
"""


class GeminiIntentClassifier(BaseIntentClassifier):
    """Conversational intent classifier powered by Google Gemini."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        offer_extractor: Optional[BaseOfferExtractor] = None,
        client: Optional[Any] = None,
    ):
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or settings.GEMINI_API_KEY
            or settings.AI_API_KEY
        )
        self.model_name = model_name or settings.AI_MODEL_NAME
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else settings.AI_TIMEOUT_SECONDS
        self.offer_extractor = offer_extractor or GeminiOfferExtractor(
            api_key=self.api_key,
            model_name=self.model_name,
            timeout_seconds=self.timeout_seconds,
            client=client,
        )
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

    async def classify(
        self, text: str, context: Optional[Any] = None
    ) -> ConversationalIntentResult:
        """Classify user intent using Gemini with structured output and transient retries."""
        client = self._get_client()

        context_info = {}
        if context:
            if getattr(context, "state", None):
                context_info["current_state"] = getattr(context.state, "value", str(context.state))
            if getattr(context, "current_order_id", None):
                context_info["active_order_id"] = context.current_order_id
            if getattr(context, "offer", None):
                known_offer = {k: v for k, v in context.offer.model_dump().items() if v is not None}
                if known_offer:
                    context_info["in_progress_offer"] = known_offer

        context_str = f"\nConversation Context: {json.dumps(context_info)}\n" if context_info else ""
        prompt = f"{context_str}Farmer message to classify: \"{text}\""

        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=GEMINI_INTENT_SYSTEM_INSTRUCTION,
            temperature=0.0,
            response_mime_type="application/json",
        )

        max_retries = 2
        retry_delay = 0.5
        response = None

        for attempt in range(max_retries + 1):
            try:
                coro = client.aio.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
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
                        "Transient error during Gemini intent classification (attempt %d/%d): %s. Retrying in %.1fs...",
                        attempt + 1,
                        max_retries,
                        sanitized,
                        retry_delay,
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 1.5
                    continue
                raise exc

        raw_text = getattr(response, "text", "") or ""
        parsed = self._parse_json(raw_text)

        intent_str = parsed.get("intent", "UNKNOWN").upper()
        try:
            intent = IntentType(intent_str)
        except ValueError:
            intent = IntentType.UNKNOWN

        confidence = float(parsed.get("confidence", 1.0))
        entities = ExtractedEntities(
            order_id=parsed.get("order_id") or (getattr(context, "current_order_id", None) if context else None),
            runner_id=parsed.get("runner_id"),
            new_pickup_location=parsed.get("new_pickup_location"),
            new_pickup_time=parsed.get("new_pickup_time"),
            problem_reason=parsed.get("problem_reason"),
            clarification_subject=parsed.get("clarification_subject"),
            counter_rate=float(parsed["counter_rate"]) if parsed.get("counter_rate") is not None else None,
            future_quantity=float(parsed["future_quantity"]) if parsed.get("future_quantity") is not None else None,
            future_timing=parsed.get("future_timing"),
            amended_field=parsed.get("amended_field"),
            knowledge_topic=parsed.get("knowledge_topic"),
            recommendation_produce=parsed.get("recommendation_produce"),
            explicit_preference=parsed.get("explicit_preference"),
            handoff_reason=parsed.get("handoff_reason"),
        )

        # Re-use existing offer extraction system for OFFER_PRODUCE and AMEND_OFFER
        if intent == IntentType.OFFER_PRODUCE:
            has_active_order = bool(context and getattr(context, "current_order_id", None))
            current_offer = None if has_active_order else (getattr(context, "offer", None) if context else None)
            entities.offer = await self.offer_extractor.extract(text, current_offer=current_offer)
        elif intent == IntentType.AMEND_OFFER:
            entities.offer = await self.offer_extractor.extract(text, current_offer=None)

        return ConversationalIntentResult(
            intent=intent,
            confidence=confidence,
            entities=entities,
            raw_query=text,
            classifier_name="GeminiIntentClassifier",
            fallback_occurred=False,
        )

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse raw model text into a dictionary safely."""
        cleaned = text.strip()
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
        else:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1:
                cleaned = cleaned[start : end + 1]

        try:
            return json.loads(cleaned)
        except Exception:
            return {}
