"""Core conversational extraction and dialogue state orchestrator.

Implements SRS Phase 1:
- Natural language produce offer extraction
- Multi-turn conversation context preservation
- Targeted clarification questions
- Authoritative backend pricing boundary
- Provider-independent extractor interface
"""

import re
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, List, Any
from pydantic import BaseModel, Field

from orca.domain.schemas import ExtractedOffer, InboundMessage, OutboundMessage
from orca.domain.state_machine import OrderState
from orca.services.pricing import pricing_service
from orca.services.billing import billing_service
from orca.core.regional import get_regional_profile


class DialogueContext(BaseModel):
    """Preserves conversational dialogue state and extracted attributes across turns."""

    conversation_id: str
    farmer_id: str
    region_code: str = "GLOBAL_DEFAULT"
    state: OrderState = OrderState.OFFER_RECEIVED
    offer: ExtractedOffer = Field(default_factory=ExtractedOffer)
    history: List[Tuple[str, str]] = Field(default_factory=list)
    last_clarification_field: Optional[str] = None


class BaseOfferExtractor(ABC):
    """Abstract interface for natural-language offer extraction.
    
    Allows swapping in LLM providers (Gemini, Claude, OpenAI) or local pattern extractors.
    """

    @abstractmethod
    async def extract(
        self, text: str, current_offer: Optional[ExtractedOffer] = None
    ) -> ExtractedOffer:
        """Extract structured offer attributes from natural language message."""
        pass


class RuleBasedPatternExtractor(BaseOfferExtractor):
    """Robust, provider-independent pattern extractor for core produce offers.
    
    Handles produce types, quantities, units, availability, pickup locations,
    and farmer-provided prices without requiring external paid API calls.
    """

    # Common unit normalizations
    UNIT_MAP = {
        "kg": "kg",
        "kgs": "kg",
        "kilogram": "kg",
        "kilograms": "kg",
        "ton": "ton",
        "tons": "ton",
        "tonne": "ton",
        "tonnes": "ton",
        "quintal": "quintal",
        "quintals": "quintal",
        "lb": "lb",
        "lbs": "lb",
        "pound": "lb",
        "pounds": "lb",
        "bag": "bag",
        "bags": "bag",
    }

    PRODUCE_NORMALIZATION = {
        "potato": "potato",
        "potatoes": "potato",
        "tomato": "tomato",
        "tomatoes": "tomato",
        "onion": "onion",
        "onions": "onion",
    }

    # Common unsupported produce for testing/demonstration
    UNSUPPORTED_KNOWN = [
        "apple", "apples", "mango", "mangoes", "pineapple", "pineapples",
        "banana", "bananas", "orange", "oranges", "wheat", "rice", "papaya"
    ]

    async def extract(
        self, text: str, current_offer: Optional[ExtractedOffer] = None
    ) -> ExtractedOffer:
        """Extract attributes and merge with previous turn's context."""
        offer = current_offer.model_copy() if current_offer else ExtractedOffer()
        cleaned_text = text.strip()
        lower_text = cleaned_text.lower()

        # 1. Produce Extraction
        for raw, normalized in self.PRODUCE_NORMALIZATION.items():
            if re.search(rf"\b{raw}\b", lower_text):
                offer.produce_type = normalized
                break

        if not offer.produce_type:
            # Check for unsupported produce mentions
            for unsupp in self.UNSUPPORTED_KNOWN:
                if re.search(rf"\b{unsupp}\b", lower_text):
                    offer.produce_type = unsupp.rstrip("s")
                    break

        # 2. Quantity & Unit Extraction
        # Pattern A: Number + Unit, e.g. "10 kg", "20.5 kilograms", "5 tons"
        qty_unit_pattern = r"(?<![\$\d])(?P<qty>\d+(?:\.\d+)?)\s*(?P<unit>kg|kgs|kilograms?|tons?|tonnes?|quintals?|lbs?|pounds?|bags?)\b"
        match_qty_unit = re.search(qty_unit_pattern, lower_text)
        if match_qty_unit:
            offer.quantity = float(match_qty_unit.group("qty"))
            raw_unit = match_qty_unit.group("unit")
            offer.unit = self.UNIT_MAP.get(raw_unit, raw_unit)
        else:
            # Pattern B: Number followed directly by produce (missing unit!), e.g. "10 potatoes"
            bare_qty_match = re.search(r"\b(?P<qty>\d+(?:\.\d+)?)\s*(?:potatoes?|tomatoes?|onions?)\b", lower_text)
            if bare_qty_match and offer.quantity is None:
                offer.quantity = float(bare_qty_match.group("qty"))
                # Unit left as None

            # Pattern C: Standalone number when answering a quantity clarification
            if offer.quantity is None and re.fullmatch(r"\d+(?:\.\d+)?", cleaned_text):
                offer.quantity = float(cleaned_text)

            # Pattern D: Standalone unit when answering a unit clarification
            if offer.unit is None and lower_text in self.UNIT_MAP:
                offer.unit = self.UNIT_MAP[lower_text]

        # 3. Farmer-Provided Price Extraction
        # E.g. "$0.50/kg", "0.50 per kg", "want $0.50", "price 0.50", "at $0.50"
        price_match = re.search(
            r"(?:for|\$|price\s*(?:of|is|:)?|want|at)\s*\$?\s*(?P<price>\d+(?:\.\d+)?)\s*(?:usd|\$)?\s*(?:per|\/)?\s*(?:kg|ton|lb|bag)?",
            lower_text,
        )
        # Avoid matching quantities as prices
        if price_match:
            potential_price = float(price_match.group("price"))
            # Make sure it didn't just match the quantity
            if offer.quantity is None or abs(potential_price - offer.quantity) > 1e-4:
                offer.offered_rate = potential_price

        # 4. Pickup Location Extraction
        # E.g. "in Springfield", "at Farm 4, Route 12", "location: Greenfield", "from Springfield"
        loc_match = re.search(
            r"(?:location\s*[:=]|in|at|from)\s+([A-Z][a-zA-Z0-9\s,.-]+?)(?=(?:\s+(?:available|ready|for|pickup|this|tomorrow|on|at\s+\d|\$|\d|i\s+want)|$))",
            cleaned_text,
        )
        if loc_match:
            loc = loc_match.group(1).strip().rstrip(",.")
            if loc.lower() not in ["the morning", "the afternoon", "the farm"]:
                offer.pickup_location = loc
            elif loc.lower() == "the farm":
                offer.pickup_location = "The Farm"
        elif "springfield" in lower_text:
            offer.pickup_location = "Springfield"
        elif "greenfield" in lower_text:
            offer.pickup_location = "Greenfield"

        # 5. Availability Window & Pickup Date/Time Extraction
        # E.g. "this week", "tomorrow", "this Friday at 10 AM", "ready this Friday"
        avail_match = re.search(
            r"\b(this\s+week|tomorrow(?:\s+morning|\s+afternoon)?|today|ready\s+immediately|this\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?:\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?)?|(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?:\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?)?)\b",
            lower_text,
        )
        if avail_match:
            avail_str = avail_match.group(1).strip()
            offer.availability_window = avail_str
            if "at " in avail_str:
                offer.pickup_datetime = avail_str

        return offer


class AgentOrchestrator:
    """Conversational agent orchestrator managing extraction, dialogue state, and rate lookup."""

    def __init__(self, extractor: Optional[BaseOfferExtractor] = None):
        self.extractor = extractor or RuleBasedPatternExtractor()
        # In-memory context storage by conversation_id
        self._conversations: Dict[str, DialogueContext] = {}

    def get_or_create_context(
        self, conversation_id: str, farmer_id: str = "farmer_default", region_code: str = "GLOBAL_DEFAULT"
    ) -> DialogueContext:
        """Fetch existing conversation context or initialize a new one."""
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = DialogueContext(
                conversation_id=conversation_id,
                farmer_id=farmer_id,
                region_code=region_code,
            )
        return self._conversations[conversation_id]

    async def process_message(
        self,
        conversation_id: str,
        message_text: str,
        farmer_id: str = "farmer_default",
        region_code: str = "GLOBAL_DEFAULT",
    ) -> OutboundMessage:
        """Process an inbound farmer message through conversational extraction and dialogue state flow."""
        context = self.get_or_create_context(conversation_id, farmer_id, region_code)
        context.history.append(("farmer", message_text))

        # 1. Check for greeting or introductory messages
        normalized_words = set(re.findall(r"[a-z]+", message_text.lower()))
        greeting_words = {"hello", "hi", "hey", "greetings"}
        is_greeting = bool(normalized_words & greeting_words) or "good morning" in message_text.lower() or "good afternoon" in message_text.lower()

        extracted = await self.extractor.extract(message_text, context.offer)

        # If user sent a greeting without any produce offer
        if is_greeting and not extracted.produce_type and not extracted.quantity and not context.offer.produce_type:
            supported = pricing_service.get_supported_produce(region_code)
            reply = (
                f"Hello! I am ORCA, your agricultural procurement assistant. "
                f"I can help you sell your produce at authoritative fixed market rates. "
                f"We currently procure: {', '.join(supported)}. "
                f"What produce do you have available for sale today?"
            )
            context.history.append(("agent", reply))
            return OutboundMessage(
                recipient_id=farmer_id,
                text=reply,
                metadata={"state": context.state.value},
            )

        context.offer = extracted

        # 2. Check Produce Validity
        supported_produce = pricing_service.get_supported_produce(region_code)
        if context.offer.produce_type and context.offer.produce_type not in supported_produce:
            reply = (
                f"Thank you for contacting us. However, we currently only procure: {', '.join(supported_produce)}. "
                f"We do not currently purchase '{context.offer.produce_type}' in this region."
            )
            context.state = OrderState.DETAILS_PENDING
            context.history.append(("agent", reply))
            return OutboundMessage(
                recipient_id=farmer_id,
                text=reply,
                metadata={"state": context.state.value},
            )

        # 3. Detect Missing Required Information
        missing_fields: List[str] = []
        if not context.offer.produce_type:
            missing_fields.append("produce_type")
        if context.offer.quantity is None or context.offer.quantity <= 0:
            missing_fields.append("quantity")
        if not context.offer.unit:
            missing_fields.append("unit")
        if not context.offer.pickup_location:
            missing_fields.append("pickup_location")
        if not context.offer.availability_window and not context.offer.pickup_datetime:
            missing_fields.append("availability")

        # 4. Generate Targeted Clarification Question if info is incomplete
        if missing_fields:
            context.state = OrderState.DETAILS_PENDING
            clarification_reply = self._generate_clarification_question(context.offer, missing_fields, region_code)
            context.history.append(("agent", clarification_reply))
            return OutboundMessage(
                recipient_id=farmer_id,
                text=clarification_reply,
                metadata={"state": context.state.value, "missing_fields": missing_fields},
            )

        # 5. All Required Fields Present -> Authoritative Rate Lookup
        # Critical Rule (FR-012, FR-013): Authoritative rate comes strictly from backend service
        authoritative_rate = pricing_service.get_applicable_rate(
            produce_type=context.offer.produce_type,
            region_code=region_code,
        )
        if not authoritative_rate:
            reply = f"System Error: No rate configuration found for {context.offer.produce_type}."
            return OutboundMessage(recipient_id=farmer_id, text=reply)

        # 6. Deterministic Billing Calculation (FR-014)
        bill = billing_service.calculate_bill(
            produce_type=context.offer.produce_type,
            quantity=context.offer.quantity,
            unit=context.offer.unit,
            rate_per_unit=authoritative_rate.rate_per_unit,
            region_code=region_code,
        )

        # 7. Transition Dialogue State: RATE_VALIDATED -> AWAITING_FARMER_CONFIRMATION
        context.state = OrderState.AWAITING_FARMER_CONFIRMATION

        # 8. Compose Transaction Summary (Addressing farmer-provided price if applicable)
        summary_reply = self._generate_transaction_summary(
            context.offer, authoritative_rate.rate_per_unit, bill.currency, bill.total_amount
        )
        context.history.append(("agent", summary_reply))

        return OutboundMessage(
            recipient_id=farmer_id,
            text=summary_reply,
            metadata={
                "state": context.state.value,
                "produce": context.offer.produce_type,
                "quantity": context.offer.quantity,
                "unit": context.offer.unit,
                "authoritative_rate": authoritative_rate.rate_per_unit,
                "total_amount": bill.total_amount,
                "currency": bill.currency,
            },
        )

    def _generate_clarification_question(
        self, offer: ExtractedOffer, missing_fields: List[str], region_code: str
    ) -> str:
        """Ask a targeted clarification question for ONLY the missing attributes."""
        supported = pricing_service.get_supported_produce(region_code)

        if "produce_type" in missing_fields:
            return f"What produce do you have available for sale? (Supported: {', '.join(supported)})"

        produce = offer.produce_type

        # Quantity and unit missing
        if "quantity" in missing_fields and "unit" in missing_fields:
            return f"How much {produce} do you have available? Please specify both the quantity and measurement unit (e.g., 50 kg or 2 tons)."

        if "quantity" in missing_fields:
            return f"Could you please specify the quantity of {produce} you have available?"

        if "unit" in missing_fields:
            return f"Could you please specify the measurement unit for the {offer.quantity} {produce} (e.g., kg, tons, or bags)?"

        if "pickup_location" in missing_fields and "availability" in missing_fields:
            return f"Where should we collect the {offer.quantity} {offer.unit} of {produce}, and when will it be ready for pickup?"

        if "pickup_location" in missing_fields:
            return f"Where is the {offer.quantity} {offer.unit} of {produce} located for collection? Please provide the pickup location or farm address."

        if "availability" in missing_fields:
            return f"When will the {offer.quantity} {offer.unit} of {produce} be ready for pickup? Please specify your availability window or date."

        return "Could you please provide the missing details for your offer?"

    def _generate_transaction_summary(
        self,
        offer: ExtractedOffer,
        authoritative_rate: float,
        currency: str,
        total_amount: float,
    ) -> str:
        """Format transaction confirmation summary according to SRS guidelines."""
        price_note = ""
        if offer.offered_rate is not None and abs(offer.offered_rate - authoritative_rate) > 1e-4:
            price_note = (
                f"\nNote: You suggested a price of {currency} {offer.offered_rate:.2f} per {offer.unit}. "
                f"Our authoritative fixed procurement rate is {currency} {authoritative_rate:.2f} per {offer.unit}.\n"
            )

        timing = offer.pickup_datetime or offer.availability_window or "As scheduled"

        return (
            f"Thank you! Here is the transaction summary for your produce offer:\n"
            f"- Produce: {offer.quantity:g} {offer.unit} of {offer.produce_type}\n"
            f"- Authoritative Rate: {currency} {authoritative_rate:.2f} per {offer.unit}\n"
            f"- Total Payout: {currency} {total_amount:.2f}\n"
            f"- Pickup Location: {offer.pickup_location}\n"
            f"- Pickup Availability: {timing}\n"
            f"{price_note}\n"
            f"Please reply with 'Confirm' if you accept this transaction so we can arrange pickup and payment."
        )


orchestrator = AgentOrchestrator()
