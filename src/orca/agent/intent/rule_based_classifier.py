"""Deterministic rule-based intent classifier and entity extractor."""

import re
from typing import Optional, Any
from orca.domain.intents import (
    IntentType,
    ExtractedEntities,
    ConversationalIntentResult,
)
from orca.agent.intent.base import BaseIntentClassifier
from orca.agent.extractors.base import BaseOfferExtractor, parse_pickup_timing
from orca.agent.extractors.rule_based import RuleBasedPatternExtractor


class RuleBasedIntentClassifier(BaseIntentClassifier):
    """Deterministic regex and keyword-based intent classifier.
    
    Acts as the primary lightweight classifier and safe fallback when
    an external AI service is unavailable.
    """

    def __init__(self, offer_extractor: Optional[BaseOfferExtractor] = None):
        self.offer_extractor = offer_extractor or RuleBasedPatternExtractor()

    async def classify(
        self, text: str, context: Optional[Any] = None
    ) -> ConversationalIntentResult:
        """Classify user intent and extract entities deterministically."""
        cleaned = text.strip()
        lower_text = cleaned.lower()
        no_punct = re.sub(r"[^\w\s]", " ", lower_text).strip()

        entities = ExtractedEntities()

        # 1. Extract Order ID if present in text or fall back to context
        order_match = re.search(r"\b(ORD-[A-Fa-f0-9]{4,12})\b", text, re.IGNORECASE)
        if order_match:
            entities.order_id = order_match.group(1).upper()
        elif context and getattr(context, "current_order_id", None):
            entities.order_id = context.current_order_id

        # 1b. Check for Pricing Objection Intent (OBJECTION_PRICING)
        pricing_objection_patterns = [
            r"\b(?:too\s+low|price\s+is\s+too\s+low|rate\s+is\s+too\s+low|pay\s+(?:is\s+)?too\s+low|too\s+cheap|too\s+little)\b",
            r"\b(?:can\s+you\s+pay|can\s+you\s+give|can\s+you\s+do|can\s+you\s+offer)\s+(?:more|\$?\d+(?:\.\d+)?)\b",
            r"\b(?:pay\s+more|pay\s+better|pay\s+higher|better\s+price|better\s+rate|higher\s+price|higher\s+rate)\b",
            r"\b(?:increase\s+(?:the\s+)?(?:price|rate)|raise\s+(?:the\s+)?(?:price|rate))\b",
            r"\b(?:market\s+(?:rate|price)\s+is|mandi\s+(?:rate|price)\s+is|market\s+pays|mandi\s+pays)\b",
            r"\b(?:not\s+enough|that(?:'s|\s+is)\s+not\s+enough|give\s+me\s+more|i\s+want\s+more\s+per)\b",
            r"\b(?:can\s+i\s+get\s+more|give\s+a\s+better\s+rate|give\s+a\s+better\s+price)\b",
        ]
        is_pricing_objection = any(re.search(pat, lower_text) for pat in pricing_objection_patterns)

        if is_pricing_objection and not re.search(r"\bwhy\s+is\s+(?:the\s+)?(?:rate|price)\b", lower_text):
            counter_match = re.search(
                r"(?:(?:pay|do|offer|give|want|make\s+it|at\s+least|rate\s+is|price\s+is|pays?|to)\s+)?\$?\s*(\d+\.\d{1,2})\s*(?:usd|\$|\s*(?:per|\/)\s*(?:kg|ton|lb|bag))?",
                lower_text,
            )
            if counter_match:
                try:
                    val = float(counter_match.group(1))
                    if not re.search(rf"\b{re.escape(counter_match.group(1))}\s*(?:am|pm|kg|tons?|quintals?|lbs?|bags?)\b", lower_text):
                        entities.counter_rate = val
                except ValueError:
                    pass

            if entities.counter_rate is None:
                p_match = re.search(r"\$\s*(\d+(?:\.\d+)?)|(\d+\.\d{1,2})\s*(?:per|\/)\s*(?:kg|ton|lb|bag)", lower_text)
                if p_match:
                    entities.counter_rate = float(p_match.group(1) or p_match.group(2))

            return ConversationalIntentResult(
                intent=IntentType.OBJECTION_PRICING,
                confidence=0.95,
                entities=entities,
                raw_query=text,
            )

        # 1c. Check for Offer Amendment Intent (AMEND_OFFER)
        # 1c. Check for Offer Amendment Intent (AMEND_OFFER)
        amend_explicit_patterns = [
            r"\b(?:change\s+that\s+to|change\s+to|make\s+that|make\s+it|update\s+to|update\s+quantity\s+to|change\s+quantity\s+to|instead\s+of|i\s+meant|correct\s+that\s+to|amend|modify\s+offer)\b",
            r"\b(?:change|update|switch)\s+(?:the\s+)?(?:produce|crop|item|\w+)\s+to\b",
            r"\b(?:change|update)\s+(?:the\s+)?(?:quantity|amount)\s+to\b",
            r"\b(?:actually\s*,?\s*(?:make\s+(?:that|it)|change\s+to|update\s+to|switch\s+to))\b",
        ]
        has_amend_phrase = any(re.search(pat, lower_text) for pat in amend_explicit_patterns)

        state_val = getattr(getattr(context, "state", None), "value", str(getattr(context, "state", "")))
        is_awaiting_confirm = state_val in ["AWAITING_FARMER_CONFIRMATION", "RATE_VALIDATED"]
        has_active_order = bool(
            getattr(context, "current_order_id", None)
            or state_val in [
                "ORDER_CONFIRMED", "PAYMENT_PENDING", "PAYMENT_CONFIRMED",
                "COLLECTION_PENDING", "COLLECTION_ASSIGNED", "PICKED_UP", "COMPLETED", "EXCEPTION"
            ]
        )
        has_context_offer = bool(context and getattr(context, "offer", None) and context.offer.produce_type and context.offer.quantity)

        # Check if message is an intra-turn self-correction within an initial offer
        has_initial_correction = bool(
            re.search(r"\b(?:\d+(?:\.\d+)?)\s*(?:kg|kgs|kilos?|tons?|quintals?|lbs?|bags?)?(?:\s+(?:of\s+)?\w+)?\s*(?:—|-|,)?\s*(?:actually|wait|no(?:\s+(?:make\s+it|make\s+that))?|i\s+meant|sorry)(?:\s*,)?\s*(?:make\s+(?:it|that)\s+)?(?:\d+(?:\.\d+)?)\b", lower_text)
            or re.search(r"\b(?:actually|in\s+fact|wait)\s*,?\s*(?:it'?s\s+|make\s+(?:it|that)\s+)?(?:\d+(?:\.\d+)?)\s*(?:kg|kgs|kilos?|tons?|quintals?|lbs?|bags?)?\s*,?\s*(?:not\s+\d+(?:\.\d+)?)", lower_text)
        )

        is_amendment = False
        if has_context_offer or has_active_order:
            # When an active offer or confirmed order exists:
            has_amend_kw = has_amend_phrase or bool(re.search(r"\bactually\b", lower_text))
            if has_amend_kw:
                is_amendment = True
            elif is_awaiting_confirm and has_context_offer:
                is_amendment = (
                    bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:kg|tons?|quintals?|lbs?|bags?)\b", lower_text))
                    or bool(re.search(r"\b(?:change|update|move|pickup\s+at|switch)\b", lower_text))
                ) and not bool(re.search(r"\b(?:status|check|where|when|why|how|what|which|cancel|decline|reject)\b", lower_text)) \
                  and not bool(re.search(r"\b(?:pay|payment|payout)\b", lower_text)) \
                  and not (no_punct in {"confirm", "confirmed", "yes", "accept", "proceed", "deal"})
        else:
            # When NO active offer or order exists:
            # Self-corrections on a fresh offer (e.g. "I have 50 kg of potatoes, actually 75 kg") are OFFER_PRODUCE
            if has_initial_correction:
                is_amendment = False
            elif has_amend_phrase and not any(w in lower_text for w in ["i have", "i got", "i want to sell", "selling", "available"]):
                is_amendment = True

        if is_amendment:
            raw_cand = await self.offer_extractor.extract(text, current_offer=None)
            entities.offer = raw_cand
            if raw_cand.produce_type is not None:
                entities.amended_field = "produce"
            elif raw_cand.quantity is not None:
                entities.amended_field = "quantity"
            elif raw_cand.pickup_location is not None:
                entities.new_pickup_location = raw_cand.pickup_location
                entities.amended_field = "pickup_location"
            elif raw_cand.availability_window or raw_cand.pickup_datetime:
                entities.new_pickup_time = raw_cand.availability_window or raw_cand.pickup_datetime
                entities.amended_field = "availability"

            loc_match = re.search(r"(?i)(?:location\s*[:=]|(?:change|move|update)\s+(?:pickup|collection|location)\s+to|pickup\s+(?:location\s+)?(?:to|at))\s+([A-Za-z0-9\s,.-]+?)$", cleaned)
            if loc_match and not entities.new_pickup_location:
                loc_candidate = loc_match.group(1).strip().rstrip(".,")
                loc_cand_lower = loc_candidate.lower()
                is_time = (
                    loc_cand_lower in ["the morning", "the afternoon", "the evening", "tomorrow", "today"]
                    or bool(re.search(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b", loc_cand_lower))
                )
                is_qty = bool(re.search(r"\b\d+\s*(?:kg|tons?|quintals?|lbs?|bags?)\b", loc_cand_lower) or loc_candidate.isdigit())
                is_prod = (
                    loc_cand_lower in ["potato", "tomato", "onion", "potatoes", "tomatoes", "onions", "papaya", "mango", "mangoes", "produce", "crop"]
                    or (entities.offer and entities.offer.produce_type and loc_cand_lower == entities.offer.produce_type.lower())
                )
                if not is_time and not is_qty and not is_prod and len(loc_candidate) > 1:
                    entities.new_pickup_location = loc_candidate
                    if not entities.amended_field:
                        entities.amended_field = "pickup_location"
                    if not entities.offer:
                        entities.offer = ExtractedOffer(pickup_location=entities.new_pickup_location)
                    else:
                        entities.offer.pickup_location = entities.new_pickup_location

            avail_pattern = r"(?i)\b((?:this\s+week|ready\s+immediately|today|tomorrow)(?:\s+(?:morning|afternoon|evening))?(?:\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?)?|(?:this\s+|next\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?:\s+(?:morning|afternoon|evening))?(?:\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?)?)\b"
            avail_match = re.search(avail_pattern, text)
            if avail_match and not entities.new_pickup_time:
                _, norm_time = parse_pickup_timing(avail_match.group(1).strip())
                if norm_time:
                    entities.new_pickup_time = norm_time
                    if not entities.amended_field:
                        entities.amended_field = "availability"
                    if not entities.offer:
                        entities.offer = ExtractedOffer(availability_window=norm_time, pickup_datetime=norm_time)
                    else:
                        entities.offer.availability_window = norm_time
                        entities.offer.pickup_datetime = norm_time

            return ConversationalIntentResult(
                intent=IntentType.AMEND_OFFER,
                confidence=0.95,
                entities=entities,
                raw_query=text,
            )

        # 2. Check for Confirmation Intent (supports conversational prefixes/fillers)
        # Exclude questions, cancellations, and status inquiries
        is_inquiry_or_negation = bool(re.search(
            r"\b(?:status|check|where|when|why|how|what|which|cancel|decline|reject|don't|dont|do\s+not)\b",
            lower_text,
        ))
        is_payment_directed = bool(re.search(r"\b(?:pay|payment|payout)\b", lower_text))

        confirm_patterns = [
            r"\b(?:confirm|confirmed)\b",
            r"\b(?:accept|accepted|i\s+accept)\b",
            r"\b(?:agree|agreed|i\s+agree)\b",
            r"\b(?:proceed|let(?:'s|\s+us)?\s+proceed)\b",
            r"\b(?:deal|sounds\s+good|looks\s+good|that\s+works)\b",
            r"^(?:yes|yep|yeah|sure|ok|okay)(?:\s+(?:please|thanks|thank\s+you))?$",
        ]
        has_confirm_term = any(re.search(pat, lower_text) for pat in confirm_patterns)

        if (
            (has_confirm_term or no_punct in {"yes", "yes confirm", "yes i confirm", "yes i agree", "yes i accept", "sure", "ok", "okay"})
            and not is_inquiry_or_negation
            and not is_payment_directed
        ):
            return ConversationalIntentResult(
                intent=IntentType.CONFIRM_ORDER,
                confidence=1.0,
                entities=entities,
                raw_query=text,
            )

        # 3. Check for Payment Authorization Intent (supports conversational prefixes/fillers)
        is_payment_inquiry = bool(re.search(
            r"\b(?:status|did\s+(?:you|my|the)|has\s+(?:my|the)|have\s+you|was\s+(?:my|the)|where\s+is|check\s+(?:my|the)|cleared|receipt|payout\s+status)\b",
            lower_text,
        ))
        pay_action_patterns = [
            r"\b(?:pay\s+now|make\s+(?:the\s+)?payment|process\s+(?:the\s+)?payment|proceed\s+(?:with|to)\s+(?:the\s+)?payment|retry\s+(?:the\s+)?payment|repay|send\s+(?:the\s+)?payment|execute\s+(?:the\s+)?payment)\b",
            r"\bpay(?:\s+for\s+(?:it|this|the\s+order|my\s+order))?\b",
        ]
        has_pay_action = any(re.search(pat, lower_text) for pat in pay_action_patterns)

        if has_pay_action and not is_payment_inquiry:
            return ConversationalIntentResult(
                intent=IntentType.AUTHORIZE_PAYMENT,
                confidence=1.0,
                entities=entities,
                raw_query=text,
            )

        # 3b. Check for Personalized Recommendation Intent (PERSONALIZE_RECOMMENDATION)
        is_explicit_offer = bool(re.search(
            r"\b(?:\d+\s*(?:kg|kilos?|tons?|tonnes?|bags?|quintals?)|i\s+have|i\s+got|selling\s+\d+)\b",
            lower_text,
        ))

        # 3b-1. Check for Explicit Stated Preference
        pref_match = re.search(
            r"\b(?:i\s+usually\s+sell|i\s+typically\s+sell|i\s+normally\s+sell|i\s+prefer\s+selling|i\s+prefer\s+to\s+sell|i\s+always\s+sell|my\s+preference\s+is)\s+([A-Za-z]+)\b",
            lower_text,
        )
        if pref_match and not is_explicit_offer:
            cand_pref = pref_match.group(1).strip().lower()
            if cand_pref not in ["to", "my", "produce", "crops", "it", "this"]:
                from orca.services.farmer_profile import _normalize_commodity_name
                entities.explicit_preference = _normalize_commodity_name(cand_pref)
                return ConversationalIntentResult(
                    intent=IntentType.PERSONALIZE_RECOMMENDATION,
                    confidence=0.95,
                    entities=entities,
                    raw_query=text,
                )

        # 3b-2. Check for Personalized Recommendation Query
        personalized_patterns = [
            r"\bwhat\s+(?:would\s+you\s+recommend|do\s+you\s+recommend)\s+for\s+me\b",
            r"\brecommend(?:\s+something)?\s+for\s+me\b",
            r"\bwhat\s+should\s+i\s+(?:sell|grow)\b",
            r"\bwhat\s+would\s+be\s+good\s+for\s+me\b",
            r"\bwhat\s+is\s+good\s+for\s+me\b",
            r"\bbased\s+on\s+my\s+(?:previous|past)?\s*(?:orders|sales|history)\b",
            r"\bwhat\s+did\s+i\s+sell\s+(?:before|previously|last)\b",
            r"\bwhat\s+do\s+i\s+usually\s+sell\b",
            r"\bwhat\s+have\s+i\s+sold\b",
            r"\bcheck\s+my\s+(?:sales\s+)?history\b",
        ]
        has_personalized_match = any(re.search(pat, lower_text) for pat in personalized_patterns)
        if has_personalized_match and not is_explicit_offer:
            return ConversationalIntentResult(
                intent=IntentType.PERSONALIZE_RECOMMENDATION,
                confidence=0.95,
                entities=entities,
                raw_query=text,
            )

        # 3c. Check for Generic Recommendation Intent (RECOMMEND_PRODUCE)
        recommend_patterns = [
            r"\bwhat\s+can\s+i\s+sell\s+through\s+orca\b",
            r"\b(?:what|which)\s+(?:crops|produce)\s+can\s+i\s+sell\b",
            r"\bwhat\s+produce\s+do\s+you\s+accept\b",
            r"\bwhat\s+can\s+i\s+offer\b",
            r"\bwhat\s+else\s+can\s+i\s+sell\b",
            r"(?<!how\s)\bcan\s+i\s+sell\s+([A-Za-z]+)(?:\s+through\s+orca)?\b",
            r"\b(?:what\s+do\s+you\s+recommend|recommend\s+(?:crops|produce|what\s+to\s+sell))\b",
        ]
        has_recommend_match = any(re.search(pat, lower_text) for pat in recommend_patterns)

        if has_recommend_match and not is_explicit_offer:
            can_sell_match = re.search(r"(?<!how\s)\bcan\s+i\s+sell\s+([A-Za-z]+)(?:\s+through\s+orca)?\b", lower_text)
            if can_sell_match:
                candidate_prod = can_sell_match.group(1).strip()
                if candidate_prod.lower() not in ["it", "this", "my", "produce", "crops", "through", "orca"]:
                    entities.recommendation_produce = candidate_prod

            return ConversationalIntentResult(
                intent=IntentType.RECOMMEND_PRODUCE,
                confidence=0.95,
                entities=entities,
                raw_query=text,
            )

        # 4. Check for Supported Produce Inquiry
        if re.search(
            r"\b(?:what|which)\s+(?:do\s+you|produce|crops|vegetables|items|goods)?\s*(?:buy|purchase|procure|accept|take)\b"
            r"|\bwhat\s+can\s+i\s+sell\b"
            r"|\bare\s+you\s+(?:buying|purchasing|procuring|accepting)\b"
            r"|\bdo\s+you\s+(?:buy|purchase|procure)\b",
            lower_text,
        ):
            return ConversationalIntentResult(
                intent=IntentType.INQUIRE_SUPPORTED_PRODUCE,
                confidence=0.95,
                entities=entities,
                raw_query=text,
            )

        # 5. Check for Payment Status Inquiry
        has_order_ref = bool(entities.order_id or (context and getattr(context, "current_order_id", None)))
        is_payment_status = bool(re.search(
            r"\b(?:payment\s+status|status\s+of\s+(?:(?:my|the|an?)\s+)?payment|check\s+(?:(?:my|the|an?)\s+)?payment|where\s+is\s+(?:(?:my|the|an?)\s+)?(?:payment|payout)|(?:did|has)\s+(?:(?:my|the|an?)\s+)?payment\s+(?:go\s+through|been\s+made|cleared)|did\s+you\s+pay|payout\s+status)\b",
            lower_text,
        )) or (has_order_ref and any(w in lower_text for w in ["paid", "payment", "payout"]) and any(w in lower_text for w in ["when", "status", "check", "received", "cleared"]))

        if is_payment_status:
            return ConversationalIntentResult(
                intent=IntentType.INQUIRE_PAYMENT_STATUS,
                confidence=0.95,
                entities=entities,
                raw_query=text,
            )

        # 6. Check for Collection / Pickup Status Inquiry (supports natural collection questions)
        is_reschedule = bool(re.search(
            r"\b(?:change|reschedule|update|postpone|move)\s+(?:the\s+|my\s+)?(?:pickup|collection|timing|time|date|location|schedule)\b"
            r"|\b(?:change|reschedule|update|postpone|move)\s+to\b"
            r"|\bnew\s+pickup\s+(?:time|location|date)\b"
            r"|\bpickup\s+(?:tomorrow|later|next)\s+instead\b",
            lower_text,
        ))

        is_collection_inquiry = bool(re.search(
            r"\b(?:"
            r"collection\s+status|"
            r"pickup\s+status|"
            r"status\s+of\s+(?:the\s+|my\s+)?(?:collection|pickup)|"
            r"where\s+is\s+(?:my\s+|the\s+)?(?:runner|driver)|"
            r"when\s+will\s+(?:the\s+)?runner|"
            r"who\s+is\s+(?:coming\s+to\s+pick\s+it\s+up|my\s+runner|the\s+runner|picking\s+(?:it|the\s+produce)\s+up)|"
            r"who\s+is\s+coming\s+for\s+(?:the\s+|my\s+)?(?:pickup|collection)|"
            r"when\s+(?:is|will\s+be)\s+(?:the\s+|my\s+)?(?:pickup|collection)|"
            r"(?:what\s+is\s+(?:the\s+)?)?pickup\s+time|"
            r"when\s+will\s+(?:you\s+collect|it\s+be\s+collected|my\s+produce\s+be\s+picked\s+up|it\s+be\s+picked\s+up)|"
            r"when\s+will\s+you\s+collect|"
            r"is\s+the\s+runner\s+coming|"
            r"track\s+(?:my\s+)?(?:pickup|collection)|"
            r"is\s+it\s+(?:done|picked\s+up|completed|collected)"
            r")\b",
            lower_text,
        ))

        if is_collection_inquiry and not is_reschedule:
            return ConversationalIntentResult(
                intent=IntentType.INQUIRE_COLLECTION_STATUS,
                confidence=0.95,
                entities=entities,
                raw_query=text,
            )

        # 7. Check for Order Status Inquiry (supports colloquial conversational phrases)
        is_order_inquiry = bool(re.search(
            r"\b(?:"
            r"order\s+status|"
            r"status\s+of\s+(?:my\s+|the\s+)?order|"
            r"what(?:'s|\s+is)\s+(?:the\s+)?status\s+of\s+(?:my\s+|the\s+)?order|"
            r"what(?:'s|\s+is)\s+happening\s+with\s+(?:my\s+|the\s+)?order|"
            r"(?:any\s+)?update\s+on\s+(?:my\s+|the\s+)?order|"
            r"how\s+is\s+(?:my\s+|the\s+)?order|"
            r"(?:can\s+you\s+|could\s+you\s+)?check\s+(?:on\s+)?(?:my\s+|the\s+)?order|"
            r"where\s+is\s+(?:my\s+|the\s+)?order|"
            r"track\s+(?:my\s+|the\s+)?order|"
            r"order\s+details"
            r")\b",
            lower_text,
        )) or (entities.order_id and any(w in lower_text for w in ["status", "check", "where", "details", "update", "happening", "progress"]))

        is_order_clarification_response = bool(
            entities.order_id and (
                (context and getattr(context, "active_clarification", None) == "order_id")
                or re.fullmatch(r"(?i)\s*(?:order\s*(?:id)?\s*[:=]?\s*)?ord-[a-f0-9-]+\s*", text)
            )
        )

        if is_order_inquiry or is_order_clarification_response:
            # If farmer was specifically asked about payment or collection in context, resume that inquiry
            if context and getattr(context, "last_intent", None) == IntentType.INQUIRE_PAYMENT_STATUS.value:
                return ConversationalIntentResult(
                    intent=IntentType.INQUIRE_PAYMENT_STATUS,
                    confidence=0.95,
                    entities=entities,
                    raw_query=text,
                )
            elif context and getattr(context, "last_intent", None) == IntentType.INQUIRE_COLLECTION_STATUS.value:
                return ConversationalIntentResult(
                    intent=IntentType.INQUIRE_COLLECTION_STATUS,
                    confidence=0.95,
                    entities=entities,
                    raw_query=text,
                )
            return ConversationalIntentResult(
                intent=IntentType.INQUIRE_ORDER_STATUS,
                confidence=0.95,
                entities=entities,
                raw_query=text,
            )

        # 8. Check for Pickup Change / Reschedule Request
        if re.search(
            r"\b(?:change|reschedule|update|postpone|move)\s+(?:the\s+|my\s+)?(?:pickup|collection|timing|time|date|location|schedule)\b"
            r"|\bnew\s+pickup\s+(?:time|location|date)\b"
            r"|\bpickup\s+(?:tomorrow|later|next)\s+instead\b",
            lower_text,
        ):
            avail_pattern = r"(?i)\b((?:this\s+week|ready\s+immediately|today|tomorrow)(?:\s+(?:morning|afternoon|evening))?(?:\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?)?|(?:this\s+|next\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?:\s+(?:morning|afternoon|evening))?(?:\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?)?)\b"
            avail_match = re.search(avail_pattern, text)
            if avail_match:
                _, norm_time = parse_pickup_timing(avail_match.group(1).strip())
                if norm_time:
                    entities.new_pickup_time = norm_time
            loc_match = re.search(r"(?i)(?:location\s*[:=]|to|at)\s+([A-Za-z0-9\s,.-]+?)$", cleaned)
            if loc_match:
                entities.new_pickup_location = loc_match.group(1).strip()

            return ConversationalIntentResult(
                intent=IntentType.REQUEST_PICKUP_CHANGE,
                confidence=0.90,
                entities=entities,
                raw_query=text,
            )

        # 9. Check for Collection Problem Report
        is_hypothetical_dispute = bool(re.search(
            r"\b(?:what\s+happens\s+if|what\s+if|how\s+(?:are|do\s+you\s+handle)\s+disputes)\b",
            lower_text,
        ))
        if not is_hypothetical_dispute and re.search(
            r"\b(?:runner|pickup)\s+(?:did\s+not|didn\'t|never)\s+(?:show|shown|arrive|arrived|come|came)\b"
            r"|\bproblem\s+with\s+(?:pickup|collection|runner)\b"
            r"|\brunner\s+(?:is\s+)?late\b"
            r"|\brunner\s+rejected\b"
            r"|\bissue\s+with\s+(?:pickup|collection|runner)\b"
            r"|\bfailed\s+pickup\b",
            lower_text,
        ):
            entities.problem_reason = text
            return ConversationalIntentResult(
                intent=IntentType.REPORT_COLLECTION_PROBLEM,
                confidence=0.90,
                entities=entities,
                raw_query=text,
            )

        # 9b. Check for Human Assistance / Escalation Request (REQUEST_HUMAN_ASSISTANCE)
        human_assist_patterns = [
            r"\b(?:i\s+need\s+help|need\s+help\s+with\s+this|help\s+me\s+with\s+this|can\s+someone\s+help\s+me)\b",
            r"\b(?:talk\s+to\s+a\s+human|speak\s+to\s+a\s+human|speak\s+with\s+a\s+human|want\s+a\s+human|need\s+a\s+human)\b",
            r"\b(?:connect\s+(?:me\s+)?to\s+(?:an?\s+)?(?:agent|person|human|representative|support|someone))\b",
            r"\b(?:transfer\s+(?:me\s+)?to\s+(?:an?\s+)?(?:agent|person|human|representative|support|someone))\b",
            r"\b(?:can\s+(?:someone|a\s+person|an\s+agent)\s+call\s+me|call\s+me\s+please|have\s+someone\s+call\s+me)\b",
            r"\b(?:(?:can\s+i|i\s+want\s+to|i\s+need\s+to)\s+(?:talk|speak)\s+(?:to|with)\s+(?:someone|a\s+person|an\s+agent|a\s+representative|a\s+human))\b",
            r"\b(?:speak\s+with\s+a\s+person|talk\s+with\s+a\s+person|need\s+a\s+person)\b",
            r"\b(?:human\s+support|human\s+assistance|customer\s+support|agent\s+assistance)\b",
            r"\b(?:payment\s+dispute|discrepancy\s+in\s+(?:my\s+)?payment|i\s+was\s+underpaid|payment\s+is\s+wrong|incorrect\s+payout)\b",
        ]
        has_human_assist = any(re.search(pat, lower_text) for pat in human_assist_patterns)
        if has_human_assist and not is_explicit_offer and not is_hypothetical_dispute:
            if re.search(r"\b(?:payment\s+dispute|discrepancy\s+in\s+(?:my\s+)?payment|i\s+was\s+underpaid|payment\s+is\s+wrong|incorrect\s+payout)\b", lower_text):
                entities.handoff_reason = "PAYMENT_DISPUTE"
            else:
                entities.handoff_reason = "FARMER_REQUEST"

            return ConversationalIntentResult(
                intent=IntentType.REQUEST_HUMAN_ASSISTANCE,
                confidence=0.95,
                entities=entities,
                raw_query=text,
            )

        # 10. Check for Business Information Inquiry (INQUIRE_BUSINESS_INFO)
        business_info_patterns = [
            r"\b(?:what\s+is\s+orca|who\s+(?:is|are)\s+orca|who\s+are\s+you|about\s+orca|tell\s+me\s+about\s+orca)\b",
            r"\bhow\s+does\s+(?:orca|your\s+platform|your\s+system|this\s+platform)\s+work\b",
            r"\bwhat\s+does\s+orca\s+do\b",
            r"\b(?:what\s+is\s+your|tell\s+me\s+about\s+your)\s+(?:procurement\s+model|business\s+model|platform\s+model|company\s+model)\b",
            r"\b(?:can\s+you\s+)?(?:explain|tell\s+me)\s+(?:how\s+)?(?:you\s+guys|you\s+all|orca)\s+work\b",
            r"\b(?:tell\s+me\s+about\s+your|what\s+is\s+your)\s+procurement\s+process\b",
            r"\bhow\s+can\s+i\s+sell\s+(?:my\s+)?(?:produce|crops|harvest)\s+through\s+orca\b",
            r"\bhow\s+do\s+i\s+sell\s+(?:through|with)\s+orca\b",
            r"\bhow\s+does\s+orca\s+(?:pricing|collection|payment|procurement)\s+work\b",
        ]
        if any(re.search(pat, lower_text) for pat in business_info_patterns):
            if re.search(r"\b(?:pricing|prices|rate|rates)\b", lower_text):
                entities.knowledge_topic = "PRICING_POLICY"
            elif re.search(r"\b(?:pickup|collection|runner|logistics)\b", lower_text):
                entities.knowledge_topic = "LOGISTICS_POLICY"
            elif re.search(r"\b(?:payment|payout|paid)\b", lower_text):
                entities.knowledge_topic = "PAYMENT_POLICY"
            elif re.search(r"\b(?:procurement\s+process|sell|selling|harvest)\b", lower_text):
                entities.knowledge_topic = "PROCUREMENT_PROCESS"
            else:
                entities.knowledge_topic = "COMPANY_INFO"

            return ConversationalIntentResult(
                intent=IntentType.INQUIRE_BUSINESS_INFO,
                confidence=0.95,
                entities=entities,
                raw_query=text,
            )

        # 11. Check for Operational FAQ Inquiry (INQUIRE_OPERATIONAL_FAQ)
        is_logistics_faq = bool(re.search(
            r"\b(?:"
            r"how\s+does\s+(?:the\s+)?(?:pickup|collection|runner\s+pickup|runner\s+collection)\s+work|"
            r"how\s+do\s+pickups\s+work|"
            r"who\s+picks\s+up\s+(?:the\s+)?produce|"
            r"who\s+collects\s+(?:the\s+)?produce|"
            r"what(?:'s|\s+is)\s+(?:the\s+|your\s+)?(?:pickup|collection)\s+policy|"
            r"how\s+does\s+your\s+collection\s+process\s+work"
            r")\b",
            lower_text,
        ))

        is_payment_faq = bool(re.search(
            r"\b(?:"
            r"how\s+do\s+payments\s+work|"
            r"how\s+does\s+(?:the\s+)?(?:payment|payout)\s+work|"
            r"what(?:'s|\s+is)\s+(?:the\s+|your\s+)?(?:payment|payout)\s+policy|"
            r"what\s+payment\s+methods?\s+do\s+you|"
            r"how\s+do\s+i\s+receive\s+payment|"
            r"when\s+do\s+farmers\s+get\s+paid"
            r")\b",
            lower_text,
        ))
        if not is_payment_faq and re.search(r"\bwhen\s+do\s+i\s+get\s+paid\b", lower_text):
            has_order_ctx = bool(context and (getattr(context, "current_order_id", None) or getattr(context, "state", None) in [
                "ORDER_CONFIRMED", "PAYMENT_PENDING", "PAYMENT_CONFIRMED", "COLLECTION_PENDING", "COLLECTION_ASSIGNED", "PICKED_UP", "COMPLETED"
            ]))
            if not has_order_ctx and not bool(re.search(r"\b(?:my\s+order|order\s+ord-|for\s+ord-)\b", lower_text)):
                is_payment_faq = True

        is_dispute_faq = bool(re.search(
            r"\b(?:"
            r"what\s+happens\s+if\s+(?:the\s+)?(?:runner\s+is\s+late|runner\s+does\s+not\s+show|pickup\s+fails|there\s+is\s+a\s+collection\s+problem)|"
            r"what\s+if\s+(?:the\s+)?(?:runner\s+does\s+not\s+show|runner\s+is\s+late|pickup\s+fails)|"
            r"how\s+(?:are|do\s+you\s+handle)\s+(?:disputes|complaints)|"
            r"what\s+happens\s+if\s+a\s+pickup\s+fails"
            r")\b",
            lower_text,
        ))

        is_procurement_faq = bool(re.search(
            r"\b(?:"
            r"how\s+does\s+(?:an?\s+)?(?:offer\s+amendment|amendment)\s+work|"
            r"can\s+i\s+(?:amend|change|modify)\s+my\s+offer\s+before\s+confirming|"
            r"what\s+happens\s+after\s+(?:i\s+confirm|order\s+confirmation)|"
            r"how\s+does\s+order\s+confirmation\s+work"
            r")\b",
            lower_text,
        )) and not bool(entities.order_id) and not (context and getattr(context, "current_order_id", None))

        is_pricing_faq = bool(re.search(
            r"\b(?:"
            r"how\s+is\s+(?:the\s+)?(?:total\s+)?(?:billing|bill)\s+calculated|"
            r"how\s+is\s+total\s+calculated|"
            r"how\s+does\s+billing\s+work"
            r")\b",
            lower_text,
        ))

        is_capabilities_faq = bool(re.search(
            r"\b(?:"
            r"what\s+can\s+orca\s+do\s+and\s+what\s+are\s+its\s+limitations|"
            r"what\s+are\s+(?:the\s+)?capabilities\s+of\s+orca|"
            r"can\s+orca\s+give\s+agricultural\s+loans\s+or\s+insurance|"
            r"what\s+are\s+your\s+limitations"
            r")\b",
            lower_text,
        ))

        is_general_operational_faq = bool(re.search(
            r"\b(?:"
            r"what\s+are\s+your\s+operational\s+policies|"
            r"how\s+do\s+your\s+operations\s+work|"
            r"tell\s+me\s+about\s+your\s+policies|"
            r"what\s+are\s+your\s+policies"
            r")\b",
            lower_text,
        ))

        if (
            is_logistics_faq
            or is_payment_faq
            or is_dispute_faq
            or is_procurement_faq
            or is_pricing_faq
            or is_capabilities_faq
            or is_general_operational_faq
        ):
            if is_logistics_faq:
                entities.knowledge_topic = "LOGISTICS_POLICY"
            elif is_payment_faq:
                entities.knowledge_topic = "PAYMENT_POLICY"
            elif is_dispute_faq:
                entities.knowledge_topic = "DISPUTE_RESOLUTION"
            elif is_procurement_faq:
                entities.knowledge_topic = "PROCUREMENT_PROCESS"
            elif is_pricing_faq:
                entities.knowledge_topic = "PRICING_POLICY"
            elif is_capabilities_faq:
                entities.knowledge_topic = "SYSTEM_CAPABILITIES"
            else:
                entities.knowledge_topic = None

            return ConversationalIntentResult(
                intent=IntentType.INQUIRE_OPERATIONAL_FAQ,
                confidence=0.92,
                entities=entities,
                raw_query=text,
            )

        # 12. Check for Clarification Request
        if re.search(
            r"\b(?:why\s+is\s+(?:the\s+)?(?:rate|price)|what\s+is\s+(?:an?\s+)?(?:availability\s+window|pickup\s+window)|can\s+you\s+explain|what\s+does\s+.*\s+mean)\b",
            lower_text,
        ):
            entities.clarification_subject = text
            return ConversationalIntentResult(
                intent=IntentType.REQUEST_CLARIFICATION,
                confidence=0.90,
                entities=entities,
                raw_query=text,
            )

        # 11. Produce Offer / Selling Intent Check
        # Rule 1 & 2: Evaluate raw current message text alone for actual offer evidence
        raw_offer = await self.offer_extractor.extract(text, current_offer=None)

        current_has_produce = bool(raw_offer.produce_type)
        current_has_qty = raw_offer.quantity is not None
        current_has_loc = bool(raw_offer.pickup_location)
        current_has_timing = bool(raw_offer.availability_window or raw_offer.pickup_datetime)

        offer_keywords = {
            "have", "got", "selling", "sell", "offer", "harvest",
            "crates", "bags", "kilos", "kg", "tons", "tonnes", "quintals", "acres"
        }
        current_has_keywords = any(w in lower_text.split() for w in offer_keywords)

        has_active_order = bool(
            context and getattr(context, "current_order_id", None)
        )

        # Rule 3: If an active order exists, an unmatched message must NEVER fall back
        # to OFFER_PRODUCE merely because historical context.offer exists.
        if has_active_order:
            if (current_has_produce and current_has_qty) or (current_has_keywords and current_has_produce):
                entities.offer = raw_offer
                return ConversationalIntentResult(
                    intent=IntentType.OFFER_PRODUCE,
                    confidence=0.90,
                    entities=entities,
                    raw_query=text,
                )
            return ConversationalIntentResult(
                intent=IntentType.UNKNOWN,
                confidence=0.5,
                entities=entities,
                raw_query=text,
            )

        # When no active order exists:
        # Rule 5: Preserve legitimate multi-turn offer completion when genuinely in DETAILS_PENDING or OFFER_RECEIVED
        is_context_pending_offer = False
        if context and not has_active_order:
            state_val = getattr(context.state, "value", str(getattr(context, "state", "")))
            if state_val in ["OFFER_RECEIVED", "DETAILS_PENDING"]:
                is_context_pending_offer = True

        has_direct_offer_evidence = current_has_produce or current_has_qty or current_has_keywords
        has_multiturn_completion_evidence = is_context_pending_offer and (current_has_loc or current_has_timing)

        if has_direct_offer_evidence or has_multiturn_completion_evidence:
            current_offer = getattr(context, "offer", None) if context else None
            merged_offer = await self.offer_extractor.extract(text, current_offer=current_offer)
            entities.offer = merged_offer
            if merged_offer and merged_offer.future_availability:
                fa_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:kg|tons?|quintals?|lbs?|bags?)?\s*(.*)", merged_offer.future_availability)
                if fa_m:
                    entities.future_quantity = float(fa_m.group(1))
                    entities.future_timing = fa_m.group(2).strip() or "later"
            return ConversationalIntentResult(
                intent=IntentType.OFFER_PRODUCE,
                confidence=0.90,
                entities=entities,
                raw_query=text,
            )

        # 12. Fallback: UNKNOWN intent (first-class path)
        return ConversationalIntentResult(
            intent=IntentType.UNKNOWN,
            confidence=0.5,
            entities=entities,
            raw_query=text,
        )
