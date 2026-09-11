"""Deterministic pattern and regex offer extractor."""

import re
from typing import Optional
from orca.domain.schemas import ExtractedOffer
from orca.agent.extractors.base import BaseOfferExtractor, parse_pickup_timing


class RuleBasedPatternExtractor(BaseOfferExtractor):
    """Robust, provider-independent pattern extractor for core produce offers.
    
    Handles produce types, quantities, units, availability, pickup locations,
    and farmer-provided prices without requiring external paid API calls.
    """

    # Common unit normalizations
    UNIT_MAP = {
        "kg": "kg",
        "kgs": "kg",
        "kilo": "kg",
        "kilos": "kg",
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
        "banana", "bananas", "orange", "oranges", "wheat", "rice", "papaya",
        "carrot", "carrots", "cabbage", "cabbages",
    ]

    async def extract(
        self, text: str, current_offer: Optional[ExtractedOffer] = None
    ) -> ExtractedOffer:
        """Extract attributes and merge with previous turn's context."""
        offer = current_offer.model_copy() if current_offer else ExtractedOffer()
        cleaned_text = text.strip()
        lower_text = cleaned_text.lower()

        # 1. Produce Extraction
        # First: Check for substitution / commodity amendment clauses (e.g. "change potatoes to papaya",
        # "switch onions to mangoes", "make this tomato instead of potatoes", "papaya instead of potatoes")
        subst_patterns = [
            r"\b(?:change|switch|update|turn)\s+(?:the\s+)?(?:produce|crop|item|\w+)?\s*(?:from\s+\w+\s+)?to\s+([A-Za-z]+)",
            r"\bmake\s+(?:this|it|the\s+(?:produce|crop))\s+([A-Za-z]+)\s+instead\s+of\s+\w+",
            r"\b([A-Za-z]+)\s+instead\s+of\s+(?:potatoes?|tomatoes?|onions?|\w+)",
            r"\binstead\s+of\s+\w+\s*,\s*(?:use|make\s+it\s+|have\s+)?([A-Za-z]+)",
            r"\b(?:actually\s*,?\s*)?(?:change|switch)\s+to\s+([A-Za-z]+)",
        ]

        non_produce_words = {
            "kg", "kgs", "kilo", "kilos", "kilogram", "kilograms", "ton", "tons", "tonne", "tonnes",
            "quintal", "quintals", "lb", "lbs", "pound", "pounds", "bag", "bags",
            "pickup", "location", "farm", "springfield", "greenfield", "warehouse", "address",
            "today", "tomorrow", "morning", "afternoon", "evening", "friday", "monday", "tuesday",
            "wednesday", "thursday", "saturday", "sunday", "delhi", "pune",
            "that", "it", "this", "my", "your", "the", "a", "an", "confirm", "yes", "no", "more", "less"
        }

        for spat in subst_patterns:
            sm = re.search(spat, lower_text)
            if sm:
                cand_crop = sm.group(1).lower().strip(".,")
                if cand_crop not in non_produce_words and len(cand_crop) > 2 and not cand_crop.isdigit():
                    if cand_crop in self.PRODUCE_NORMALIZATION:
                        offer.produce_type = self.PRODUCE_NORMALIZATION[cand_crop]
                    else:
                        if cand_crop.endswith("es") and len(cand_crop) > 4:
                            offer.produce_type = cand_crop[:-2]
                        elif cand_crop.endswith("s") and len(cand_crop) > 3:
                            offer.produce_type = cand_crop[:-1]
                        else:
                            offer.produce_type = cand_crop
                    break

        if not offer.produce_type:
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
        # Pattern 0: Partial availability ("200 kg of potatoes total ... only 50 kg ready")
        partial_match = re.search(
            r"\b(?P<total>\d+(?:\.\d+)?)\s*(?P<t_unit>kg|kgs|kilos?|kilograms?|tons?|tonnes?|quintals?|lbs?|pounds?|bags?)(?:\s+(?:of\s+)?\w+)?\s*(?:total|in\s+total|overall)\b.*?\b(?:only\s+)?(?P<ready>\d+(?:\.\d+)?)\s*(?P<r_unit>kg|kgs|kilos?|kilograms?|tons?|tonnes?|quintals?|lbs?|pounds?|bags?)\s*(?:is\s+)?(?:ready|available)",
            lower_text,
        )
        # Pattern 0b: Intra-turn self-correction
        # E.g. "50 kg of potatoes, actually 75 kg" or "50 kilos, actually make that 75"
        # or "50 kg — actually 75 kg" or "Actually, it's 75 kg, not 50 kg"
        correction_reverse = re.search(
            r"\b(?:actually|in\s+fact|wait)\s*,?\s*(?:it'?s\s+|make\s+(?:it|that)\s+)?(?P<qty>\d+(?:\.\d+)?)\s*(?P<unit>kg|kgs|kilos?|kilograms?|tons?|tonnes?|quintals?|lbs?|pounds?|bags?)?\s*,?\s*(?:not\s+\d+(?:\.\d+)?)",
            lower_text,
        )
        correction_forward = re.search(
            r"\b(?P<old_qty>\d+(?:\.\d+)?)\s*(?P<old_unit>kg|kgs|kilos?|kilograms?|tons?|tonnes?|quintals?|lbs?|pounds?|bags?)?(?:\s+(?:of\s+)?\w+)?\s*(?:—|-|,)?\s*(?:actually|wait|no(?:\s+(?:make\s+it|make\s+that))?|i\s+meant|sorry)(?:\s*,)?\s*(?:make\s+(?:it|that)\s+)?(?P<qty>\d+(?:\.\d+)?)\s*(?P<unit>kg|kgs|kilos?|kilograms?|tons?|tonnes?|quintals?|lbs?|pounds?|bags?)?\b",
            lower_text,
        )
        
        if partial_match:
            total_qty = float(partial_match.group("total"))
            ready_qty = float(partial_match.group("ready"))
            raw_unit = partial_match.group("r_unit")
            offer.quantity = ready_qty
            offer.unit = self.UNIT_MAP.get(raw_unit, raw_unit)
            rem_qty = total_qty - ready_qty
            if rem_qty > 0:
                offer.future_availability = f"{rem_qty:g} {offer.unit} later"
        elif correction_reverse:
            offer.quantity = float(correction_reverse.group("qty"))
            raw_unit = correction_reverse.group("unit")
            if raw_unit:
                offer.unit = self.UNIT_MAP.get(raw_unit, raw_unit)
            elif not offer.unit:
                offer.unit = "kg"
        elif correction_forward:
            offer.quantity = float(correction_forward.group("qty"))
            raw_unit = correction_forward.group("unit") or correction_forward.group("old_unit")
            if raw_unit:
                offer.unit = self.UNIT_MAP.get(raw_unit, raw_unit)
            elif not offer.unit:
                offer.unit = "kg"
        else:
            # Pattern A: Number + Unit, e.g. "10 kg", "20.5 kilograms", "5 tons", "300 kilos"
            qty_unit_pattern = r"(?<![\$\d])(?P<qty>\d+(?:\.\d+)?)\s*(?P<unit>kg|kgs|kilos?|kilograms?|tons?|tonnes?|quintals?|lbs?|pounds?|bags?)\b"
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
            r"(?:(?:\$|price\s*(?:of|is|:)?|want)\s*\$?\s*(?P<price1>\d+(?:\.\d+)?)\s*(?:usd|\$)?(?:\s*(?:per|\/)\s*(?:kg|ton|lb|bag))?|(?:for|at)\s+\$\s*(?P<price2>\d+(?:\.\d+)?)|(?:for|at)\s+(?P<price3>\d+(?:\.\d+)?)\s*(?:usd|\$|\s*(?:per|\/)\s*(?:kg|ton|lb|bag))\b)",
            lower_text,
        )
        if price_match:
            price_str = price_match.group("price1") or price_match.group("price2") or price_match.group("price3")
            if price_str:
                potential_price = float(price_str)
                # Ensure it is not part of a time expression (e.g. 10 am, 2 pm)
                if not re.search(rf"\b{price_str}\s*(?:am|pm|o\'clock|hours?|mins?)\b", lower_text):
                    # Make sure it didn't just match the quantity
                    if offer.quantity is None or abs(potential_price - offer.quantity) > 1e-4:
                        offer.offered_rate = potential_price

        # 4. Pickup Location Extraction
        # Semantic, clause-aware location extraction:
        # Avoid generic bare "to" which misidentifies commodities/quantities as locations!
        loc_patterns = [
            r"(?i)(?:location\s*[:=]|pickup\s+(?:location\s+)?(?:to|in|at)|move\s+(?:pickup|collection|location)\s+to|change\s+(?:pickup|collection|location)\s+to|deliver\s+to)\s+([A-Za-z0-9\s,.-]+?)(?=(?:\s+(?:available|ready|for|pickup|this|today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|next|on|at\s+\d|\$|\d+(?:\.\d+)?\s*(?:kg|ton|quintal|lb|bag)|i\s+want|and\s+can)|$))",
            r"(?i)(?:\bin\b|\bat\b|\bfrom\b|\bnear\b)\s+([A-Za-z0-9\s,.-]+?)(?=(?:\s+(?:available|ready|for|pickup|this|today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|next|on|at\s+\d|\$|\d+(?:\.\d+)?\s*(?:kg|ton|quintal|lb|bag)|i\s+want|and\s+can)|$))",
        ]
        for lpat in loc_patterns:
            loc_match = re.search(lpat, cleaned_text)
            if loc_match:
                loc = loc_match.group(1).strip().rstrip(",.")
                loc_lower = loc.lower()
                is_time_like = bool(
                    re.search(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b", loc_lower)
                    or loc_lower in ["the morning", "the afternoon", "the evening", "tomorrow", "today"]
                )
                is_qty_like = bool(re.search(r"\b\d+\s*(?:kg|kgs|ton|tons|quintal|lb|bags?)\b", loc_lower) or loc.isdigit())
                is_commodity = bool(
                    loc_lower in self.PRODUCE_NORMALIZATION
                    or loc_lower.rstrip("s") in self.UNSUPPORTED_KNOWN
                    or (offer.produce_type and loc_lower == offer.produce_type.lower())
                    or loc_lower in ["papaya", "mango", "mangoes", "produce", "crop"]
                )
                is_amend_kw = bool(re.search(r"\b(?:actually|make\s+that|change\s+to)\b", loc_lower))
                if not is_time_like and not is_qty_like and not is_commodity and not is_amend_kw and len(loc) > 1:
                    offer.pickup_location = loc
                    break

        if not offer.pickup_location:
            if re.search(r"\bspringfield\b", lower_text):
                offer.pickup_location = "Springfield"
            elif re.search(r"\bgreenfield\b", lower_text):
                offer.pickup_location = "Greenfield"

        # 5. Availability Window & Pickup Date/Time Extraction
        avail_pattern = r"""(?i)\b(
            (?:this\s+week|ready\s+immediately|today|tomorrow)
            (?:\s+(?:morning|afternoon|evening))?
            (?:\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?)?
            |
            (?:this\s+|next\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)
            (?:\s+(?:morning|afternoon|evening))?
            (?:\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?)?
        )\b"""
        avail_match = re.search(avail_pattern, cleaned_text, re.VERBOSE)
        if avail_match:
            raw_avail = avail_match.group(1).strip()
            _, normalized_timing = parse_pickup_timing(raw_avail)
            offer.availability_window = normalized_timing
            if "at " in normalized_timing.lower():
                offer.pickup_datetime = normalized_timing
        else:
            # Check for standalone time expression (e.g. answering a time clarification)
            time_match = re.search(r"(?i)\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", cleaned_text)
            if time_match:
                raw_time = time_match.group(0).strip()
                if not raw_time.lower().startswith("at "):
                    raw_time = f"at {raw_time}"
                if offer.availability_window and "at " not in offer.availability_window.lower():
                    combined = f"{offer.availability_window} {raw_time}"
                    _, normalized_timing = parse_pickup_timing(combined)
                    offer.availability_window = normalized_timing
                    offer.pickup_datetime = normalized_timing
                else:
                    _, normalized_timing = parse_pickup_timing(raw_time)
                    offer.availability_window = normalized_timing
                    offer.pickup_datetime = normalized_timing

        return offer
