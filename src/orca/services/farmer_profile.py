"""Factual Farmer Profile Service.

Tracks strictly observed facts from ORCA interactions:
- Produce offered in conversational turns
- Orders completed through the order/collection lifecycle
- Explicit preferences directly stated by the farmer

Contains NO speculative scoring, NO inferred attributes, and NO predictive modeling.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional
from orca.domain.farmer_profile import (
    FarmerProfile,
    FactualOfferObservation,
    FactualCompletedOrder,
)


def _normalize_commodity_name(name: str) -> str:
    """Normalize plural/singular produce strings (e.g. 'potatoes' -> 'potato')."""
    clean = name.strip().lower()
    if clean.endswith("es"):
        stem = clean[:-2]
        if stem in ["potato", "tomato"]:
            return stem
    if clean.endswith("s"):
        stem = clean[:-1]
        if stem in ["onion", "potato", "tomato", "cabbage"]:
            return stem
    return clean


class FarmerProfileService:
    """Lightweight in-memory profile service tracking factual farmer interaction history."""

    def __init__(self):
        self._profiles: Dict[str, FarmerProfile] = {}

    def clear(self) -> None:
        """Clear all in-memory farmer profiles (used in demo reset and unit tests)."""
        self._profiles.clear()

    def get_profile(self, farmer_id: str) -> FarmerProfile:
        """Retrieve existing profile or return a blank factual profile."""
        if farmer_id not in self._profiles:
            self._profiles[farmer_id] = FarmerProfile(farmer_id=farmer_id)
        return self._profiles[farmer_id]

    def record_observed_offer(
        self,
        farmer_id: str,
        produce: str,
        quantity: Optional[float] = None,
        unit: Optional[str] = None,
        location: Optional[str] = None,
    ) -> FarmerProfile:
        """Record an explicit produce offer made during a conversation turn."""
        profile = self.get_profile(farmer_id)
        norm_produce = _normalize_commodity_name(produce)

        obs = FactualOfferObservation(
            produce=norm_produce,
            quantity=quantity,
            unit=unit.lower() if unit else None,
            location=location,
            timestamp=datetime.now(timezone.utc),
        )
        profile.observed_offers.append(obs)
        profile.last_offered_produce = norm_produce

        if norm_produce not in profile.observed_produce_history:
            profile.observed_produce_history.append(norm_produce)

        if unit and unit.lower() not in profile.observed_units:
            profile.observed_units.append(unit.lower())

        if location and not profile.location:
            profile.location = location

        profile.updated_at = datetime.now(timezone.utc)
        self._profiles[farmer_id] = profile
        return profile

    def record_completed_order(
        self,
        farmer_id: str,
        produce: str,
        quantity: Optional[float] = None,
        unit: Optional[str] = None,
        order_id: Optional[str] = None,
    ) -> FarmerProfile:
        """Record a completed purchase/sale order upon order completion."""
        profile = self.get_profile(farmer_id)
        norm_produce = _normalize_commodity_name(produce)

        order_record = FactualCompletedOrder(
            order_id=order_id,
            produce=norm_produce,
            quantity=quantity,
            unit=unit.lower() if unit else None,
            completed_at=datetime.now(timezone.utc),
        )
        profile.completed_orders.append(order_record)
        profile.completed_order_count = len(profile.completed_orders)
        profile.last_completed_produce = norm_produce

        if norm_produce not in profile.observed_produce_history:
            profile.observed_produce_history.append(norm_produce)

        if unit and unit.lower() not in profile.observed_units:
            profile.observed_units.append(unit.lower())

        profile.updated_at = datetime.now(timezone.utc)
        self._profiles[farmer_id] = profile
        return profile

    def record_explicit_preference(
        self,
        farmer_id: str,
        preference: str,
    ) -> FarmerProfile:
        """Record an explicitly stated farmer crop preference (e.g. 'I usually sell onions')."""
        profile = self.get_profile(farmer_id)
        norm_pref = _normalize_commodity_name(preference)

        if norm_pref not in profile.explicit_preferences:
            profile.explicit_preferences.append(norm_pref)

        profile.updated_at = datetime.now(timezone.utc)
        self._profiles[farmer_id] = profile
        return profile


farmer_profile_service = FarmerProfileService()
