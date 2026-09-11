"""Factual Farmer Profile and Grounded Personalization Domain Contracts.

Strictly observational and grounded in verified ORCA interactions.
Contains ZERO predictive ML, NO farmer tiers, NO wealth/farm size inference,
and NO speculative scoring.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class FactualOfferObservation(BaseModel):
    """Observable record of an explicit produce offer made by the farmer."""

    produce: str
    quantity: Optional[float] = None
    unit: Optional[str] = None
    location: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FactualCompletedOrder(BaseModel):
    """Observable record of an order completed through ORCA."""

    order_id: Optional[str] = None
    produce: str
    quantity: Optional[float] = None
    unit: Optional[str] = None
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FarmerProfile(BaseModel):
    """Minimal factual farmer profile strictly derived from observed ORCA interactions.

    Fields are limited strictly to explicitly observed facts.
    Contains NO speculative attributes, tiers, risk scores, or wealth indicators.
    """

    farmer_id: str
    location: Optional[str] = None
    observed_produce_history: List[str] = Field(default_factory=list)
    observed_units: List[str] = Field(default_factory=list)
    completed_order_count: int = 0
    last_offered_produce: Optional[str] = None
    last_completed_produce: Optional[str] = None
    explicit_preferences: List[str] = Field(default_factory=list)
    observed_offers: List[FactualOfferObservation] = Field(default_factory=list)
    completed_orders: List[FactualCompletedOrder] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PersonalizedRecommendationItem(BaseModel):
    """A single personalized recommendation grounded in an observed fact."""

    produce: str
    factual_reason: str
    grounding_source: str = "profile_history"  # "profile_history" | "explicit_preference"
    rate: Optional[float] = None
    currency: Optional[str] = None
    unit: Optional[str] = None


class PersonalizedRecommendationResult(BaseModel):
    """Structured result of a personalized procurement recommendation query."""

    matched: bool
    grounding_source: str = "profile_history"  # "profile_history" | "explicit_preference" | "insufficient_history"
    recommendations: List[PersonalizedRecommendationItem] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    clarification_required: bool = False
    message: Optional[str] = None
