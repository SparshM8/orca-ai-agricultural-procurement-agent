"""Grounded procurement recommendation domain contracts.

Strictly grounded in verified ORCA backend capabilities, supported commodities,
and authoritative procurement policies.
Contains ZERO predictive market forecasts, profitability optimization, or speculative claims.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class RecommendationCategory(str, Enum):
    """Categorical classification of grounded recommendations."""

    SUPPORTED_PRODUCE = "SUPPORTED_PRODUCE"
    CAPABILITY = "CAPABILITY"
    GENERAL = "GENERAL"


class ProcurementRecommendation(BaseModel):
    """A single grounded procurement recommendation based strictly on verified ORCA facts."""

    recommendation_id: str
    produce: str
    reason: str
    supporting_knowledge: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_type: str = "CONFIGURED_POLICY"


class RecommendationResult(BaseModel):
    """Structured result of a procurement recommendation query."""

    matched: bool
    recommendations: List[ProcurementRecommendation] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    clarification_required: bool = False
    message: Optional[str] = None
    category: RecommendationCategory = RecommendationCategory.GENERAL
