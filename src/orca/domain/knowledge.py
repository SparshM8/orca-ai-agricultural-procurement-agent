"""Knowledge domain contracts, category taxonomy, and query result models.

Strictly grounded in ORCA SRS v2.0 and authoritative procurement behavior.
Contains zero hardcoded prices and zero unsupported operational claims.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class KnowledgeCategory(str, Enum):
    """Categorical taxonomy for grounded ORCA business knowledge and operational FAQs."""

    COMPANY_INFO = "COMPANY_INFO"
    PROCUREMENT_PROCESS = "PROCUREMENT_PROCESS"
    PRICING_POLICY = "PRICING_POLICY"
    LOGISTICS_POLICY = "LOGISTICS_POLICY"
    PAYMENT_POLICY = "PAYMENT_POLICY"
    DISPUTE_RESOLUTION = "DISPUTE_RESOLUTION"
    SYSTEM_CAPABILITIES = "SYSTEM_CAPABILITIES"


class KnowledgeArticle(BaseModel):
    """Grounded informational knowledge article representing verified ORCA business rules."""

    article_id: str
    category: KnowledgeCategory
    title: str
    summary: str
    content: str
    keywords: List[str] = Field(default_factory=list)
    related_topics: List[str] = Field(default_factory=list)
    source_type: str = "CONFIGURED_POLICY"  # e.g., "SRS_DOCUMENTATION", "SYSTEM_BEHAVIOR", "CONFIGURED_POLICY"
    verified: bool = True


class KnowledgeResult(BaseModel):
    """Structured result of a business knowledge lookup.
    
    Prevents weak keyword matches or hallucinations from being treated
    as authoritative answers.
    """

    matched: bool
    answerable: bool
    confidence: float = Field(ge=0.0, le=1.0)
    source_article_id: Optional[str] = None
    category: Optional[KnowledgeCategory] = None
    answer: Optional[str] = None
    clarification_required: bool = False
