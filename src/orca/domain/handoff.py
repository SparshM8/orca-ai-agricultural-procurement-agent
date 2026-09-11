"""Human Handoff and Operational Escalation Domain Models.

Minimal, factual representation of operational exceptions requiring human attention.
Contains ZERO raw model prompts, ZERO raw conversation transcripts, and ZERO sensitive credentials.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class HandoffStatus(str, Enum):
    """Lifecycle status of a human handoff case."""

    OPEN = "OPEN"
    ASSIGNED = "ASSIGNED"
    RESOLVED = "RESOLVED"
    CANCELLED = "CANCELLED"


class HandoffReason(str, Enum):
    """Categorical classification of why human assistance was requested or escalated."""

    PAYMENT_DISPUTE = "PAYMENT_DISPUTE"
    COLLECTION_FAILURE = "COLLECTION_FAILURE"
    OPERATIONAL_EXCEPTION = "OPERATIONAL_EXCEPTION"
    KNOWLEDGE_GAP = "KNOWLEDGE_GAP"
    FARMER_REQUEST = "FARMER_REQUEST"
    OTHER = "OTHER"


class HumanHandoffCase(BaseModel):
    """Minimal, factual operational record of a case escalated to human assistance."""

    case_id: str
    farmer_id: str
    order_id: Optional[str] = None
    reason: HandoffReason
    status: HandoffStatus = HandoffStatus.OPEN
    summary: str
    source_intent: str = "REQUEST_HUMAN_ASSISTANCE"
    assigned_to: Optional[str] = None
    resolution_notes: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
