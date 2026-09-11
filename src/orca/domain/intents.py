"""Intent types, entity structures, and tool execution contracts."""

from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

from orca.domain.schemas import ExtractedOffer


class IntentType(str, Enum):
    """Supported conversational intents for agricultural procurement dialogue."""

    OFFER_PRODUCE = "OFFER_PRODUCE"
    INQUIRE_SUPPORTED_PRODUCE = "INQUIRE_SUPPORTED_PRODUCE"
    INQUIRE_ORDER_STATUS = "INQUIRE_ORDER_STATUS"
    INQUIRE_PAYMENT_STATUS = "INQUIRE_PAYMENT_STATUS"
    INQUIRE_COLLECTION_STATUS = "INQUIRE_COLLECTION_STATUS"
    REQUEST_CLARIFICATION = "REQUEST_CLARIFICATION"
    CONFIRM_ORDER = "CONFIRM_ORDER"
    AUTHORIZE_PAYMENT = "AUTHORIZE_PAYMENT"
    REQUEST_PICKUP_CHANGE = "REQUEST_PICKUP_CHANGE"
    REPORT_COLLECTION_PROBLEM = "REPORT_COLLECTION_PROBLEM"
    OBJECTION_PRICING = "OBJECTION_PRICING"
    AMEND_OFFER = "AMEND_OFFER"
    INQUIRE_BUSINESS_INFO = "INQUIRE_BUSINESS_INFO"
    INQUIRE_OPERATIONAL_FAQ = "INQUIRE_OPERATIONAL_FAQ"
    RECOMMEND_PRODUCE = "RECOMMEND_PRODUCE"
    PERSONALIZE_RECOMMENDATION = "PERSONALIZE_RECOMMENDATION"
    REQUEST_HUMAN_ASSISTANCE = "REQUEST_HUMAN_ASSISTANCE"
    UNKNOWN = "UNKNOWN"


class ExtractedEntities(BaseModel):
    """Entities extracted across conversational dialogue turns."""

    offer: Optional[ExtractedOffer] = None
    order_id: Optional[str] = None
    runner_id: Optional[str] = None
    new_pickup_location: Optional[str] = None
    new_pickup_time: Optional[str] = None
    problem_reason: Optional[str] = None
    clarification_subject: Optional[str] = None
    counter_rate: Optional[float] = Field(default=None, description="Farmer proposed counter rate if bargaining")
    future_quantity: Optional[float] = Field(default=None, description="Staged or future harvest quantity")
    future_timing: Optional[str] = Field(default=None, description="Estimated window for future harvest")
    amended_field: Optional[str] = Field(default=None, description="Specific offer field amended, e.g. quantity, location")
    knowledge_topic: Optional[str] = Field(default=None, description="Topic or subject for business knowledge lookup")
    recommendation_produce: Optional[str] = Field(default=None, description="Produce commodity mentioned in recommendation query")
    recommendation_topic: Optional[str] = Field(default=None, description="Topic or category for recommendation query")
    explicit_preference: Optional[str] = Field(default=None, description="Explicitly stated farmer produce preference")
    handoff_reason: Optional[str] = Field(default=None, description="Reason category for human escalation")


class ConversationalIntentResult(BaseModel):
    """Structured result of intent classification and entity extraction.
    
    The LLM outputs only intent and entities; deterministic Python code
    is solely responsible for tool selection and execution.
    """

    intent: IntentType = IntentType.UNKNOWN
    confidence: float = 1.0
    entities: ExtractedEntities = Field(default_factory=ExtractedEntities)
    raw_query: str = ""
    classifier_name: str = "RuleBasedIntentClassifier"
    fallback_occurred: bool = False
    fallback_reason: Optional[str] = None


class ToolExecutionResult(BaseModel):
    """Standardized result returned by approved backend tools in the ToolRegistry."""

    tool_name: str
    success: bool
    data: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    state_updated: Optional[str] = None
