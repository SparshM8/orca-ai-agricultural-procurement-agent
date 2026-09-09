"""Pydantic schemas and DTOs for agent interaction, validation, and API contracts."""

from datetime import datetime, timezone
from typing import Optional, List
from pydantic import BaseModel, Field
from orca.domain.state_machine import OrderState


class InboundMessage(BaseModel):
    """Normalized inbound message from any messaging adapter."""

    sender_id: str = Field(description="Normalized sender identity, e.g., phone number")
    channel: str = Field(default="whatsapp", description="whatsapp, web, console")
    text: str = Field(description="Farmer message content")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    raw_payload: Optional[dict] = None


class OutboundMessage(BaseModel):
    """Normalized outbound message to be sent via channel adapter."""

    recipient_id: str = Field(description="Target recipient ID")
    channel: str = Field(default="whatsapp")
    text: str = Field(description="Agent response content")
    metadata: Optional[dict] = None


class ExtractedOffer(BaseModel):
    """Structured produce offer extracted from natural language by the LLM."""

    produce_type: Optional[str] = Field(default=None, description="Produce name, e.g. potato")
    quantity: Optional[float] = Field(default=None, description="Extracted numerical quantity")
    unit: Optional[str] = Field(default=None, description="Extracted unit, e.g., kg, ton, bag")
    offered_rate: Optional[float] = Field(default=None, description="Farmer proposed rate if specified")
    pickup_location: Optional[str] = Field(default=None, description="Pickup location or address")
    availability_window: Optional[str] = Field(default=None, description="When produce is ready for pickup")
    pickup_datetime: Optional[str] = Field(default=None, description="Specific date/time for pickup")
    farmer_confirmed: Optional[bool] = Field(default=None, description="Explicit confirmation yes/no")


class BillSummary(BaseModel):
    """Deterministic transaction/bill summary calculated by backend."""

    produce_type: str
    quantity: float
    unit: str
    rate_per_unit: float
    currency: str
    subtotal: float
    tax_amount: float = 0.0
    total_amount: float
    calculation_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TransactionSummary(BaseModel):
    """Summary presented to farmer for final confirmation."""

    order_id: Optional[str] = None
    produce_type: str
    quantity: float
    unit: str
    rate_per_unit: float
    currency: str
    total_amount: float
    pickup_location: str
    pickup_datetime: Optional[str] = None
    status: OrderState
    formatted_summary: str
