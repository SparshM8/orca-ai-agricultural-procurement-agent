"""Domain models representing core agricultural procurement entities (SRS Section 13)."""

from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field
from orca.domain.state_machine import OrderState


class Farmer(BaseModel):
    """Farmer domain entity."""

    id: str = Field(description="Unique farmer identifier")
    name: str = Field(description="Farmer display or full name")
    phone: str = Field(description="Phone number / WhatsApp identity (E.164)")
    pickup_location: Optional[str] = Field(default=None, description="Default pickup address/coordinates")
    region_code: str = Field(default="GLOBAL_DEFAULT", description="Associated regional profile code")
    payment_details_ref: Optional[str] = Field(default=None, description="Reference to payout account or token")
    status: str = Field(default="ACTIVE")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ProduceListing(BaseModel):
    """Produce offered by farmer during conversation."""

    id: str = Field(description="Unique listing identifier")
    farmer_id: str = Field(description="Associated farmer ID")
    produce_type: str = Field(description="Normalized produce name, e.g., potato, tomato")
    quantity: float = Field(description="Offered quantity")
    unit: str = Field(description="Measurement unit, e.g., kg, ton")
    offered_rate: Optional[float] = Field(default=None, description="Optional rate suggested by farmer")
    availability_window: Optional[str] = Field(default=None, description="Dates or window of availability")
    status: str = Field(default="PENDING")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Rate(BaseModel):
    """Authoritative rate entry configured by administrator or market service."""

    id: str = Field(description="Unique rate identifier")
    produce_type: str = Field(description="Produce type, e.g., potato")
    region_code: str = Field(description="Applicable region code")
    rate_per_unit: float = Field(description="Authoritative price per unit")
    unit: str = Field(description="Base unit, e.g., kg")
    currency: str = Field(default="USD", description="Currency code")
    effective_from: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    effective_to: Optional[datetime] = None
    source: str = Field(default="ADMIN_CONFIGURED", description="Source: GOVT_MSP, BUYER_CONFIG, MARKET")
    is_active: bool = True


class Order(BaseModel):
    """Procurement transaction order record."""

    id: str = Field(description="Unique order identifier (UUID or semantic format)")
    farmer_id: str = Field(description="Farmer ID")
    produce_type: str = Field(description="Produce item")
    quantity: float = Field(description="Final validated quantity")
    unit: str = Field(description="Validated unit")
    validated_rate: float = Field(description="Authoritative rate applied")
    currency: str = Field(default="USD")
    total_amount: float = Field(description="Deterministic total (quantity * validated_rate + tax)")
    pickup_location: str = Field(description="Specified pickup location")
    pickup_datetime: Optional[datetime] = Field(default=None, description="Scheduled collection time")
    status: OrderState = Field(default=OrderState.OFFER_RECEIVED)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Payment(BaseModel):
    """Payment record associated with an order."""

    id: str = Field(description="Unique payment record ID")
    order_id: str = Field(description="Referenced order ID")
    amount: float = Field(description="Payment amount")
    currency: str = Field(default="USD")
    status: str = Field(default="PENDING", description="PENDING, INITIATED, SUCCESS, FAILED")
    provider_reference: Optional[str] = Field(default=None, description="External transaction ID from payment gateway")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CollectionTask(BaseModel):
    """Logistics pickup/collection task for runner."""

    id: str = Field(description="Unique collection task ID")
    order_id: str = Field(description="Associated order ID")
    runner_id: Optional[str] = Field(default=None, description="Assigned runner ID")
    pickup_location: str = Field(description="Physical location for collection")
    scheduled_datetime: Optional[datetime] = Field(default=None)
    status: str = Field(default="PENDING", description="PENDING, ASSIGNED, PICKED_UP, COMPLETED, FAILED")
    completed_at: Optional[datetime] = None


class Conversation(BaseModel):
    """Conversation session linking messaging channel, farmer, and current order."""

    id: str = Field(description="Unique conversation / session identifier")
    farmer_id: str = Field(description="Farmer ID")
    channel: str = Field(default="whatsapp", description="whatsapp, web, console")
    channel_user_id: str = Field(description="Channel-specific sender ID (e.g. WhatsApp phone number)")
    state: OrderState = Field(default=OrderState.OFFER_RECEIVED)
    current_order_id: Optional[str] = Field(default=None)
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
