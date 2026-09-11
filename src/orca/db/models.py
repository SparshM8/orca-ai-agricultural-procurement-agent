"""SQLAlchemy ORM models representing persistent procurement records."""

from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import String, Float, Boolean, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from orca.db.session import Base


class FarmerModel(Base):
    """Persistent farmer profile."""

    __tablename__ = "farmers"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    phone: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    pickup_location: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    region_code: Mapped[str] = mapped_column(String(32), default="GLOBAL_DEFAULT")
    payment_details_ref: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="ACTIVE")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class RateModel(Base):
    """Persistent authoritative rate entry."""

    __tablename__ = "rates"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    produce_type: Mapped[str] = mapped_column(String(64), index=True)
    region_code: Mapped[str] = mapped_column(String(32), index=True)
    rate_per_unit: Mapped[float] = mapped_column(Float)
    unit: Mapped[str] = mapped_column(String(32), default="kg")
    currency: Mapped[str] = mapped_column(String(8), default="USD")
    source: Mapped[str] = mapped_column(String(64), default="CONFIGURED_BASELINE")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class OrderModel(Base):
    """Persistent procurement order record."""

    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    farmer_id: Mapped[str] = mapped_column(String(64), index=True)
    produce_type: Mapped[str] = mapped_column(String(64))
    quantity: Mapped[float] = mapped_column(Float)
    unit: Mapped[str] = mapped_column(String(32))
    validated_rate: Mapped[float] = mapped_column(Float)
    currency: Mapped[str] = mapped_column(String(8), default="USD")
    total_amount: Mapped[float] = mapped_column(Float)
    pickup_location: Mapped[str] = mapped_column(String(256))
    pickup_datetime: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    pickup_time_str: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(64), default="ORDER_CONFIRMED")
    conversation_id: Mapped[Optional[str]] = mapped_column(
        String(128), unique=True, nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class PaymentModel(Base):
    """Persistent payment record."""

    __tablename__ = "payments"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    order_id: Mapped[str] = mapped_column(String(64), index=True)
    amount: Mapped[float] = mapped_column(Float)
    currency: Mapped[str] = mapped_column(String(8), default="USD")
    status: Mapped[str] = mapped_column(String(32), default="PENDING")
    provider_reference: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class CollectionTaskModel(Base):
    """Persistent collection and pickup task."""

    __tablename__ = "collection_tasks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    order_id: Mapped[str] = mapped_column(String(64), index=True)
    runner_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    produce_type: Mapped[str] = mapped_column(String(64), default="")
    quantity: Mapped[float] = mapped_column(Float, default=0.0)
    unit: Mapped[str] = mapped_column(String(32), default="kg")
    pickup_location: Mapped[str] = mapped_column(String(256))
    scheduled_datetime: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    scheduled_time_str: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    farmer_id: Mapped[str] = mapped_column(String(64), default="")
    farmer_contact: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="PENDING")
    failure_reason: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    assigned_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    picked_up_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class ConversationModel(Base):
    """Persistent conversation session."""

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    farmer_id: Mapped[str] = mapped_column(String(64), index=True)
    channel: Mapped[str] = mapped_column(String(32), default="whatsapp")
    channel_user_id: Mapped[str] = mapped_column(String(128), index=True)
    state: Mapped[str] = mapped_column(String(64), default="OFFER_RECEIVED")
    current_order_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
