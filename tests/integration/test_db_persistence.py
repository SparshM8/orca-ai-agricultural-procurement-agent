"""Integration tests verifying database persistence and repository operations."""

import pytest
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from orca.db.session import Base
from orca.db.models import OrderModel, RateModel
from orca.db.repository import init_db, OrderRepository, RateRepository


@pytest.fixture
async def test_db_session():
    """Create an isolated in-memory SQLite database for testing persistence."""
    test_engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=False,
    )
    await init_db(test_engine)

    session_maker = async_sessionmaker(
        bind=test_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_maker() as session:
        yield session

    await test_engine.dispose()


@pytest.mark.asyncio
async def test_rate_repository_seeds_and_queries(test_db_session: AsyncSession):
    """Verify rates are seeded and can be queried from database."""
    rate_repo = RateRepository(test_db_session)
    produce_list = await rate_repo.get_supported_produce("GLOBAL_DEFAULT")
    assert "potato" in produce_list
    assert "tomato" in produce_list
    assert "onion" in produce_list

    rate = await rate_repo.get_applicable_rate("potato", "GLOBAL_DEFAULT")
    assert rate is not None
    assert rate.rate_per_unit == 0.40
    assert rate.currency == "USD"


@pytest.mark.asyncio
async def test_order_repository_persistence_and_status_update(test_db_session: AsyncSession):
    """Verify orders can be persisted, retrieved, and updated in database."""
    order_repo = OrderRepository(test_db_session)

    new_order = OrderModel(
        id="ORD-TEST-001",
        farmer_id="farmer_42",
        produce_type="potato",
        quantity=50.0,
        unit="kg",
        validated_rate=0.40,
        currency="USD",
        total_amount=20.00,
        pickup_location="Warehouse 4, Route 12",
        status="ORDER_CONFIRMED",
        conversation_id="conv_session_test_001",
    )

    saved_order = await order_repo.save(new_order)
    assert saved_order.id == "ORD-TEST-001"

    # Fetch by ID
    fetched = await order_repo.get_by_id("ORD-TEST-001")
    assert fetched is not None
    assert fetched.produce_type == "potato"
    assert fetched.total_amount == 20.00
    assert fetched.status == "ORDER_CONFIRMED"

    # Fetch by conversation_id (for idempotency)
    by_conv = await order_repo.get_by_conversation_id("conv_session_test_001")
    assert by_conv is not None
    assert by_conv.id == "ORD-TEST-001"

    # Update status to PAYMENT_PENDING
    updated = await order_repo.update_status("ORD-TEST-001", "PAYMENT_PENDING")
    assert updated is not None
    assert updated.status == "PAYMENT_PENDING"

    # Verify persisted in database
    verified = await order_repo.get_by_id("ORD-TEST-001")
    assert verified is not None
    assert verified.status == "PAYMENT_PENDING"
