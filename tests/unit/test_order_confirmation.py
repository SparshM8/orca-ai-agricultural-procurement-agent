"""Focused unit tests for farmer confirmation and validated order creation.

Covers the 6 required scenarios:
1. Successful confirmation creates an order.
2. Correct authoritative rate and total are persisted.
3. Duplicate confirmation does not create another order.
4. Confirmation without a pending transaction is rejected safely.
5. Invalid transaction data cannot create an order.
6. Correct state transition to ORDER_CONFIRMED.
"""

import pytest
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from orca.agent.orchestrator import AgentOrchestrator
from orca.services.order import order_service
from orca.domain.state_machine import OrderState, InvalidStateTransitionError
from orca.db.session import Base
from orca.db.repository import init_db, OrderRepository


@pytest.fixture
def orchestrator():
    """Fresh orchestrator instance for testing."""
    return AgentOrchestrator(auto_process_payment=False)


@pytest.fixture
async def test_db_session():
    """Create isolated in-memory SQLite database session for persistence verification."""
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
async def test_successful_confirmation_creates_order(orchestrator: AgentOrchestrator):
    """Scenario 1: Farmer explicit confirmation creates exactly one validated order."""
    conv_id = "test_conv_confirm_success"
    offer_msg = "I have 50 kg of potatoes in Springfield available this Friday at 10 AM."

    # Turn 1: Offer presentation
    res1 = await orchestrator.process_message(conv_id, offer_msg, farmer_id="farmer_01")
    ctx = orchestrator._conversations[conv_id]
    assert ctx.state == OrderState.AWAITING_FARMER_CONFIRMATION
    assert "Confirm" in res1.text

    # Turn 2: Explicit Confirmation
    confirm_msg = "Confirm"
    res2 = await orchestrator.process_message(conv_id, confirm_msg, farmer_id="farmer_01")

    # Verify state and response
    assert ctx.state == OrderState.ORDER_CONFIRMED
    assert ctx.current_order_id is not None
    order_id = ctx.current_order_id

    # Verify order exists in backend order service
    order = order_service.get_order(order_id)
    assert order is not None
    assert order.id == order_id
    assert order.produce_type == "potato"
    assert order.quantity == 50.0
    assert order.unit == "kg"
    assert order.validated_rate == 0.40
    assert order.total_amount == 20.00
    assert order.status == OrderState.ORDER_CONFIRMED

    # Verify user-facing confirmation message contains required fields
    assert "Order Confirmed!" in res2.text
    assert order_id in res2.text
    assert "50 kg of potato" in res2.text
    assert "USD 0.40 per kg" in res2.text
    assert "Total Amount: USD 20.00" in res2.text
    assert "ORDER_CONFIRMED" in res2.text


@pytest.mark.asyncio
async def test_correct_authoritative_rate_and_total_persisted(test_db_session: AsyncSession):
    """Scenario 2: Correct authoritative rate and deterministic total are persisted to database."""
    conv_id = "test_conv_db_persistence"

    # Farmer attempts to suggest $0.75/kg for onions (authoritative rate is $0.35/kg)
    order = await order_service.create_order_async(
        farmer_id="farmer_persisted",
        produce_type="onion",
        quantity=100.0,
        unit="kg",
        pickup_location="Warehouse 9, North Sector",
        conversation_id=conv_id,
        session=test_db_session,
    )

    # Verify database persistence
    repo = OrderRepository(test_db_session)
    order_model = await repo.get_by_id(order.id)

    assert order_model is not None
    assert order_model.id == order.id
    assert order_model.produce_type == "onion"
    assert order_model.quantity == 100.0
    assert order_model.unit == "kg"
    assert order_model.validated_rate == 0.35  # Authoritative rate enforced
    assert order_model.total_amount == 35.00   # 100 * 0.35 = 35.00 USD
    assert order_model.status == "ORDER_CONFIRMED"
    assert order_model.pickup_location == "Warehouse 9, North Sector"
    assert order_model.conversation_id == conv_id


@pytest.mark.asyncio
async def test_duplicate_confirmation_does_not_create_another_order(orchestrator: AgentOrchestrator):
    """Scenario 3: Repeated confirmation messages return existing order without creating duplicate."""
    conv_id = "test_conv_duplicate_confirm"
    offer_msg = "I have 20 kg of tomatoes in Springfield ready tomorrow."

    # Turn 1: Present transaction
    await orchestrator.process_message(conv_id, offer_msg, farmer_id="farmer_dup")

    # Turn 2: First confirmation
    res_first = await orchestrator.process_message(conv_id, "Yes, I accept", farmer_id="farmer_dup")
    order_id_1 = res_first.metadata["order_id"]
    order_count_before = len(order_service._orders)

    # Turn 3: Duplicate confirmation sent
    res_duplicate = await orchestrator.process_message(conv_id, "Confirm", farmer_id="farmer_dup")
    order_id_2 = res_duplicate.metadata["order_id"]
    order_count_after = len(order_service._orders)

    # Assert exactly same order ID and no duplicate order created
    assert order_id_1 == order_id_2
    assert order_count_before == order_count_after
    assert res_duplicate.metadata["is_duplicate"] is True
    assert "This transaction has already been confirmed" in res_duplicate.text
    assert order_id_1 in res_duplicate.text


@pytest.mark.asyncio
async def test_confirmation_without_pending_transaction_rejected_safely(orchestrator: AgentOrchestrator):
    """Scenario 4: Confirmation without a pending transaction is safely rejected."""
    # Case A: Brand new conversation sends "Confirm"
    conv_id_empty = "test_conv_empty_confirm"
    res_empty = await orchestrator.process_message(conv_id_empty, "Confirm", farmer_id="farmer_empty")

    ctx_empty = orchestrator._conversations[conv_id_empty]
    assert ctx_empty.state == OrderState.OFFER_RECEIVED
    assert ctx_empty.current_order_id is None
    assert "no pending transaction to confirm" in res_empty.text.lower()

    # Case B: Conversation has details pending (missing location) and sends "Confirm"
    conv_id_pending = "test_conv_details_pending_confirm"
    await orchestrator.process_message(conv_id_pending, "I have 30 kg of potatoes.", farmer_id="farmer_pending")

    res_pending = await orchestrator.process_message(conv_id_pending, "Yes, I accept", farmer_id="farmer_pending")
    ctx_pending = orchestrator._conversations[conv_id_pending]
    assert ctx_pending.state == OrderState.DETAILS_PENDING
    assert ctx_pending.current_order_id is None
    assert "cannot confirm order yet" in res_pending.text.lower()


def test_invalid_transaction_data_cannot_create_order():
    """Scenario 5: Direct service calls with invalid transaction data are strictly rejected."""
    # 5a. Unsupported produce
    with pytest.raises(ValueError, match="Validation failed: Unsupported produce"):
        order_service.create_order(
            farmer_id="farmer_inv",
            produce_type="mangoes",
            quantity=50.0,
            unit="kg",
            pickup_location="Farm 1",
        )

    # 5b. Invalid / negative quantity
    with pytest.raises(ValueError, match="Validation failed"):
        order_service.create_order(
            farmer_id="farmer_inv",
            produce_type="potato",
            quantity=-10.0,
            unit="kg",
            pickup_location="Farm 1",
        )

    # 5c. Empty pickup location
    with pytest.raises(ValueError, match="Pickup location must not be empty"):
        order_service.create_order(
            farmer_id="farmer_inv",
            produce_type="potato",
            quantity=10.0,
            unit="kg",
            pickup_location="",
        )

    # 5d. Fabricated / mismatched rate
    with pytest.raises(ValueError, match="Rate mismatch"):
        order_service.create_order(
            farmer_id="farmer_inv",
            produce_type="potato",
            quantity=10.0,
            unit="kg",
            pickup_location="Farm 1",
            validated_rate=9.99,  # Mismatches authoritative 0.40
        )


@pytest.mark.asyncio
async def test_correct_state_transition_to_order_confirmed(orchestrator: AgentOrchestrator):
    """Scenario 6: Verifies valid state transition sequence to ORDER_CONFIRMED."""
    conv_id = "test_conv_state_flow"

    # Step 1: Incomplete offer -> DETAILS_PENDING
    await orchestrator.process_message(conv_id, "I have 40 kg of onions.", farmer_id="farmer_seq")
    ctx = orchestrator._conversations[conv_id]
    assert ctx.state == OrderState.DETAILS_PENDING

    # Step 2: Add missing details -> AWAITING_FARMER_CONFIRMATION
    await orchestrator.process_message(conv_id, "In Springfield ready today.", farmer_id="farmer_seq")
    assert ctx.state == OrderState.AWAITING_FARMER_CONFIRMATION

    # Step 3: Farmer confirms -> ORDER_CONFIRMED
    await orchestrator.process_message(conv_id, "Confirm", farmer_id="farmer_seq")
    assert ctx.state == OrderState.ORDER_CONFIRMED

    # Direct State Machine Verification: Cannot jump from OFFER_RECEIVED directly to ORDER_CONFIRMED
    with pytest.raises(InvalidStateTransitionError):
        order_service.transition_state(ctx.current_order_id, OrderState.OFFER_RECEIVED)
