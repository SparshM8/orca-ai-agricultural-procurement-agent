"""Focused unit tests for the payment flow after order confirmation.

Covers the 6 required scenarios:
1. Payment record created for a confirmed order.
2. Payment amount matches the order total.
3. Successful payment moves order to PAYMENT_CONFIRMED.
4. Failed payment is handled correctly.
5. Duplicate payment request/event does not create duplicate payment records.
6. Payment cannot be created for an invalid order state.
"""

import pytest
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from orca.domain.models import Order
from orca.domain.state_machine import OrderState
from orca.services.order import order_service
from orca.services.payment import PaymentService, payment_service
from orca.adapters.payment.demo import DemoPaymentAdapter
from orca.agent.orchestrator import AgentOrchestrator
from orca.db.session import Base
from orca.db.repository import init_db, PaymentRepository, OrderRepository


@pytest.fixture
def clean_order():
    """Create a unique confirmed order for testing payment."""
    order = order_service.create_order(
        farmer_id="farmer_pay_test",
        produce_type="potato",
        quantity=50.0,
        unit="kg",
        pickup_location="Springfield Depot",
        conversation_id="conv_pay_test_001",
    )
    return order


@pytest.fixture
async def test_db_session():
    """In-memory SQLite database session for persistence testing."""
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
async def test_payment_record_created_for_confirmed_order(clean_order: Order):
    """Scenario 1: Payment record created for a confirmed order with unique ID and timestamp."""
    payment, is_success, msg = await payment_service.initiate_order_payment(clean_order.id)

    assert payment is not None
    assert payment.id.startswith("PAY-")
    assert payment.order_id == clean_order.id
    assert payment.timestamp is not None
    assert is_success is True


@pytest.mark.asyncio
async def test_payment_amount_matches_order_total():
    """Scenario 2: Payment amount strictly matches the persisted order total."""
    # 75 kg of potatoes at $0.40/kg = $30.00 USD total
    order = order_service.create_order(
        farmer_id="farmer_amount_test",
        produce_type="potato",
        quantity=75.0,
        unit="kg",
        pickup_location="North Farm",
        conversation_id="conv_amount_test_002",
    )
    assert order.total_amount == 30.00

    payment, is_success, _ = await payment_service.initiate_order_payment(order.id)
    assert payment.amount == 30.00
    assert payment.currency == "USD"
    assert payment.amount == order.total_amount


@pytest.mark.asyncio
async def test_successful_payment_moves_order_to_payment_confirmed():
    """Scenario 3: Successful payment transitions order to PAYMENT_CONFIRMED."""
    order = order_service.create_order(
        farmer_id="farmer_success_test",
        produce_type="tomato",
        quantity=20.0,
        unit="kg",
        pickup_location="East Greenhouse",
        conversation_id="conv_success_pay_003",
    )
    assert order.status == OrderState.ORDER_CONFIRMED

    payment, is_success, msg = await payment_service.initiate_order_payment(order.id)

    assert is_success is True
    assert payment.status == "SUCCESS"
    assert payment.provider_reference is not None
    assert payment.provider_reference.startswith("DEMO_TX_")
    # Verify order advanced through PAYMENT_PENDING to PAYMENT_CONFIRMED
    assert order.status == OrderState.PAYMENT_CONFIRMED


@pytest.mark.asyncio
async def test_failed_payment_handled_correctly():
    """Scenario 4: Failed payment keeps order in PAYMENT_PENDING with retry guidance."""
    failing_adapter = DemoPaymentAdapter(default_should_succeed=False)
    custom_payment_service = PaymentService(adapter=failing_adapter)

    order = order_service.create_order(
        farmer_id="farmer_fail_test",
        produce_type="onion",
        quantity=40.0,
        unit="kg",
        pickup_location="South Cellar",
        conversation_id="conv_fail_pay_004",
    )

    payment, is_success, msg = await custom_payment_service.initiate_order_payment(order.id)

    assert is_success is False
    assert payment.status == "FAILED"
    # Order remains in PAYMENT_PENDING
    assert order.status == OrderState.PAYMENT_PENDING
    assert "Retry payment" in msg

    # Now test retry with a succeeding adapter outcome
    failing_adapter.set_order_outcome(order.id, should_succeed=True)
    retry_payment, retry_success, retry_msg = await custom_payment_service.initiate_order_payment(order.id)

    assert retry_success is True
    assert retry_payment.status == "SUCCESS"
    assert order.status == OrderState.PAYMENT_CONFIRMED


@pytest.mark.asyncio
async def test_duplicate_payment_request_idempotent():
    """Scenario 5: Repeated payment requests do not create duplicate payment records."""
    order = order_service.create_order(
        farmer_id="farmer_idem_test",
        produce_type="potato",
        quantity=10.0,
        unit="kg",
        pickup_location="West Field",
        conversation_id="conv_idem_pay_005",
    )

    # First payment
    p1, s1, _ = await payment_service.initiate_order_payment(order.id)
    payments_count_before = len(payment_service._payments)

    # Duplicate payment request
    p2, s2, msg2 = await payment_service.initiate_order_payment(order.id)
    payments_count_after = len(payment_service._payments)

    assert p1.id == p2.id
    assert payments_count_before == payments_count_after
    assert s2 is True
    assert "already completed" in msg2.lower()


@pytest.mark.asyncio
async def test_payment_cannot_be_created_for_invalid_order_state():
    """Scenario 6: Payment cannot be initiated for missing or unconfirmed orders."""
    # 6a. Non-existent order
    with pytest.raises(ValueError, match="Order not found"):
        await payment_service.initiate_order_payment("ORD-NONEXISTENT-999")

    # 6b. Order in illegal state (e.g. COMPLETED or CANCELLED)
    order = order_service.create_order(
        farmer_id="farmer_invalid_state",
        produce_type="potato",
        quantity=15.0,
        unit="kg",
        pickup_location="Farm 2",
        conversation_id="conv_invalid_state_006",
    )
    # Manually change state to CANCELLED
    order.status = OrderState.CANCELLED

    with pytest.raises(ValueError, match="Invalid order state for payment"):
        await payment_service.initiate_order_payment(order.id)


@pytest.mark.asyncio
async def test_payment_database_persistence(test_db_session: AsyncSession):
    """Bonus verification: Payment records are persisted and verifiable in the database."""
    order = await order_service.create_order_async(
        farmer_id="farmer_db_pay",
        produce_type="potato",
        quantity=100.0,
        unit="kg",
        pickup_location="Central Hub",
        conversation_id="conv_db_pay_007",
        session=test_db_session,
    )

    payment, is_success, _ = await payment_service.initiate_order_payment(
        order_id=order.id,
        session=test_db_session,
    )

    repo = PaymentRepository(test_db_session)
    persisted_payment = await repo.get_by_id(payment.id)

    assert persisted_payment is not None
    assert persisted_payment.id == payment.id
    assert persisted_payment.order_id == order.id
    assert persisted_payment.amount == 40.00  # 100 * 0.40
    assert persisted_payment.currency == "USD"
    assert persisted_payment.status == "SUCCESS"
    assert persisted_payment.provider_reference is not None


@pytest.mark.asyncio
async def test_end_to_end_conversation_order_confirmed_to_payment_confirmed():
    """End-to-end conversational flow from offer to PAYMENT_CONFIRMED."""
    orchestrator = AgentOrchestrator(auto_process_payment=True)
    conv_id = "test_conv_e2e_payment"

    # Step 1: Offer
    await orchestrator.process_message(
        conv_id,
        "I have 50 kg of potatoes in Springfield available this Friday at 10 AM.",
        farmer_id="farmer_e2e",
    )
    ctx = orchestrator._conversations[conv_id]
    assert ctx.state == OrderState.AWAITING_FARMER_CONFIRMATION

    # Step 2: Farmer Confirms -> Order created -> Payment processed -> PAYMENT_CONFIRMED
    res = await orchestrator.process_message(conv_id, "Confirm", farmer_id="farmer_e2e")

    assert ctx.state == OrderState.PAYMENT_CONFIRMED
    assert "Order and Payment Confirmed!" in res.text
    assert res.metadata["order_id"] is not None
    assert res.metadata["status"] == "PAYMENT_CONFIRMED"
    assert res.metadata["payment_status"] == "SUCCESS"
    assert "Total Amount: USD 20.00" in res.text
    assert "Payment Status: SUCCESS" in res.text
