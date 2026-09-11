"""Focused unit tests for the collection and runner workflow (FR-026 to FR-031).

Covers the 8 required scenarios:
1. Collection task created once after successful payment.
2. Duplicate collection creation is prevented.
3. Runner can view and accept an available task.
4. Runner can reject a task safely (returning order to COLLECTION_PENDING).
5. Runner can confirm pickup (moving order to PICKED_UP then COMPLETED).
6. Pickup moves the order through the correct states.
7. Failed pickup moves to EXCEPTION and reschedules safely to COLLECTION_PENDING.
8. Invalid state transitions are strictly rejected.
Bonus:
9. Database persistence for collection tasks.
10. End-to-end full workflow from farmer offer through COMPLETED.
"""

import uuid
import pytest
from datetime import datetime, timezone
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from orca.domain.models import Order
from orca.domain.state_machine import OrderState, InvalidStateTransitionError
from orca.services.order import order_service
from orca.services.payment import payment_service
from orca.services.collection import CollectionService, collection_service
from orca.adapters.logistics.demo import DemoLogisticsAdapter
from orca.agent.orchestrator import AgentOrchestrator
from orca.db.session import Base
from orca.db.repository import init_db, CollectionRepository, OrderRepository


@pytest.fixture
def paid_order():
    """Create a unique order and advance it to PAYMENT_CONFIRMED."""
    order = order_service.create_order(
        farmer_id="farmer_coll_test",
        produce_type="potato",
        quantity=60.0,
        unit="kg",
        pickup_location="Depot 4, Route 12",
        conversation_id=f"conv_coll_test_{uuid.uuid4().hex[:8]}",
    )
    # Advance to PAYMENT_PENDING then PAYMENT_CONFIRMED
    order_service.transition_state(order.id, OrderState.PAYMENT_PENDING)
    order_service.transition_state(order.id, OrderState.PAYMENT_CONFIRMED)
    return order


@pytest.fixture
async def test_db_session():
    """In-memory SQLite database session for persistence verification."""
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
async def test_collection_task_created_once_after_successful_payment(paid_order: Order):
    """Scenario 1: Collection task created with complete payload and transitions order to COLLECTION_PENDING."""
    task = await collection_service.create_collection_task(
        order_id=paid_order.id,
        pickup_location=paid_order.pickup_location,
        scheduled_time_str="Friday at 10 AM",
    )

    assert task is not None
    assert task.id.startswith("COL-")
    assert task.order_id == paid_order.id
    assert task.produce_type == "potato"
    assert task.quantity == 60.0
    assert task.unit == "kg"
    assert task.pickup_location == "Depot 4, Route 12"
    assert task.scheduled_time_str == "Friday at 10 AM"
    assert task.farmer_id == "farmer_coll_test"
    assert task.status == "PENDING"
    assert task.runner_id is None
    assert task.created_at is not None

    # Verify order advanced to COLLECTION_PENDING
    assert paid_order.status == OrderState.COLLECTION_PENDING


@pytest.mark.asyncio
async def test_duplicate_collection_creation_prevented(paid_order: Order):
    """Scenario 2: Repeated collection creation returns existing task without duplicates."""
    task1 = await collection_service.create_collection_task(paid_order.id)
    count_before = len(collection_service._tasks)

    task2 = await collection_service.create_collection_task(paid_order.id)
    count_after = len(collection_service._tasks)

    assert task1.id == task2.id
    assert count_before == count_after


@pytest.mark.asyncio
async def test_runner_can_view_and_accept_task(paid_order: Order):
    """Scenario 3: Runner views available task and accepts it, transitioning order to COLLECTION_ASSIGNED."""
    task = await collection_service.create_collection_task(paid_order.id)

    # Runner views available tasks
    available = collection_service.get_available_tasks()
    assert task.id in [t.id for t in available]

    # Runner accepts
    accepted_task = await collection_service.accept_task(
        task_id=task.id, runner_id="runner_alex_01"
    )

    assert accepted_task.status == "ASSIGNED"
    assert accepted_task.runner_id == "runner_alex_01"
    assert accepted_task.assigned_at is not None
    assert paid_order.status == OrderState.COLLECTION_ASSIGNED

    # Task no longer in available pending tasks
    available_after = collection_service.get_available_tasks()
    assert task.id not in [t.id for t in available_after]


@pytest.mark.asyncio
async def test_runner_can_reject_task_safely(paid_order: Order):
    """Scenario 4: Runner rejection unassigns runner and returns order to COLLECTION_PENDING."""
    task = await collection_service.create_collection_task(paid_order.id)
    await collection_service.accept_task(task.id, runner_id="runner_alex_01")
    assert paid_order.status == OrderState.COLLECTION_ASSIGNED

    # Runner rejects task
    rejected_task = await collection_service.reject_task(
        task_id=task.id,
        runner_id="runner_alex_01",
        reason="Vehicle breakdown",
    )

    assert rejected_task.status == "PENDING"
    assert rejected_task.runner_id is None
    assert rejected_task.assigned_at is None
    # Order returned to COLLECTION_PENDING via backend state machine
    assert paid_order.status == OrderState.COLLECTION_PENDING

    # Available again for other runners
    available = collection_service.get_available_tasks()
    assert task.id in [t.id for t in available]

    # Another runner can now accept it
    reassigned = await collection_service.accept_task(task.id, runner_id="runner_ben_02")
    assert reassigned.status == "ASSIGNED"
    assert reassigned.runner_id == "runner_ben_02"
    assert paid_order.status == OrderState.COLLECTION_ASSIGNED


@pytest.mark.asyncio
async def test_runner_can_confirm_pickup_and_order_completes(paid_order: Order):
    """Scenario 5: Runner confirms pickup, moving order through PICKED_UP to COMPLETED."""
    task = await collection_service.create_collection_task(paid_order.id)
    await collection_service.accept_task(task.id, runner_id="runner_alex_01")

    # Runner confirms pickup
    confirmed_task = await collection_service.confirm_pickup(
        task_id=task.id, runner_id="runner_alex_01"
    )

    assert confirmed_task.status == "COMPLETED"
    assert confirmed_task.picked_up_at is not None
    assert confirmed_task.completed_at is not None
    assert paid_order.status == OrderState.COMPLETED


@pytest.mark.asyncio
async def test_pickup_moves_order_through_correct_states():
    """Scenario 6: Order progresses through the strict state machine sequence."""
    order = order_service.create_order(
        farmer_id="farmer_seq_test",
        produce_type="onion",
        quantity=30.0,
        unit="kg",
        pickup_location="Farm 7",
        conversation_id="conv_seq_002",
    )
    assert order.status == OrderState.ORDER_CONFIRMED

    # 1. Advance to PAYMENT_PENDING
    order_service.transition_state(order.id, OrderState.PAYMENT_PENDING)
    assert order.status == OrderState.PAYMENT_PENDING

    # 2. Advance to PAYMENT_CONFIRMED
    order_service.transition_state(order.id, OrderState.PAYMENT_CONFIRMED)
    assert order.status == OrderState.PAYMENT_CONFIRMED

    # 3. Create collection -> COLLECTION_PENDING
    task = await collection_service.create_collection_task(order.id)
    assert order.status == OrderState.COLLECTION_PENDING

    # 4. Runner accept -> COLLECTION_ASSIGNED
    await collection_service.accept_task(task.id, runner_id="runner_01")
    assert order.status == OrderState.COLLECTION_ASSIGNED

    # 5. Confirm pickup -> PICKED_UP -> COMPLETED
    await collection_service.confirm_pickup(task.id, runner_id="runner_01")
    assert order.status == OrderState.COMPLETED


@pytest.mark.asyncio
async def test_failed_pickup_moves_to_exception_and_reschedules(paid_order: Order):
    """Scenario 7: Pickup failure moves order to EXCEPTION, rescheduling returns order to COLLECTION_PENDING."""
    task = await collection_service.create_collection_task(paid_order.id)
    await collection_service.accept_task(task.id, runner_id="runner_alex_01")
    assert paid_order.status == OrderState.COLLECTION_ASSIGNED

    # Report failure
    failed_task = await collection_service.fail_pickup(
        task_id=task.id,
        runner_id="runner_alex_01",
        reason="Produce not ready at location",
    )

    assert failed_task.status == "FAILED"
    assert failed_task.failure_reason == "Produce not ready at location"
    assert paid_order.status == OrderState.EXCEPTION

    # Reschedule collection
    rescheduled_task = await collection_service.reschedule_task(
        task_id=task.id,
        new_time_str="Tomorrow at 3 PM",
    )

    assert rescheduled_task.status == "PENDING"
    assert rescheduled_task.runner_id is None
    assert rescheduled_task.failure_reason is None
    assert rescheduled_task.scheduled_time_str == "Tomorrow at 3 PM"
    assert paid_order.status == OrderState.COLLECTION_PENDING


@pytest.mark.asyncio
async def test_invalid_state_transitions_rejected(paid_order: Order):
    """Scenario 8: Invalid runner actions or state transitions are strictly rejected."""
    task = await collection_service.create_collection_task(paid_order.id)

    # 8a. Confirming pickup when task is still PENDING
    with pytest.raises(ValueError, match="must be ASSIGNED"):
        await collection_service.confirm_pickup(task.id, runner_id="runner_unassigned")

    # 8b. Non-assigned runner confirming pickup
    await collection_service.accept_task(task.id, runner_id="runner_correct")
    with pytest.raises(ValueError, match="Runner mismatch"):
        await collection_service.confirm_pickup(task.id, runner_id="runner_impostor")

    # 8c. Illegal state transition on order directly
    with pytest.raises(InvalidStateTransitionError):
        # Cannot transition from COLLECTION_ASSIGNED directly to COMPLETED without PICKED_UP
        order_service.transition_state(paid_order.id, OrderState.COMPLETED)


@pytest.mark.asyncio
async def test_collection_database_persistence(test_db_session: AsyncSession):
    """Scenario 9: Collection tasks are persisted and auditable in the database."""
    order = await order_service.create_order_async(
        farmer_id="farmer_db_coll",
        produce_type="tomato",
        quantity=50.0,
        unit="kg",
        pickup_location="Warehouse East",
        conversation_id="conv_db_coll_003",
        session=test_db_session,
    )
    order_service.transition_state(order.id, OrderState.PAYMENT_PENDING)
    order_service.transition_state(order.id, OrderState.PAYMENT_CONFIRMED)

    # Create task with DB persistence
    task = await collection_service.create_collection_task(
        order_id=order.id,
        session=test_db_session,
    )

    repo = CollectionRepository(test_db_session)
    persisted_task = await repo.get_by_id(task.id)

    assert persisted_task is not None
    assert persisted_task.id == task.id
    assert persisted_task.order_id == order.id
    assert persisted_task.produce_type == "tomato"
    assert persisted_task.quantity == 50.0
    assert persisted_task.unit == "kg"
    assert persisted_task.status == "PENDING"

    # Accept task and verify DB update
    await collection_service.accept_task(
        task_id=task.id,
        runner_id="runner_persisted_01",
        session=test_db_session,
    )

    updated_task = await repo.get_by_id(task.id)
    assert updated_task.status == "ASSIGNED"
    assert updated_task.runner_id == "runner_persisted_01"
    assert updated_task.assigned_at is not None


@pytest.mark.asyncio
async def test_end_to_end_conversation_offer_to_completed():
    """Scenario 10: Complete lifecycle from farmer natural language offer to COMPLETED."""
    orchestrator = AgentOrchestrator(auto_create_collection_task=True)
    conv_id = "test_conv_full_lifecycle"
    farmer_id = "farmer_e2e_full"

    # 1. Farmer Offer
    res1 = await orchestrator.process_message(
        conv_id,
        "I have 50 kg of potatoes in Springfield available this Friday at 10 AM.",
        farmer_id=farmer_id,
    )
    ctx = orchestrator._conversations[conv_id]
    assert ctx.state == OrderState.AWAITING_FARMER_CONFIRMATION

    # 2. Farmer Confirms -> ORDER_CONFIRMED -> PAYMENT_PENDING
    res2 = await orchestrator.process_message(conv_id, "Confirm", farmer_id=farmer_id)
    assert ctx.state == OrderState.PAYMENT_PENDING
    order_id = res2.metadata["order_id"]

    # 3. Farmer Pays -> PAYMENT_CONFIRMED -> COLLECTION_PENDING (Collection task created)
    res3 = await orchestrator.process_message(conv_id, "Pay", farmer_id=farmer_id)
    assert ctx.state == OrderState.COLLECTION_PENDING
    assert "Payment Confirmed!" in res3.text
    assert "Collection Details:" in res3.text
    assert "COLLECTION_PENDING" in res3.text
    task_id = res3.metadata["collection_task_id"]
    assert task_id is not None

    # 4. Farmer checks status
    res_status1 = await orchestrator.process_message(conv_id, "Status update please", farmer_id=farmer_id)
    assert "pending runner assignment" in res_status1.text

    # 5. Runner accepts task
    await collection_service.accept_task(task_id, runner_id="runner_dave_05")
    order = order_service.get_order(order_id)
    assert order.status == OrderState.COLLECTION_ASSIGNED

    # 6. Farmer checks status again
    res_status2 = await orchestrator.process_message(conv_id, "Where is the runner?", farmer_id=farmer_id)
    assert "runner_dave_05" in res_status2.text
    assert "COLLECTION_ASSIGNED" in res_status2.text

    # 7. Runner confirms pickup
    await collection_service.confirm_pickup(task_id, runner_id="runner_dave_05")
    assert order.status == OrderState.COMPLETED

    # 8. Farmer checks final status
    res_status3 = await orchestrator.process_message(conv_id, "Is it done?", farmer_id=farmer_id)
    assert "completed" in res_status3.text.lower()
    assert "COMPLETED" in res_status3.text
