"""Collection and logistics coordination service (FR-026 to FR-031).

Enforces:
- Order transition to COLLECTION_PENDING after successful payment.
- Idempotent creation of exactly one collection task per order.
- Complete task payload: order ID, produce, quantity, pickup location, schedule, farmer reference.
- Runner workflow: view available, accept, reject safely, confirm pickup.
- Strict backend state machine progression:
  PAYMENT_CONFIRMED -> COLLECTION_PENDING -> COLLECTION_ASSIGNED -> PICKED_UP -> COMPLETED.
- Pickup failure and rescheduling via EXCEPTION without state corruption.
- Persistent task state and timestamps in database.
- Provider-independent adapter integration.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from orca.domain.models import CollectionTask
from orca.domain.state_machine import OrderState
from orca.services.order import order_service
from orca.adapters.logistics.base import BaseLogisticsAdapter
from orca.adapters.logistics.demo import DemoLogisticsAdapter
from orca.db.models import CollectionTaskModel


class CollectionService:
    """Manages pickup tasks, runner assignment, state machine progression, and DB persistence."""

    def __init__(self, adapter: Optional[BaseLogisticsAdapter] = None):
        self.adapter = adapter or DemoLogisticsAdapter()
        # In-memory index: task_id -> CollectionTask
        self._tasks: Dict[str, CollectionTask] = {}
        # Idempotency index: order_id -> task_id
        self._order_tasks: Dict[str, str] = {}

    def clear(self) -> None:
        """Clear all in-memory collection tasks and idempotency indices (used in demo reset / testing)."""
        self._tasks.clear()
        self._order_tasks.clear()

    def get_task(self, task_id: str) -> Optional[CollectionTask]:
        """Retrieve task by task ID."""
        return self._tasks.get(task_id)

    def get_task_by_order(self, order_id: str) -> Optional[CollectionTask]:
        """Find task associated with order (idempotency lookup)."""
        task_id = self._order_tasks.get(order_id)
        if task_id:
            return self._tasks.get(task_id)
        for t in self._tasks.values():
            if t.order_id == order_id:
                return t
        return None

    async def create_collection_task(
        self,
        order_id: str,
        pickup_location: Optional[str] = None,
        scheduled_datetime: Optional[datetime] = None,
        scheduled_time_str: Optional[str] = None,
        session: Optional[AsyncSession] = None,
    ) -> CollectionTask:
        """Create a collection task for a confirmed and paid order.

        FR-026: Collection task associated with confirmed order.
        FR-027: Payload includes produce, quantity, location, time, farmer info.
        FR-028: Prevent duplicate task creation.
        """
        order = order_service.get_order(order_id)
        if not order:
            raise ValueError(f"Order not found: {order_id}")

        # Idempotency check: if task already exists for this order, return it
        existing_task = self.get_task_by_order(order_id)
        if existing_task:
            return existing_task

        # Valid order states for collection creation
        if order.status not in [OrderState.PAYMENT_CONFIRMED, OrderState.COLLECTION_PENDING]:
            raise ValueError(
                f"Invalid order state for collection: Order {order_id} is in state {order.status.value}, "
                f"must be in PAYMENT_CONFIRMED or COLLECTION_PENDING."
            )

        # Advance order state to COLLECTION_PENDING if in PAYMENT_CONFIRMED
        if order.status == OrderState.PAYMENT_CONFIRMED:
            order_service.transition_state(order_id, OrderState.COLLECTION_PENDING)
            await self._update_order_status_db(order_id, OrderState.COLLECTION_PENDING.value, session)

        task_id = f"COL-{uuid.uuid4().hex[:8].upper()}"
        resolved_location = pickup_location or order.pickup_location
        resolved_datetime = scheduled_datetime or order.pickup_datetime
        resolved_time_str = scheduled_time_str or order.pickup_time_str

        task = CollectionTask(
            id=task_id,
            order_id=order_id,
            runner_id=None,
            produce_type=order.produce_type,
            quantity=order.quantity,
            unit=order.unit,
            pickup_location=resolved_location,
            scheduled_datetime=resolved_datetime,
            scheduled_time_str=resolved_time_str,
            farmer_id=order.farmer_id,
            farmer_contact=f"+FARMER_{order.farmer_id}",
            status="PENDING",
            failure_reason=None,
            created_at=datetime.now(timezone.utc),
        )

        self._tasks[task_id] = task
        self._order_tasks[order_id] = task_id

        # Dispatch notification to provider adapter (simulation layer only)
        await self.adapter.dispatch_collection(
            order_id=order_id,
            pickup_location=resolved_location,
            scheduled_datetime=resolved_datetime,
            contact_phone=task.farmer_contact or "",
        )

        # Persist task to database
        await self._persist_task(task, session)

        return task

    def get_available_tasks(self) -> List[CollectionTask]:
        """View all available collection tasks waiting for runner assignment (status == PENDING)."""
        return [t for t in self._tasks.values() if t.status == "PENDING"]

    def get_all_tasks(self, status: Optional[str] = None) -> List[CollectionTask]:
        """View all collection tasks, optionally filtered by status."""
        tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        return sorted(tasks, key=lambda t: t.created_at or datetime.min, reverse=True)

    def get_assigned_tasks(self, runner_id: str) -> List[CollectionTask]:
        """View collection tasks assigned to a specific runner."""
        tasks = [t for t in self._tasks.values() if t.runner_id == runner_id]
        return sorted(tasks, key=lambda t: t.created_at or datetime.min, reverse=True)

    async def accept_task(
        self,
        task_id: str,
        runner_id: str,
        session: Optional[AsyncSession] = None,
    ) -> CollectionTask:
        """Runner accepts an available task, transitioning order to COLLECTION_ASSIGNED."""
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"Collection task not found: {task_id}")

        if not runner_id or not runner_id.strip():
            raise ValueError("Runner ID must be specified to accept task.")

        if task.status != "PENDING":
            raise ValueError(
                f"Task {task_id} cannot be accepted: current status is {task.status}, must be PENDING."
            )

        # State machine transition: COLLECTION_PENDING -> COLLECTION_ASSIGNED
        order_service.transition_state(task.order_id, OrderState.COLLECTION_ASSIGNED)
        await self._update_order_status_db(task.order_id, OrderState.COLLECTION_ASSIGNED.value, session)

        task.runner_id = runner_id.strip()
        task.status = "ASSIGNED"
        task.assigned_at = datetime.now(timezone.utc)

        await self._persist_task(task, session)
        return task

    async def reject_task(
        self,
        task_id: str,
        runner_id: str,
        reason: Optional[str] = None,
        session: Optional[AsyncSession] = None,
    ) -> CollectionTask:
        """Runner safely rejects a task.

        If task was assigned to runner, releases runner and returns order to COLLECTION_PENDING.
        If task was unassigned (PENDING), task remains PENDING for other runners.
        """
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"Collection task not found: {task_id}")

        if task.status == "ASSIGNED":
            if task.runner_id != runner_id:
                raise ValueError(
                    f"Runner {runner_id} cannot reject task {task_id} assigned to {task.runner_id}."
                )

            # Revert order through state machine: COLLECTION_ASSIGNED -> COLLECTION_PENDING
            order_service.transition_state(task.order_id, OrderState.COLLECTION_PENDING)
            await self._update_order_status_db(task.order_id, OrderState.COLLECTION_PENDING.value, session)

            task.runner_id = None
            task.assigned_at = None
            task.status = "PENDING"
            await self._persist_task(task, session)
            return task

        elif task.status == "PENDING":
            # Runner simply declines an open task; remains available for other runners
            return task

        else:
            raise ValueError(f"Cannot reject task {task_id} in status {task.status}.")

    async def confirm_pickup(
        self,
        task_id: str,
        runner_id: str,
        session: Optional[AsyncSession] = None,
    ) -> CollectionTask:
        """Runner confirms physical pickup of produce.

        Transitions order through valid state machine:
        COLLECTION_ASSIGNED -> PICKED_UP -> COMPLETED.
        """
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"Collection task not found: {task_id}")

        if task.status != "ASSIGNED":
            raise ValueError(
                f"Cannot confirm pickup for task {task_id}: task is in status {task.status}, must be ASSIGNED."
            )

        if task.runner_id != runner_id:
            raise ValueError(
                f"Runner mismatch: Runner {runner_id} cannot confirm pickup for task assigned to {task.runner_id}."
            )

        # Step 1: Advance order to PICKED_UP
        order_service.transition_state(task.order_id, OrderState.PICKED_UP)
        await self._update_order_status_db(task.order_id, OrderState.PICKED_UP.value, session)
        task.picked_up_at = datetime.now(timezone.utc)
        task.status = "PICKED_UP"

        # Step 2: Advance order to COMPLETED
        order_service.transition_state(task.order_id, OrderState.COMPLETED)
        await self._update_order_status_db(task.order_id, OrderState.COMPLETED.value, session)
        task.completed_at = datetime.now(timezone.utc)
        task.status = "COMPLETED"

        await self._persist_task(task, session)
        return task

    async def fail_pickup(
        self,
        task_id: str,
        runner_id: str,
        reason: str,
        session: Optional[AsyncSession] = None,
    ) -> CollectionTask:
        """Runner reports failed pickup. Transitions order to EXCEPTION without corrupting state."""
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"Collection task not found: {task_id}")

        if task.status != "ASSIGNED":
            raise ValueError(
                f"Cannot report failure for task {task_id}: task is in status {task.status}, must be ASSIGNED."
            )

        if task.runner_id != runner_id:
            raise ValueError(
                f"Runner mismatch: Runner {runner_id} cannot report failure for task assigned to {task.runner_id}."
            )

        # Transition order: COLLECTION_ASSIGNED -> EXCEPTION
        order_service.transition_state(task.order_id, OrderState.EXCEPTION)
        await self._update_order_status_db(task.order_id, OrderState.EXCEPTION.value, session)

        task.status = "FAILED"
        task.failure_reason = reason

        await self._persist_task(task, session)
        return task

    async def reschedule_task(
        self,
        task_id: str,
        new_datetime: Optional[datetime] = None,
        new_time_str: Optional[str] = None,
        session: Optional[AsyncSession] = None,
    ) -> CollectionTask:
        """Reschedule a failed task from EXCEPTION back to COLLECTION_PENDING."""
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"Collection task not found: {task_id}")

        if task.status != "FAILED":
            raise ValueError(
                f"Cannot reschedule task {task_id}: task is in status {task.status}, must be FAILED."
            )

        # Transition order: EXCEPTION -> COLLECTION_PENDING
        order_service.transition_state(task.order_id, OrderState.COLLECTION_PENDING)
        await self._update_order_status_db(task.order_id, OrderState.COLLECTION_PENDING.value, session)

        task.status = "PENDING"
        task.runner_id = None
        task.assigned_at = None
        task.failure_reason = None
        if new_datetime:
            task.scheduled_datetime = new_datetime
        if new_time_str:
            task.scheduled_time_str = new_time_str

        await self._persist_task(task, session)
        return task

    async def _update_order_status_db(
        self, order_id: str, new_status: str, session: Optional[AsyncSession] = None
    ) -> None:
        """Sync order status to database repository."""
        from orca.db.repository import OrderRepository

        async def _do_update(s: AsyncSession):
            repo = OrderRepository(s)
            await repo.update_status(order_id, new_status)

        if session:
            await _do_update(session)
        else:
            try:
                from orca.db.session import async_session_factory
                async with async_session_factory() as s:
                    await _do_update(s)
            except Exception:
                pass

    async def _persist_task(
        self, task: CollectionTask, session: Optional[AsyncSession] = None
    ) -> Optional[CollectionTaskModel]:
        """Persist or update collection task in database."""
        from orca.db.repository import CollectionRepository

        async def _do_persist(s: AsyncSession):
            repo = CollectionRepository(s)
            existing = await repo.get_by_id(task.id)
            if existing:
                return await repo.update_status(
                    task_id=task.id,
                    status=task.status,
                    runner_id=task.runner_id,
                    assigned_at=task.assigned_at,
                    picked_up_at=task.picked_up_at,
                    completed_at=task.completed_at,
                    failure_reason=task.failure_reason,
                    scheduled_datetime=task.scheduled_datetime,
                    scheduled_time_str=task.scheduled_time_str,
                )
            model = CollectionTaskModel(
                id=task.id,
                order_id=task.order_id,
                runner_id=task.runner_id,
                produce_type=task.produce_type,
                quantity=task.quantity,
                unit=task.unit,
                pickup_location=task.pickup_location,
                scheduled_datetime=task.scheduled_datetime,
                scheduled_time_str=task.scheduled_time_str,
                farmer_id=task.farmer_id,
                farmer_contact=task.farmer_contact,
                status=task.status,
                failure_reason=task.failure_reason,
                created_at=task.created_at,
                assigned_at=task.assigned_at,
                picked_up_at=task.picked_up_at,
                completed_at=task.completed_at,
            )
            return await repo.save(model)

        if session:
            return await _do_persist(session)
        else:
            try:
                from orca.db.session import async_session_factory
                async with async_session_factory() as s:
                    return await _do_persist(s)
            except Exception:
                return None


collection_service = CollectionService()
