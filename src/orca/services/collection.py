"""Collection and logistics coordination service (FR-026 to FR-031)."""

import uuid
from datetime import datetime, timezone
from typing import Dict, Optional
from orca.domain.models import CollectionTask
from orca.domain.state_machine import OrderState
from orca.services.order import order_service


class CollectionService:
    """Manages pickup tasks, runner assignment, and completion."""

    def __init__(self):
        self._tasks: Dict[str, CollectionTask] = {}

    def create_collection_task(
        self,
        order_id: str,
        pickup_location: str,
        scheduled_datetime: Optional[datetime] = None,
    ) -> CollectionTask:
        """Create a collection task for a confirmed and paid order."""
        task_id = f"COL-{uuid.uuid4().hex[:8].upper()}"
        task = CollectionTask(
            id=task_id,
            order_id=order_id,
            pickup_location=pickup_location,
            scheduled_datetime=scheduled_datetime,
            status="PENDING",
        )
        self._tasks[task_id] = task

        # Advance order state to COLLECTION_PENDING
        order_service.transition_state(order_id, OrderState.COLLECTION_PENDING)
        return task

    def assign_runner(self, task_id: str, runner_id: str) -> CollectionTask:
        """Assign runner to collection task."""
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"Collection task not found: {task_id}")

        task.runner_id = runner_id
        task.status = "ASSIGNED"
        order_service.transition_state(task.order_id, OrderState.COLLECTION_ASSIGNED)
        return task

    def mark_picked_up(self, task_id: str) -> CollectionTask:
        """Mark collection as picked up by runner."""
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"Collection task not found: {task_id}")

        task.status = "PICKED_UP"
        order_service.transition_state(task.order_id, OrderState.PICKED_UP)
        return task

    def complete_collection(self, task_id: str) -> CollectionTask:
        """Complete collection and advance order to COMPLETED."""
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"Collection task not found: {task_id}")

        task.status = "COMPLETED"
        task.completed_at = datetime.now(timezone.utc)
        order_service.transition_state(task.order_id, OrderState.COMPLETED)
        return task

    def get_task(self, task_id: str) -> Optional[CollectionTask]:
        """Retrieve task by ID."""
        return self._tasks.get(task_id)


collection_service = CollectionService()
