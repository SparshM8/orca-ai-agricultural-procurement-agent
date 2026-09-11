"""Demo / Local Logistics Adapter (FR-026 to FR-031).

Constraint:
This is an adapter/simulation layer only. It does not hold authoritative state.
Authoritative task and order state reside in CollectionService and the database.
"""

import uuid
from typing import Any, Dict, Optional
from datetime import datetime
from orca.adapters.logistics.base import BaseLogisticsAdapter


class DemoLogisticsAdapter(BaseLogisticsAdapter):
    """Demo / local runner dispatch adapter for simulation and testing."""

    def __init__(self, should_dispatch_succeed: bool = True):
        self.should_dispatch_succeed = should_dispatch_succeed
        self.dispatched_events: list = []

    async def dispatch_collection(
        self,
        order_id: str,
        pickup_location: str,
        scheduled_datetime: Optional[datetime],
        contact_phone: str,
    ) -> Dict[str, Any]:
        """Simulate publishing or broadcasting collection task to runner network."""
        dispatch_ref = f"DEMO_DISPATCH_{uuid.uuid4().hex[:6].upper()}"
        event = {
            "order_id": order_id,
            "pickup_location": pickup_location,
            "scheduled_datetime": scheduled_datetime,
            "contact_phone": contact_phone,
            "dispatch_ref": dispatch_ref,
            "status": "DISPATCHED" if self.should_dispatch_succeed else "FAILED",
        }
        self.dispatched_events.append(event)
        return event

    async def get_dispatch_status(self, task_id: str) -> Dict[str, Any]:
        """Query external dispatch status."""
        return {
            "task_id": task_id,
            "status": "ACTIVE",
            "provider": "DEMO_LOGISTICS",
        }
