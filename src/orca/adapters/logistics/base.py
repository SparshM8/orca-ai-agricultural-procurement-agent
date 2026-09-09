"""Abstract logistics and runner dispatch adapter interface (FR-026 to FR-031)."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime


class BaseLogisticsAdapter(ABC):
    """Abstract logistics provider and runner dispatch adapter."""

    @abstractmethod
    async def dispatch_collection(
        self,
        order_id: str,
        pickup_location: str,
        scheduled_datetime: Optional[datetime],
        contact_phone: str,
    ) -> Dict[str, Any]:
        """Dispatch or publish collection task to runner network."""
        pass

    @abstractmethod
    async def get_dispatch_status(self, task_id: str) -> Dict[str, Any]:
        """Query collection and runner status."""
        pass
