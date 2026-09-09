"""External integration adapters (Messaging, Payment, Logistics)."""

from orca.adapters.messaging.base import BaseMessagingAdapter
from orca.adapters.payment.base import BasePaymentAdapter
from orca.adapters.logistics.base import BaseLogisticsAdapter

__all__ = [
    "BaseMessagingAdapter",
    "BasePaymentAdapter",
    "BaseLogisticsAdapter",
]
