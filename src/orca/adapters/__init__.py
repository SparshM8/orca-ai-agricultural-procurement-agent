"""External integration adapters (Messaging, Payment, Logistics)."""

from orca.adapters.messaging.base import BaseMessagingAdapter, ChannelProvider
from orca.adapters.payment.base import BasePaymentAdapter
from orca.adapters.logistics.base import BaseLogisticsAdapter

__all__ = [
    "ChannelProvider",
    "BaseMessagingAdapter",
    "BasePaymentAdapter",
    "BaseLogisticsAdapter",
]
