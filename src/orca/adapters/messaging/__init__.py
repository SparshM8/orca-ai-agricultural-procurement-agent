"""Messaging adapters package."""

from orca.adapters.messaging.base import ChannelProvider, BaseMessagingAdapter
from orca.adapters.messaging.demo import DemoChannelAdapter
from orca.adapters.messaging.factory import (
    register_channel_provider,
    get_channel_provider,
)

__all__ = [
    "ChannelProvider",
    "BaseMessagingAdapter",
    "DemoChannelAdapter",
    "register_channel_provider",
    "get_channel_provider",
]
