"""Channel provider registry and factory."""

from typing import Dict, Optional
from orca.adapters.messaging.base import ChannelProvider
from orca.adapters.messaging.demo import DemoChannelAdapter

# Registry of channel providers
_CHANNELS: Dict[str, ChannelProvider] = {
    "web_chat": DemoChannelAdapter(name="web_chat"),
    "demo": DemoChannelAdapter(name="demo"),
    "simulator": DemoChannelAdapter(name="simulator"),
}


def register_channel_provider(name: str, provider: ChannelProvider) -> None:
    """Register a custom or specialized channel provider (e.g. WhatsAppChannelAdapter)."""
    _CHANNELS[name.lower()] = provider


def get_channel_provider(name: Optional[str] = None) -> ChannelProvider:
    """Retrieve the channel provider for the requested channel name.
    
    Defaults to 'web_chat' if unspecified.
    """
    key = (name or "web_chat").lower()
    if key not in _CHANNELS:
        # Dynamically instantiate a DemoChannelAdapter for unknown mock channels
        _CHANNELS[key] = DemoChannelAdapter(name=key)
    return _CHANNELS[key]
