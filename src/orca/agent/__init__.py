"""AI Agent module containing prompt architecture and tool definitions."""

from orca.agent.prompts import (
    SYSTEM_BASE_PROMPT,
    CLARIFICATION_PROMPT_TEMPLATE,
    CONFIRMATION_PROMPT_TEMPLATE,
)
from orca.agent.tools import AgentToolRegistry, tool_registry

__all__ = [
    "SYSTEM_BASE_PROMPT",
    "CLARIFICATION_PROMPT_TEMPLATE",
    "CONFIRMATION_PROMPT_TEMPLATE",
    "AgentToolRegistry",
    "tool_registry",
]
