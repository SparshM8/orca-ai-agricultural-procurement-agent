"""AI Agent module containing prompt architecture, tool definitions, and conversational orchestrator."""

from orca.agent.prompts import (
    SYSTEM_BASE_PROMPT,
    CLARIFICATION_PROMPT_TEMPLATE,
    CONFIRMATION_PROMPT_TEMPLATE,
)
from orca.agent.tools import AgentToolRegistry, tool_registry
from orca.agent.orchestrator import (
    DialogueContext,
    BaseOfferExtractor,
    RuleBasedPatternExtractor,
    AgentOrchestrator,
    orchestrator,
)

__all__ = [
    "SYSTEM_BASE_PROMPT",
    "CLARIFICATION_PROMPT_TEMPLATE",
    "CONFIRMATION_PROMPT_TEMPLATE",
    "AgentToolRegistry",
    "tool_registry",
    "DialogueContext",
    "BaseOfferExtractor",
    "RuleBasedPatternExtractor",
    "AgentOrchestrator",
    "orchestrator",
]
