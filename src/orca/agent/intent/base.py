"""Abstract base class for conversational intent classifiers."""

from abc import ABC, abstractmethod
from typing import Optional, Any
from orca.domain.intents import ConversationalIntentResult


class BaseIntentClassifier(ABC):
    """Base contract for intent classification and entity extraction."""

    @abstractmethod
    async def classify(
        self, text: str, context: Optional[Any] = None
    ) -> ConversationalIntentResult:
        """Classify user intent and extract relevant domain entities.
        
        Args:
            text: The natural language message from the user.
            context: Optional DialogueContext representing conversation history and state.
            
        Returns:
            ConversationalIntentResult containing intent and extracted entities.
        """
        pass
