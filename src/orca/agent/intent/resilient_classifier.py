"""Resilient intent classifier with automatic fallback handling."""

import logging
from typing import Optional, Any
from orca.domain.intents import ConversationalIntentResult
from orca.agent.intent.base import BaseIntentClassifier

logger = logging.getLogger(__name__)


class ResilientIntentClassifier(BaseIntentClassifier):
    """Wraps a primary intent classifier (e.g. Gemini) with automatic fallback.
    
    Ensures that network errors, 503 unavailable, 429 quota exhaustion,
    or timeouts never crash the conversation.
    """

    def __init__(self, primary: BaseIntentClassifier, fallback: BaseIntentClassifier):
        self.primary = primary
        self.fallback = fallback
        self.last_fallback_occurred: bool = False
        self.last_error: Optional[Exception] = None

    async def classify(
        self, text: str, context: Optional[Any] = None
    ) -> ConversationalIntentResult:
        """Attempt primary intent classification, falling back gracefully on failure."""
        self.last_fallback_occurred = False
        self.last_error = None
        try:
            res = await self.primary.classify(text, context)
            res.classifier_name = self.primary.__class__.__name__
            res.fallback_occurred = False
            return res
        except Exception as exc:
            self.last_fallback_occurred = True
            self.last_error = exc
            logger.warning(
                "Primary intent classifier (%s) failed with error: %s. Falling back to %s.",
                self.primary.__class__.__name__,
                exc,
                self.fallback.__class__.__name__,
            )
            res = await self.fallback.classify(text, context)
            res.classifier_name = self.fallback.__class__.__name__
            res.fallback_occurred = True
            res.fallback_reason = str(exc)
            return res
