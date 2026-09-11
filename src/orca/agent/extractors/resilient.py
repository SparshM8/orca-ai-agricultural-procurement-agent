"""Resilient offer extractor with automatic fallback handling."""

import logging
from typing import Optional
from orca.domain.schemas import ExtractedOffer
from orca.agent.extractors.base import BaseOfferExtractor

logger = logging.getLogger(__name__)


class ResilientOfferExtractor(BaseOfferExtractor):
    """Wraps a primary AI extractor with an automatic fallback (e.g. RuleBasedPatternExtractor).
    
    Ensures that network errors, model timeouts, or parsing errors never crash
    the farmer session or raise 500 exceptions.
    """

    def __init__(self, primary: BaseOfferExtractor, fallback: BaseOfferExtractor):
        self.primary = primary
        self.fallback = fallback
        self.last_fallback_occurred: bool = False
        self.last_error: Optional[Exception] = None

    async def extract(
        self, text: str, current_offer: Optional[ExtractedOffer] = None
    ) -> ExtractedOffer:
        """Attempt primary extraction, gracefully falling back on error."""
        self.last_fallback_occurred = False
        self.last_error = None
        try:
            return await self.primary.extract(text, current_offer)
        except Exception as exc:
            self.last_fallback_occurred = True
            self.last_error = exc
            logger.warning(
                "Primary extractor (%s) failed with error: %s. Falling back to %s.",
                self.primary.__class__.__name__,
                exc,
                self.fallback.__class__.__name__,
            )
            return await self.fallback.extract(text, current_offer)
