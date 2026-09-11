"""Extractor factory for dynamic provider selection."""

import os
from typing import Optional
from orca.core.config import settings
from orca.agent.extractors.base import BaseOfferExtractor
from orca.agent.extractors.rule_based import RuleBasedPatternExtractor
from orca.agent.extractors.ai_extractor import OpenAICompatibleAIExtractor
from orca.agent.extractors.gemini import GeminiOfferExtractor
from orca.agent.extractors.resilient import ResilientOfferExtractor


def get_offer_extractor(
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
    fallback_on_failure: Optional[bool] = None,
) -> BaseOfferExtractor:
    """Factory creating the configured offer extractor.
    
    Default is 'rule_based', requiring zero external AI services.
    """
    selected_provider = (provider or settings.AI_PROVIDER).lower()
    use_fallback = fallback_on_failure if fallback_on_failure is not None else settings.AI_FALLBACK_ON_FAILURE

    if selected_provider == "gemini":
        gemini_extractor = GeminiOfferExtractor(
            api_key=api_key or os.environ.get("GEMINI_API_KEY") or settings.GEMINI_API_KEY or settings.AI_API_KEY,
            model_name=model_name or settings.AI_MODEL_NAME,
            timeout_seconds=timeout_seconds if timeout_seconds is not None else settings.AI_TIMEOUT_SECONDS,
        )
        if use_fallback:
            return ResilientOfferExtractor(primary=gemini_extractor, fallback=RuleBasedPatternExtractor())
        return gemini_extractor

    if selected_provider in ("openai_compatible", "ollama", "local", "vllm"):
        ai_extractor = OpenAICompatibleAIExtractor(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
        )
        if use_fallback:
            return ResilientOfferExtractor(primary=ai_extractor, fallback=RuleBasedPatternExtractor())
        return ai_extractor

    # Default provider: rule-based
    return RuleBasedPatternExtractor()
