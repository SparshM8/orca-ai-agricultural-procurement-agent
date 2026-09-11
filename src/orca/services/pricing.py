"""Authoritative pricing service (FR-011 to FR-015).

Deterministic rate lookup and validation. Prevents LLM price fabrication.
"""

from typing import Optional, List, Dict
from orca.domain.models import Rate
from orca.core.regional import get_regional_profile


class PricingService:
    """Authoritative rate lookup and produce validation service."""

    def __init__(self):
        # In-memory baseline rates for development / test mode
        self._rates: Dict[str, Rate] = {
            "GLOBAL_DEFAULT:potato": Rate(
                id="rate_potato_001",
                produce_type="potato",
                region_code="GLOBAL_DEFAULT",
                rate_per_unit=0.40,
                unit="kg",
                currency="USD",
                source="CONFIGURED_BASELINE",
            ),
            "GLOBAL_DEFAULT:tomato": Rate(
                id="rate_tomato_001",
                produce_type="tomato",
                region_code="GLOBAL_DEFAULT",
                rate_per_unit=0.60,
                unit="kg",
                currency="USD",
                source="CONFIGURED_BASELINE",
            ),
            "GLOBAL_DEFAULT:onion": Rate(
                id="rate_onion_001",
                produce_type="onion",
                region_code="GLOBAL_DEFAULT",
                rate_per_unit=0.35,
                unit="kg",
                currency="USD",
                source="CONFIGURED_BASELINE",
            ),
        }

    def reset_baseline_rates(self) -> None:
        """Reset in-memory rates back to baseline."""
        self.__init__()

    def get_supported_produce(self, region_code: str = "GLOBAL_DEFAULT") -> List[str]:
        """List supported produce items for a given region."""
        supported = [
            rate.produce_type
            for rate in self._rates.values()
            if rate.region_code == region_code and rate.is_active
        ]
        return sorted(list(set(supported)))

    def get_applicable_rate(
        self,
        produce_type: str,
        region_code: str = "GLOBAL_DEFAULT",
        unit: Optional[str] = None,
    ) -> Optional[Rate]:
        """Retrieve authoritative purchase rate for produce in region.

        Critical Rule (FR-013): Returns configured rate from backend source of truth.
        """
        key = f"{region_code}:{produce_type.strip().lower()}"
        rate = self._rates.get(key)
        if not rate:
            # Fallback to GLOBAL_DEFAULT if region-specific rate is not found
            key_default = f"GLOBAL_DEFAULT:{produce_type.strip().lower()}"
            rate = self._rates.get(key_default)
        return rate

    def validate_produce_offer(
        self,
        produce_type: str,
        quantity: float,
        unit: str,
        region_code: str = "GLOBAL_DEFAULT",
    ) -> bool:
        """Validate whether produce and quantity meet minimum trading rules."""
        profile = get_regional_profile(region_code)
        if unit.lower() not in [u.lower() for u in profile.standard_units]:
            return False
        if quantity <= 0:
            return False
        return produce_type.lower() in self.get_supported_produce(region_code)


pricing_service = PricingService()
