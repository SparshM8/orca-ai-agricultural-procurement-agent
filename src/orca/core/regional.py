"""Regional configuration profiles and currency/unit abstractions.

Implements SRS v2.0 Global Scope Addendum:
'Global by architecture. Regional by configuration.'
"""

from enum import Enum
from typing import Dict
from pydantic import BaseModel, Field


class PricingStrategy(str, Enum):
    FIXED_PROCUREMENT_RATE = "FIXED_PROCUREMENT_RATE"
    BUYER_DEFINED = "BUYER_DEFINED"
    FARMER_DEFINED = "FARMER_DEFINED"
    NEGOTIATED = "NEGOTIATED"
    MARKET_LINKED = "MARKET_LINKED"
    GOVERNMENT_REFERENCE = "GOVERNMENT_REFERENCE"


class RegionalProfile(BaseModel):
    """Regional configuration profile for decoupling locale, currency, units, and providers."""

    region_code: str = Field(description="Unique region identifier, e.g., GLOBAL_DEFAULT, US_CA, IN_MH")
    country_name: str = Field(description="Full country name")
    currency_code: str = Field(description="ISO 4217 Currency Code, e.g., USD, EUR, INR, KES")
    currency_symbol: str = Field(description="Display symbol, e.g., $, €, ₹, KSh")
    standard_units: list[str] = Field(default_factory=lambda: ["kg", "quintal", "ton", "lb"])
    default_unit: str = "kg"
    pricing_strategy: PricingStrategy = PricingStrategy.FIXED_PROCUREMENT_RATE
    payment_provider_key: str = "SANDBOX_MOCK"
    logistics_provider_key: str = "MOCK_RUNNER_POOL"
    tax_rate_percentage: float = 0.0
    locale: str = "en-US"


# Pre-configured baseline regional profiles
REGIONAL_PROFILES: Dict[str, RegionalProfile] = {
    "GLOBAL_DEFAULT": RegionalProfile(
        region_code="GLOBAL_DEFAULT",
        country_name="Global Baseline",
        currency_code="USD",
        currency_symbol="$",
        standard_units=["kg", "quintal", "ton", "lb"],
        default_unit="kg",
        pricing_strategy=PricingStrategy.FIXED_PROCUREMENT_RATE,
        payment_provider_key="SANDBOX_MOCK",
        logistics_provider_key="MOCK_RUNNER_POOL",
        tax_rate_percentage=0.0,
        locale="en-US",
    ),
    "US_STANDARD": RegionalProfile(
        region_code="US_STANDARD",
        country_name="United States",
        currency_code="USD",
        currency_symbol="$",
        standard_units=["lb", "ton", "kg"],
        default_unit="lb",
        pricing_strategy=PricingStrategy.BUYER_DEFINED,
        payment_provider_key="SANDBOX_MOCK",
        logistics_provider_key="MOCK_RUNNER_POOL",
        tax_rate_percentage=0.0,
        locale="en-US",
    ),
    "IN_MH": RegionalProfile(
        region_code="IN_MH",
        country_name="India (Maharashtra)",
        currency_code="INR",
        currency_symbol="₹",
        standard_units=["kg", "quintal", "ton"],
        default_unit="kg",
        pricing_strategy=PricingStrategy.FIXED_PROCUREMENT_RATE,
        payment_provider_key="SANDBOX_MOCK",
        logistics_provider_key="MOCK_RUNNER_POOL",
        tax_rate_percentage=0.0,
        locale="en-IN",
    ),
    "KE_RV": RegionalProfile(
        region_code="KE_RV",
        country_name="Kenya (Rift Valley)",
        currency_code="KES",
        currency_symbol="KSh",
        standard_units=["kg", "bag", "ton"],
        default_unit="kg",
        pricing_strategy=PricingStrategy.FIXED_PROCUREMENT_RATE,
        payment_provider_key="SANDBOX_MOCK",
        logistics_provider_key="MOCK_RUNNER_POOL",
        tax_rate_percentage=0.0,
        locale="en-KE",
    ),
}


def get_regional_profile(region_code: str = "GLOBAL_DEFAULT") -> RegionalProfile:
    """Retrieve regional profile by code with fallback to GLOBAL_DEFAULT."""
    return REGIONAL_PROFILES.get(region_code, REGIONAL_PROFILES["GLOBAL_DEFAULT"])
