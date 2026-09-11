"""Grounded Personalization Service for ORCA.

Generates explainable, factual recommendations derived strictly from observed
farmer interaction history or explicitly stated preferences.

Contains ZERO predictive ML, NO scoring models, NO inferred wealth,
and NO automated trading actions.
"""

from typing import Optional, List
from orca.domain.farmer_profile import (
    FarmerProfile,
    PersonalizedRecommendationItem,
    PersonalizedRecommendationResult,
)
from orca.services.farmer_profile import farmer_profile_service, _normalize_commodity_name
from orca.services.pricing import pricing_service


class PersonalizationService:
    """Authoritative service for explainable, factual personalized recommendations."""

    def __init__(self):
        pass

    def get_personalized_recommendations(
        self,
        farmer_id: str,
        query: Optional[str] = None,
        region_code: str = "GLOBAL_DEFAULT",
    ) -> PersonalizedRecommendationResult:
        """Generate personalized recommendations grounded strictly in observed farmer history."""
        profile = farmer_profile_service.get_profile(farmer_id)
        supported_crops = pricing_service.get_supported_produce(region_code)

        # 1. Check if farmer has explicitly stated preferences
        if profile.explicit_preferences:
            recommendations: List[PersonalizedRecommendationItem] = []
            unsupported_prefs: List[str] = []

            for pref in profile.explicit_preferences:
                norm_pref = _normalize_commodity_name(pref)
                if norm_pref in supported_crops:
                    rate_obj = pricing_service.get_applicable_rate(norm_pref, region_code)
                    rate_val = rate_obj.rate_per_unit if rate_obj else None
                    curr = rate_obj.currency if rate_obj else "USD"
                    unit = rate_obj.unit if rate_obj else "kg"

                    reason = (
                        f"{norm_pref.title()} is relevant because you explicitly stated a preference for {norm_pref}, "
                        f"and {norm_pref} is currently supported by ORCA at an authoritative rate of "
                        f"{curr} {rate_val:.2f} per {unit}."
                    )
                    recommendations.append(
                        PersonalizedRecommendationItem(
                            produce=norm_pref.title(),
                            factual_reason=reason,
                            grounding_source="explicit_preference",
                            rate=rate_val,
                            currency=curr,
                            unit=unit,
                        )
                    )
                else:
                    unsupported_prefs.append(norm_pref.title())

            if recommendations:
                unsupported_msg = ""
                if unsupported_prefs:
                    unsupported_msg = (
                        f" Note: You also mentioned a preference for {', '.join(unsupported_prefs)}, "
                        f"which is not currently procured by ORCA in this region."
                    )

                items_text = "\n".join([f"- {r.produce}: {r.factual_reason}" for r in recommendations])
                msg = (
                    f"Based on your stated preferences:\n"
                    f"{items_text}{unsupported_msg}"
                )
                return PersonalizedRecommendationResult(
                    matched=True,
                    grounding_source="explicit_preference",
                    recommendations=recommendations,
                    confidence=1.0,
                    message=msg,
                )
            elif unsupported_prefs:
                msg = (
                    f"You previously stated a preference for {', '.join(unsupported_prefs)}, "
                    f"but that produce is not currently supported in ORCA's procurement catalog. "
                    f"Currently supported crops are: {', '.join(supported_crops)}."
                )
                return PersonalizedRecommendationResult(
                    matched=False,
                    grounding_source="explicit_preference",
                    recommendations=[],
                    confidence=0.9,
                    message=msg,
                )

        # 2. Check if farmer has completed order history
        if profile.completed_orders:
            recommendations: List[PersonalizedRecommendationItem] = []
            unsupported_completed: List[str] = []
            seen_produce = set()

            for order in profile.completed_orders:
                norm_p = _normalize_commodity_name(order.produce)
                if norm_p in seen_produce:
                    continue
                seen_produce.add(norm_p)

                if norm_p in supported_crops:
                    rate_obj = pricing_service.get_applicable_rate(norm_p, region_code)
                    rate_val = rate_obj.rate_per_unit if rate_obj else None
                    curr = rate_obj.currency if rate_obj else "USD"
                    unit = rate_obj.unit if rate_obj else "kg"

                    reason = (
                        f"{norm_p.title()} is relevant because you previously completed an order for {norm_p} through ORCA, "
                        f"and {norm_p} is currently supported by ORCA at an authoritative rate of "
                        f"{curr} {rate_val:.2f} per {unit}."
                    )
                    recommendations.append(
                        PersonalizedRecommendationItem(
                            produce=norm_p.title(),
                            factual_reason=reason,
                            grounding_source="profile_history",
                            rate=rate_val,
                            currency=curr,
                            unit=unit,
                        )
                    )
                else:
                    unsupported_completed.append(norm_p.title())

            if recommendations:
                unsupported_msg = ""
                if unsupported_completed:
                    unsupported_msg = (
                        f" Note: You previously completed sales for {', '.join(unsupported_completed)}, "
                        f"which is not currently procured by ORCA in this region."
                    )

                items_text = "\n".join([f"- {r.produce}: {r.factual_reason}" for r in recommendations])
                msg = (
                    f"You have previously completed sales through ORCA for the following supported crops:\n"
                    f"{items_text}{unsupported_msg}"
                )
                return PersonalizedRecommendationResult(
                    matched=True,
                    grounding_source="profile_history",
                    recommendations=recommendations,
                    confidence=1.0,
                    message=msg,
                )
            elif unsupported_completed:
                msg = (
                    f"You previously completed sales for {', '.join(unsupported_completed)}, "
                    f"but that produce is not currently supported in ORCA's procurement catalog. "
                    f"Currently supported crops are: {', '.join(supported_crops)}."
                )
                return PersonalizedRecommendationResult(
                    matched=False,
                    grounding_source="profile_history",
                    recommendations=[],
                    confidence=0.9,
                    message=msg,
                )

        # 3. Check if farmer has observed offer history (uncompleted offers)
        if profile.observed_produce_history:
            recommendations: List[PersonalizedRecommendationItem] = []
            unsupported_observed: List[str] = []

            for prod in profile.observed_produce_history:
                norm_p = _normalize_commodity_name(prod)
                if norm_p in supported_crops:
                    rate_obj = pricing_service.get_applicable_rate(norm_p, region_code)
                    rate_val = rate_obj.rate_per_unit if rate_obj else None
                    curr = rate_obj.currency if rate_obj else "USD"
                    unit = rate_obj.unit if rate_obj else "kg"

                    reason = (
                        f"{norm_p.title()} is relevant because you previously offered {norm_p} through ORCA, "
                        f"and {norm_p} is currently supported by ORCA at an authoritative rate of "
                        f"{curr} {rate_val:.2f} per {unit}."
                    )
                    recommendations.append(
                        PersonalizedRecommendationItem(
                            produce=norm_p.title(),
                            factual_reason=reason,
                            grounding_source="profile_history",
                            rate=rate_val,
                            currency=curr,
                            unit=unit,
                        )
                    )
                else:
                    unsupported_observed.append(norm_p.title())

            if recommendations:
                items_text = "\n".join([f"- {r.produce}: {r.factual_reason}" for r in recommendations])
                msg = (
                    f"Based on your previously offered produce:\n"
                    f"{items_text}"
                )
                return PersonalizedRecommendationResult(
                    matched=True,
                    grounding_source="profile_history",
                    recommendations=recommendations,
                    confidence=0.95,
                    message=msg,
                )
            elif unsupported_observed:
                msg = (
                    f"You previously offered {', '.join(unsupported_observed)}, "
                    f"but that produce is not currently supported in ORCA's procurement catalog. "
                    f"Currently supported crops are: {', '.join(supported_crops)}."
                )
                return PersonalizedRecommendationResult(
                    matched=False,
                    grounding_source="profile_history",
                    recommendations=[],
                    confidence=0.9,
                    message=msg,
                )

        # 4. No history / insufficient history
        supported_str = ", ".join(supported_crops)
        msg = (
            f"I don't have enough history about your previous sales or offers to personalize a recommendation yet. "
            f"I can show you ORCA's currently supported produce: {supported_str}. "
            f"What produce do you have available?"
        )
        return PersonalizedRecommendationResult(
            matched=False,
            grounding_source="insufficient_history",
            recommendations=[],
            confidence=1.0,
            clarification_required=True,
            message=msg,
        )


personalization_service = PersonalizationService()
