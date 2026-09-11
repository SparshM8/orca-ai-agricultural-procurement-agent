"""Authoritative Procurement Service.

Coordinates produce offer evaluation, policy constraints validation,
and grounded pricing rationale without duplicating pricing/billing arithmetic
or mutating domain state.
"""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from orca.domain.schemas import ExtractedOffer, BillSummary
from orca.domain.models import Rate
from orca.domain.procurement import (
    ProcurementConstraints,
    ProcurementEvaluation,
    FutureHarvestDeclaration,
    get_procurement_policy,
)
from orca.services.pricing import pricing_service, PricingService
from orca.services.billing import billing_service, BillingService
from orca.agent.extractors.base import parse_pickup_timing


class ProcurementService:
    """Authoritative evaluation service for agricultural procurement offers."""

    def __init__(
        self,
        pricing: Optional[PricingService] = None,
        billing: Optional[BillingService] = None,
    ):
        self.pricing = pricing or pricing_service
        self.billing = billing or billing_service

    def validate_constraints(
        self,
        quantity: Optional[float],
        unit: Optional[str],
        pickup_datetime: Optional[datetime] = None,
        region_code: str = "GLOBAL_DEFAULT",
        produce_type: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> List[str]:
        """Validate quantity, unit, and scheduling horizon against active policy."""
        policy = get_procurement_policy(region_code=region_code, produce_type=produce_type)
        return policy.validate_constraints(
            quantity=quantity,
            unit=unit,
            pickup_datetime=pickup_datetime,
            now=now,
        )

    def get_pricing_rationale(
        self,
        produce_type: str,
        region_code: str = "GLOBAL_DEFAULT",
    ) -> Dict[str, Any]:
        """Provide grounded fixed-rate policy rationale for a given produce type.
        
        Returns authoritative rate and non-negotiable policy benefits without
        inventing unapproved commercial concessions.
        """
        policy = get_procurement_policy(region_code=region_code, produce_type=produce_type)
        rate = self.pricing.get_applicable_rate(produce_type, region_code)
        supported = produce_type.strip().lower() in [
            p.lower() for p in self.pricing.get_supported_produce(region_code)
        ]

        return {
            "produce_type": produce_type,
            "region_code": region_code,
            "supported": supported,
            "rate_per_unit": rate.rate_per_unit if rate else None,
            "unit": rate.unit if rate else None,
            "currency": rate.currency if rate else "USD",
            "source": rate.source if rate else None,
            "fixed_rate_model": True,
            "policy_message": policy.fixed_rate_policy_message,
            "benefits": [
                "Guaranteed farm-gate collection by a local vetted runner",
                "Zero transportation deductions or transit risk for farmer",
                "Instant electronic payment verification upon pickup",
                "Transparent authoritative pricing with zero middleman broker fees",
            ],
        }

    def evaluate_offer(
        self,
        offer: ExtractedOffer,
        region_code: str = "GLOBAL_DEFAULT",
        now: Optional[datetime] = None,
    ) -> ProcurementEvaluation:
        """Deterministically evaluate an ExtractedOffer against backend constraints.
        
        Evaluates:
        1. Produce catalog support
        2. Required field presence (produce, quantity, unit, location, timing)
        3. Quantity thresholds (min/max from active policy)
        4. Unit validity from active policy
        5. Scheduling horizon (past check, max advance days)
        6. Authoritative rate lookup via PricingService
        7. Pricing discrepancy (farmer offered rate vs authoritative rate)
        8. Deterministic bill calculation via BillingService (only if valid)
        
        CRITICAL: Never mutates OrderState, orders, payments, or collections.
        """
        policy = get_procurement_policy(region_code=region_code, produce_type=offer.produce_type)
        missing_fields: List[str] = []
        rejection_reasons: List[str] = []

        # 1. Produce type validation
        produce_supported = False
        if not offer.produce_type or not offer.produce_type.strip():
            missing_fields.append("produce_type")
        else:
            supported_crops = self.pricing.get_supported_produce(region_code)
            if offer.produce_type.strip().lower() in [c.lower() for c in supported_crops]:
                produce_supported = True
            else:
                produce_supported = False
                rejection_reasons.append(f"UNSUPPORTED_PRODUCE:{offer.produce_type.strip()}")

        # 2. Check presence of other mandatory transaction fields
        if offer.quantity is None:
            missing_fields.append("quantity")

        if not offer.unit or not offer.unit.strip():
            missing_fields.append("unit")

        if not offer.pickup_location or not offer.pickup_location.strip():
            missing_fields.append("pickup_location")

        timing_str = offer.pickup_datetime or offer.availability_window
        if not timing_str or not timing_str.strip():
            missing_fields.append("availability_window")

        # 3. Parse pickup timing if provided
        target_dt = None
        if timing_str and timing_str.strip():
            target_dt, _ = parse_pickup_timing(timing_str)

        # 4. Policy constraint evaluation
        violations = policy.validate_constraints(
            quantity=offer.quantity,
            unit=offer.unit,
            pickup_datetime=target_dt,
            now=now,
        )
        if violations:
            rejection_reasons.extend(violations)

        quantity_within_limits = not any(v.startswith("QUANTITY_") for v in violations)
        timing_within_limits = not any(v.startswith("SCHEDULE_") for v in violations)

        # 5. Authoritative rate lookup (strictly backend)
        rate: Optional[Rate] = None
        authoritative_rate_val: Optional[float] = None
        if produce_supported and offer.produce_type:
            rate = self.pricing.get_applicable_rate(offer.produce_type, region_code)
            if rate:
                authoritative_rate_val = rate.rate_per_unit

        # 6. Pricing discrepancy detection
        pricing_discrepancy = False
        farmer_offered_rate = offer.offered_rate
        if farmer_offered_rate is not None and authoritative_rate_val is not None:
            if abs(farmer_offered_rate - authoritative_rate_val) > 1e-4:
                pricing_discrepancy = True

        # 7. Bill calculation (delegated to BillingService)
        bill_summary: Optional[BillSummary] = None
        is_fully_acceptable = (
            produce_supported
            and not missing_fields
            and not rejection_reasons
            and rate is not None
            and offer.quantity is not None
            and offer.quantity > 0
            and offer.unit is not None
        )

        if is_fully_acceptable and rate and offer.quantity and offer.unit:
            bill_summary = self.billing.calculate_bill(
                produce_type=rate.produce_type,
                quantity=offer.quantity,
                unit=offer.unit,
                rate_per_unit=rate.rate_per_unit,
                region_code=region_code,
            )

        # 8. Guidance message synthesis
        guidance: Optional[str] = None
        if not produce_supported and offer.produce_type:
            supported_list = ", ".join(p.title() for p in self.pricing.get_supported_produce(region_code))
            guidance = f"Produce '{offer.produce_type}' is not procured in {region_code}. We currently purchase: {supported_list}."
        elif missing_fields:
            guidance = f"Missing required offer details: {', '.join(missing_fields)}."
        elif rejection_reasons:
            guidance = f"Offer does not meet procurement constraints: {'; '.join(rejection_reasons)}."
        elif pricing_discrepancy:
            guidance = (
                f"Your suggested rate of {rate.currency if rate else 'USD'} {farmer_offered_rate:.2f} differs "
                f"from our authoritative fixed rate of {rate.currency if rate else 'USD'} {authoritative_rate_val:.2f} per {offer.unit}. "
                f"ORCA purchases strictly at authoritative fixed rates."
            )
        elif is_fully_acceptable and bill_summary:
            guidance = (
                f"Valid offer: {bill_summary.quantity:g} {bill_summary.unit} of {bill_summary.produce_type} "
                f"at authoritative rate {bill_summary.currency} {bill_summary.rate_per_unit:.2f} per {bill_summary.unit}. "
                f"Total payout: {bill_summary.currency} {bill_summary.total_amount:.2f}."
            )

        return ProcurementEvaluation(
            is_acceptable=is_fully_acceptable,
            produce_supported=produce_supported,
            rate=rate,
            bill_summary=bill_summary,
            missing_fields=missing_fields,
            rejection_reasons=rejection_reasons,
            pricing_discrepancy=pricing_discrepancy,
            farmer_offered_rate=farmer_offered_rate,
            authoritative_rate=authoritative_rate_val,
            quantity_within_limits=quantity_within_limits,
            timing_within_limits=timing_within_limits,
            guidance_message=guidance,
            policy_id=policy.policy_id,
        )


procurement_service = ProcurementService()
