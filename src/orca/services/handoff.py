"""Human Handoff & Exception Escalation Service.

Provides deterministic management of operational cases escalated to human personnel.
Stores minimal, factual case metadata and never alters order/payment state directly.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union
from orca.domain.handoff import (
    HumanHandoffCase,
    HandoffStatus,
    HandoffReason,
)


class HandoffService:
    """In-memory service managing operational human handoff cases."""

    def __init__(self):
        self._cases: Dict[str, HumanHandoffCase] = {}

    def clear(self) -> None:
        """Clear all in-memory cases (used in demo reset and unit tests)."""
        self._cases.clear()

    def create_case(
        self,
        farmer_id: str,
        reason: Union[HandoffReason, str] = HandoffReason.FARMER_REQUEST,
        summary: str = "",
        source_intent: str = "REQUEST_HUMAN_ASSISTANCE",
        order_id: Optional[str] = None,
    ) -> HumanHandoffCase:
        """Create a new human handoff case deterministically."""
        resolved_reason: HandoffReason
        if isinstance(reason, HandoffReason):
            resolved_reason = reason
        else:
            try:
                resolved_reason = HandoffReason(reason.upper())
            except (ValueError, AttributeError):
                resolved_reason = HandoffReason.OTHER

        case_id = f"HC-{uuid.uuid4().hex[:6].upper()}"
        case = HumanHandoffCase(
            case_id=case_id,
            farmer_id=farmer_id,
            order_id=order_id,
            reason=resolved_reason,
            status=HandoffStatus.OPEN,
            summary=summary or f"Assistance requested for {farmer_id}",
            source_intent=source_intent,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        self._cases[case_id] = case
        return case

    def get_case(self, case_id: str) -> Optional[HumanHandoffCase]:
        """Retrieve a specific case by case_id."""
        return self._cases.get(case_id)

    def list_open_cases(self) -> List[HumanHandoffCase]:
        """List all currently OPEN or ASSIGNED cases."""
        return [
            c for c in self._cases.values()
            if c.status in [HandoffStatus.OPEN, HandoffStatus.ASSIGNED]
        ]

    def list_all_cases(
        self,
        farmer_id: Optional[str] = None,
        status: Optional[Union[HandoffStatus, str]] = None,
    ) -> List[HumanHandoffCase]:
        """List all cases ordered by creation timestamp with optional filtering."""
        cases = sorted(list(self._cases.values()), key=lambda c: c.created_at, reverse=True)
        if farmer_id:
            cases = [c for c in cases if c.farmer_id == farmer_id]
        if status:
            stat_val = status.value if isinstance(status, HandoffStatus) else str(status).upper()
            cases = [c for c in cases if c.status.value == stat_val]
        return cases

    def assign_case(self, case_id: str, assigned_to: str) -> HumanHandoffCase:
        """Assign an open case to a specific human support agent."""
        case = self.get_case(case_id)
        if not case:
            raise ValueError(f"Handoff case not found: {case_id}")
        case.assigned_to = assigned_to
        case.status = HandoffStatus.ASSIGNED
        case.updated_at = datetime.now(timezone.utc)
        return case

    def resolve_case(self, case_id: str, resolution_notes: Optional[str] = None) -> HumanHandoffCase:
        """Mark a case as resolved with optional resolution notes."""
        case = self.get_case(case_id)
        if not case:
            raise ValueError(f"Handoff case not found: {case_id}")
        case.status = HandoffStatus.RESOLVED
        if resolution_notes:
            case.resolution_notes = resolution_notes
        case.updated_at = datetime.now(timezone.utc)
        return case

    def cancel_case(self, case_id: str) -> HumanHandoffCase:
        """Cancel a handoff case."""
        case = self.get_case(case_id)
        if not case:
            raise ValueError(f"Handoff case not found: {case_id}")
        case.status = HandoffStatus.CANCELLED
        case.updated_at = datetime.now(timezone.utc)
        return case


handoff_service = HandoffService()
