"""Deterministic Python intent-to-tool mapping and dispatch planning."""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

from orca.domain.intents import IntentType, ConversationalIntentResult
from orca.domain.schemas import ExtractedOffer


class ToolInvocationPlan(BaseModel):
    """Deterministic plan specifying which approved tool to invoke and with what arguments.
    
    Constructed strictly in Python based on validated intent and conversation context.
    The LLM has zero ability to specify or select tool names.
    """

    intent: IntentType
    tool_name: Optional[str] = None
    arguments: Dict[str, Any] = Field(default_factory=dict)
    is_state_advancing: bool = False
    validation_error: Optional[str] = None


class IntentToToolMapper:
    """Deterministic mapper resolving ConversationalIntentResult to an approved tool."""

    @staticmethod
    def map_intent_to_tool(
        intent_result: ConversationalIntentResult, context: Optional[Any] = None
    ) -> ToolInvocationPlan:
        """Map classified intent to an approved tool invocation plan."""
        intent = intent_result.intent
        entities = intent_result.entities
        order_id = entities.order_id or (getattr(context, "current_order_id", None) if context else None)
        region_code = getattr(context, "region_code", "GLOBAL_DEFAULT") if context else "GLOBAL_DEFAULT"

        # 1. First-class UNKNOWN intent: No tool execution
        if intent == IntentType.UNKNOWN:
            return ToolInvocationPlan(
                intent=IntentType.UNKNOWN,
                tool_name=None,
                arguments={},
                is_state_advancing=False,
            )

        # 2. OFFER_PRODUCE: evaluate offer details against authoritative pricing
        if intent == IntentType.OFFER_PRODUCE:
            return ToolInvocationPlan(
                intent=IntentType.OFFER_PRODUCE,
                tool_name="evaluate_produce_offer",
                arguments={
                    "offer": entities.offer,
                    "region_code": region_code,
                },
                is_state_advancing=True,
            )

        # 3. INQUIRE_SUPPORTED_PRODUCE: retrieve catalog of procured crops
        if intent == IntentType.INQUIRE_SUPPORTED_PRODUCE:
            return ToolInvocationPlan(
                intent=IntentType.INQUIRE_SUPPORTED_PRODUCE,
                tool_name="get_supported_produce",
                arguments={"region_code": region_code},
                is_state_advancing=False,
            )

        # 4. INQUIRE_ORDER_STATUS: inspect existing order details
        if intent == IntentType.INQUIRE_ORDER_STATUS:
            return ToolInvocationPlan(
                intent=IntentType.INQUIRE_ORDER_STATUS,
                tool_name="get_order_status",
                arguments={"order_id": order_id},
                is_state_advancing=False,
                validation_error="No active order ID specified or found in conversation." if not order_id else None,
            )

        # 5. INQUIRE_PAYMENT_STATUS: inspect payment record
        if intent == IntentType.INQUIRE_PAYMENT_STATUS:
            return ToolInvocationPlan(
                intent=IntentType.INQUIRE_PAYMENT_STATUS,
                tool_name="get_payment_status",
                arguments={"order_id": order_id},
                is_state_advancing=False,
                validation_error="No active order ID specified or found in conversation." if not order_id else None,
            )

        # 6. INQUIRE_COLLECTION_STATUS: inspect runner / pickup logistics task
        if intent == IntentType.INQUIRE_COLLECTION_STATUS:
            return ToolInvocationPlan(
                intent=IntentType.INQUIRE_COLLECTION_STATUS,
                tool_name="get_collection_status",
                arguments={"order_id": order_id},
                is_state_advancing=False,
                validation_error="No active order ID specified or found in conversation." if not order_id else None,
            )

        # 7. CONFIRM_ORDER: create order and initiate payment request
        if intent == IntentType.CONFIRM_ORDER:
            return ToolInvocationPlan(
                intent=IntentType.CONFIRM_ORDER,
                tool_name="confirm_and_create_order",
                arguments={"region_code": region_code},
                is_state_advancing=True,
            )

        # 8. AUTHORIZE_PAYMENT: process payment for confirmed order
        if intent == IntentType.AUTHORIZE_PAYMENT:
            return ToolInvocationPlan(
                intent=IntentType.AUTHORIZE_PAYMENT,
                tool_name="authorize_payment",
                arguments={"order_id": order_id},
                is_state_advancing=True,
                validation_error="No active order ID specified or found in conversation to authorize payment for." if not order_id else None,
            )

        # 9. REQUEST_PICKUP_CHANGE: reschedule pickup timing or location
        if intent == IntentType.REQUEST_PICKUP_CHANGE:
            return ToolInvocationPlan(
                intent=IntentType.REQUEST_PICKUP_CHANGE,
                tool_name="request_pickup_reschedule",
                arguments={
                    "order_id": order_id,
                    "new_time_str": entities.new_pickup_time,
                    "new_location": entities.new_pickup_location,
                },
                is_state_advancing=True,
                validation_error="No active order ID found to reschedule." if not order_id else None,
            )

        # 10. REPORT_COLLECTION_PROBLEM: report issue with pickup or runner
        if intent == IntentType.REPORT_COLLECTION_PROBLEM:
            return ToolInvocationPlan(
                intent=IntentType.REPORT_COLLECTION_PROBLEM,
                tool_name="report_collection_problem",
                arguments={
                    "order_id": order_id,
                    "reason": entities.problem_reason or intent_result.raw_query,
                },
                is_state_advancing=True,
                validation_error="No active order ID found to report a problem against." if not order_id else None,
            )

        # 11. REQUEST_CLARIFICATION: explain requirement, terms, or rate lookup
        if intent == IntentType.REQUEST_CLARIFICATION:
            return ToolInvocationPlan(
                intent=IntentType.REQUEST_CLARIFICATION,
                tool_name="explain_requirement",
                arguments={
                    "subject": entities.clarification_subject or intent_result.raw_query,
                    "region_code": region_code,
                },
                is_state_advancing=False,
            )

        # 12. OBJECTION_PRICING: evaluate pricing objection against authoritative policy
        if intent == IntentType.OBJECTION_PRICING:
            produce_type = (
                (entities.offer.produce_type if entities.offer else None)
                or (getattr(getattr(context, "offer", None), "produce_type", None) if context else None)
            )
            counter_rate = (
                entities.counter_rate
                or (entities.offer.offered_rate if entities.offer else None)
            )
            return ToolInvocationPlan(
                intent=IntentType.OBJECTION_PRICING,
                tool_name="handle_pricing_objection",
                arguments={
                    "produce": produce_type,
                    "counter_rate": counter_rate,
                    "region_code": region_code,
                },
                is_state_advancing=False,
                validation_error="No produce identified to evaluate pricing objection." if not produce_type else None,
            )

        # 13. AMEND_OFFER: evaluate candidate amended offer safely without premature context mutation
        if intent == IntentType.AMEND_OFFER:
            current_offer = getattr(context, "offer", None)
            candidate_dict = current_offer.model_dump() if current_offer else {}
            if entities.offer:
                for k, v in entities.offer.model_dump(exclude_none=True).items():
                    candidate_dict[k] = v
            if entities.new_pickup_location:
                candidate_dict["pickup_location"] = entities.new_pickup_location
            if entities.new_pickup_time:
                candidate_dict["availability_window"] = entities.new_pickup_time
            if entities.counter_rate is not None:
                candidate_dict["offered_rate"] = entities.counter_rate

            candidate_offer = ExtractedOffer(**candidate_dict)

            return ToolInvocationPlan(
                intent=IntentType.AMEND_OFFER,
                tool_name="evaluate_procurement_offer",
                arguments={
                    "offer": candidate_offer,
                    "region_code": region_code,
                },
                is_state_advancing=False,
            )

        # 14. INQUIRE_BUSINESS_INFO: grounded company, mission, or procurement model inquiry
        if intent == IntentType.INQUIRE_BUSINESS_INFO:
            return ToolInvocationPlan(
                intent=IntentType.INQUIRE_BUSINESS_INFO,
                tool_name="get_business_knowledge",
                arguments={
                    "query": intent_result.raw_query,
                    "topic": entities.knowledge_topic,
                },
                is_state_advancing=False,
            )

        # 15. INQUIRE_OPERATIONAL_FAQ: grounded operational policy, logistics, payment, or dispute FAQ
        if intent == IntentType.INQUIRE_OPERATIONAL_FAQ:
            return ToolInvocationPlan(
                intent=IntentType.INQUIRE_OPERATIONAL_FAQ,
                tool_name="get_business_knowledge",
                arguments={
                    "query": intent_result.raw_query,
                    "topic": entities.knowledge_topic,
                },
                is_state_advancing=False,
            )

        # 16. RECOMMEND_PRODUCE: grounded procurement recommendations
        if intent == IntentType.RECOMMEND_PRODUCE:
            return ToolInvocationPlan(
                intent=IntentType.RECOMMEND_PRODUCE,
                tool_name="get_procurement_recommendations",
                arguments={
                    "query": intent_result.raw_query,
                    "produce": entities.recommendation_produce,
                    "region_code": region_code,
                },
                is_state_advancing=False,
            )

        # 17. PERSONALIZE_RECOMMENDATION: grounded personalized recommendation based strictly on observed history/preferences
        if intent == IntentType.PERSONALIZE_RECOMMENDATION:
            farmer_id = getattr(context, "farmer_id", "farmer_default") if context else "farmer_default"
            return ToolInvocationPlan(
                intent=IntentType.PERSONALIZE_RECOMMENDATION,
                tool_name="get_personalized_recommendations",
                arguments={
                    "farmer_id": farmer_id,
                    "region_code": region_code,
                    "query": intent_result.raw_query,
                },
                is_state_advancing=False,
            )

        # 18. REQUEST_HUMAN_ASSISTANCE: operational escalation to human support
        if intent == IntentType.REQUEST_HUMAN_ASSISTANCE:
            farmer_id = getattr(context, "farmer_id", "farmer_default") if context else "farmer_default"
            reason = entities.handoff_reason or "FARMER_REQUEST"
            if not entities.handoff_reason and "payment" in intent_result.raw_query.lower() and any(
                w in intent_result.raw_query.lower() for w in ["wrong", "dispute", "discrepancy", "underpaid", "incorrect"]
            ):
                reason = "PAYMENT_DISPUTE"

            return ToolInvocationPlan(
                intent=IntentType.REQUEST_HUMAN_ASSISTANCE,
                tool_name="create_human_handoff",
                arguments={
                    "farmer_id": farmer_id,
                    "order_id": order_id,
                    "reason": reason,
                    "summary": intent_result.raw_query,
                    "source_intent": "REQUEST_HUMAN_ASSISTANCE",
                },
                is_state_advancing=False,
            )

        # Fallback default
        return ToolInvocationPlan(intent=IntentType.UNKNOWN, tool_name=None, arguments={})
