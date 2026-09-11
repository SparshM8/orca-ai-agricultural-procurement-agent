"""Core conversational extraction and dialogue state orchestrator.

Implements SRS Phase 1 & 2:
- Natural language produce offer extraction
- Conversational intent classification and approved tool execution
- Multi-turn conversation context preservation
- Targeted clarification questions
- Authoritative backend pricing boundary
- Provider-independent extractor and channel interface
"""

import re
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple, List, Any
from pydantic import BaseModel, Field

from orca.domain.schemas import ExtractedOffer, InboundMessage, OutboundMessage
from orca.domain.state_machine import OrderState, validate_transition
from orca.services.pricing import pricing_service
from orca.services.billing import billing_service
from orca.services.order import order_service
from orca.services.payment import payment_service
from orca.services.collection import collection_service
from orca.core.regional import get_regional_profile
from orca.core.config import settings
from orca.agent.extractors import (
    BaseOfferExtractor,
    RuleBasedPatternExtractor,
    OpenAICompatibleAIExtractor,
    GeminiOfferExtractor,
    ResilientOfferExtractor,
    get_offer_extractor,
    parse_pickup_timing,
)
from orca.domain.intents import IntentType, ConversationalIntentResult, ToolExecutionResult
from orca.agent.intent.base import BaseIntentClassifier
from orca.agent.intent.rule_based_classifier import RuleBasedIntentClassifier
from orca.agent.intent.gemini_classifier import GeminiIntentClassifier
from orca.agent.intent.resilient_classifier import ResilientIntentClassifier
import uuid
from orca.domain.trace import AgentTrace
from orca.agent.intent.mapping import IntentToToolMapper, ToolInvocationPlan
from orca.agent.tools import tool_registry


class DialogueContext(BaseModel):
    """Preserves conversational dialogue state and extracted attributes across turns."""

    conversation_id: str
    farmer_id: str
    region_code: str = "GLOBAL_DEFAULT"
    state: OrderState = OrderState.OFFER_RECEIVED
    offer: ExtractedOffer = Field(default_factory=ExtractedOffer)
    current_order_id: Optional[str] = None
    history: List[Tuple[str, str]] = Field(default_factory=list)
    last_clarification_field: Optional[str] = None
    last_intent: Optional[str] = None
    last_tool: Optional[str] = None
    last_tool_result: Optional[Dict[str, Any]] = None
    order_history: List[str] = Field(default_factory=list)
    active_clarification: Optional[str] = None
    traces: List[AgentTrace] = Field(default_factory=list)
    last_trace: Optional[AgentTrace] = None
    negotiation_attempts: int = 0
    amendment_history: List[str] = Field(default_factory=list)
    future_availability_notes: Optional[str] = None
    clarification_queue: List[str] = Field(default_factory=list)
    is_demo: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def add_trace(self, trace: AgentTrace, max_history: int = 50) -> None:
        """Record an observational trace, keeping history bounded to max_history."""
        self.last_trace = trace
        self.traces.append(trace)
        if len(self.traces) > max_history:
            self.traces = self.traces[-max_history:]

    def get_recent_traces(self, limit: int = 10) -> List[AgentTrace]:
        """Return the most recent traces up to limit."""
        return self.traces[-limit:]


def is_confirmation_intent(text: str) -> bool:
    """Identify if user message represents an explicit confirmation."""
    cleaned = re.sub(r"[^\w\s]", "", text.strip().lower())
    confirm_phrases = {
        "confirm",
        "confirmed",
        "yes",
        "yes confirm",
        "yes confirmed",
        "yes i accept",
        "i accept",
        "accept",
        "accepted",
        "yes i agree",
        "i agree",
        "agree",
        "proceed",
        "sounds good",
        "ok",
        "okay",
        "sure",
        "deal",
    }
    if cleaned in confirm_phrases:
        return True
    words = cleaned.split()
    if words and words[0] in ["confirm", "confirmed"]:
        return True
    if len(words) >= 2 and words[0] == "yes" and words[1] in ["i", "confirm", "confirmed", "accept", "agree", "proceed"]:
        return True
    if cleaned.startswith("i accept") or cleaned.startswith("i agree"):
        return True
    return False


def is_payment_intent(text: str) -> bool:
    """Identify if user message represents an intent to authorize, make, or retry payment."""
    cleaned = re.sub(r"[^\w\s]", "", text.strip().lower())
    payment_phrases = {
        "pay",
        "pay now",
        "make payment",
        "make the payment",
        "process payment",
        "proceed with payment",
        "proceed to payment",
        "retry payment",
        "repay",
        "retry",
        "send payment",
        "execute payment",
        "please pay",
        "yes pay",
    }
    if cleaned in payment_phrases:
        return True
    words = cleaned.split()
    if words and words[0] in ["pay", "repay", "retry"]:
        return True
    if len(words) >= 2 and words[0] == "make" and words[1] == "payment":
        return True
    return False


class AgentOrchestrator:
    """Conversational agent orchestrator managing extraction, dialogue state, rate lookup, and payment."""

    def __init__(
        self,
        extractor: Optional[BaseOfferExtractor] = None,
        create_payment_request_on_confirm: bool = True,
        auto_create_collection_task: bool = True,
        auto_process_payment: bool = False,
        intent_classifier: Optional[BaseIntentClassifier] = None,
    ):
        self.extractor = extractor or get_offer_extractor()
        self.create_payment_request_on_confirm = create_payment_request_on_confirm
        self.auto_create_collection_task = auto_create_collection_task
        self.auto_process_payment = auto_process_payment
        # In-memory context storage by conversation_id
        self._conversations: Dict[str, DialogueContext] = {}

        # Initialize intent classification pipeline
        if intent_classifier is not None:
            self.intent_classifier = intent_classifier
        else:
            selected_provider = settings.AI_PROVIDER.strip().lower()
            if selected_provider == "gemini":
                primary = GeminiIntentClassifier(offer_extractor=self.extractor)
                fallback = RuleBasedIntentClassifier(offer_extractor=self.extractor)
                self.intent_classifier = ResilientIntentClassifier(primary=primary, fallback=fallback)
            else:
                self.intent_classifier = RuleBasedIntentClassifier(offer_extractor=self.extractor)

    def clear(self) -> None:
        """Clear all in-memory conversations, contexts, and traces (used in demo reset / testing)."""
        self._conversations.clear()

    def get_or_create_context(
        self, conversation_id: str, farmer_id: str = "farmer_default", region_code: str = "GLOBAL_DEFAULT"
    ) -> DialogueContext:
        """Fetch existing conversation context or initialize a new one."""
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = DialogueContext(
                conversation_id=conversation_id,
                farmer_id=farmer_id,
                region_code=region_code,
            )
        return self._conversations[conversation_id]

    def get_context(self, conversation_id: str) -> Optional[DialogueContext]:
        """Fetch existing conversation context if present."""
        return self._conversations.get(conversation_id)

    def get_context_by_order_id(self, order_id: str) -> Optional[DialogueContext]:
        """Fetch conversation context associated with an order ID."""
        for ctx in self._conversations.values():
            if ctx.current_order_id == order_id:
                return ctx
        return None

    async def process_message(
        self,
        conversation_id: str,
        message_text: str,
        farmer_id: str = "farmer_default",
        region_code: str = "GLOBAL_DEFAULT",
    ) -> OutboundMessage:
        """Process an inbound farmer message through conversational intent, tool execution, and dialogue state flow."""
        trace_id = f"trc_{uuid.uuid4().hex[:12]}"
        context = self.get_or_create_context(conversation_id, farmer_id, region_code)
        state_before = context.state.value

        trace_data: Dict[str, Any] = {
            "trace_id": trace_id,
            "detected_intent": "UNKNOWN",
            "confidence": 1.0,
            "extracted_entities": {},
            "classifier_used": getattr(self.intent_classifier, "__class__", type(self.intent_classifier)).__name__,
            "fallback_occurred": False,
            "fallback_reason": None,
            "selected_tool": None,
            "tool_arguments": None,
            "tool_success": None,
            "tool_error": None,
            "response_path": "deterministic",
            "error": None,
        }

        outbound: Optional[OutboundMessage] = None
        try:
            outbound = await self._process_message_core(
                context=context,
                conversation_id=conversation_id,
                message_text=message_text,
                farmer_id=farmer_id,
                region_code=region_code,
                trace_data=trace_data,
            )
            return outbound
        except Exception as exc:
            trace_data["error"] = str(exc)
            raise
        finally:
            state_after = context.state.value
            trace = AgentTrace.create_sanitized(
                trace_id=trace_id,
                conversation_id=conversation_id,
                sender_id=farmer_id,
                state_before=state_before,
                state_after=state_after,
                input_text=message_text,
                detected_intent=trace_data.get("detected_intent", "UNKNOWN"),
                confidence=trace_data.get("confidence", 1.0),
                extracted_entities=trace_data.get("extracted_entities", {}),
                classifier_used=trace_data.get("classifier_used", "RuleBasedIntentClassifier"),
                fallback_occurred=trace_data.get("fallback_occurred", False),
                fallback_reason=trace_data.get("fallback_reason"),
                selected_tool=trace_data.get("selected_tool"),
                tool_arguments=trace_data.get("tool_arguments"),
                tool_success=trace_data.get("tool_success"),
                tool_error=trace_data.get("tool_error"),
                response_path=trace_data.get("response_path", "deterministic"),
                error=trace_data.get("error"),
            )
            context.add_trace(trace)
            if outbound is not None:
                if outbound.metadata is None:
                    outbound.metadata = {}
                outbound.metadata["trace_id"] = trace_id

    async def _process_message_core(
        self,
        context: DialogueContext,
        conversation_id: str,
        message_text: str,
        farmer_id: str = "farmer_default",
        region_code: str = "GLOBAL_DEFAULT",
        trace_data: Optional[Dict[str, Any]] = None,
    ) -> OutboundMessage:
        if trace_data is None:
            trace_data = {}

        # 1. Authoritative backend sync: sync context.state with order if exists
        if context.current_order_id:
            order_rec = order_service.get_order(context.current_order_id)
            if order_rec:
                context.state = order_rec.status
            if context.current_order_id not in context.order_history:
                context.order_history.append(context.current_order_id)

        context.history.append(("farmer", message_text))

        # 2. Intent Classification via configured classifier
        intent_res = await self.intent_classifier.classify(message_text, context=context)
        intent = intent_res.intent
        context.last_intent = intent.value

        trace_data["detected_intent"] = intent.value
        trace_data["confidence"] = intent_res.confidence
        trace_data["extracted_entities"] = intent_res.entities.model_dump()
        trace_data["classifier_used"] = getattr(intent_res, "classifier_name", None) or getattr(self.intent_classifier, "__class__", type(self.intent_classifier)).__name__
        trace_data["fallback_occurred"] = getattr(intent_res, "fallback_occurred", False)
        trace_data["fallback_reason"] = getattr(intent_res, "fallback_reason", None)

        # Sync explicit order_id if extracted
        if intent_res.entities.order_id:
            context.current_order_id = intent_res.entities.order_id
            if intent_res.entities.order_id not in context.order_history:
                context.order_history.append(intent_res.entities.order_id)

        # Sync explicit farmer preference if stated
        if intent_res.entities.explicit_preference:
            try:
                from orca.services.farmer_profile import farmer_profile_service
                farmer_profile_service.record_explicit_preference(
                    farmer_id=farmer_id,
                    preference=intent_res.entities.explicit_preference,
                )
            except Exception:
                pass

        # 3. Deterministic Intent-to-Tool Mapping Plan
        plan = IntentToToolMapper.map_intent_to_tool(intent_res, context=context)
        trace_data["selected_tool"] = plan.tool_name
        trace_data["tool_arguments"] = plan.arguments

        # -------------------------------------------------------------
        # Branch A: Payment Authorization (AUTHORIZE_PAYMENT)
        # -------------------------------------------------------------
        if intent == IntentType.AUTHORIZE_PAYMENT:
            # Case A1: Already confirmed and paid / in collection states
            if (
                context.state in [
                    OrderState.PAYMENT_CONFIRMED,
                    OrderState.COLLECTION_PENDING,
                    OrderState.COLLECTION_ASSIGNED,
                    OrderState.PICKED_UP,
                    OrderState.COMPLETED,
                ]
                and context.current_order_id
            ):
                order = order_service.get_order(context.current_order_id)
                payment = payment_service.get_payment_for_order(context.current_order_id)
                coll_task = collection_service.get_task_by_order(context.current_order_id)
                reply = (
                    f"Payment has already been confirmed for this order.\n"
                    f"- Order ID: {context.current_order_id}\n"
                    f"- Produce: {context.offer.quantity:g} {context.offer.unit} of {context.offer.produce_type}\n"
                    f"- Total Amount: {order.currency if order else 'USD'} {order.total_amount if order else 0.0:.2f}\n"
                    f"- Payment ID: {payment.id if payment else 'N/A'}\n"
                    f"- Payment Status: SUCCESS (Ref: {payment.provider_reference if payment else 'N/A'})\n"
                    f"- Order Status: {order.status.value if order else context.state.value}\n"
                    f"- Collection Task: {coll_task.id if coll_task else 'N/A'} ({coll_task.status if coll_task else 'N/A'})\n\n"
                    f"No further action is required. Our logistics runner will collect the produce as scheduled."
                )
                context.history.append(("agent", reply))
                return OutboundMessage(
                    recipient_id=farmer_id,
                    text=reply,
                    metadata={
                        "state": context.state.value,
                        "order_id": context.current_order_id,
                        "is_duplicate": True,
                        "payment_status": "SUCCESS",
                        "collection_task_id": coll_task.id if coll_task else None,
                    },
                )

            # Case A2: In PAYMENT_PENDING -> Execute payment via DemoPaymentAdapter
            if (
                context.state == OrderState.PAYMENT_PENDING
                or (
                    context.current_order_id
                    and order_service.get_order(context.current_order_id)
                    and order_service.get_order(context.current_order_id).status == OrderState.PAYMENT_PENDING
                )
            ):
                order = order_service.get_order(context.current_order_id)
                payment, is_paid, pay_msg = await payment_service.initiate_order_payment(
                    context.current_order_id
                )
                context.last_tool = "create_payment_request"
                context.last_tool_result = {"payment_id": payment.id, "status": payment.status, "success": is_paid}
                trace_data["tool_success"] = is_paid
                trace_data["tool_error"] = None if is_paid else pay_msg
                if is_paid:
                    if self.auto_create_collection_task:
                        timing_str = (
                            context.offer.pickup_datetime
                            or context.offer.availability_window
                            or (order.pickup_time_str if order else None)
                        )
                        target_dt, _ = parse_pickup_timing(timing_str)
                        coll_task = await collection_service.create_collection_task(
                            order_id=context.current_order_id,
                            scheduled_datetime=target_dt or (order.pickup_datetime if order else None),
                            scheduled_time_str=timing_str,
                        )
                        context.last_tool = "create_collection_task"
                        context.state = OrderState.COLLECTION_PENDING
                        reply = (
                            f"Payment Confirmed!\n"
                            f"- Order ID: {context.current_order_id}\n"
                            f"- Produce: {context.offer.quantity:g} {context.offer.unit} of {context.offer.produce_type}\n"
                            f"- Authoritative Rate: {order.currency if order else 'USD'} {order.validated_rate if order else 0.0:.2f} per {context.offer.unit}\n"
                            f"- Total Amount: {order.currency if order else 'USD'} {order.total_amount if order else 0.0:.2f}\n"
                            f"- Payment ID: {payment.id}\n"
                            f"- Payment Status: SUCCESS (Ref: {payment.provider_reference})\n"
                            f"- Order Status: COLLECTION_PENDING\n\n"
                            f"Collection Details:\n"
                            f"- Task ID: {coll_task.id}\n"
                            f"- Pickup Location: {coll_task.pickup_location}\n"
                            f"- Scheduled Window: {coll_task.scheduled_time_str or 'As scheduled'}\n"
                            f"- Collection Status: PENDING\n\n"
                            f"Thank you! Your payment has been confirmed and your collection task has been published to our runner network. A local runner will be assigned for pickup shortly."
                        )
                        context.history.append(("agent", reply))
                        return OutboundMessage(
                            recipient_id=farmer_id,
                            text=reply,
                            metadata={
                                "state": context.state.value,
                                "order_id": context.current_order_id,
                                "produce": context.offer.produce_type,
                                "quantity": context.offer.quantity,
                                "unit": context.offer.unit,
                                "authoritative_rate": order.validated_rate if order else 0.0,
                                "total_amount": order.total_amount if order else 0.0,
                                "currency": order.currency if order else "USD",
                                "status": OrderState.COLLECTION_PENDING.value,
                                "payment_id": payment.id,
                                "payment_status": "SUCCESS",
                                "provider_reference": payment.provider_reference,
                                "collection_task_id": coll_task.id,
                                "collection_status": coll_task.status,
                            },
                        )
                    else:
                        context.state = OrderState.PAYMENT_CONFIRMED
                        reply = (
                            f"Payment Confirmed!\n"
                            f"- Order ID: {context.current_order_id}\n"
                            f"- Produce: {context.offer.quantity:g} {context.offer.unit} of {context.offer.produce_type}\n"
                            f"- Authoritative Rate: {order.currency if order else 'USD'} {order.validated_rate if order else 0.0:.2f} per {context.offer.unit}\n"
                            f"- Total Amount: {order.currency if order else 'USD'} {order.total_amount if order else 0.0:.2f}\n"
                            f"- Payment ID: {payment.id}\n"
                            f"- Payment Status: SUCCESS (Ref: {payment.provider_reference})\n"
                            f"- Order Status: PAYMENT_CONFIRMED\n"
                            f"- Pickup Location: {order.pickup_location if order else context.offer.pickup_location}\n\n"
                            f"Thank you! Your payment has been confirmed and our logistics runner has been scheduled for pickup."
                        )
                        context.history.append(("agent", reply))
                        return OutboundMessage(
                            recipient_id=farmer_id,
                            text=reply,
                            metadata={
                                "state": context.state.value,
                                "order_id": context.current_order_id,
                                "produce": context.offer.produce_type,
                                "quantity": context.offer.quantity,
                                "unit": context.offer.unit,
                                "authoritative_rate": order.validated_rate if order else 0.0,
                                "total_amount": order.total_amount if order else 0.0,
                                "currency": order.currency if order else "USD",
                                "status": OrderState.PAYMENT_CONFIRMED.value,
                                "payment_id": payment.id,
                                "payment_status": "SUCCESS",
                                "provider_reference": payment.provider_reference,
                            },
                        )
                else:
                    context.state = OrderState.PAYMENT_PENDING
                    reply = (
                        f"Payment Failed: {pay_msg}\n"
                        f"- Order ID: {context.current_order_id}\n"
                        f"- Total Amount: {order.currency if order else 'USD'} {order.total_amount if order else 0.0:.2f}\n"
                        f"- Order Status: PAYMENT_PENDING\n"
                        f"- Payment Status: FAILED\n\n"
                        f"Please reply 'Pay' or 'Retry payment' to attempt processing again."
                    )
                    context.history.append(("agent", reply))
                    return OutboundMessage(
                        recipient_id=farmer_id,
                        text=reply,
                        metadata={
                            "state": context.state.value,
                            "order_id": context.current_order_id,
                            "status": OrderState.PAYMENT_PENDING.value,
                            "payment_id": payment.id,
                            "payment_status": "FAILED",
                        },
                    )

            # Case A3: Farmer says "Pay" while still in AWAITING_FARMER_CONFIRMATION
            if context.state == OrderState.AWAITING_FARMER_CONFIRMATION:
                reply = "Please confirm your order first by replying 'Confirm'. Once confirmed, we will generate your payment request."
                context.history.append(("agent", reply))
                return OutboundMessage(
                    recipient_id=farmer_id,
                    text=reply,
                    metadata={"state": context.state.value},
                )

            # Case A4: No active order to pay for
            reply = "There is no active order to pay for. Please provide your produce offer details first."
            context.history.append(("agent", reply))
            return OutboundMessage(
                recipient_id=farmer_id,
                text=reply,
                metadata={"state": context.state.value},
            )

        # -------------------------------------------------------------
        # Branch B: Explicit Confirmation (CONFIRM_ORDER)
        # -------------------------------------------------------------
        if intent == IntentType.CONFIRM_ORDER:
            # Case B1: Duplicate confirmation on already confirmed order / payment
            if (context.state in [OrderState.ORDER_CONFIRMED, OrderState.PAYMENT_CONFIRMED]) and context.current_order_id:
                existing_order = order_service.get_order(context.current_order_id)
                payment = payment_service.get_payment_for_order(context.current_order_id)
                reply = (
                    f"This transaction has already been confirmed.\n"
                    f"- Order ID: {context.current_order_id}\n"
                    f"- Produce: {context.offer.quantity:g} {context.offer.unit} of {context.offer.produce_type}\n"
                    f"- Authoritative Rate: {existing_order.currency if existing_order else 'USD'} {existing_order.validated_rate if existing_order else 0.0:.2f} per {context.offer.unit}\n"
                    f"- Total Amount: {existing_order.currency if existing_order else 'USD'} {existing_order.total_amount if existing_order else 0.0:.2f}\n"
                    f"- Payment Status: {payment.status if payment else 'N/A'}\n"
                    f"- Status: {context.state.value}\n\n"
                    f"Our logistics runner will be assigned for pickup as scheduled."
                )
                context.history.append(("agent", reply))
                return OutboundMessage(
                    recipient_id=farmer_id,
                    text=reply,
                    metadata={
                        "state": context.state.value,
                        "order_id": context.current_order_id,
                        "is_duplicate": True,
                        "payment_status": payment.status if payment else None,
                    },
                )

            # Case B2: In PAYMENT_PENDING state
            if context.state == OrderState.PAYMENT_PENDING and context.current_order_id:
                order = order_service.get_order(context.current_order_id)
                payment = payment_service.get_payment_for_order(context.current_order_id)
                reply = (
                    f"Your order has already been confirmed and is awaiting payment.\n"
                    f"- Order ID: {context.current_order_id}\n"
                    f"- Produce: {context.offer.quantity:g} {context.offer.unit} of {context.offer.produce_type}\n"
                    f"- Total Amount: {order.currency if order else 'USD'} {order.total_amount if order else 0.0:.2f}\n"
                    f"- Payment ID: {payment.id if payment else 'N/A'}\n"
                    f"- Payment Status: PAYMENT_PENDING\n"
                    f"- Order Status: PAYMENT_PENDING\n\n"
                    f"Please reply 'Pay' or 'Make payment' to process your payout."
                )
                context.history.append(("agent", reply))
                return OutboundMessage(
                    recipient_id=farmer_id,
                    text=reply,
                    metadata={
                        "state": context.state.value,
                        "order_id": context.current_order_id,
                        "status": OrderState.PAYMENT_PENDING.value,
                        "payment_status": "PENDING",
                    },
                )

            # Case B3: Awaiting confirmation -> Create validated order idempotently!
            if context.state == OrderState.AWAITING_FARMER_CONFIRMATION:
                if (
                    not context.offer.produce_type
                    or context.offer.quantity is None
                    or context.offer.quantity <= 0
                    or not context.offer.unit
                    or not context.offer.pickup_location
                ):
                    context.state = OrderState.DETAILS_PENDING
                    reply = "Cannot confirm transaction: some required details are missing or invalid."
                    context.history.append(("agent", reply))
                    return OutboundMessage(recipient_id=farmer_id, text=reply)

                timing_str = context.offer.pickup_datetime or context.offer.availability_window
                target_dt, _ = parse_pickup_timing(timing_str)

                order = await order_service.create_order_async(
                    farmer_id=farmer_id,
                    produce_type=context.offer.produce_type,
                    quantity=context.offer.quantity,
                    unit=context.offer.unit,
                    pickup_location=context.offer.pickup_location,
                    region_code=region_code,
                    conversation_id=conversation_id,
                    pickup_datetime=target_dt,
                    pickup_time_str=timing_str,
                )

                context.current_order_id = order.id
                if order.id not in context.order_history:
                    context.order_history.append(order.id)
                context.last_tool = "create_order"
                context.last_tool_result = {"order_id": order.id, "status": order.status.value, "total_amount": order.total_amount}
                trace_data["tool_success"] = True

                if self.create_payment_request_on_confirm:
                    payment, _ = await payment_service.create_payment_request(order.id)
                    context.last_tool = "create_payment_request"

                    if self.auto_process_payment:
                        payment, is_paid, pay_msg = await payment_service.initiate_order_payment(order.id)
                        if is_paid:
                            if self.auto_create_collection_task:
                                coll_task = await collection_service.create_collection_task(
                                    order_id=order.id,
                                    scheduled_datetime=target_dt or order.pickup_datetime,
                                    scheduled_time_str=timing_str or order.pickup_time_str,
                                )
                                context.last_tool = "create_collection_task"
                            context.state = OrderState.PAYMENT_CONFIRMED
                            confirmation_reply = (
                                f"Order and Payment Confirmed!\n"
                                f"- Order ID: {order.id}\n"
                                f"- Produce: {order.quantity:g} {order.unit} of {order.produce_type}\n"
                                f"- Authoritative Rate: {order.currency} {order.validated_rate:.2f} per {order.unit}\n"
                                f"- Total Amount: {order.currency} {order.total_amount:.2f}\n"
                                f"- Payment Status: SUCCESS (Ref: {payment.provider_reference})\n"
                                f"- Order Status: PAYMENT_CONFIRMED\n"
                                f"- Pickup Location: {order.pickup_location}\n"
                                f"- Scheduled Window: {order.pickup_time_str or timing_str or 'As scheduled'}\n\n"
                                f"Thank you! Payment has been recorded and your order is confirmed for pickup."
                            )
                        else:
                            context.state = OrderState.PAYMENT_PENDING
                            confirmation_reply = (
                                f"Order Confirmed, but Payment Failed.\n"
                                f"- Order ID: {order.id}\n"
                                f"- Produce: {order.quantity:g} {order.unit} of {order.produce_type}\n"
                                f"- Total Amount: {order.currency} {order.total_amount:.2f}\n"
                                f"- Order Status: PAYMENT_PENDING\n"
                                f"- Payment Status: FAILED\n\n"
                                f"{pay_msg}"
                            )
                        context.history.append(("agent", confirmation_reply))
                        return OutboundMessage(
                            recipient_id=farmer_id,
                            text=confirmation_reply,
                            metadata={
                                "state": context.state.value,
                                "order_id": order.id,
                                "produce": order.produce_type,
                                "quantity": order.quantity,
                                "unit": order.unit,
                                "authoritative_rate": order.validated_rate,
                                "total_amount": order.total_amount,
                                "currency": order.currency,
                                "status": order.status.value,
                                "payment_id": payment.id,
                                "payment_status": payment.status,
                            },
                        )

                    context.state = OrderState.PAYMENT_PENDING
                    timing_display = order.pickup_time_str or timing_str or "As scheduled"
                    confirmation_reply = (
                        f"Order Confirmed!\n"
                        f"- Order ID: {order.id}\n"
                        f"- Produce: {order.quantity:g} {order.unit} of {order.produce_type}\n"
                        f"- Authoritative Rate: {order.currency} {order.validated_rate:.2f} per {order.unit}\n"
                        f"- Total Amount: {order.currency} {order.total_amount:.2f}\n"
                        f"- Pickup Location: {order.pickup_location}\n"
                        f"- Scheduled Window: {timing_display}\n"
                        f"- Order Status: ORDER_CONFIRMED\n\n"
                        f"Payment Request:\n"
                        f"- Payment ID: {payment.id}\n"
                        f"- Payout Amount: {order.currency} {order.total_amount:.2f}\n"
                        f"- Payment Status: PAYMENT_PENDING\n\n"
                        f"Please reply 'Pay' or 'Make payment' to process your payout."
                    )
                    context.history.append(("agent", confirmation_reply))
                    return OutboundMessage(
                        recipient_id=farmer_id,
                        text=confirmation_reply,
                        metadata={
                            "state": context.state.value,
                            "order_id": order.id,
                            "produce": order.produce_type,
                            "quantity": order.quantity,
                            "unit": order.unit,
                            "authoritative_rate": order.validated_rate,
                            "total_amount": order.total_amount,
                            "currency": order.currency,
                            "status": OrderState.PAYMENT_PENDING.value,
                            "payment_id": payment.id,
                            "payment_status": payment.status,
                            "scheduled_time_str": timing_display,
                        },
                    )

                context.state = OrderState.ORDER_CONFIRMED
                timing_display = order.pickup_time_str or timing_str or "As scheduled"
                confirmation_reply = (
                    f"Order Confirmed! Your procurement order has been successfully created.\n"
                    f"- Order ID: {order.id}\n"
                    f"- Produce: {order.quantity:g} {order.unit} of {order.produce_type}\n"
                    f"- Authoritative Rate: {order.currency} {order.validated_rate:.2f} per {order.unit}\n"
                    f"- Total Amount: {order.currency} {order.total_amount:.2f}\n"
                    f"- Pickup Location: {order.pickup_location}\n"
                    f"- Scheduled Window: {timing_display}\n"
                    f"- Status: ORDER_CONFIRMED\n\n"
                    f"Thank you for confirming. Your order is now registered with our procurement desk."
                )
                context.history.append(("agent", confirmation_reply))
                return OutboundMessage(
                    recipient_id=farmer_id,
                    text=confirmation_reply,
                    metadata={
                        "state": context.state.value,
                        "order_id": order.id,
                        "produce": order.produce_type,
                        "quantity": order.quantity,
                        "unit": order.unit,
                        "authoritative_rate": order.validated_rate,
                        "total_amount": order.total_amount,
                        "currency": order.currency,
                        "status": order.status.value,
                    },
                )

            # Case B4: Not awaiting confirmation
            if not context.offer.produce_type:
                reply = (
                    "There is no pending transaction to confirm. "
                    "Please let us know what agricultural produce you would like to sell."
                )
            else:
                supported_produce = pricing_service.get_supported_produce(region_code)
                if context.offer.produce_type not in supported_produce:
                    reply = (
                        f"Cannot confirm transaction: '{context.offer.produce_type}' is not currently procured in this region. "
                        f"We currently procure: {', '.join(supported_produce)}."
                    )
                else:
                    reply = (
                        "Cannot confirm order yet because some required details are missing. "
                        "Please provide the remaining information before we can generate your order."
                    )
            context.history.append(("agent", reply))
            return OutboundMessage(
                recipient_id=farmer_id,
                text=reply,
                metadata={"state": context.state.value},
            )

        # -------------------------------------------------------------
        # Branch C: Cancellation
        # -------------------------------------------------------------
        if context.state == OrderState.AWAITING_FARMER_CONFIRMATION and any(
            w in message_text.lower().split() for w in ["cancel", "decline", "reject"]
        ):
            context.state = OrderState.CANCELLED
            reply = "Your transaction offer has been cancelled. Please let us know if you would like to start a new offer."
            context.history.append(("agent", reply))
            return OutboundMessage(
                recipient_id=farmer_id,
                text=reply,
                metadata={"state": context.state.value},
            )

        # -------------------------------------------------------------
        # Branch D: Supported Produce Inquiry (INQUIRE_SUPPORTED_PRODUCE)
        # -------------------------------------------------------------
        if intent == IntentType.INQUIRE_SUPPORTED_PRODUCE:
            tool_res = await tool_registry.execute_tool("get_supported_produce", plan.arguments)
            context.last_tool = "get_supported_produce"
            context.last_tool_result = tool_res.data
            trace_data["tool_success"] = tool_res.success
            trace_data["tool_error"] = tool_res.error_message
            crops = tool_res.data.get("items", [])
            rates_info = []
            for p in crops:
                r = pricing_service.get_applicable_rate(p, region_code)
                if r:
                    rates_info.append(f"- {p.title()}: {r.currency} {r.rate_per_unit:.2f} per {r.unit}")
                else:
                    rates_info.append(f"- {p.title()}")
            rates_str = "\n".join(rates_info)
            reply = (
                f"We currently procure the following produce at fixed authoritative rates:\n"
                f"{rates_str}\n\n"
                f"To sell your harvest, tell me the produce type, quantity, pickup location, and when it will be ready."
            )
            context.history.append(("agent", reply))
            return OutboundMessage(
                recipient_id=farmer_id,
                text=reply,
                metadata={
                    "state": context.state.value,
                    "intent": "INQUIRE_SUPPORTED_PRODUCE",
                    "supported_produce": crops,
                },
            )

        # -------------------------------------------------------------
        # Branch E: Order Status Inquiry (INQUIRE_ORDER_STATUS)
        # -------------------------------------------------------------
        if intent == IntentType.INQUIRE_ORDER_STATUS:
            order_id = plan.arguments.get("order_id")
            if not order_id:
                context.active_clarification = "order_id"
                reply = "Please specify the Order ID you would like to check (for example, ORD-...)."
                context.history.append(("agent", reply))
                return OutboundMessage(
                    recipient_id=farmer_id,
                    text=reply,
                    metadata={"state": context.state.value, "missing_entity": "order_id"},
                )

            context.active_clarification = None
            tool_res = await tool_registry.execute_tool("get_order_status", {"order_id": order_id})
            context.last_tool = "get_order_status"
            context.last_tool_result = tool_res.data
            trace_data["tool_success"] = tool_res.success
            trace_data["tool_error"] = tool_res.error_message
            data = tool_res.data

            if not data.get("found"):
                reply = f"I could not find an order with ID '{order_id}'. Please check the order number and try again."
                context.history.append(("agent", reply))
                return OutboundMessage(
                    recipient_id=farmer_id,
                    text=reply,
                    metadata={"state": context.state.value, "order_id": order_id, "found": False},
                )

            order = order_service.get_order(order_id)
            coll_task = collection_service.get_task_by_order(order_id)
            payment = payment_service.get_payment_for_order(order_id)

            reply = (
                f"Collection & Order Status:\n"
                f"- Order ID: {order_id}\n"
                f"- Order Status: {data['status']}\n"
                f"- Produce: {data['quantity']:g} {data['unit']} of {data['produce']}\n"
                f"- Total Payout: {data['currency']} {data['total_amount']:.2f} ({'PAID' if payment and payment.status == 'SUCCESS' else (payment.status if payment else 'PENDING')})\n"
                f"- Collection Task ID: {coll_task.id if coll_task else 'N/A'}\n"
                f"- Pickup Location: {data['pickup_location']}\n"
                f"- Scheduled Window: {coll_task.scheduled_time_str if coll_task and coll_task.scheduled_time_str else (data.get('pickup_time_str') or 'As scheduled')}\n"
                f"- Collection Status: {coll_task.status if coll_task else 'N/A'}\n"
            )
            if coll_task and coll_task.runner_id:
                reply += f"- Assigned Runner: {coll_task.runner_id}\n"
            if order and order.status == OrderState.COMPLETED:
                reply += "\nYour produce pickup has been completed! Thank you for selling with ORCA."
            elif order and order.status == OrderState.PICKED_UP:
                reply += "\nYour produce has been picked up by the runner and is currently en route."
            elif order and order.status == OrderState.COLLECTION_ASSIGNED:
                reply += "\nA runner has been assigned and is en route to collect your produce."
            elif coll_task and coll_task.status == "PENDING":
                reply += "\nYour collection task is pending runner assignment. A runner will be assigned shortly."

            context.history.append(("agent", reply))
            return OutboundMessage(
                recipient_id=farmer_id,
                text=reply,
                metadata={
                    "state": context.state.value,
                    "order_id": order_id,
                    "collection_task_id": coll_task.id if coll_task else None,
                    "status": data["status"],
                },
            )

        # -------------------------------------------------------------
        # Branch F: Payment Status Inquiry (INQUIRE_PAYMENT_STATUS)
        # -------------------------------------------------------------
        if intent == IntentType.INQUIRE_PAYMENT_STATUS:
            order_id = plan.arguments.get("order_id")
            if not order_id:
                context.active_clarification = "order_id"
                reply = "Please specify the Order ID to check your payment status (for example, ORD-...)."
                context.history.append(("agent", reply))
                return OutboundMessage(
                    recipient_id=farmer_id,
                    text=reply,
                    metadata={"state": context.state.value, "missing_entity": "order_id"},
                )

            context.active_clarification = None
            tool_res = await tool_registry.execute_tool("get_payment_status", {"order_id": order_id})
            context.last_tool = "get_payment_status"
            context.last_tool_result = tool_res.data
            trace_data["tool_success"] = tool_res.success
            trace_data["tool_error"] = tool_res.error_message
            data = tool_res.data
            pay_status = data.get("status")
            order = order_service.get_order(order_id)

            if pay_status == "NOT_FOUND":
                reply = f"No payment record found for order {order_id}. Please ensure your order has been confirmed."
            else:
                reply = (
                    f"Payment Status for Order {order_id}:\n"
                    f"- Payment ID: {data.get('payment_id')}\n"
                    f"- Payment Status: {pay_status}\n"
                    f"- Provider Reference: {data.get('provider_reference') or 'N/A'}\n"
                    f"- Order Status: {order.status.value if order else 'UNKNOWN'}\n"
                    f"- Total Amount: {order.currency if order else 'USD'} {order.total_amount if order else 0.0:.2f}"
                )
            context.history.append(("agent", reply))
            return OutboundMessage(
                recipient_id=farmer_id,
                text=reply,
                metadata={
                    "state": context.state.value,
                    "order_id": order_id,
                    "payment_status": pay_status,
                },
            )

        # -------------------------------------------------------------
        # Branch G: Collection Status Inquiry (INQUIRE_COLLECTION_STATUS)
        # -------------------------------------------------------------
        if intent == IntentType.INQUIRE_COLLECTION_STATUS:
            order_id = plan.arguments.get("order_id")
            if not order_id:
                context.active_clarification = "order_id"
                reply = "Please provide your Order ID so I can check your collection and runner status."
                context.history.append(("agent", reply))
                return OutboundMessage(
                    recipient_id=farmer_id,
                    text=reply,
                    metadata={"state": context.state.value, "missing_entity": "order_id"},
                )

            context.active_clarification = None
            tool_res = await tool_registry.execute_tool("get_collection_status", {"order_id": order_id})
            context.last_tool = "get_collection_status"
            context.last_tool_result = tool_res.data
            trace_data["tool_success"] = tool_res.success
            trace_data["tool_error"] = tool_res.error_message
            data = tool_res.data
            order = order_service.get_order(order_id)
            coll_task = collection_service.get_task_by_order(order_id)

            if data.get("status") == "NOT_FOUND":
                reply = f"No collection task found for order {order_id}. Collection tasks are scheduled once payment is confirmed."
            else:
                reply = (
                    f"Collection & Order Status:\n"
                    f"- Order ID: {order_id}\n"
                    f"- Order Status: {order.status.value if order else context.state.value}\n"
                    f"- Produce: {order.quantity if order else context.offer.quantity:g} {order.unit if order else context.offer.unit} of {order.produce_type if order else context.offer.produce_type}\n"
                    f"- Total Payout: {order.currency if order else 'USD'} {order.total_amount if order else 0.0:.2f} (PAID)\n"
                    f"- Collection Task ID: {data.get('task_id')}\n"
                    f"- Pickup Location: {order.pickup_location if order else context.offer.pickup_location}\n"
                    f"- Scheduled Window: {coll_task.scheduled_time_str if coll_task and coll_task.scheduled_time_str else (order.pickup_time_str if order and order.pickup_time_str else 'As scheduled')}\n"
                    f"- Collection Status: {data.get('status')}\n"
                )
                if data.get("runner_id"):
                    reply += f"- Assigned Runner: {data.get('runner_id')}\n"
                if order and order.status == OrderState.COMPLETED:
                    reply += "\nYour produce pickup has been completed! Thank you for selling with ORCA."
                elif order and order.status == OrderState.PICKED_UP:
                    reply += "\nYour produce has been picked up by the runner and is currently en route."
                elif order and order.status == OrderState.COLLECTION_ASSIGNED:
                    reply += "\nA runner has been assigned and is en route to collect your produce."
                else:
                    reply += "\nYour collection task is pending runner assignment. A runner will be assigned shortly."

            context.history.append(("agent", reply))
            return OutboundMessage(
                recipient_id=farmer_id,
                text=reply,
                metadata={
                    "state": context.state.value,
                    "order_id": order_id,
                    "collection_task_id": data.get("task_id"),
                    "status": order.status.value if order else context.state.value,
                },
            )

        # -------------------------------------------------------------
        # Branch H: Pickup Reschedule Request (REQUEST_PICKUP_CHANGE)
        # -------------------------------------------------------------
        if intent == IntentType.REQUEST_PICKUP_CHANGE:
            order_id = plan.arguments.get("order_id")
            if not order_id:
                context.active_clarification = "order_id"
                reply = "Please provide the Order ID for the pickup you would like to reschedule."
                context.history.append(("agent", reply))
                return OutboundMessage(
                    recipient_id=farmer_id,
                    text=reply,
                    metadata={"state": context.state.value, "missing_entity": "order_id"},
                )

            context.active_clarification = None
            tool_res = await tool_registry.execute_tool("request_pickup_reschedule", plan.arguments)
            context.last_tool = "request_pickup_reschedule"
            context.last_tool_result = tool_res.data
            trace_data["tool_success"] = tool_res.success
            trace_data["tool_error"] = tool_res.error_message

            if not tool_res.success:
                reply = f"Could not reschedule pickup: {tool_res.error_message}"
            else:
                d = tool_res.data
                reply = (
                    f"Pickup reschedule request processed successfully.\n"
                    f"- Order ID: {order_id}\n"
                    f"- Task ID: {d.get('task_id')}\n"
                    f"- New Scheduled Window: {d.get('scheduled_time_str') or 'Updated'}\n"
                    f"- Pickup Location: {d.get('pickup_location')}\n"
                    f"- Collection Status: {d.get('status')}"
                )
            context.history.append(("agent", reply))
            return OutboundMessage(
                recipient_id=farmer_id,
                text=reply,
                metadata={
                    "state": context.state.value,
                    "order_id": order_id,
                    "rescheduled": tool_res.success,
                },
            )

        # -------------------------------------------------------------
        # Branch I: Report Collection Problem (REPORT_COLLECTION_PROBLEM)
        # -------------------------------------------------------------
        if intent == IntentType.REPORT_COLLECTION_PROBLEM:
            order_id = plan.arguments.get("order_id")
            if not order_id:
                context.active_clarification = "order_id"
                reply = "Please provide your Order ID so we can record and resolve the collection issue."
                context.history.append(("agent", reply))
                return OutboundMessage(
                    recipient_id=farmer_id,
                    text=reply,
                    metadata={"state": context.state.value, "missing_entity": "order_id"},
                )

            context.active_clarification = None
            tool_res = await tool_registry.execute_tool("report_collection_problem", plan.arguments)
            context.last_tool = "report_collection_problem"
            context.last_tool_result = tool_res.data
            trace_data["tool_success"] = tool_res.success
            trace_data["tool_error"] = tool_res.error_message

            if not tool_res.success:
                reply = f"Could not record collection problem: {tool_res.error_message}"
            else:
                d = tool_res.data
                handoff_note = ""
                try:
                    from orca.services.handoff import handoff_service
                    from orca.domain.handoff import HandoffReason
                    hc = handoff_service.create_case(
                        farmer_id=farmer_id,
                        order_id=order_id,
                        reason=HandoffReason.COLLECTION_FAILURE,
                        summary=f"Collection problem reported for Order {order_id}: {d.get('reason', 'unspecified issue')}",
                        source_intent=intent.value,
                    )
                    handoff_note = f"\nA human support case ({hc.case_id}) has been created for logistics team escalation."
                except Exception:
                    pass

                reply = (
                    f"We have recorded the collection issue for Order {order_id}.\n"
                    f"- Issue Reason: {d.get('reason')}\n"
                    f"- Task ID: {d.get('task_id')}\n"
                    f"- Task Status: {d.get('status')}\n"
                    f"- Order Status: {d.get('order_status', 'EXCEPTION')}\n\n"
                    f"Our logistics support team has flagged this exception and will reassign or reschedule your pickup.{handoff_note}"
                )
            context.history.append(("agent", reply))
            return OutboundMessage(
                recipient_id=farmer_id,
                text=reply,
                metadata={
                    "state": context.state.value,
                    "order_id": order_id,
                    "problem_reported": True,
                },
            )

        # -------------------------------------------------------------
        # Branch J: Clarification Request (REQUEST_CLARIFICATION)
        # -------------------------------------------------------------
        if intent == IntentType.REQUEST_CLARIFICATION:
            tool_res = await tool_registry.execute_tool("explain_requirement", plan.arguments)
            context.last_tool = "explain_requirement"
            context.last_tool_result = tool_res.data
            trace_data["tool_success"] = tool_res.success
            trace_data["tool_error"] = tool_res.error_message
            d = tool_res.data
            topic = d.get("topic")
            explanation = d.get("explanation")
            if topic == "pricing":
                rates = d.get("rates", {})
                rates_text = "\n".join([f"- {p.title()}: {r}" for p, r in rates.items()])
                reply = f"{explanation}\n\nCurrent procurement rates:\n{rates_text}"
            elif topic == "availability_window":
                reply = f"{explanation}\n\nPlease let us know when your harvest will be ready (for example: 'tomorrow at 10 AM' or 'Friday afternoon')."
            else:
                reply = explanation
            context.history.append(("agent", reply))
            return OutboundMessage(
                recipient_id=farmer_id,
                text=reply,
                metadata={"state": context.state.value, "intent": "REQUEST_CLARIFICATION", "topic": topic},
            )

        # -------------------------------------------------------------
        # Branch K: Pricing Objection (OBJECTION_PRICING)
        # -------------------------------------------------------------
        if intent == IntentType.OBJECTION_PRICING:
            context.negotiation_attempts += 1
            tool_res = await tool_registry.execute_tool("handle_pricing_objection", plan.arguments)
            context.last_tool = "handle_pricing_objection"
            context.last_tool_result = tool_res.data
            trace_data["tool_success"] = tool_res.success
            trace_data["tool_error"] = tool_res.error_message

            data = tool_res.data
            produce = data.get("produce") or context.offer.produce_type or "produce"
            auth_rate = data.get("authoritative_rate")
            currency = data.get("currency", "USD")
            unit = data.get("unit") or context.offer.unit or "kg"
            counter_rate = data.get("farmer_counter_rate")

            total_amount = None
            if auth_rate is not None and context.offer.quantity:
                total_amount = auth_rate * context.offer.quantity

            reply_lines = []
            if counter_rate is not None:
                reply_lines.append(
                    f"We noted your suggested rate of {currency} {counter_rate:.2f} per {unit}. "
                    f"However, ORCA operates on authoritative fixed market rates, "
                    f"and our purchase price cannot be negotiated or altered."
                )
            else:
                reply_lines.append(
                    f"We understand your pricing concern. However, ORCA operates on "
                    f"authoritative fixed market rates, and our purchase price cannot be negotiated or altered."
                )

            reply_lines.append(
                f"\nOur fixed rate of {currency} {auth_rate:.2f} per {unit} for {produce.title()} includes substantial benefits:\n"
                f"- Guaranteed free farm-gate collection by a local runner (zero transportation costs for you)\n"
                f"- Full price transparency with zero intermediary deductions or commission fees\n"
                f"- Instant electronic payment directly upon physical verification"
            )
            if total_amount is not None and context.offer.quantity:
                reply_lines.append(
                    f"\nFor your lot of {context.offer.quantity:g} {unit}, your guaranteed total payout is {currency} {total_amount:.2f}."
                )
            reply_lines.append(
                f"\nWould you like to proceed with your transaction at this authoritative rate (reply 'Confirm'), or would you prefer to decline?"
            )

            reply = "\n".join(reply_lines)
            context.history.append(("agent", reply))
            return OutboundMessage(
                recipient_id=farmer_id,
                text=reply,
                metadata={
                    "state": context.state.value,
                    "intent": "OBJECTION_PRICING",
                    "authoritative_rate": auth_rate,
                    "farmer_counter_rate": counter_rate,
                    "negotiation_attempts": context.negotiation_attempts,
                    "fixed_rate": True,
                    "total_amount": total_amount,
                },
            )

        # -------------------------------------------------------------
        # Branch L: Offer Amendment (AMEND_OFFER)
        # -------------------------------------------------------------
        if intent == IntentType.AMEND_OFFER:
            POST_CONFIRM_STATES = [
                OrderState.ORDER_CONFIRMED,
                OrderState.PAYMENT_PENDING,
                OrderState.PAYMENT_CONFIRMED,
                OrderState.COLLECTION_PENDING,
                OrderState.COLLECTION_ASSIGNED,
                OrderState.PICKED_UP,
                OrderState.COMPLETED,
                OrderState.EXCEPTION,
            ]
            has_active_confirmed_order = bool(
                (context.current_order_id and context.state in POST_CONFIRM_STATES)
                or context.state in POST_CONFIRM_STATES
            )

            if has_active_confirmed_order:
                existing_order = order_service.get_order(context.current_order_id) if context.current_order_id else None
                raw_cand = intent_res.entities.offer

                # Check if amendment attempts to modify quantity or produce
                is_qty_or_produce_change = False
                if raw_cand:
                    if raw_cand.quantity is not None and existing_order and abs(raw_cand.quantity - existing_order.quantity) > 1e-4:
                        is_qty_or_produce_change = True
                    if raw_cand.produce_type and existing_order and raw_cand.produce_type.lower() != existing_order.produce_type.lower():
                        is_qty_or_produce_change = True
                if intent_res.entities.amended_field in ["quantity", "produce"]:
                    is_qty_or_produce_change = True
                if any(w in message_text.lower() for w in ["kg", "tons", "quintal", "quantity", "produce", "potatoes", "tomatoes", "onions"]):
                    if raw_cand and (raw_cand.quantity is not None or raw_cand.produce_type):
                        is_qty_or_produce_change = True

                if is_qty_or_produce_change:
                    trace_data["tool_success"] = False
                    trace_data["tool_error"] = "POST_CONFIRMATION_LOCK: confirmed orders cannot modify produce or quantity"
                    reply = (
                        f"Cannot modify confirmed order: Order {context.current_order_id} has already been confirmed "
                        f"for {existing_order.quantity if existing_order else context.offer.quantity:g} "
                        f"{existing_order.unit if existing_order else context.offer.unit} of "
                        f"{existing_order.produce_type if existing_order else context.offer.produce_type}.\n"
                        f"Produce type and quantity cannot be altered on a confirmed order. "
                        f"If you have additional produce to sell, please start a new offer. "
                        f"If you need to cancel this order, please let us know."
                    )
                    context.history.append(("agent", reply))
                    return OutboundMessage(
                        recipient_id=farmer_id,
                        text=reply,
                        metadata={
                            "state": context.state.value,
                            "order_id": context.current_order_id,
                            "amendment_rejected": True,
                            "reason": "POST_CONFIRMATION_LOCK",
                        },
                    )
                else:
                    # Timing or location amendment on confirmed order -> route to reschedule!
                    new_time = intent_res.entities.new_pickup_time or (raw_cand.availability_window if raw_cand else None)
                    new_loc = intent_res.entities.new_pickup_location or (raw_cand.pickup_location if raw_cand else None)
                    tool_res = await tool_registry.execute_tool(
                        "request_pickup_reschedule",
                        {
                            "order_id": context.current_order_id,
                            "new_time_str": new_time,
                            "new_location": new_loc,
                        },
                    )
                    context.last_tool = "request_pickup_reschedule"
                    context.last_tool_result = tool_res.data
                    trace_data["tool_success"] = tool_res.success
                    trace_data["tool_error"] = tool_res.error_message

                    if tool_res.success:
                        d = tool_res.data
                        reply = (
                            f"Pickup reschedule request processed successfully for Order {context.current_order_id}.\n"
                            f"- New Scheduled Window: {d.get('scheduled_time_str') or 'Updated'}\n"
                            f"- Pickup Location: {d.get('pickup_location')}\n"
                            f"- Collection Status: {d.get('status')}"
                        )
                    else:
                        reply = f"Could not update pickup details: {tool_res.error_message}"

                    context.history.append(("agent", reply))
                    return OutboundMessage(
                        recipient_id=farmer_id,
                        text=reply,
                        metadata={
                            "state": context.state.value,
                            "order_id": context.current_order_id,
                            "rescheduled": tool_res.success,
                        },
                    )

            # Pre-confirmation amendment:
            candidate_offer = plan.arguments.get("offer")
            if not candidate_offer or not isinstance(candidate_offer, ExtractedOffer):
                candidate_dict = context.offer.model_dump()
                if intent_res.entities.offer:
                    for k, v in intent_res.entities.offer.model_dump(exclude_none=True).items():
                        candidate_dict[k] = v
                if intent_res.entities.new_pickup_location:
                    candidate_dict["pickup_location"] = intent_res.entities.new_pickup_location
                if intent_res.entities.new_pickup_time:
                    candidate_dict["availability_window"] = intent_res.entities.new_pickup_time
                candidate_offer = ExtractedOffer(**candidate_dict)

            # Check if there is NO prior offer and candidate is completely empty
            if not candidate_offer.produce_type and candidate_offer.quantity is None:
                reply = (
                    "There is no active offer to amend. Please specify the produce type and quantity "
                    "you would like to sell."
                )
                context.state = OrderState.DETAILS_PENDING
                context.history.append(("agent", reply))
                return OutboundMessage(
                    recipient_id=farmer_id,
                    text=reply,
                    metadata={"state": context.state.value},
                )

            # Evaluate candidate offer authoritatively without mutating context.offer yet!
            tool_res = await tool_registry.execute_tool(
                "evaluate_procurement_offer",
                {"offer": candidate_offer, "region_code": region_code},
            )
            context.last_tool = "evaluate_procurement_offer"
            context.last_tool_result = tool_res.data
            trace_data["tool_success"] = tool_res.success
            trace_data["tool_error"] = tool_res.error_message
            eval_data = tool_res.data

            if eval_data.get("is_acceptable", False):
                # Candidate offer is accepted!
                change_desc = (
                    f"Amended offer to {candidate_offer.quantity:g} {candidate_offer.unit} of "
                    f"{candidate_offer.produce_type} at {candidate_offer.pickup_location}"
                )
                context.amendment_history.append(change_desc)

                # Mutate context.offer ONLY NOW after authoritative validation
                context.offer = candidate_offer
                context.state = OrderState.AWAITING_FARMER_CONFIRMATION
                context.active_clarification = None

                auth_rate = eval_data.get("rate", 0.0)
                currency = eval_data.get("currency", "USD")
                total_amount = eval_data.get("total_amount")
                if total_amount is None:
                    bill = billing_service.calculate_bill(
                        produce_type=candidate_offer.produce_type,
                        quantity=candidate_offer.quantity,
                        unit=candidate_offer.unit,
                        rate_per_unit=auth_rate,
                        region_code=region_code,
                    )
                    total_amount = bill.total_amount
                    currency = bill.currency

                timing = candidate_offer.pickup_datetime or candidate_offer.availability_window or "As scheduled"
                reply = (
                    f"Your offer has been updated with the amended details:\n"
                    f"- Produce: {candidate_offer.quantity:g} {candidate_offer.unit} of {candidate_offer.produce_type}\n"
                    f"- Authoritative Rate: {currency} {auth_rate:.2f} per {candidate_offer.unit}\n"
                    f"- Updated Total Payout: {currency} {total_amount:.2f}\n"
                    f"- Pickup Location: {candidate_offer.pickup_location}\n"
                    f"- Scheduled Window: {timing}\n\n"
                    f"Please reply 'Confirm' if you accept these updated terms so we can finalize your order."
                )
                context.history.append(("agent", reply))
                return OutboundMessage(
                    recipient_id=farmer_id,
                    text=reply,
                    metadata={
                        "state": context.state.value,
                        "produce": candidate_offer.produce_type,
                        "quantity": candidate_offer.quantity,
                        "unit": candidate_offer.unit,
                        "authoritative_rate": auth_rate,
                        "total_amount": total_amount,
                        "currency": currency,
                        "amended": True,
                    },
                )
            else:
                # Candidate amendment is rejected!
                # CRITICAL: context.offer remains untouched!
                reasons = eval_data.get("rejection_reasons", [])
                missing = eval_data.get("missing_fields", [])
                guidance = eval_data.get("guidance_message") or "The requested amendment does not meet procurement policy constraints."
                rejection_text = "; ".join(reasons) if reasons else (f"Missing: {', '.join(missing)}")

                if context.offer.produce_type and context.offer.quantity:
                    reply = (
                        f"Cannot update offer: {guidance}\n"
                        f"- Issue: {rejection_text}\n\n"
                        f"Your previous offer remains in effect:\n"
                        f"- Produce: {context.offer.quantity:g} {context.offer.unit} of {context.offer.produce_type}\n"
                        f"- Location: {context.offer.pickup_location or 'Pending'}\n\n"
                        f"Please reply 'Confirm' to proceed with your previous offer, or provide valid details."
                    )
                else:
                    reply = (
                        f"Cannot update offer: {guidance}\n"
                        f"- Issue: {rejection_text}\n\n"
                        f"Please provide valid offer details (produce type, quantity, pickup location, and availability window)."
                    )
                    context.state = OrderState.DETAILS_PENDING

                context.history.append(("agent", reply))
                return OutboundMessage(
                    recipient_id=farmer_id,
                    text=reply,
                    metadata={
                        "state": context.state.value,
                        "amendment_rejected": True,
                        "rejection_reasons": reasons,
                        "missing_fields": missing,
                    },
                )

        # -------------------------------------------------------------
        # Branch M: Business Knowledge & Operational FAQ
        # (INQUIRE_BUSINESS_INFO, INQUIRE_OPERATIONAL_FAQ)
        # -------------------------------------------------------------
        if intent in [IntentType.INQUIRE_BUSINESS_INFO, IntentType.INQUIRE_OPERATIONAL_FAQ]:
            # Read-only execution: execute registered get_business_knowledge tool
            tool_res = await tool_registry.execute_tool("get_business_knowledge", plan.arguments)
            context.last_tool = "get_business_knowledge"
            context.last_tool_result = tool_res.data
            trace_data["tool_success"] = tool_res.success
            trace_data["tool_error"] = tool_res.error_message

            data = tool_res.data or {}
            is_matched = data.get("matched", False)
            is_answerable = data.get("answerable", False)
            answer = data.get("answer")
            conf = data.get("confidence", 0.0)

            if is_matched and is_answerable and answer:
                reply = answer
                metadata = {
                    "state": context.state.value,
                    "intent": intent.value,
                    "matched": True,
                    "answerable": True,
                    "category": data.get("category"),
                    "source_article_id": data.get("source_article_id"),
                    "confidence": conf,
                }
            elif conf > 0.0 or data.get("clarification_required") or any(
                w in message_text.lower() for w in ["policy", "policies", "operation", "operations", "how does", "what are"]
            ):
                # Weak or ambiguous query: ask a concise clarification question with category overview
                category_name = data.get("category") or plan.arguments.get("topic")
                if category_name:
                    cat_display = str(category_name).replace("_", " ").title()
                    reply = (
                        f"Could you please clarify what specific detail you need regarding our {cat_display}? "
                        f"I can provide grounded information on procedures, timelines, and operational requirements."
                    )
                else:
                    reply = (
                        "Could you please specify which operational topic you would like to know about? "
                        "I can provide details on:\n"
                        "- Procurement Process (how to sell, offer creation, and amendments)\n"
                        "- Pricing Policy (authoritative fixed rates and billing)\n"
                        "- Logistics & Collection (runner dispatch and farm-gate pickup)\n"
                        "- Payment Policy (electronic payout timing and methods)\n"
                        "- Dispute Resolution (handling late runners or failed collections)\n"
                        "- System Capabilities (supported operations and platform boundaries)"
                    )
                metadata = {
                    "state": context.state.value,
                    "intent": intent.value,
                    "matched": False,
                    "answerable": False,
                    "clarification_required": True,
                    "category": category_name,
                    "confidence": conf,
                }
            else:
                # No match safety: clearly say no grounded info exists and never hallucinate
                reply = (
                    "I do not have verified or grounded information to answer that question. "
                    "As ORCA's agricultural procurement assistant, I can answer questions regarding "
                    "our procurement process, fixed pricing, runner collection logistics, payment policies, "
                    "and dispute resolution."
                )
                metadata = {
                    "state": context.state.value,
                    "intent": intent.value,
                    "matched": False,
                    "answerable": False,
                    "confidence": conf,
                    "no_match": True,
                }

            context.history.append(("agent", reply))
            return OutboundMessage(
                recipient_id=farmer_id,
                text=reply,
                metadata=metadata,
            )

        # -------------------------------------------------------------
        # Branch N: Grounded Procurement Recommendations (RECOMMEND_PRODUCE)
        # -------------------------------------------------------------
        if intent == IntentType.RECOMMEND_PRODUCE:
            tool_res = await tool_registry.execute_tool("get_procurement_recommendations", plan.arguments)
            context.last_tool = "get_procurement_recommendations"
            context.last_tool_result = tool_res.data
            trace_data["tool_success"] = tool_res.success
            trace_data["tool_error"] = tool_res.error_message

            data = tool_res.data or {}
            recs = data.get("recommendations", [])
            matched = data.get("matched", False)
            message = data.get("message")
            clarif = data.get("clarification_required", False)

            if matched and recs:
                if len(recs) == 1:
                    r = recs[0]
                    reply = (
                        f"Yes! We actively procure {r['produce']} through ORCA.\n"
                        f"- Rationale: {r['reason']}\n\n"
                        f"To submit an offer, tell me the quantity you have available, your farm pickup location, "
                        f"and when your harvest will be ready for collection."
                    )
                else:
                    items_text = "\n".join([f"- {r['produce']}: {r['reason']}" for r in recs])
                    reply = (
                        f"Here are the produce commodities currently procured through ORCA:\n"
                        f"{items_text}\n\n"
                        f"To start an offer for any of these crops, simply provide the produce type, quantity, "
                        f"farm pickup location, and your availability window."
                    )
                metadata = {
                    "state": context.state.value,
                    "intent": intent.value,
                    "matched": True,
                    "recommendation_count": len(recs),
                    "produce_recommended": [r["produce"] for r in recs],
                }
            elif not matched and message and not clarif:
                reply = message
                metadata = {
                    "state": context.state.value,
                    "intent": intent.value,
                    "matched": False,
                    "unsupported_produce": plan.arguments.get("produce"),
                }
            else:
                reply = (
                    "ORCA currently has insufficient verified information to provide a recommendation for that inquiry. "
                    "You can ask which crops we support, or specify the produce you would like to sell."
                )
                metadata = {
                    "state": context.state.value,
                    "intent": intent.value,
                    "matched": False,
                    "clarification_required": True,
                }

            context.history.append(("agent", reply))
            return OutboundMessage(
                recipient_id=farmer_id,
                text=reply,
                metadata=metadata,
            )

        # -------------------------------------------------------------
        # Branch P: Factual Grounded Personalization (PERSONALIZE_RECOMMENDATION)
        # -------------------------------------------------------------
        if intent == IntentType.PERSONALIZE_RECOMMENDATION:
            tool_res = await tool_registry.execute_tool("get_personalized_recommendations", plan.arguments)
            context.last_tool = "get_personalized_recommendations"
            context.last_tool_result = tool_res.data
            trace_data["tool_success"] = tool_res.success
            trace_data["tool_error"] = tool_res.error_message

            data = tool_res.data or {}
            recs = data.get("recommendations", [])
            matched = data.get("matched", False)
            message = data.get("message")
            grounding_source = data.get("grounding_source", "profile_history")

            # Update trace arguments with safe diagnostics
            trace_data["tool_arguments"] = {
                "farmer_id": farmer_id,
                "grounding_source": grounding_source,
                "matched": matched,
            }

            if matched and recs:
                reply = message or (
                    "Based on your observed interaction history with ORCA:\n"
                    + "\n".join([f"- {r['produce']}: {r['factual_reason']}" for r in recs])
                    + "\n\nTo submit an offer for any of these crops, simply share your available quantity, "
                    + "farm location, and pickup availability."
                )
                metadata = {
                    "state": context.state.value,
                    "intent": intent.value,
                    "matched": True,
                    "grounding_source": grounding_source,
                    "recommendation_count": len(recs),
                    "produce_recommended": [r["produce"] for r in recs],
                }
            elif message:
                reply = message
                metadata = {
                    "state": context.state.value,
                    "intent": intent.value,
                    "matched": False,
                    "grounding_source": grounding_source,
                    "clarification_required": data.get("clarification_required", False),
                }
            else:
                supported_crops = pricing_service.get_supported_produce(region_code)
                reply = (
                    f"I don't have enough history about your previous sales or offers to personalize a recommendation yet. "
                    f"I can show you ORCA's currently supported produce: {', '.join(supported_crops)}. "
                    f"What produce do you have available?"
                )
                metadata = {
                    "state": context.state.value,
                    "intent": intent.value,
                    "matched": False,
                    "grounding_source": "insufficient_history",
                    "clarification_required": True,
                }

            context.history.append(("agent", reply))
            return OutboundMessage(
                recipient_id=farmer_id,
                text=reply,
                metadata=metadata,
            )

        # -------------------------------------------------------------
        # Branch Q: Human Assistance Request (REQUEST_HUMAN_ASSISTANCE)
        # -------------------------------------------------------------
        if intent == IntentType.REQUEST_HUMAN_ASSISTANCE:
            tool_res = await tool_registry.execute_tool("create_human_handoff", plan.arguments)
            context.last_tool = "create_human_handoff"
            context.last_tool_result = tool_res.data
            trace_data["tool_success"] = tool_res.success
            trace_data["tool_error"] = tool_res.error_message

            case_id = (tool_res.data or {}).get("case_id", "HC-SUPPORT")
            handoff_reason = plan.arguments.get("reason", "FARMER_REQUEST")

            trace_data["tool_arguments"] = {
                "farmer_id": farmer_id,
                "handoff_case_id": case_id,
                "handoff_reason": handoff_reason,
            }

            if handoff_reason == "PAYMENT_DISPUTE":
                if context.current_order_id:
                    reply = (
                        f"Understood. I have created a payment review case {case_id} for our finance team "
                        f"regarding Order {context.current_order_id}. Your order status ({context.state.value}) remains unchanged. "
                        f"A finance specialist will review the transaction and follow up with you."
                    )
                else:
                    reply = (
                        f"Understood. I have created support case {case_id} for our finance team "
                        f"to review your payment discrepancy inquiry. A specialist will follow up with you."
                    )
            elif context.current_order_id:
                reply = (
                    f"Understood. I have created support case {case_id} for a human team member. "
                    f"Your order {context.current_order_id} remains unchanged with status {context.state.value}. "
                    f"An agent will follow up with you."
                )
            else:
                reply = (
                    f"Understood. I have created support case {case_id} for a human team member. "
                    f"An agent will review your request and follow up with you."
                )

            context.history.append(("agent", reply))
            return OutboundMessage(
                recipient_id=farmer_id,
                text=reply,
                metadata={
                    "state": context.state.value,
                    "intent": intent.value,
                    "handoff_case_id": case_id,
                    "handoff_reason": handoff_reason,
                    "order_id": context.current_order_id,
                },
            )

        # -------------------------------------------------------------
        # Branch O: Produce Offer Evaluation (OFFER_PRODUCE)
        # -------------------------------------------------------------
        if intent == IntentType.OFFER_PRODUCE:
            # DEFENSIVE ORCHESTRATOR SAFETY GUARD:
            # If the transaction is already beyond the offer-evaluation stage,
            # never route back through offer evaluation or recalculate billing
            # unless the incoming message contains genuine new offer evidence (produce + quantity).
            STATES_BEYOND_OFFER = [
                OrderState.ORDER_CONFIRMED,
                OrderState.PAYMENT_PENDING,
                OrderState.PAYMENT_CONFIRMED,
                OrderState.COLLECTION_PENDING,
                OrderState.COLLECTION_ASSIGNED,
                OrderState.PICKED_UP,
                OrderState.COMPLETED,
                OrderState.EXCEPTION,
            ]
            has_active_order = bool(context.current_order_id or context.state in STATES_BEYOND_OFFER)
            if has_active_order:
                # Check for genuine new offer evidence in message_text alone
                raw_new_offer = await self.extractor.extract(message_text, current_offer=None)
                has_genuine_new_offer = bool(raw_new_offer.produce_type and raw_new_offer.quantity and raw_new_offer.quantity > 0)
                if not has_genuine_new_offer:
                    # Guard activated: protect active order from any backward state regression
                    existing_order = order_service.get_order(context.current_order_id) if context.current_order_id else None
                    if existing_order:
                        context.state = existing_order.status

                    if context.state == OrderState.PAYMENT_PENDING:
                        reply = (
                            f"Your order {context.current_order_id} is currently awaiting payment.\n"
                            f"- Produce: {existing_order.quantity:g} {existing_order.unit} of {existing_order.produce_type}\n"
                            f"- Total Amount: {existing_order.currency} {existing_order.total_amount:.2f}\n"
                            f"- Order Status: PAYMENT_PENDING\n\n"
                            f"Please reply 'Pay' to complete payment, or ask 'What's happening with my order?' to check status."
                        )
                    elif context.state in [
                        OrderState.PAYMENT_CONFIRMED,
                        OrderState.COLLECTION_PENDING,
                        OrderState.COLLECTION_ASSIGNED,
                        OrderState.PICKED_UP,
                        OrderState.COMPLETED,
                    ]:
                        reply = (
                            f"Your order {context.current_order_id} is active with status {context.state.value}.\n"
                            f"You can ask 'What's happening with my order?' or 'When is the pickup?' for status and logistics updates."
                        )
                    else:
                        reply = (
                            f"You have an active order ({context.current_order_id}) in status {context.state.value}."
                        )

                    context.history.append(("agent", reply))
                    return OutboundMessage(
                        recipient_id=farmer_id,
                        text=reply,
                        metadata={
                            "state": context.state.value,
                            "order_id": context.current_order_id,
                            "guard_triggered": True,
                        },
                    )

            # Ambiguous conflicting location detection in the same turn
            loc_candidates = []
            for city in ["springfield", "delhi", "greenfield", "shelbyville", "farm a", "farm b", "farm 1", "farm 2"]:
                if re.search(rf"\b{city}\b", message_text.lower()):
                    loc_candidates.append(city.title())

            has_ambiguous_location_conflict = len(set(loc_candidates)) > 1 and not any(
                w in message_text.lower() for w in ["actually", "instead", "not", "moved from", "changed from"]
            )
            if has_ambiguous_location_conflict:
                context.state = OrderState.DETAILS_PENDING
                context.active_clarification = "pickup_location"
                cities_str = " and ".join(sorted(set(loc_candidates)))
                reply = (
                    f"You mentioned multiple locations ({cities_str}). "
                    f"Could you please clarify which address is the correct pickup location for collection?"
                )
                context.history.append(("agent", reply))
                return OutboundMessage(
                    recipient_id=farmer_id,
                    text=reply,
                    metadata={
                        "state": context.state.value,
                        "missing_fields": ["pickup_location"],
                        "conflict_detected": True,
                    },
                )

            extracted = intent_res.entities.offer
            if not extracted or (not extracted.produce_type and not extracted.quantity and not extracted.pickup_location):
                extracted = await self.extractor.extract(message_text, context.offer)

            context.offer = extracted

            # Record observed offer facts into farmer profile
            if context.offer and context.offer.produce_type:
                try:
                    from orca.services.farmer_profile import farmer_profile_service
                    farmer_profile_service.record_observed_offer(
                        farmer_id=farmer_id,
                        produce=context.offer.produce_type,
                        quantity=context.offer.quantity,
                        unit=context.offer.unit,
                        location=context.offer.pickup_location,
                    )
                except Exception:
                    pass

            # Check and record partial/future availability
            if intent_res.entities.future_quantity is not None:
                u = context.offer.unit or (intent_res.entities.offer.unit if intent_res.entities.offer else "kg")
                tim = intent_res.entities.future_timing or "later"
                context.future_availability_notes = f"{intent_res.entities.future_quantity:g} {u or 'kg'} {tim}"
            elif context.offer.future_availability:
                context.future_availability_notes = context.offer.future_availability

            # Check Produce Validity
            supported_produce = pricing_service.get_supported_produce(region_code)
            if context.offer.produce_type and context.offer.produce_type not in supported_produce:
                reply = (
                    f"Thank you for contacting us. However, we currently only procure: {', '.join(supported_produce)}. "
                    f"We do not currently purchase '{context.offer.produce_type}' in this region."
                )
                context.state = OrderState.DETAILS_PENDING
                context.history.append(("agent", reply))
                return OutboundMessage(
                    recipient_id=farmer_id,
                    text=reply,
                    metadata={"state": context.state.value},
                )

            # Detect Missing Required Information
            missing_fields: List[str] = []
            if not context.offer.produce_type:
                missing_fields.append("produce_type")
            if context.offer.quantity is None or context.offer.quantity <= 0:
                missing_fields.append("quantity")
            if not context.offer.unit:
                missing_fields.append("unit")
            if not context.offer.pickup_location:
                missing_fields.append("pickup_location")
            if not context.offer.availability_window and not context.offer.pickup_datetime:
                missing_fields.append("availability")

            # Generate Targeted Clarification Question if info is incomplete
            if missing_fields:
                context.state = OrderState.DETAILS_PENDING
                context.active_clarification = missing_fields[0]
                clarification_reply = self._generate_clarification_question(context.offer, missing_fields, region_code)
                context.history.append(("agent", clarification_reply))
                return OutboundMessage(
                    recipient_id=farmer_id,
                    text=clarification_reply,
                    metadata={"state": context.state.value, "missing_fields": missing_fields},
                )

            # All Required Fields Present -> Authoritative Rate Lookup
            authoritative_rate = pricing_service.get_applicable_rate(
                produce_type=context.offer.produce_type,
                region_code=region_code,
            )
            if not authoritative_rate:
                supported = pricing_service.get_supported_produce(region_code)
                reply = (
                    f"Thank you for contacting us. However, we do not currently have an active procurement rate for '{context.offer.produce_type}' in this region. "
                    f"We currently procure: {', '.join(supported)}."
                )
                context.state = OrderState.DETAILS_PENDING
                context.history.append(("agent", reply))
                return OutboundMessage(
                    recipient_id=farmer_id,
                    text=reply,
                    metadata={"state": context.state.value, "produce": context.offer.produce_type},
                )

            # Deterministic Billing Calculation
            bill = billing_service.calculate_bill(
                produce_type=context.offer.produce_type,
                quantity=context.offer.quantity,
                unit=context.offer.unit,
                rate_per_unit=authoritative_rate.rate_per_unit,
                region_code=region_code,
            )
            context.last_tool = "calculate_order_total"
            context.last_tool_result = bill.model_dump()
            trace_data["tool_success"] = True
            context.state = OrderState.AWAITING_FARMER_CONFIRMATION
            context.active_clarification = None

            summary_reply = self._generate_transaction_summary(
                context.offer,
                authoritative_rate.rate_per_unit,
                bill.currency,
                bill.total_amount,
                future_availability_notes=context.future_availability_notes,
            )
            context.history.append(("agent", summary_reply))
            return OutboundMessage(
                recipient_id=farmer_id,
                text=summary_reply,
                metadata={
                    "state": context.state.value,
                    "produce": context.offer.produce_type,
                    "quantity": context.offer.quantity,
                    "unit": context.offer.unit,
                    "authoritative_rate": authoritative_rate.rate_per_unit,
                    "total_amount": bill.total_amount,
                    "currency": bill.currency,
                    "pickup_location": context.offer.pickup_location,
                    "pickup_availability": context.offer.pickup_datetime or context.offer.availability_window,
                    "pickup_datetime": context.offer.pickup_datetime,
                },
            )

        # -------------------------------------------------------------
        # Branch L: UNKNOWN Intent (First-class path, zero tools executed)
        # -------------------------------------------------------------
        if context.state == OrderState.PAYMENT_PENDING and context.current_order_id:
            order = order_service.get_order(context.current_order_id)
            reply = (
                f"Order {context.current_order_id} is awaiting payment authorization of "
                f"{order.currency if order else 'USD'} {order.total_amount if order else 0.0:.2f}.\n"
                f"Please reply 'Pay' or 'Make payment' to proceed with payment."
            )
            context.history.append(("agent", reply))
            return OutboundMessage(
                recipient_id=farmer_id,
                text=reply,
                metadata={
                    "state": context.state.value,
                    "order_id": context.current_order_id,
                    "status": OrderState.PAYMENT_PENDING.value,
                    "payment_status": "PENDING",
                },
            )

        supported = pricing_service.get_supported_produce(region_code)
        reply = (
            f"Hello! I am ORCA, your agricultural procurement assistant. "
            f"I can help you sell your produce at authoritative fixed market rates. "
            f"We currently procure: {', '.join(supported)}. "
            f"You can also ask about an order status, payment status, pickup logistics, or get help. "
            f"What produce do you have available for sale today?"
        )
        context.history.append(("agent", reply))
        return OutboundMessage(
            recipient_id=farmer_id,
            text=reply,
            metadata={"state": context.state.value, "intent": "UNKNOWN"},
        )

    def _generate_clarification_question(
        self, offer: ExtractedOffer, missing_fields: List[str], region_code: str
    ) -> str:
        """Ask a targeted clarification question for ONLY the missing attributes."""
        supported = pricing_service.get_supported_produce(region_code)

        if "produce_type" in missing_fields:
            return f"What produce do you have available for sale? (Supported: {', '.join(supported)})"

        produce = offer.produce_type

        # Quantity and unit missing
        if "quantity" in missing_fields and "unit" in missing_fields:
            return f"How much {produce} do you have available? Please specify both the quantity and measurement unit (e.g., 50 kg or 2 tons)."

        if "quantity" in missing_fields:
            return f"Could you please specify the quantity of {produce} you have available?"

        if "unit" in missing_fields:
            return f"Could you please specify the measurement unit for the {offer.quantity} {produce} (e.g., kg, tons, or bags)?"

        if "pickup_location" in missing_fields and "availability" in missing_fields:
            return f"Where should we collect the {offer.quantity} {offer.unit} of {produce}, and when will it be ready for pickup?"

        if "pickup_location" in missing_fields:
            return f"Where is the {offer.quantity} {offer.unit} of {produce} located for collection? Please provide the pickup location or farm address."

        if "availability" in missing_fields:
            return f"When will the {offer.quantity} {offer.unit} of {produce} be ready for pickup? Please specify your availability window or date."

        return "Could you please provide the missing details for your offer?"

    def _generate_transaction_summary(
        self,
        offer: ExtractedOffer,
        authoritative_rate: float,
        currency: str,
        total_amount: float,
        future_availability_notes: Optional[str] = None,
    ) -> str:
        """Format transaction confirmation summary according to SRS guidelines."""
        price_note = ""
        if offer.offered_rate is not None and abs(offer.offered_rate - authoritative_rate) > 1e-4:
            price_note = (
                f"\nNote: You suggested a price of {currency} {offer.offered_rate:.2f} per {offer.unit}. "
                f"Our authoritative fixed procurement rate is {currency} {authoritative_rate:.2f} per {offer.unit}.\n"
            )

        future_note = ""
        if future_availability_notes:
            future_note = (
                f"\nNote: We have also recorded your future availability of {future_availability_notes} "
                f"for subsequent collection when ready.\n"
            )

        timing = offer.pickup_datetime or offer.availability_window or "As scheduled"

        return (
            f"Thank you! Here is the transaction summary for your produce offer:\n"
            f"- Produce: {offer.quantity:g} {offer.unit} of {offer.produce_type}\n"
            f"- Authoritative Rate: {currency} {authoritative_rate:.2f} per {offer.unit}\n"
            f"- Total Payout: {currency} {total_amount:.2f}\n"
            f"- Pickup Location: {offer.pickup_location}\n"
            f"- Pickup Availability: {timing}\n"
            f"{price_note}"
            f"{future_note}\n"
            f"Please reply with 'Confirm' if you accept this transaction so we can arrange pickup and payment."
        )


orchestrator = AgentOrchestrator()
