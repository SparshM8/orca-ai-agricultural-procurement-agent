"""Agent tool definitions and bindings according to SRS Section 8.

All tools expose typed schemas to the LLM and delegate to backend deterministic services.
The LLM does not execute operations directly.
"""

from typing import Dict, Any, List, Optional
import inspect

from orca.domain.intents import ToolExecutionResult
from orca.domain.schemas import ExtractedOffer
from orca.agent.extractors.base import parse_pickup_timing
from orca.services.pricing import pricing_service
from orca.services.billing import billing_service
from orca.services.order import order_service
from orca.services.payment import payment_service
from orca.services.collection import collection_service
from orca.services.knowledge import knowledge_service
from orca.domain.knowledge import KnowledgeCategory, KnowledgeResult
from orca.core.regional import get_regional_profile


class AgentToolRegistry:
    """Registry and dispatcher for LLM agent tools."""

    @staticmethod
    def get_supported_produce(region_code: str = "GLOBAL_DEFAULT") -> List[str]:
        """Tool 1: get_supported_produce() - retrieve supported produce types."""
        return pricing_service.get_supported_produce(region_code)

    @staticmethod
    def get_applicable_rate(
        produce: str,
        location: str = "GLOBAL_DEFAULT",
        date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Tool 2: get_applicable_rate(produce, location, date) - authoritative rate lookup."""
        rate = pricing_service.get_applicable_rate(produce_type=produce, region_code=location)
        if not rate:
            return {
                "found": False,
                "error": f"No authoritative rate configured for produce '{produce}' in region '{location}'",
            }
        return {
            "found": True,
            "produce": rate.produce_type,
            "rate": rate.rate_per_unit,
            "unit": rate.unit,
            "currency": rate.currency,
            "source": rate.source,
        }

    @staticmethod
    def validate_quantity(
        quantity: float,
        unit: str,
        region_code: str = "GLOBAL_DEFAULT",
    ) -> Dict[str, Any]:
        """Tool 3: validate_quantity(quantity, unit) - check validity of quantity and unit."""
        profile = get_regional_profile(region_code)
        is_unit_valid = unit.lower() in [u.lower() for u in profile.standard_units]
        is_qty_valid = quantity > 0
        return {
            "valid": is_unit_valid and is_qty_valid,
            "quantity": quantity,
            "unit": unit,
            "supported_units": profile.standard_units,
        }

    @staticmethod
    def calculate_order_total(
        quantity: float,
        rate: float,
        region_code: str = "GLOBAL_DEFAULT",
    ) -> Dict[str, Any]:
        """Tool 4: calculate_order_total(quantity, rate) - deterministic total calculation."""
        profile = get_regional_profile(region_code)
        subtotal = round(quantity * rate, 2)
        tax = round(subtotal * (profile.tax_rate_percentage / 100.0), 2)
        total = round(subtotal + tax, 2)
        return {
            "quantity": quantity,
            "rate": rate,
            "currency": profile.currency_code,
            "subtotal": subtotal,
            "tax": tax,
            "total": total,
        }

    @staticmethod
    def create_order(
        farmer_id: str,
        produce: str,
        quantity: float,
        unit: str,
        pickup_location: str,
        validated_rate: Optional[float] = None,
        conversation_id: Optional[str] = None,
        region_code: str = "GLOBAL_DEFAULT",
    ) -> Dict[str, Any]:
        """Tool 5: create_order(validated_order) - create an auditable order.
        
        Cannot bypass backend validation or fabricate prices.
        """
        try:
            order = order_service.create_order(
                farmer_id=farmer_id,
                produce_type=produce,
                quantity=quantity,
                unit=unit,
                pickup_location=pickup_location,
                validated_rate=validated_rate,
                region_code=region_code,
                conversation_id=conversation_id,
            )
            return {
                "success": True,
                "order_id": order.id,
                "status": order.status.value,
                "total_amount": order.total_amount,
                "currency": order.currency,
            }
        except ValueError as err:
            return {
                "success": False,
                "error": str(err),
            }

    @staticmethod
    def create_bill(order_id: str) -> Dict[str, Any]:
        """Tool 6: create_bill(order_id) - generate transaction summary/bill."""
        order = order_service.get_order(order_id)
        if not order:
            return {"error": f"Order not found: {order_id}"}
        bill = billing_service.calculate_bill(
            produce_type=order.produce_type,
            quantity=order.quantity,
            unit=order.unit,
            rate_per_unit=order.validated_rate,
        )
        return bill.model_dump()

    @staticmethod
    async def create_payment_request(order_id: str) -> Dict[str, Any]:
        """Tool 7: create_payment_request(order_id) - initiate payment workflow."""
        try:
            payment, is_success, msg = await payment_service.initiate_order_payment(order_id)
            return {
                "success": is_success,
                "payment_id": payment.id,
                "order_id": payment.order_id,
                "amount": payment.amount,
                "currency": payment.currency,
                "status": payment.status,
                "provider_reference": payment.provider_reference,
                "message": msg,
            }
        except ValueError as err:
            return {"success": False, "error": str(err)}

    @staticmethod
    def get_payment_status(order_id: str) -> Dict[str, Any]:
        """Tool 8: get_payment_status(order_id) - query payment status."""
        payment = payment_service.get_payment_for_order(order_id)
        if not payment:
            return {"status": "NOT_FOUND"}
        return {
            "payment_id": payment.id,
            "order_id": payment.order_id,
            "status": payment.status,
            "provider_reference": payment.provider_reference,
        }

    @staticmethod
    def create_collection_task(order_id: str) -> Dict[str, Any]:
        """Tool 9: create_collection_task(order_id) - schedule logistics pickup."""
        order = order_service.get_order(order_id)
        if not order:
            return {"error": f"Order not found: {order_id}"}
        task = collection_service.create_collection_task(
            order_id=order.id,
            pickup_location=order.pickup_location,
            scheduled_datetime=order.pickup_datetime,
        )
        return {
            "task_id": task.id,
            "order_id": task.order_id,
            "pickup_location": task.pickup_location,
            "status": task.status,
        }

    @staticmethod
    def get_collection_status(order_id: str) -> Dict[str, Any]:
        """Tool 10: get_collection_status(order_id) - query collection task status."""
        for task in collection_service._tasks.values():
            if task.order_id == order_id:
                return {
                    "task_id": task.id,
                    "status": task.status,
                    "runner_id": task.runner_id,
                    "completed_at": str(task.completed_at) if task.completed_at else None,
                }
        return {"status": "NOT_FOUND"}

    @staticmethod
    def send_confirmation(message: str) -> Dict[str, Any]:
        """Tool 11: send_confirmation(message) - acknowledge action to farmer."""
        return {"sent": True, "message_preview": message}

    @staticmethod
    def get_order_status(order_id: str) -> Dict[str, Any]:
        """Tool 12: get_order_status(order_id) - retrieve order and lifecycle state."""
        order = order_service.get_order(order_id)
        if not order:
            return {"found": False, "error": f"Order not found: {order_id}"}
        return {
            "found": True,
            "order_id": order.id,
            "produce": order.produce_type,
            "quantity": order.quantity,
            "unit": order.unit,
            "rate": order.validated_rate,
            "total_amount": order.total_amount,
            "currency": order.currency,
            "status": order.status.value,
            "pickup_location": order.pickup_location,
            "pickup_time_str": order.pickup_time_str,
            "created_at": str(order.created_at) if order.created_at else None,
        }

    @staticmethod
    async def request_pickup_reschedule(
        order_id: str,
        new_time_str: Optional[str] = None,
        new_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Tool 13: request_pickup_reschedule(order_id, new_time_str, new_location) - update logistics pickup."""
        order = order_service.get_order(order_id)
        if not order:
            return {"success": False, "error": f"Order not found: {order_id}"}
        task = collection_service.get_task_by_order(order_id)
        target_dt, norm_time = parse_pickup_timing(new_time_str) if new_time_str else (None, None)
        resolved_time_str = norm_time or new_time_str

        if task:
            if task.status == "FAILED":
                await collection_service.reschedule_task(
                    task_id=task.id,
                    new_datetime=target_dt,
                    new_time_str=resolved_time_str,
                )
            else:
                if resolved_time_str:
                    task.scheduled_time_str = resolved_time_str
                    task.scheduled_datetime = target_dt
                    order.pickup_time_str = resolved_time_str
                    order.pickup_datetime = target_dt
                if new_location:
                    task.pickup_location = new_location
                    order.pickup_location = new_location
        else:
            if resolved_time_str:
                order.pickup_time_str = resolved_time_str
                order.pickup_datetime = target_dt
            if new_location:
                order.pickup_location = new_location

        return {
            "success": True,
            "order_id": order_id,
            "task_id": task.id if task else None,
            "status": task.status if task else "SCHEDULED",
            "scheduled_time_str": task.scheduled_time_str if task else order.pickup_time_str,
            "pickup_location": task.pickup_location if task else order.pickup_location,
        }

    @staticmethod
    async def report_collection_problem(order_id: str, reason: str) -> Dict[str, Any]:
        """Tool 14: report_collection_problem(order_id, reason) - log logistics issue."""
        order = order_service.get_order(order_id)
        if not order:
            return {"success": False, "error": f"Order not found: {order_id}"}
        task = collection_service.get_task_by_order(order_id)
        if not task:
            return {"success": False, "error": f"No collection task found for order {order_id}"}

        if task.status == "ASSIGNED" and task.runner_id:
            await collection_service.fail_pickup(
                task_id=task.id,
                runner_id=task.runner_id,
                reason=reason,
            )
            return {
                "success": True,
                "order_id": order_id,
                "task_id": task.id,
                "status": "FAILED",
                "order_status": "EXCEPTION",
                "reason": reason,
            }
        else:
            task.failure_reason = reason
            return {
                "success": True,
                "order_id": order_id,
                "task_id": task.id,
                "status": task.status,
                "reason": reason,
            }

    @staticmethod
    def explain_requirement(subject: str, region_code: str = "GLOBAL_DEFAULT") -> Dict[str, Any]:
        """Tool 15: explain_requirement(subject, region_code) - clarify terms, pricing, or policies."""
        subj_lower = subject.lower()
        if any(k in subj_lower for k in ["rate", "price", "pricing"]):
            supported = pricing_service.get_supported_produce(region_code)
            rates = {}
            for p in supported:
                r = pricing_service.get_applicable_rate(p, region_code)
                if r:
                    rates[p] = f"{r.currency} {r.rate_per_unit:.2f} per {r.unit}"
            return {
                "topic": "pricing",
                "explanation": "ORCA sets authoritative procurement purchase rates based on fair market benchmarks in your region. Rates cannot be altered during conversation.",
                "rates": rates,
            }
        elif any(k in subj_lower for k in ["window", "timing", "availability", "time", "pickup"]):
            return {
                "topic": "availability_window",
                "explanation": "The availability window is the time or day your produce will be ready at your farm for collection by our runner (e.g., 'tomorrow at 10 AM' or 'Friday morning').",
            }
        else:
            supported = pricing_service.get_supported_produce(region_code)
            return {
                "topic": "general",
                "explanation": f"ORCA purchases produce directly from farmers with transparent pricing and guaranteed runner pickup. We currently purchase: {', '.join(p.title() for p in supported)}.",
                "supported_produce": supported,
            }

    @staticmethod
    def handle_pricing_objection(
        produce: Optional[str] = None,
        counter_rate: Optional[float] = None,
        region_code: str = "GLOBAL_DEFAULT",
    ) -> Dict[str, Any]:
        """Tool 16: handle_pricing_objection(produce, counter_rate, region_code)
        
        Retrieves authoritative rate and grounded fixed-rate policy explanation.
        Preserves counter-rate only as an informational discrepancy field.
        Never alters authoritative rate, creates orders, or mutates state.
        """
        from orca.services.procurement import procurement_service

        if not produce:
            return {
                "success": False,
                "error": "No produce specified for pricing objection.",
            }

        rationale = procurement_service.get_pricing_rationale(
            produce_type=produce, region_code=region_code
        )
        auth_rate = rationale.get("rate_per_unit")

        discrepancy = False
        if counter_rate is not None and auth_rate is not None:
            discrepancy = abs(counter_rate - auth_rate) > 1e-4

        return {
            "success": True,
            "produce": produce,
            "region_code": region_code,
            "supported": rationale.get("supported", False),
            "authoritative_rate": auth_rate,
            "currency": rationale.get("currency", "USD"),
            "unit": rationale.get("unit"),
            "farmer_counter_rate": counter_rate,
            "discrepancy": discrepancy,
            "policy_message": rationale.get("policy_message"),
            "benefits": rationale.get("benefits", []),
            "fixed_rate_model": True,
        }

    @staticmethod
    def evaluate_procurement_offer(
        offer: Optional[ExtractedOffer] = None,
        region_code: str = "GLOBAL_DEFAULT",
        produce: Optional[str] = None,
        quantity: Optional[float] = None,
        unit: Optional[str] = None,
        pickup_location: Optional[str] = None,
        availability_window: Optional[str] = None,
        pickup_datetime: Optional[str] = None,
        offered_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Tool 17: evaluate_procurement_offer(...)
        
        Authoritative evaluation of an offer or candidate amendment against backend policy.
        Never mutates OrderState, orders, payments, or database.
        """
        from orca.services.procurement import procurement_service

        if offer is not None:
            candidate_dict = offer.model_dump()
        else:
            candidate_dict = {}

        if produce is not None:
            candidate_dict["produce_type"] = produce
        if quantity is not None:
            candidate_dict["quantity"] = quantity
        if unit is not None:
            candidate_dict["unit"] = unit
        if pickup_location is not None:
            candidate_dict["pickup_location"] = pickup_location
        if availability_window is not None:
            candidate_dict["availability_window"] = availability_window
        if pickup_datetime is not None:
            candidate_dict["pickup_datetime"] = pickup_datetime
        if offered_rate is not None:
            candidate_dict["offered_rate"] = offered_rate

        candidate_offer = ExtractedOffer(**candidate_dict)

        evaluation = procurement_service.evaluate_offer(
            offer=candidate_offer, region_code=region_code
        )

        return {
            "success": evaluation.is_acceptable,
            "is_acceptable": evaluation.is_acceptable,
            "produce_supported": evaluation.produce_supported,
            "produce": candidate_offer.produce_type,
            "quantity": candidate_offer.quantity,
            "unit": candidate_offer.unit,
            "pickup_location": candidate_offer.pickup_location,
            "availability_window": candidate_offer.availability_window or candidate_offer.pickup_datetime,
            "rate": evaluation.authoritative_rate,
            "currency": evaluation.rate.currency if evaluation.rate else "USD",
            "bill_summary": evaluation.bill_summary.model_dump() if evaluation.bill_summary else None,
            "total_amount": evaluation.bill_summary.total_amount if evaluation.bill_summary else None,
            "missing_fields": evaluation.missing_fields,
            "rejection_reasons": evaluation.rejection_reasons,
            "pricing_discrepancy": evaluation.pricing_discrepancy,
            "farmer_offered_rate": evaluation.farmer_offered_rate,
            "guidance_message": evaluation.guidance_message,
            "policy_id": evaluation.policy_id,
            "error": evaluation.guidance_message if not evaluation.is_acceptable else None,
        }

    @staticmethod
    def get_business_knowledge(
        query: str,
        topic: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Tool 18: get_business_knowledge(query, topic) - grounded Q&A and operational FAQ lookup.
        
        Delegates strictly to authoritative BusinessKnowledgeService.
        Never mutates orders, payments, collection tasks, or OrderState.
        Never fabricates prices or unsupported operational promises.
        """
        category_enum: Optional[KnowledgeCategory] = None
        if topic:
            try:
                category_enum = KnowledgeCategory(topic.upper())
            except (ValueError, KeyError):
                category_enum = None

        res = knowledge_service.lookup(query=query, category=category_enum)
        res_dict = res.model_dump()
        res_dict["success"] = res.answerable
        return res_dict

    @staticmethod
    def get_procurement_recommendations(
        query: str,
        produce: Optional[str] = None,
        region_code: str = "GLOBAL_DEFAULT",
    ) -> Dict[str, Any]:
        """Tool 19: get_procurement_recommendations(query, produce, region_code) - grounded crop recommendations.
        
        Delegates strictly to authoritative RecommendationService.
        Never mutates orders, payments, collection tasks, or OrderState.
        Never fabricates profit estimates, speculative demand, or live prices.
        """
        from orca.services.recommendation import recommendation_service

        res = recommendation_service.get_recommendations(
            query=query, produce=produce, region_code=region_code
        )
        res_dict = res.model_dump()
        res_dict["success"] = res.matched
        return res_dict

    @staticmethod
    def get_personalized_recommendations(
        farmer_id: str,
        query: Optional[str] = None,
        region_code: str = "GLOBAL_DEFAULT",
    ) -> Dict[str, Any]:
        """Tool 20: get_personalized_recommendations(farmer_id, query, region_code) - factual grounded personalized recommendation.
        
        Delegates strictly to authoritative PersonalizationService.
        Never mutates orders, payments, collection tasks, or OrderState.
        Never uses farmer tiers, wealth profiling, or predictive ML.
        """
        from orca.services.personalization import personalization_service

        res = personalization_service.get_personalized_recommendations(
            farmer_id=farmer_id, query=query, region_code=region_code
        )
        res_dict = res.model_dump()
        res_dict["success"] = res.matched
        return res_dict

    @staticmethod
    def create_human_handoff(
        farmer_id: str,
        order_id: Optional[str] = None,
        reason: str = "FARMER_REQUEST",
        summary: str = "",
        source_intent: str = "REQUEST_HUMAN_ASSISTANCE",
    ) -> Dict[str, Any]:
        """Tool 21: create_human_handoff(...) - operational exception escalation to human staff."""
        from orca.services.handoff import handoff_service
        case = handoff_service.create_case(
            farmer_id=farmer_id,
            order_id=order_id,
            reason=reason,
            summary=summary,
            source_intent=source_intent,
        )
        return {
            "success": True,
            "case_id": case.case_id,
            "status": case.status.value,
            "reason": case.reason.value,
            "farmer_id": case.farmer_id,
            "order_id": case.order_id,
            "summary": case.summary,
        }

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolExecutionResult:
        """Deterministic tool execution boundary. Validates tool name and dispatches safely."""
        APPROVED_TOOLS = {
            "get_supported_produce": self.get_supported_produce,
            "get_applicable_rate": self.get_applicable_rate,
            "validate_quantity": self.validate_quantity,
            "calculate_order_total": self.calculate_order_total,
            "create_order": self.create_order,
            "create_bill": self.create_bill,
            "create_payment_request": self.create_payment_request,
            "get_payment_status": self.get_payment_status,
            "create_collection_task": self.create_collection_task,
            "get_collection_status": self.get_collection_status,
            "get_order_status": self.get_order_status,
            "request_pickup_reschedule": self.request_pickup_reschedule,
            "report_collection_problem": self.report_collection_problem,
            "explain_requirement": self.explain_requirement,
            "send_confirmation": self.send_confirmation,
            "handle_pricing_objection": self.handle_pricing_objection,
            "evaluate_procurement_offer": self.evaluate_procurement_offer,
            "get_business_knowledge": self.get_business_knowledge,
            "get_procurement_recommendations": self.get_procurement_recommendations,
            "get_personalized_recommendations": self.get_personalized_recommendations,
            "create_human_handoff": self.create_human_handoff,
        }

        if tool_name not in APPROVED_TOOLS:
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                error_message=f"Tool '{tool_name}' is not an approved backend tool.",
            )

        fn = APPROVED_TOOLS[tool_name]
        try:
            # Filter arguments to those accepted by the target callable
            sig = inspect.signature(fn)
            filtered_args = {k: v for k, v in arguments.items() if k in sig.parameters}

            if inspect.iscoroutinefunction(fn):
                res = await fn(**filtered_args)
            else:
                res = fn(**filtered_args)

            if isinstance(res, dict):
                data = res
                success = res.get("success", True) if "success" in res else not bool(res.get("error"))
                err = res.get("error")
            elif isinstance(res, list):
                data = {"items": res}
                success = True
                err = None
            else:
                data = {"result": res}
                success = True
                err = None

            return ToolExecutionResult(
                tool_name=tool_name,
                success=success,
                data=data,
                error_message=err,
            )
        except Exception as exc:
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                data={},
                error_message=str(exc),
            )


tool_registry = AgentToolRegistry()
