"""Agent tool definitions and bindings according to SRS Section 8.

All tools expose typed schemas to the LLM and delegate to backend deterministic services.
The LLM does not execute operations directly.
"""

from typing import Dict, Any, List, Optional
from orca.services.pricing import pricing_service
from orca.services.billing import billing_service
from orca.services.order import order_service
from orca.services.payment import payment_service
from orca.services.collection import collection_service
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
        validated_rate: float,
        pickup_location: str,
        conversation_id: Optional[str] = None,
        region_code: str = "GLOBAL_DEFAULT",
    ) -> Dict[str, Any]:
        """Tool 5: create_order(validated_order) - create an auditable order."""
        order = order_service.create_order(
            farmer_id=farmer_id,
            produce_type=produce,
            quantity=quantity,
            unit=unit,
            validated_rate=validated_rate,
            pickup_location=pickup_location,
            region_code=region_code,
            conversation_id=conversation_id,
        )
        return {
            "order_id": order.id,
            "status": order.status.value,
            "total_amount": order.total_amount,
            "currency": order.currency,
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
    def create_payment_request(order_id: str) -> Dict[str, Any]:
        """Tool 7: create_payment_request(order_id) - initiate payment workflow."""
        order = order_service.get_order(order_id)
        if not order:
            return {"error": f"Order not found: {order_id}"}
        payment = payment_service.create_payment_request(
            order_id=order.id,
            amount=order.total_amount,
            currency=order.currency,
        )
        return {
            "payment_id": payment.id,
            "order_id": payment.order_id,
            "amount": payment.amount,
            "currency": payment.currency,
            "status": payment.status,
        }

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


tool_registry = AgentToolRegistry()
