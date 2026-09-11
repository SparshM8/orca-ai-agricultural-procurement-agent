from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Query, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse

from orca.core.config import settings
from orca.domain.schemas import InboundMessage
from orca.domain.state_machine import OrderState
from orca.services.pricing import pricing_service
from orca.services.order import order_service
from orca.services.payment import payment_service
from orca.services.collection import collection_service
from orca.services.channel import channel_service
from orca.adapters.messaging import get_channel_provider
from orca.agent.orchestrator import orchestrator
from orca.db.session import engine
from orca.db.repository import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan managing database initialization, demo seeding, and teardown."""
    await init_db(engine)
    try:
        from orca.services.demo_seed import seed_demo_data
        seed_demo_data()
    except Exception:
        pass
    yield
    await engine.dispose()


class RunnerActionPayload(BaseModel):
    runner_id: str
    reason: Optional[str] = None


class RunnerReschedulePayload(BaseModel):
    new_time_str: Optional[str] = None


TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "ui" / "templates" / "index.html"
STATIC_LOGO_PATH = Path(__file__).resolve().parent.parent / "ui" / "static" / "logo.png"
ROOT_LOGO_PATH = Path("D:/ORCA/orcalogo.png")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Global AI Agricultural Procurement Agent API",
        lifespan=lifespan,
    )

    # Enable CORS for local testing & multi-origin browser access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -------------------------------------------------------------------------
    # Core & Existing Endpoints
    # -------------------------------------------------------------------------
    @app.get("/health", tags=["System"])
    async def health_check():
        """Health check probe."""
        return {
            "status": "healthy",
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
        }

    @app.get("/produce", tags=["Procurement"])
    async def get_produce(region: str = settings.DEFAULT_REGION):
        """Retrieve supported produce for a given region (SRS Section 14)."""
        produce = pricing_service.get_supported_produce(region_code=region)
        return {"region": region, "supported_produce": produce}

    @app.get("/rates", tags=["Procurement"])
    async def get_rates(produce: str, region: str = settings.DEFAULT_REGION):
        """Retrieve authoritative purchase rate (SRS Section 14)."""
        rate = pricing_service.get_applicable_rate(produce_type=produce, region_code=region)
        if not rate:
            return {"found": False, "message": f"No rate for {produce} in {region}"}
        return {"found": True, "rate": rate}

    @app.post("/agent/message", tags=["Agent"])
    async def process_agent_message(message: InboundMessage):
        """Process normalized inbound conversation message (SRS Section 14).
        
        Delegates through the decoupled ChannelService layer for consistency and deduplication.
        """
        channel_name = message.channel or "web_chat"
        payload = message.model_dump()
        try:
            result = await channel_service.handle_inbound(channel_name, payload)
            return result
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err))

    # -------------------------------------------------------------------------
    # Channel Webhook & Simulation Endpoints
    # -------------------------------------------------------------------------
    @app.post("/webhook/{channel_name}", tags=["Channels"])
    @app.post("/api/channels/{channel_name}/webhook", tags=["Channels"])
    async def handle_channel_webhook(channel_name: str, payload: Dict[str, Any], request: Request):
        """Receive and normalize channel webhook events (e.g. simulator, web_chat, future WhatsApp)."""
        headers = dict(request.headers)
        try:
            result = await channel_service.handle_inbound(channel_name, payload, headers=headers)
            return result
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err))
        except Exception as err:
            raise HTTPException(status_code=500, detail=f"Channel processing error: {str(err)}")

    @app.get("/api/channels/{channel_name}/messages", tags=["Channels"])
    async def get_channel_sent_messages(channel_name: str, recipient_id: Optional[str] = None):
        """Inspect outbound messages sent through the specified channel adapter."""
        provider = get_channel_provider(channel_name)
        if hasattr(provider, "get_sent_messages"):
            messages = provider.get_sent_messages(recipient_id=recipient_id)
            return {"channel": channel_name, "count": len(messages), "messages": [m.model_dump() for m in messages]}
        return {"channel": channel_name, "count": 0, "messages": []}

    # -------------------------------------------------------------------------
    # Farmer Presentation Endpoints
    # -------------------------------------------------------------------------
    @app.get("/api/farmer/context/{sender_id}", tags=["Farmer"])
    async def get_farmer_context(sender_id: str):
        """Retrieve conversational context, current order state, message history, and support handoff for a farmer."""
        from orca.services.handoff import handoff_service
        farmer_cases = handoff_service.list_all_cases(farmer_id=sender_id)
        latest_case = farmer_cases[0].model_dump() if farmer_cases else None

        conv_id = f"conv_{sender_id}"
        ctx = orchestrator.get_context(conv_id)
        if not ctx:
            return {
                "found": False,
                "sender_id": sender_id,
                "conversation_id": conv_id,
                "state": "OFFER_RECEIVED",
                "current_order_id": None,
                "history": [],
                "handoff_case": latest_case,
            }

        order_info = None
        pay_info = None
        coll_info = None
        if ctx.current_order_id:
            order = order_service.get_order(ctx.current_order_id)
            if order:
                order_info = order.model_dump()
                ctx.state = order.status
            pay = payment_service.get_payment_for_order(ctx.current_order_id)
            if pay:
                pay_info = pay.model_dump()
            coll = collection_service.get_task_by_order(ctx.current_order_id)
            if coll:
                coll_info = coll.model_dump()

        return {
            "found": True,
            "sender_id": sender_id,
            "conversation_id": conv_id,
            "state": ctx.state.value,
            "current_order_id": ctx.current_order_id,
            "offer": ctx.offer.model_dump(),
            "order": order_info,
            "payment": pay_info,
            "collection_task": coll_info,
            "handoff_case": latest_case,
            "history": [{"role": r, "text": t} for r, t in ctx.history],
        }

    # -------------------------------------------------------------------------
    # Runner Console Endpoints
    # -------------------------------------------------------------------------
    @app.get("/api/runner/tasks/available", tags=["Runner"])
    async def get_available_runner_tasks():
        """Retrieve all collection tasks currently waiting for runner assignment (status == PENDING)."""
        tasks = collection_service.get_available_tasks()
        return {"tasks": [t.model_dump() for t in tasks]}

    @app.get("/api/runner/tasks/assigned", tags=["Runner"])
    async def get_assigned_runner_tasks(runner_id: str = Query(..., description="Runner ID")):
        """Retrieve tasks assigned to a specific runner."""
        tasks = collection_service.get_assigned_tasks(runner_id=runner_id)
        return {"runner_id": runner_id, "tasks": [t.model_dump() for t in tasks]}

    @app.get("/api/runner/tasks/{task_id}", tags=["Runner"])
    async def get_runner_task_detail(task_id: str):
        """Retrieve single task detail with related order information."""
        task = collection_service.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        order = order_service.get_order(task.order_id)
        return {
            "task": task.model_dump(),
            "order": order.model_dump() if order else None,
        }

    @app.post("/api/runner/tasks/{task_id}/accept", tags=["Runner"])
    async def runner_accept_task(task_id: str, payload: RunnerActionPayload):
        """Runner accepts an available task, advancing order to COLLECTION_ASSIGNED."""
        try:
            task = await collection_service.accept_task(task_id=task_id, runner_id=payload.runner_id)
            return {"status": "accepted", "task": task.model_dump()}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/runner/tasks/{task_id}/reject", tags=["Runner"])
    async def runner_reject_task(task_id: str, payload: RunnerActionPayload):
        """Runner safely rejects a task, returning order to COLLECTION_PENDING."""
        try:
            task = await collection_service.reject_task(
                task_id=task_id, runner_id=payload.runner_id, reason=payload.reason
            )
            return {"status": "rejected", "task": task.model_dump()}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/runner/tasks/{task_id}/confirm-pickup", tags=["Runner"])
    async def runner_confirm_pickup(task_id: str, payload: RunnerActionPayload):
        """Runner confirms physical pickup, completing order through state machine."""
        try:
            task = await collection_service.confirm_pickup(task_id=task_id, runner_id=payload.runner_id)
            return {"status": "completed", "task": task.model_dump()}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/runner/tasks/{task_id}/fail", tags=["Runner"])
    async def runner_fail_pickup(task_id: str, payload: RunnerActionPayload):
        """Runner reports pickup failure, transitioning order to EXCEPTION."""
        if not payload.reason:
            raise HTTPException(status_code=400, detail="Failure reason is required.")
        try:
            task = await collection_service.fail_pickup(
                task_id=task_id, runner_id=payload.runner_id, reason=payload.reason
            )
            return {"status": "failed", "task": task.model_dump()}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/runner/tasks/{task_id}/reschedule", tags=["Runner"])
    async def runner_reschedule_task(task_id: str, payload: RunnerReschedulePayload):
        """Reschedule a failed task from EXCEPTION back to COLLECTION_PENDING."""
        try:
            task = await collection_service.reschedule_task(
                task_id=task_id, new_time_str=payload.new_time_str
            )
            return {"status": "rescheduled", "task": task.model_dump()}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # -------------------------------------------------------------------------
    # Operations / Admin Dashboard Endpoints
    # -------------------------------------------------------------------------
    @app.get("/api/admin/orders", tags=["Admin"])
    async def get_all_admin_orders(status: Optional[str] = None):
        """View all orders with payment & collection status for operational visibility."""
        orders = order_service.get_all_orders(status=status)
        result = []
        for o in orders:
            pay = payment_service.get_payment_for_order(o.id)
            coll = collection_service.get_task_by_order(o.id)
            result.append({
                "id": o.id,
                "farmer_id": o.farmer_id,
                "produce_type": o.produce_type,
                "quantity": o.quantity,
                "unit": o.unit,
                "validated_rate": o.validated_rate,
                "currency": o.currency,
                "total_amount": o.total_amount,
                "pickup_location": o.pickup_location,
                "status": o.status.value,
                "created_at": o.created_at.isoformat() if o.created_at else None,
                "updated_at": o.updated_at.isoformat() if o.updated_at else None,
                "payment": {
                    "id": pay.id,
                    "amount": pay.amount,
                    "currency": pay.currency,
                    "status": pay.status,
                    "provider_reference": pay.provider_reference,
                    "timestamp": pay.timestamp.isoformat() if pay.timestamp else None,
                } if pay else None,
                "collection_task": {
                    "id": coll.id,
                    "status": coll.status,
                    "runner_id": coll.runner_id,
                    "pickup_location": coll.pickup_location,
                    "scheduled_time_str": coll.scheduled_time_str,
                    "failure_reason": coll.failure_reason,
                    "created_at": coll.created_at.isoformat() if coll.created_at else None,
                } if coll else None,
            })
        return {"total": len(result), "orders": result}

    @app.get("/api/admin/orders/{order_id}", tags=["Admin"])
    async def get_admin_order_detail(order_id: str):
        """Detailed order view with payment, collection, progression, and conversation transcript."""
        order = order_service.get_order(order_id)
        if not order:
            raise HTTPException(status_code=404, detail=f"Order {order_id} not found")

        pay = payment_service.get_payment_for_order(order_id)
        coll = collection_service.get_task_by_order(order_id)
        context = orchestrator.get_context_by_order_id(order_id)

        # Build chronological state progression
        order_st = order.status
        progression = [
            {"state": "OFFER_RECEIVED", "label": "Offer Received", "reached": True},
            {
                "state": "ORDER_CONFIRMED",
                "label": "Order Confirmed",
                "reached": True,
                "timestamp": order.created_at.isoformat() if order.created_at else None,
            },
            {
                "state": "PAYMENT_PENDING",
                "label": "Payment Pending",
                "reached": pay is not None,
            },
            {
                "state": "PAYMENT_CONFIRMED",
                "label": "Payment Confirmed",
                "reached": pay is not None and pay.status == "SUCCESS",
                "timestamp": pay.timestamp.isoformat() if (pay and pay.status == "SUCCESS") else None,
                "detail": f"Ref: {pay.provider_reference}" if (pay and pay.provider_reference) else None,
            },
            {
                "state": "COLLECTION_PENDING",
                "label": "Collection Pending",
                "reached": coll is not None,
            },
            {
                "state": "COLLECTION_ASSIGNED",
                "label": "Runner Assigned",
                "reached": coll is not None and coll.status in ["ASSIGNED", "PICKED_UP", "COMPLETED"],
                "detail": f"Runner: {coll.runner_id}" if coll and coll.runner_id else None,
                "timestamp": coll.assigned_at.isoformat() if (coll and coll.assigned_at) else None,
            },
            {
                "state": "PICKED_UP",
                "label": "Produce Picked Up",
                "reached": coll is not None and coll.status in ["PICKED_UP", "COMPLETED"],
                "timestamp": coll.picked_up_at.isoformat() if (coll and coll.picked_up_at) else None,
            },
            {
                "state": "COMPLETED",
                "label": "Completed",
                "reached": order_st == OrderState.COMPLETED,
                "timestamp": coll.completed_at.isoformat() if (coll and coll.completed_at) else None,
            },
        ]
        if order_st == OrderState.EXCEPTION or (coll and coll.status == "FAILED"):
            progression.append({
                "state": "EXCEPTION",
                "label": "Exception / Pickup Failed",
                "reached": True,
                "detail": coll.failure_reason if coll else "Unknown error",
            })

        transcript = []
        safe_traces = []
        if context:
            for role, text in context.history:
                transcript.append({"role": role, "text": text})
            if hasattr(context, "traces"):
                for t in context.traces:
                    safe_args = None
                    if t.tool_arguments and isinstance(t.tool_arguments, dict):
                        safe_args = {
                            k: v for k, v in t.tool_arguments.items()
                            if k in ("produce_type", "quantity", "unit", "pickup_location", "order_id", "counter_rate", "rate")
                        }
                    safe_traces.append({
                        "trace_id": t.trace_id,
                        "timestamp": t.timestamp,
                        "intent": t.detected_intent,
                        "confidence": t.confidence,
                        "classifier": t.classifier_used,
                        "fallback_occurred": t.fallback_occurred,
                        "fallback_reason": t.fallback_reason,
                        "selected_tool": t.selected_tool,
                        "tool_success": t.tool_success,
                        "state_before": t.state_before,
                        "state_after": t.state_after,
                        "tool_arguments": safe_args,
                    })

        return {
            "order": order.model_dump(),
            "payment": pay.model_dump() if pay else None,
            "collection_task": coll.model_dump() if coll else None,
            "progression": progression,
            "traces": safe_traces,
            "conversation": {
                "conversation_id": context.conversation_id if context else None,
                "farmer_id": context.farmer_id if context else order.farmer_id,
                "state": context.state.value if context else order.status.value,
                "transcript": transcript,
                "offer": context.offer.model_dump() if context else None,
                "traces": safe_traces,
            } if context else None,
        }

    @app.get("/api/admin/tasks", tags=["Admin"])
    async def get_all_admin_tasks(status: Optional[str] = None):
        """View all collection tasks."""
        tasks = collection_service.get_all_tasks(status=status)
        return {"total": len(tasks), "tasks": [t.model_dump() for t in tasks]}

    @app.get("/api/admin/payments", tags=["Admin"])
    async def get_all_admin_payments():
        """View all payment records."""
        payments = payment_service.get_all_payments()
        return {"total": len(payments), "payments": [p.model_dump() for p in payments]}

    @app.get("/api/admin/stats", tags=["Admin"])
    async def get_admin_stats():
        """Retrieve high-level procurement metrics for executive overview."""
        orders = order_service.get_all_orders()
        total_orders = len(orders)
        total_gmv = sum(o.total_amount for o in orders)
        completed = sum(1 for o in orders if o.status == OrderState.COMPLETED)
        pending_collection = sum(
            1 for o in orders if o.status in [OrderState.COLLECTION_PENDING, OrderState.COLLECTION_ASSIGNED]
        )
        exceptions = sum(1 for o in orders if o.status == OrderState.EXCEPTION)
        available_tasks = len(collection_service.get_available_tasks())
        from orca.services.handoff import handoff_service
        open_handoffs = len(handoff_service.list_open_cases())

        return {
            "total_orders": total_orders,
            "total_gmv": round(total_gmv, 2),
            "currency": "USD",
            "completed_orders": completed,
            "pending_collection": pending_collection,
            "exceptions": exceptions,
            "available_runner_tasks": available_tasks,
            "open_handoffs": open_handoffs,
        }

    @app.get("/api/admin/handoffs", tags=["Admin"])
    async def list_admin_handoffs():
        """List all operational human handoff cases."""
        from orca.services.handoff import handoff_service
        cases = handoff_service.list_all_cases()
        return {
            "count": len(cases),
            "cases": [c.model_dump() for c in cases],
        }

    @app.get("/api/admin/handoffs/{case_id}", tags=["Admin"])
    async def get_admin_handoff_case(case_id: str):
        """Retrieve details of a specific human handoff case."""
        from orca.services.handoff import handoff_service
        case = handoff_service.get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail=f"Handoff case '{case_id}' not found.")
        return case.model_dump()

    @app.post("/api/admin/handoffs/{case_id}/resolve", tags=["Admin"])
    async def resolve_admin_handoff(case_id: str, payload: Dict[str, Any] = Body(default_factory=dict)):
        """Mark a human handoff case as resolved."""
        from orca.services.handoff import handoff_service
        notes = payload.get("notes", "Resolved by operational support")
        try:
            case = handoff_service.resolve_case(case_id, resolution_notes=notes)
            return {"status": "success", "case": case.model_dump()}
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.post("/api/admin/demo/seed", tags=["Admin"])
    async def seed_demo():
        """Idempotently seed demonstration data across all services (Phase 8A)."""
        from orca.services.demo_seed import seed_demo_data
        result = seed_demo_data()
        return result

    @app.get("/api/demo/samples", tags=["Demo"])
    async def get_demo_scenarios():
        """Retrieve structured sample scenarios and transcripts for guided demonstration."""
        from orca.services.demo_seed import get_demo_samples
        return {"samples": get_demo_samples()}

    @app.post("/api/admin/demo/reset", tags=["Admin"])
    async def reset_demo_state(reseed: bool = Query(default=False, description="Optionally reseed demo data after reset")):
        """Reset all demo orders, payments, collection tasks, and conversational contexts.
        
        Reseeds baseline rates and returns a deterministic success response.
        Safe to execute repeatedly without side effects.
        """
        # 1. Clear in-memory services
        order_service.clear()
        payment_service.clear()
        collection_service.clear()
        orchestrator.clear()
        channel_service.clear()
        pricing_service.reset_baseline_rates()
        try:
            from orca.services.farmer_profile import farmer_profile_service
            farmer_profile_service.clear()
        except Exception:
            pass
        try:
            from orca.services.handoff import handoff_service
            handoff_service.clear()
        except Exception:
            pass

        # 2. Reset database tables and reseed
        try:
            from orca.db.repository import reset_db
            await reset_db(engine)
        except Exception:
            # Safe fallback if engine is in-memory or not initialized
            pass

        if reseed:
            from orca.services.demo_seed import seed_demo_data
            seed_demo_data()

        return {
            "status": "success",
            "message": "Demo state successfully reset to clean baseline." + (" Demo data re-seeded." if reseed else ""),
            "cleared": {
                "orders": 0,
                "payments": 0,
                "collection_tasks": 0,
                "conversations": 0,
                "traces": 0,
            },
            "reseeded": reseed,
            "baseline_rates_active": pricing_service.get_supported_produce(),
        }

    # -------------------------------------------------------------------------
    # Web Presentation UI
    # -------------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse, tags=["UI"], include_in_schema=False)
    @app.get("/ui", response_class=HTMLResponse, tags=["UI"], include_in_schema=False)
    async def serve_ui():
        """Serve the presentation demo single-page application."""
        if TEMPLATE_PATH.exists():
            return HTMLResponse(content=TEMPLATE_PATH.read_text(encoding="utf-8"))
        return HTMLResponse(
            content="""<!DOCTYPE html>
<html>
<head><title>ORCA Agricultural Procurement</title></head>
<body>
  <h2>ORCA Presentation UI</h2>
  <p>Template file <code>src/orca/ui/templates/index.html</code> is being generated...</p>
</body>
</html>""",
            status_code=200,
        )

    @app.get("/static/logo.png", include_in_schema=False)
    @app.get("/logo.png", include_in_schema=False)
    async def serve_logo():
        """Serve the official ORCA AI SOLUTIONS logo."""
        if STATIC_LOGO_PATH.exists():
            return FileResponse(STATIC_LOGO_PATH, media_type="image/png")
        if ROOT_LOGO_PATH.exists():
            return FileResponse(ROOT_LOGO_PATH, media_type="image/png")
        raise HTTPException(status_code=404, detail="Logo not found")

    return app


app = create_app()

