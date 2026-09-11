"""Demo Seed Data and Guided Product Experience Service (Phase 8A).

Provides deterministic, idempotent sample conversation and operational records for:
1. Hero Procurement Flow (5 turns, price negotiation, amendment, completion, runner fulfillment)
2. Assistance & Grounded Recommendations (3 turns, FAQ, produce catalog, grounded recommendation)
3. Human Support & Exception Handoff (1 turn, support case creation with zero state corruption)

Guarantees:
- Zero live API calls or external dependencies
- Fully tagged with metadata={"demo": True, "scenario": ...} and is_demo=True
- Idempotent: can be called multiple times without duplicate records or ID collisions
- Reversible: clear_demo_data() cleans demo records while preserving non-demo state
- Clear distinction between sample demo history and live AI sessions
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

from orca.domain.models import Order, Payment, CollectionTask
from orca.domain.state_machine import OrderState
from orca.domain.schemas import ExtractedOffer
from orca.domain.handoff import HumanHandoffCase, HandoffReason, HandoffStatus
from orca.domain.farmer_profile import FactualCompletedOrder, FactualOfferObservation
from orca.services.order import order_service
from orca.services.payment import payment_service
from orca.services.collection import collection_service
from orca.services.farmer_profile import farmer_profile_service
from orca.services.handoff import handoff_service


# -----------------------------------------------------------------------------
# Seed Identifiers
# -----------------------------------------------------------------------------
HERO_FARMER_ID = "farmer_demo_hero"
HERO_CONV_ID = f"conv_{HERO_FARMER_ID}"
HERO_ORDER_ID = "ORD-DEMO-001"
HERO_PAYMENT_ID = "PAY-DEMO-001"
HERO_TASK_ID = "COL-DEMO-001"

ASSISTANCE_FARMER_ID = "farmer_demo_assistance"
ASSISTANCE_CONV_ID = f"conv_{ASSISTANCE_FARMER_ID}"

HANDOFF_FARMER_ID = "farmer_demo_handoff"
HANDOFF_CONV_ID = f"conv_{HANDOFF_FARMER_ID}"
HANDOFF_CASE_ID = "HC-DEMO-001"


# -----------------------------------------------------------------------------
# Transcripts Definition
# -----------------------------------------------------------------------------
HERO_TRANSCRIPT: List[Dict[str, str]] = [
    {
        "role": "farmer",
        "text": "I have 50kg of potatoes in Nairobi available tomorrow at 10 AM.",
    },
    {
        "role": "agent",
        "text": "I can offer $0.40 per kg for 50 kg of potatoes in Nairobi (Total: $20.00). Pickup scheduled for Tomorrow 10:00 AM. Would you like to proceed?",
    },
    {
        "role": "farmer",
        "text": "Can you pay $0.45?",
    },
    {
        "role": "agent",
        "text": "Our purchase rate for potatoes in GLOBAL_DEFAULT is fixed at $0.40 per kg based on official regional procurement rates. We cannot adjust this rate. Would you still like to proceed with 50 kg at $0.40/kg?",
    },
    {
        "role": "farmer",
        "text": "Actually, change that to 80 kg",
    },
    {
        "role": "agent",
        "text": "Updated your offer to 80 kg of potatoes at $0.40/kg (Total: $32.00) in Nairobi. Would you like to confirm this order?",
    },
    {
        "role": "farmer",
        "text": "Confirm",
    },
    {
        "role": "agent",
        "text": "Order ORD-DEMO-001 confirmed! Total: $32.00. Payment authorization is ready. Please reply 'pay' or click authorize to proceed.",
    },
    {
        "role": "farmer",
        "text": "Pay",
    },
    {
        "role": "agent",
        "text": "Payment of $32.00 received successfully (Ref: SANDBOX-PAY-DEMO-001). Collection task COL-DEMO-001 created. Runner Runner-01 accepted, picked up, and delivered the produce. Order COMPLETED!",
    },
]

ASSISTANCE_TRANSCRIPT: List[Dict[str, str]] = [
    {
        "role": "farmer",
        "text": "How does pickup and payment work?",
    },
    {
        "role": "agent",
        "text": "Once you confirm an offer, we initiate instant mobile payment authorization. Upon payment confirmation, an ORCA runner is dispatched to your specified pickup location to inspect and collect the produce.",
    },
    {
        "role": "farmer",
        "text": "What crops do you buy?",
    },
    {
        "role": "agent",
        "text": "We currently procure the following supported produce items: Potato ($0.40/kg), Tomato ($0.60/kg), Onion ($0.35/kg), Cabbage ($0.25/kg), Maize ($0.30/kg), Beans ($0.80/kg).",
    },
    {
        "role": "farmer",
        "text": "What would you recommend for me?",
    },
    {
        "role": "agent",
        "text": "Based on your verified history with potato, we recommend continuing with Potato ($0.40/kg). We also have high procurement capacity for Onion ($0.35/kg) and Cabbage ($0.25/kg) in your region.",
    },
]

HANDOFF_TRANSCRIPT: List[Dict[str, str]] = [
    {
        "role": "farmer",
        "text": "I need to speak to a human support agent about my collection schedule.",
    },
    {
        "role": "agent",
        "text": "I understand you need human assistance. I have opened support case HC-DEMO-001 for you (Reason: FARMER_REQUEST). A human support specialist will review your case and contact you shortly.",
    },
]


def is_demo_seeded() -> bool:
    """Check if demo seed data is currently loaded in memory."""
    from orca.agent.orchestrator import orchestrator

    return (
        HERO_ORDER_ID in order_service._orders
        and HERO_CONV_ID in orchestrator._conversations
        and HANDOFF_CASE_ID in handoff_service._cases
    )


def seed_demo_data() -> Dict[str, Any]:
    """Seed sample demonstration records into in-memory services idempotently.

    Safe to invoke repeatedly without creating duplicate entities or corrupting state.
    """
    from orca.agent.orchestrator import orchestrator, DialogueContext

    now = datetime.now(timezone.utc)
    scheduled_time = now + timedelta(days=1)

    # -------------------------------------------------------------------------
    # Scenario 1: Hero Procurement Flow (Completed Order, Payment & Task)
    # -------------------------------------------------------------------------
    hero_order = Order(
        id=HERO_ORDER_ID,
        farmer_id=HERO_FARMER_ID,
        produce_type="potato",
        quantity=80.0,
        unit="kg",
        validated_rate=0.40,
        currency="USD",
        total_amount=32.00,
        pickup_location="Nairobi",
        pickup_datetime=scheduled_time,
        pickup_time_str="Tomorrow 10:00 AM",
        status=OrderState.COMPLETED,
        metadata={"demo": True, "scenario": "hero"},
        created_at=now - timedelta(minutes=15),
        updated_at=now,
    )
    order_service._orders[HERO_ORDER_ID] = hero_order
    order_service._confirmed_conversations[HERO_CONV_ID] = HERO_ORDER_ID

    hero_payment = Payment(
        id=HERO_PAYMENT_ID,
        order_id=HERO_ORDER_ID,
        amount=32.00,
        currency="USD",
        status="SUCCESS",
        provider_reference="SANDBOX-PAY-DEMO-001",
        metadata={"demo": True, "scenario": "hero"},
        timestamp=now - timedelta(minutes=8),
    )
    payment_service._payments[HERO_PAYMENT_ID] = hero_payment
    payment_service._order_payments[HERO_ORDER_ID] = HERO_PAYMENT_ID

    hero_task = CollectionTask(
        id=HERO_TASK_ID,
        order_id=HERO_ORDER_ID,
        runner_id="Runner-01",
        produce_type="potato",
        quantity=80.0,
        unit="kg",
        pickup_location="Nairobi",
        scheduled_datetime=scheduled_time,
        scheduled_time_str="Tomorrow 10:00 AM",
        farmer_id=HERO_FARMER_ID,
        status="COMPLETED",
        metadata={"demo": True, "scenario": "hero"},
        created_at=now - timedelta(minutes=8),
        assigned_at=now - timedelta(minutes=7),
        picked_up_at=now - timedelta(minutes=4),
        completed_at=now - timedelta(minutes=1),
    )
    collection_service._tasks[HERO_TASK_ID] = hero_task
    collection_service._order_tasks[HERO_ORDER_ID] = HERO_TASK_ID

    # Pre-record completed order in profile service for grounded recommendations
    profile_hero = farmer_profile_service.get_profile(HERO_FARMER_ID)
    profile_hero.location = "Nairobi"
    if not any(co.order_id == HERO_ORDER_ID for co in profile_hero.completed_orders):
        profile_hero.completed_orders.append(
            FactualCompletedOrder(
                order_id=HERO_ORDER_ID,
                produce="potato",
                quantity=80.0,
                unit="kg",
                completed_at=now - timedelta(minutes=1),
            )
        )
    profile_hero.completed_order_count = len(profile_hero.completed_orders)
    profile_hero.last_completed_produce = "potato"
    if "potato" not in profile_hero.observed_produce_history:
        profile_hero.observed_produce_history.append("potato")
    if "kg" not in profile_hero.observed_units:
        profile_hero.observed_units.append("kg")
    farmer_profile_service._profiles[HERO_FARMER_ID] = profile_hero

    # Dialogue context for hero conversation
    hero_ctx = DialogueContext(
        conversation_id=HERO_CONV_ID,
        farmer_id=HERO_FARMER_ID,
        region_code="GLOBAL_DEFAULT",
        state=OrderState.COMPLETED,
        current_order_id=HERO_ORDER_ID,
        order_history=[HERO_ORDER_ID],
        is_demo=True,
        metadata={"demo": True, "scenario": "hero"},
        offer=ExtractedOffer(
            produce_type="potato",
            quantity=80.0,
            unit="kg",
            pickup_location="Nairobi",
            pickup_datetime="Tomorrow 10:00 AM",
        ),
        history=[(m["role"], m["text"]) for m in HERO_TRANSCRIPT],
        negotiation_attempts=1,
        amendment_history=["quantity: 50 -> 80"],
    )
    orchestrator._conversations[HERO_CONV_ID] = hero_ctx

    # -------------------------------------------------------------------------
    # Scenario 2: Assistance & Grounded Recommendations
    # -------------------------------------------------------------------------
    profile_assistance = farmer_profile_service.get_profile(ASSISTANCE_FARMER_ID)
    profile_assistance.location = "Nakuru"
    past_order_id = "ORD-DEMO-PAST-002"
    if not any(co.order_id == past_order_id for co in profile_assistance.completed_orders):
        profile_assistance.completed_orders.append(
            FactualCompletedOrder(
                order_id=past_order_id,
                produce="potato",
                quantity=100.0,
                unit="kg",
                completed_at=now - timedelta(days=5),
            )
        )
    profile_assistance.completed_order_count = len(profile_assistance.completed_orders)
    profile_assistance.last_completed_produce = "potato"
    if "potato" not in profile_assistance.observed_produce_history:
        profile_assistance.observed_produce_history.append("potato")
    if "kg" not in profile_assistance.observed_units:
        profile_assistance.observed_units.append("kg")
    farmer_profile_service._profiles[ASSISTANCE_FARMER_ID] = profile_assistance

    assistance_ctx = DialogueContext(
        conversation_id=ASSISTANCE_CONV_ID,
        farmer_id=ASSISTANCE_FARMER_ID,
        region_code="GLOBAL_DEFAULT",
        state=OrderState.OFFER_RECEIVED,
        current_order_id=None,
        is_demo=True,
        metadata={"demo": True, "scenario": "assistance"},
        history=[(m["role"], m["text"]) for m in ASSISTANCE_TRANSCRIPT],
    )
    orchestrator._conversations[ASSISTANCE_CONV_ID] = assistance_ctx

    # -------------------------------------------------------------------------
    # Scenario 3: Human Escalation & Operational Handoff
    # -------------------------------------------------------------------------
    handoff_case = HumanHandoffCase(
        case_id=HANDOFF_CASE_ID,
        farmer_id=HANDOFF_FARMER_ID,
        order_id=None,
        reason=HandoffReason.FARMER_REQUEST,
        status=HandoffStatus.OPEN,
        summary="Farmer requested human assistance regarding collection schedule coordination.",
        source_intent="REQUEST_HUMAN_ASSISTANCE",
        metadata={"demo": True, "scenario": "handoff"},
        created_at=now - timedelta(minutes=5),
        updated_at=now - timedelta(minutes=5),
    )
    handoff_service._cases[HANDOFF_CASE_ID] = handoff_case

    handoff_ctx = DialogueContext(
        conversation_id=HANDOFF_CONV_ID,
        farmer_id=HANDOFF_FARMER_ID,
        region_code="GLOBAL_DEFAULT",
        state=OrderState.OFFER_RECEIVED,
        current_order_id=None,
        is_demo=True,
        metadata={"demo": True, "scenario": "handoff"},
        history=[(m["role"], m["text"]) for m in HANDOFF_TRANSCRIPT],
    )
    orchestrator._conversations[HANDOFF_CONV_ID] = handoff_ctx

    return {
        "status": "success",
        "orders_seeded": 1,
        "payments_seeded": 1,
        "collection_tasks_seeded": 1,
        "conversations_seeded": 3,
        "handoff_cases_seeded": 1,
        "profiles_seeded": 2,
    }


def clear_demo_data() -> Dict[str, int]:
    """Remove only demo records from in-memory stores, preserving live session data."""
    from orca.agent.orchestrator import orchestrator

    cleared = {
        "orders": 0,
        "payments": 0,
        "collection_tasks": 0,
        "conversations": 0,
        "handoff_cases": 0,
        "profiles": 0,
    }

    # Remove demo orders
    demo_orders = [
        oid for oid, o in list(order_service._orders.items())
        if o.metadata.get("demo") is True or oid.startswith("ORD-DEMO-")
    ]
    for oid in demo_orders:
        order_service._orders.pop(oid, None)
        cleared["orders"] += 1
    order_service._confirmed_conversations.pop(HERO_CONV_ID, None)

    # Remove demo payments
    demo_payments = [
        pid for pid, p in list(payment_service._payments.items())
        if p.metadata.get("demo") is True or pid.startswith("PAY-DEMO-")
    ]
    for pid in demo_payments:
        payment_service._payments.pop(pid, None)
        cleared["payments"] += 1
    payment_service._order_payments.pop(HERO_ORDER_ID, None)

    # Remove demo collection tasks
    demo_tasks = [
        tid for tid, t in list(collection_service._tasks.items())
        if t.metadata.get("demo") is True or tid.startswith("COL-DEMO-")
    ]
    for tid in demo_tasks:
        collection_service._tasks.pop(tid, None)
        cleared["collection_tasks"] += 1
    collection_service._order_tasks.pop(HERO_ORDER_ID, None)

    # Remove demo handoff cases
    demo_cases = [
        cid for cid, c in list(handoff_service._cases.items())
        if c.metadata.get("demo") is True or cid.startswith("HC-DEMO-")
    ]
    for cid in demo_cases:
        handoff_service._cases.pop(cid, None)
        cleared["handoff_cases"] += 1

    # Remove demo conversations
    demo_convs = [
        cid for cid, c in list(orchestrator._conversations.items())
        if c.is_demo or c.metadata.get("demo") is True or cid in [HERO_CONV_ID, ASSISTANCE_CONV_ID, HANDOFF_CONV_ID]
    ]
    for cid in demo_convs:
        orchestrator._conversations.pop(cid, None)
        cleared["conversations"] += 1

    # Remove demo profiles
    for fid in [HERO_FARMER_ID, ASSISTANCE_FARMER_ID, HANDOFF_FARMER_ID]:
        if fid in farmer_profile_service._profiles:
            farmer_profile_service._profiles.pop(fid, None)
            cleared["profiles"] += 1

    return cleared


def get_demo_samples() -> List[Dict[str, Any]]:
    """Return structured list of sample scenarios and transcripts for UI and documentation."""
    return [
        {
            "id": "hero",
            "key": "hero",
            "title": "Hero Procurement Flow",
            "badge": "COMPLETED",
            "badge_color": "success",
            "tagline": "End-to-end procurement negotiation, amendment, payment, and runner pickup",
            "farmer_id": HERO_FARMER_ID,
            "conversation_id": HERO_CONV_ID,
            "order_id": HERO_ORDER_ID,
            "state": "COMPLETED",
            "turns_count": len(HERO_TRANSCRIPT),
            "transcript": HERO_TRANSCRIPT,
            "summary": "Farmer offers 50 kg potatoes, negotiates for $0.45 (refused with authoritative explanation), amends to 80 kg ($32.00), confirms order, authorizes sandbox payment, and completes runner pickup.",
        },
        {
            "id": "assistance",
            "key": "assistance",
            "title": "FAQ & Grounded Recommendations",
            "badge": "GROUNDED",
            "badge_color": "info",
            "tagline": "Logistics FAQ, catalog inquiry, and factual personalized crop recommendations",
            "farmer_id": ASSISTANCE_FARMER_ID,
            "conversation_id": ASSISTANCE_CONV_ID,
            "order_id": None,
            "state": "OFFER_RECEIVED",
            "turns_count": len(ASSISTANCE_TRANSCRIPT),
            "transcript": ASSISTANCE_TRANSCRIPT,
            "summary": "Farmer inquires about pickup operations and produce catalog. Requests personalized recommendations, which are deterministically grounded in their verified prior sales history.",
        },
        {
            "id": "handoff",
            "key": "handoff",
            "title": "Human Support Escalation",
            "badge": "SUPPORT_CASE",
            "badge_color": "warning",
            "tagline": "Deterministic operational exception routing to human personnel with zero state drift",
            "farmer_id": HANDOFF_FARMER_ID,
            "conversation_id": HANDOFF_CONV_ID,
            "order_id": None,
            "state": "OFFER_RECEIVED",
            "turns_count": len(HANDOFF_TRANSCRIPT),
            "transcript": HANDOFF_TRANSCRIPT,
            "summary": "Farmer requests human support regarding logistics coordination. Agent creates support case HC-DEMO-001 under FARMER_REQUEST without corrupting transaction state.",
        },
    ]
