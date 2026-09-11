"""Unit tests for Demo Seed Data & Guided Product Experience (Phase 8A).

Tests scenarios A through L:
A. Demo seed execution & counts
B. Demo seed idempotency
C. Demo records metadata tagging
D. Hero procurement completed state & fulfillment
E. Hero procurement transcript preservation
F. Assistance & grounded recommendation scenario
G. Human support handoff scenario
H. Demo samples API contract
I. Clear demo data isolation
J. Admin endpoints (/api/admin/demo/seed, /api/demo/samples, /api/admin/demo/reset)
K. Live session isolation from demo sample state
L. Gemini-offline resilience with RuleBasedIntentClassifier fallback
"""

import pytest
from httpx import AsyncClient, ASGITransport

from orca.domain.state_machine import OrderState
from orca.domain.handoff import HandoffStatus, HandoffReason
from orca.domain.models import Order
from orca.agent.orchestrator import orchestrator, DialogueContext
from orca.services.order import order_service
from orca.services.payment import payment_service
from orca.services.collection import collection_service
from orca.services.farmer_profile import farmer_profile_service
from orca.services.handoff import handoff_service
from orca.services.demo_seed import (
    seed_demo_data,
    clear_demo_data,
    is_demo_seeded,
    get_demo_samples,
    HERO_FARMER_ID,
    HERO_CONV_ID,
    HERO_ORDER_ID,
    HERO_PAYMENT_ID,
    HERO_TASK_ID,
    ASSISTANCE_FARMER_ID,
    ASSISTANCE_CONV_ID,
    HANDOFF_FARMER_ID,
    HANDOFF_CONV_ID,
    HANDOFF_CASE_ID,
)
from orca.api.app import app


@pytest.fixture(autouse=True)
def clean_services_fixture():
    """Ensure clean service state before each test, then clean up after."""
    order_service.clear()
    payment_service.clear()
    collection_service.clear()
    orchestrator.clear()
    farmer_profile_service.clear()
    handoff_service.clear()
    yield
    order_service.clear()
    payment_service.clear()
    collection_service.clear()
    orchestrator.clear()
    farmer_profile_service.clear()
    handoff_service.clear()


# -----------------------------------------------------------------------------
# Test A: Demo seed execution
# -----------------------------------------------------------------------------
def test_demo_seed_execution():
    assert not is_demo_seeded()
    res = seed_demo_data()
    assert res["status"] == "success"
    assert is_demo_seeded()

    assert HERO_ORDER_ID in order_service._orders
    assert HERO_PAYMENT_ID in payment_service._payments
    assert HERO_TASK_ID in collection_service._tasks
    assert HERO_CONV_ID in orchestrator._conversations
    assert ASSISTANCE_CONV_ID in orchestrator._conversations
    assert HANDOFF_CONV_ID in orchestrator._conversations
    assert HANDOFF_CASE_ID in handoff_service._cases
    assert HERO_FARMER_ID in farmer_profile_service._profiles


# -----------------------------------------------------------------------------
# Test B: Demo seed idempotency
# -----------------------------------------------------------------------------
def test_demo_seed_idempotency():
    seed_demo_data()
    order_count_1 = len(order_service._orders)
    payment_count_1 = len(payment_service._payments)
    task_count_1 = len(collection_service._tasks)
    conv_count_1 = len(orchestrator._conversations)

    # Re-run seed
    seed_demo_data()
    assert len(order_service._orders) == order_count_1
    assert len(payment_service._payments) == payment_count_1
    assert len(collection_service._tasks) == task_count_1
    assert len(orchestrator._conversations) == conv_count_1


# -----------------------------------------------------------------------------
# Test C: Demo records metadata tagging
# -----------------------------------------------------------------------------
def test_demo_records_metadata_tagging():
    seed_demo_data()
    order = order_service.get_order(HERO_ORDER_ID)
    assert order is not None
    assert order.metadata.get("demo") is True
    assert order.metadata.get("scenario") == "hero"

    payment = payment_service.get_payment(HERO_PAYMENT_ID)
    assert payment is not None
    assert payment.metadata.get("demo") is True
    assert payment.metadata.get("scenario") == "hero"

    task = collection_service.get_task(HERO_TASK_ID)
    assert task is not None
    assert task.metadata.get("demo") is True
    assert task.metadata.get("scenario") == "hero"

    hero_ctx = orchestrator.get_context(HERO_CONV_ID)
    assert hero_ctx is not None
    assert hero_ctx.is_demo is True
    assert hero_ctx.metadata.get("demo") is True

    handoff = handoff_service.get_case(HANDOFF_CASE_ID)
    assert handoff is not None
    assert handoff.metadata.get("demo") is True
    assert handoff.metadata.get("scenario") == "handoff"


# -----------------------------------------------------------------------------
# Test D: Hero procurement state & fulfillment
# -----------------------------------------------------------------------------
def test_demo_hero_procurement_state():
    seed_demo_data()
    order = order_service.get_order(HERO_ORDER_ID)
    assert order.status == OrderState.COMPLETED
    assert order.produce_type == "potato"
    assert order.quantity == 80.0
    assert order.validated_rate == 0.40
    assert order.total_amount == 32.00
    assert order.pickup_location == "Nairobi"

    payment = payment_service.get_payment_for_order(HERO_ORDER_ID)
    assert payment.status == "SUCCESS"
    assert payment.amount == 32.00
    assert payment.provider_reference == "SANDBOX-PAY-DEMO-001"

    task = collection_service.get_task_by_order(HERO_ORDER_ID)
    assert task.status == "COMPLETED"
    assert task.runner_id == "Runner-01"


# -----------------------------------------------------------------------------
# Test E: Hero procurement transcript
# -----------------------------------------------------------------------------
def test_demo_hero_transcript():
    seed_demo_data()
    ctx = orchestrator.get_context(HERO_CONV_ID)
    assert len(ctx.history) == 10  # 5 turns = 10 messages (farmer, agent pairs)
    farmer_msgs = [text for role, text in ctx.history if role == "farmer"]
    agent_msgs = [text for role, text in ctx.history if role == "agent"]

    assert any("50kg" in m or "50 kg" in m for m in farmer_msgs)
    assert any("0.45" in m for m in farmer_msgs)
    assert any("80" in m for m in farmer_msgs)
    assert any("confirm" in m.lower() for m in farmer_msgs)
    assert any("pay" in m.lower() for m in farmer_msgs)

    assert any("fixed at $0.40" in m for m in agent_msgs)
    assert any("ORD-DEMO-001" in m for m in agent_msgs)
    assert any("SANDBOX-PAY-DEMO-001" in m for m in agent_msgs)


# -----------------------------------------------------------------------------
# Test F: Assistance & grounded recommendations scenario
# -----------------------------------------------------------------------------
def test_demo_assistance_scenario():
    seed_demo_data()
    ctx = orchestrator.get_context(ASSISTANCE_CONV_ID)
    assert len(ctx.history) == 6  # 3 turns
    profile = farmer_profile_service.get_profile(ASSISTANCE_FARMER_ID)
    assert "potato" in profile.observed_produce_history
    assert profile.completed_order_count >= 1

    last_agent_msg = [t for r, t in ctx.history if r == "agent"][-1]
    assert "potato" in last_agent_msg.lower()


# -----------------------------------------------------------------------------
# Test G: Human support handoff scenario
# -----------------------------------------------------------------------------
def test_demo_handoff_scenario():
    seed_demo_data()
    case = handoff_service.get_case(HANDOFF_CASE_ID)
    assert case is not None
    assert case.status == HandoffStatus.OPEN
    assert case.reason == HandoffReason.FARMER_REQUEST
    assert case.farmer_id == HANDOFF_FARMER_ID

    ctx = orchestrator.get_context(HANDOFF_CONV_ID)
    assert len(ctx.history) == 2
    assert "HC-DEMO-001" in ctx.history[1][1]


# -----------------------------------------------------------------------------
# Test H: Demo samples API contract
# -----------------------------------------------------------------------------
def test_get_demo_samples_api_contract():
    samples = get_demo_samples()
    assert len(samples) == 3
    keys = {s["id"] for s in samples}
    assert keys == {"hero", "assistance", "handoff"}

    for s in samples:
        assert "title" in s
        assert "badge" in s
        assert "farmer_id" in s
        assert "conversation_id" in s
        assert "transcript" in s
        assert len(s["transcript"]) > 0


# -----------------------------------------------------------------------------
# Test I: Clear demo data isolation
# -----------------------------------------------------------------------------
def test_clear_demo_data_isolation():
    seed_demo_data()

    # Create a non-demo order and context
    real_order = Order(
        id="ORD-LIVE-999",
        farmer_id="farmer_live_real",
        produce_type="cabbage",
        quantity=50.0,
        unit="kg",
        validated_rate=0.25,
        currency="USD",
        total_amount=12.50,
        pickup_location="Kisumu",
        status=OrderState.ORDER_CONFIRMED,
        metadata={"demo": False},
    )
    order_service._orders[real_order.id] = real_order
    orchestrator._conversations["conv_farmer_live_real"] = DialogueContext(
        conversation_id="conv_farmer_live_real",
        farmer_id="farmer_live_real",
        is_demo=False,
    )

    cleared = clear_demo_data()
    assert cleared["orders"] == 1
    assert cleared["conversations"] == 3

    # Verify demo removed
    assert HERO_ORDER_ID not in order_service._orders
    assert HERO_CONV_ID not in orchestrator._conversations

    # Verify real non-demo data is intact!
    assert "ORD-LIVE-999" in order_service._orders
    assert "conv_farmer_live_real" in orchestrator._conversations


# -----------------------------------------------------------------------------
# Test J: Admin endpoints
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_admin_demo_endpoints():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Seed demo
        res = await client.post("/api/admin/demo/seed")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"

        # 2. Get samples
        res_samples = await client.get("/api/demo/samples")
        assert res_samples.status_code == 200
        samples_data = res_samples.json()
        assert "samples" in samples_data
        assert len(samples_data["samples"]) == 3

        # 3. Farmer context retrieval for demo hero
        res_ctx = await client.get(f"/api/farmer/context/{HERO_FARMER_ID}")
        assert res_ctx.status_code == 200
        ctx_data = res_ctx.json()
        assert ctx_data["found"] is True
        assert ctx_data["state"] == "COMPLETED"
        assert ctx_data["order"]["id"] == HERO_ORDER_ID
        assert ctx_data["payment"]["status"] == "SUCCESS"
        assert ctx_data["collection_task"]["status"] == "COMPLETED"
        assert len(ctx_data["history"]) == 10

        # 4. Reset without reseed
        res_reset = await client.post("/api/admin/demo/reset?reseed=false")
        assert res_reset.status_code == 200
        assert not is_demo_seeded()

        # 5. Reset with reseed
        res_reseed = await client.post("/api/admin/demo/reset?reseed=true")
        assert res_reseed.status_code == 200
        assert is_demo_seeded()


# -----------------------------------------------------------------------------
# Test K: Live session isolation from demo sample state
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_live_session_isolation():
    seed_demo_data()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Start a brand new live farmer session
        live_farmer_id = "farmer_live_456"
        res = await client.post(
            "/agent/message",
            json={
                "sender_id": live_farmer_id,
                "channel": "web_chat",
                "text": "I have 100 kg of cabbage in Nakuru available tomorrow morning",
            },
        )
        assert res.status_code == 200
        live_reply = res.json()
        assert "reply" in live_reply
        assert len(live_reply["reply"]) > 0

        # Verify demo hero context is completely unaffected
        res_hero = await client.get(f"/api/farmer/context/{HERO_FARMER_ID}")
        hero_data = res_hero.json()
        assert hero_data["state"] == "COMPLETED"
        assert hero_data["order"]["id"] == HERO_ORDER_ID
        assert len(hero_data["history"]) == 10

        # Verify live farmer context exists independently
        res_live = await client.get(f"/api/farmer/context/{live_farmer_id}")
        live_data = res_live.json()
        assert live_data["sender_id"] == live_farmer_id
        assert live_data["offer"]["produce_type"] == "cabbage"


# -----------------------------------------------------------------------------
# Test L: Rule-based fallback with Gemini offline
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_rule_based_fallback_with_gemini_offline(monkeypatch):
    """Verifies that when Gemini is offline or unavailable, live messages continue seamlessly."""
    # Force Gemini to fail by overriding or setting an invalid key or mocking
    from orca.agent.intent.rule_based_classifier import RuleBasedIntentClassifier
    
    # Configure orchestrator with purely rule-based classifier (simulating offline fallback)
    rb_classifier = RuleBasedIntentClassifier()
    monkeypatch.setattr(orchestrator, "intent_classifier", rb_classifier)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/agent/message",
            json={
                "sender_id": "farmer_offline_test",
                "channel": "web_chat",
                "text": "I have 50 kg of tomatoes in Nairobi",
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert "reply" in data
        assert "$0.60" in data["reply"] or "tomato" in data["reply"].lower()
