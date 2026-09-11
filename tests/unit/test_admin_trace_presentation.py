"""Unit tests for Phase 2: Exposing safe AgentTrace in Admin Order inspection.

Verifies:
- Traces are retrievable through GET /api/admin/orders/{order_id}
- Safe diagnostic fields (trace_id, timestamp, intent, confidence, classifier, fallback, state) are present
- Private/internal secrets (API keys, tokens, raw passwords) are never exposed
- Trace remains observational and does not alter business or order state
- Existing order endpoint response structure remains backward-compatible
"""

import pytest
from httpx import AsyncClient, ASGITransport
from orca.api.app import app
from orca.agent.orchestrator import orchestrator
from orca.domain.trace import AgentTrace


@pytest.fixture
async def client():
    """Async HTTP client fixture configured for the FastAPI app."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.mark.asyncio
async def test_admin_order_traces_retrievable_and_safe(client: AsyncClient):
    """Verify GET /api/admin/orders/{order_id} includes safe traces with zero leaked credentials."""
    await client.post("/api/admin/demo/reset")
    farmer_id = "trace_test_farmer_01"

    # Inbound message
    r1 = await client.post(
        "/agent/message",
        json={
            "sender_id": farmer_id,
            "channel": "web_chat",
            "text": "I have 50 kg of potatoes in Nairobi available tomorrow at 10 AM.",
        },
    )
    assert r1.status_code == 200

    # Confirm order
    r2 = await client.post(
        "/agent/message",
        json={"sender_id": farmer_id, "channel": "web_chat", "text": "Confirm"},
    )
    assert r2.status_code == 200

    # Get farmer context to find order ID
    ctx_res = await client.get(f"/api/farmer/context/{farmer_id}")
    assert ctx_res.status_code == 200
    order_id = ctx_res.json()["current_order_id"]
    assert order_id is not None

    # Retrieve order detail from Admin API
    order_detail_res = await client.get(f"/api/admin/orders/{order_id}")
    assert order_detail_res.status_code == 200
    detail = order_detail_res.json()

    # Verify existing compatibility
    assert "order" in detail
    assert detail["order"]["id"] == order_id
    assert "progression" in detail
    assert "payment" in detail
    assert "collection_task" in detail
    assert "conversation" in detail

    # Verify traces are present and structured
    assert "traces" in detail
    traces = detail["traces"]
    assert len(traces) >= 2

    # Check diagnostic fields on each trace
    for t in traces:
        assert "trace_id" in t
        assert t["trace_id"].startswith("trc_")
        assert "timestamp" in t
        assert "intent" in t
        assert "confidence" in t
        assert "classifier" in t
        assert "fallback_occurred" in t
        assert "state_before" in t
        assert "state_after" in t

        # Verify no secret leakage
        t_str = str(t).lower()
        assert "api_key" not in t_str or "[redacted" in t_str
        assert "secret" not in t_str or "[redacted" in t_str
        assert "bearer" not in t_str or "[redacted" in t_str


@pytest.mark.asyncio
async def test_trace_is_observational_only(client: AsyncClient):
    """Verify that adding traces does not mutate order state, total amounts, or business calculation."""
    await client.post("/api/admin/demo/reset")
    farmer_id = "trace_observational_farmer"

    await client.post(
        "/agent/message",
        json={
            "sender_id": farmer_id,
            "channel": "web_chat",
            "text": "I have 50 kg of potatoes in Nairobi available tomorrow at 10 AM.",
        },
    )
    await client.post(
        "/agent/message",
        json={"sender_id": farmer_id, "channel": "web_chat", "text": "Confirm"},
    )

    ctx_res = await client.get(f"/api/farmer/context/{farmer_id}")
    order_id = ctx_res.json()["current_order_id"]

    # Inject a trace with different state in memory to test observational nature
    conv_id = f"conv_{farmer_id}"
    context = orchestrator.get_context(conv_id)
    assert context is not None

    fake_trace = AgentTrace.create_sanitized(
        conversation_id=conv_id,
        sender_id=farmer_id,
        state_before="ORDER_CONFIRMED",
        state_after="COMPLETED",  # Fake claim in trace
        input_text="Simulated message",
        detected_intent="CONFIRM_ORDER",
    )
    context.add_trace(fake_trace)

    # Re-fetch order from admin API: order status must still be ORDER_CONFIRMED (or PAYMENT_PENDING)
    detail_res = await client.get(f"/api/admin/orders/{order_id}")
    detail = detail_res.json()
    assert detail["order"]["status"] in ["ORDER_CONFIRMED", "PAYMENT_PENDING"]
    # The trace records the simulated turn without changing order.status
    matching_trace = next((t for t in detail["traces"] if t["trace_id"] == fake_trace.trace_id), None)
    assert matching_trace is not None
    assert matching_trace["state_after"] == "COMPLETED"
    assert detail["order"]["status"] != "COMPLETED"
