"""Unit tests for Demo State Reset (POST /api/admin/demo/reset).

Verifies:
- Reset from populated state clears all orders, payments, tasks, and conversations
- Reset can be executed repeatedly (idempotent / safe)
- Zero leftover orders, payments, tasks, or conversations in API endpoints
- Baseline configured rates remain available and authoritative after reset
"""

import pytest
from httpx import AsyncClient, ASGITransport
from orca.api.app import app
from orca.services.order import order_service
from orca.services.payment import payment_service
from orca.services.collection import collection_service
from orca.services.pricing import pricing_service
from orca.agent.orchestrator import orchestrator


@pytest.fixture
async def client():
    """Async HTTP client fixture configured for the FastAPI app."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.mark.asyncio
async def test_demo_reset_from_populated_state(client: AsyncClient):
    """Test resetting demo state when orders, payments, tasks, and contexts exist."""
    farmer_id = "reset_farmer_01"

    # Conversational turn
    await client.post(
        "/agent/message",
        json={
            "sender_id": farmer_id,
            "channel": "web_chat",
            "text": "I have 50 kg of potatoes in Nairobi available tomorrow at 10 AM.",
        },
    )

    # Confirm order
    await client.post(
        "/agent/message",
        json={"sender_id": farmer_id, "channel": "web_chat", "text": "Confirm"},
    )

    # Verify state is populated
    assert len(order_service.get_all_orders()) >= 1
    assert len(orchestrator._conversations) >= 1

    # 2. Execute Demo Reset
    res = await client.post("/api/admin/demo/reset")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "successfully reset" in data["message"].lower()

    # 3. Verify zero leftover records in memory and services
    assert len(order_service.get_all_orders()) == 0
    assert len(payment_service.get_all_payments()) == 0
    assert len(collection_service.get_all_tasks()) == 0
    assert len(orchestrator._conversations) == 0

    # 4. Verify baseline rates remain available
    assert "potato" in pricing_service.get_supported_produce()
    assert "tomato" in pricing_service.get_supported_produce()
    rate = pricing_service.get_applicable_rate("potato")
    assert rate is not None
    assert rate.rate_per_unit == 0.40


@pytest.mark.asyncio
async def test_demo_reset_twice(client: AsyncClient):
    """Test that resetting demo state multiple times in succession is safe and deterministic."""
    res1 = await client.post("/api/admin/demo/reset")
    assert res1.status_code == 200
    assert res1.json()["status"] == "success"

    res2 = await client.post("/api/admin/demo/reset")
    assert res2.status_code == 200
    assert res2.json()["status"] == "success"

    assert len(order_service.get_all_orders()) == 0
    assert len(payment_service.get_all_payments()) == 0
    assert len(collection_service.get_all_tasks()) == 0
    assert len(orchestrator._conversations) == 0


@pytest.mark.asyncio
async def test_zero_leftover_in_api_endpoints(client: AsyncClient):
    """Verify that after reset, all admin and runner endpoints reflect clean zero state."""
    await client.post("/api/admin/demo/reset")

    # Admin stats
    stats_res = await client.get("/api/admin/stats")
    assert stats_res.status_code == 200
    stats = stats_res.json()
    assert stats["total_orders"] == 0
    assert stats["total_gmv"] == 0.0
    assert stats["completed_orders"] == 0
    assert stats["pending_collection"] == 0
    assert stats["available_runner_tasks"] == 0

    # Admin orders
    orders_res = await client.get("/api/admin/orders")
    assert orders_res.status_code == 200
    orders_data = orders_res.json()
    assert orders_data["total"] == 0
    assert orders_data["orders"] == []

    # Runner tasks
    tasks_res = await client.get("/api/runner/tasks/available")
    assert tasks_res.status_code == 200
    assert tasks_res.json()["tasks"] == []

    # Farmer context
    farmer_res = await client.get("/api/farmer/context/farmer_001")
    assert farmer_res.status_code == 200
    assert farmer_res.json()["found"] is False


@pytest.mark.asyncio
async def test_baseline_rates_available_after_reset(client: AsyncClient):
    """Verify rate and produce discovery endpoints work immediately after reset."""
    await client.post("/api/admin/demo/reset")

    produce_res = await client.get("/produce")
    assert produce_res.status_code == 200
    produce_list = produce_res.json()["supported_produce"]
    assert "potato" in produce_list
    assert "tomato" in produce_list
    assert "onion" in produce_list

    rate_res = await client.get("/rates?produce=potato")
    assert rate_res.status_code == 200
    rate_data = rate_res.json()
    assert rate_data["found"] is True
    assert rate_data["rate"]["rate_per_unit"] == 0.40
