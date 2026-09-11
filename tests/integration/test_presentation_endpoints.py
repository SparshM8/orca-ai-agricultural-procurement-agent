"""Integration tests for Presentation and Demo Interface REST endpoints.

Verifies:
- Farmer Chat API & context endpoint (/api/farmer/context/{sender_id})
- Runner Console APIs (available, assigned, accept, reject, confirm-pickup, fail, reschedule)
- Admin Dashboard APIs (/api/admin/orders, /api/admin/orders/{id}, /api/admin/stats, tasks, payments)
- Web UI serving endpoint (/)
"""

import pytest
from httpx import AsyncClient, ASGITransport
from orca.api.app import app
from orca.domain.state_machine import OrderState


@pytest.fixture
async def client():
    """Async HTTP client fixture configured for the FastAPI app."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.mark.asyncio
async def test_ui_root_endpoint_serves_html(client: AsyncClient):
    """Verify GET / returns 200 OK with HTML content containing all three roles."""
    response = await client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    text = response.text
    assert "ORCA" in text
    assert "Farmer Chat" in text
    assert "Runner Console" in text
    assert "Admin Dashboard" in text
    assert "Reset Demo Data" in text
    assert "Agent Trace & Decision Audit" in text
    assert "typing-bubble" in text
    assert "runner-action-modal" in text
    assert "Human Escalations" in text
    assert "admin-subtabs" in text


@pytest.mark.asyncio
async def test_farmer_chat_and_context_sync(client: AsyncClient):
    """Verify full farmer interaction reflects accurately in /api/farmer/context."""
    farmer_id = "farmer_test_chat_01"

    # Turn 1: Offer submission
    offer_payload = {
        "sender_id": farmer_id,
        "channel": "web_chat",
        "text": "I have 50 kg of potatoes in Nairobi available tomorrow at 10 AM.",
    }
    r1 = await client.post("/agent/message", json=offer_payload)
    assert r1.status_code == 200
    assert "USD 0.40" in r1.json()["reply"]
    assert "- Pickup Availability: tomorrow at 10:00 AM" in r1.json()["reply"]

    # Check context endpoint
    c1 = await client.get(f"/api/farmer/context/{farmer_id}")
    assert c1.status_code == 200
    ctx_data = c1.json()
    assert ctx_data["found"] is True
    assert ctx_data["state"] == "AWAITING_FARMER_CONFIRMATION"
    assert ctx_data["offer"]["produce_type"] == "potato"
    assert ctx_data["offer"]["quantity"] == 50.0
    assert ctx_data["offer"]["availability_window"] == "tomorrow at 10:00 AM"

    # Turn 2: Farmer Confirms
    r2 = await client.post(
        "/agent/message",
        json={"sender_id": farmer_id, "channel": "web_chat", "text": "Confirm"},
    )
    assert r2.status_code == 200
    assert "Order Confirmed" in r2.json()["reply"]
    assert "- Scheduled Window: tomorrow at 10:00 AM" in r2.json()["reply"]
    order_id = r2.json()["metadata"]["order_id"]

    # Check context endpoint reflects PAYMENT_PENDING
    c2 = await client.get(f"/api/farmer/context/{farmer_id}")
    assert c2.json()["state"] == "PAYMENT_PENDING"
    assert c2.json()["current_order_id"] == order_id
    assert c2.json()["order"]["pickup_time_str"] == "tomorrow at 10:00 AM"
    assert c2.json()["payment"]["status"] == "PENDING"

    # Turn 3: Farmer Pays
    r3 = await client.post(
        "/agent/message",
        json={"sender_id": farmer_id, "channel": "web_chat", "text": "Pay"},
    )
    assert r3.status_code == 200
    assert "Payment Confirmed" in r3.json()["reply"]
    assert "- Scheduled Window: tomorrow at 10:00 AM" in r3.json()["reply"]

    # Check context endpoint reflects COLLECTION_PENDING and SUCCESS payment
    c3 = await client.get(f"/api/farmer/context/{farmer_id}")
    assert c3.json()["state"] == "COLLECTION_PENDING"
    assert c3.json()["payment"]["status"] == "SUCCESS"
    assert c3.json()["collection_task"]["status"] == "PENDING"
    assert c3.json()["collection_task"]["scheduled_time_str"] == "tomorrow at 10:00 AM"


@pytest.mark.asyncio
async def test_runner_console_lifecycle_flow(client: AsyncClient):
    """Verify runner console operations: view available, accept, reject, confirm-pickup."""
    farmer_id = "farmer_runner_test_01"

    # Setup an order ready for collection
    await client.post(
        "/agent/message",
        json={
            "sender_id": farmer_id,
            "channel": "web_chat",
            "text": "I have 30 kg of tomatoes in Nairobi ready tomorrow 2pm",
        },
    )
    await client.post(
        "/agent/message",
        json={"sender_id": farmer_id, "channel": "web_chat", "text": "Confirm"},
    )
    await client.post(
        "/agent/message",
        json={"sender_id": farmer_id, "channel": "web_chat", "text": "Pay"},
    )

    # 1. Runner checks available tasks
    r_avail = await client.get("/api/runner/tasks/available")
    assert r_avail.status_code == 200
    tasks = r_avail.json()["tasks"]
    matching_tasks = [t for t in tasks if t["produce_type"] == "tomato" and t["quantity"] == 30.0]
    assert len(matching_tasks) >= 1
    target_task = matching_tasks[0]
    task_id = target_task["id"]

    # 2. Get task detail
    r_detail = await client.get(f"/api/runner/tasks/{task_id}")
    assert r_detail.status_code == 200
    assert r_detail.json()["task"]["id"] == task_id
    assert r_detail.json()["order"]["id"] == target_task["order_id"]

    # 3. Runner accepts task
    r_accept = await client.post(
        f"/api/runner/tasks/{task_id}/accept",
        json={"runner_id": "Runner-01"},
    )
    assert r_accept.status_code == 200
    assert r_accept.json()["status"] == "accepted"
    assert r_accept.json()["task"]["status"] == "ASSIGNED"
    assert r_accept.json()["task"]["runner_id"] == "Runner-01"

    # 4. Check assigned tasks
    r_assigned = await client.get("/api/runner/tasks/assigned?runner_id=Runner-01")
    assert r_assigned.status_code == 200
    assigned_ids = [t["id"] for t in r_assigned.json()["tasks"]]
    assert task_id in assigned_ids

    # 5. Runner rejects task -> safely reverts to PENDING
    r_reject = await client.post(
        f"/api/runner/tasks/{task_id}/reject",
        json={"runner_id": "Runner-01", "reason": "Flat tire"},
    )
    assert r_reject.status_code == 200
    assert r_reject.json()["status"] == "rejected"
    assert r_reject.json()["task"]["status"] == "PENDING"
    assert r_reject.json()["task"]["runner_id"] is None

    # 6. Another runner accepts task
    r_accept2 = await client.post(
        f"/api/runner/tasks/{task_id}/accept",
        json={"runner_id": "Runner-02"},
    )
    assert r_accept2.status_code == 200
    assert r_accept2.json()["task"]["runner_id"] == "Runner-02"

    # 7. Runner confirms pickup -> moves order to COMPLETED
    r_confirm = await client.post(
        f"/api/runner/tasks/{task_id}/confirm-pickup",
        json={"runner_id": "Runner-02"},
    )
    assert r_confirm.status_code == 200
    assert r_confirm.json()["status"] == "completed"
    assert r_confirm.json()["task"]["status"] == "COMPLETED"


@pytest.mark.asyncio
async def test_runner_failure_and_reschedule_flow(client: AsyncClient):
    """Verify runner reporting failure moves order to EXCEPTION, and rescheduling restores PENDING."""
    farmer_id = "farmer_runner_fail_01"

    await client.post(
        "/agent/message",
        json={
            "sender_id": farmer_id,
            "channel": "web_chat",
            "text": "I have 40 kg of onions in Nairobi ready tomorrow 11am",
        },
    )
    await client.post(
        "/agent/message",
        json={"sender_id": farmer_id, "channel": "web_chat", "text": "Confirm"},
    )
    await client.post(
        "/agent/message",
        json={"sender_id": farmer_id, "channel": "web_chat", "text": "Pay"},
    )

    r_avail = await client.get("/api/runner/tasks/available")
    tasks = [t for t in r_avail.json()["tasks"] if t["produce_type"] == "onion"]
    task_id = tasks[0]["id"]

    # Accept
    await client.post(f"/api/runner/tasks/{task_id}/accept", json={"runner_id": "Runner-01"})

    # Report failure without reason rejected
    bad_fail = await client.post(f"/api/runner/tasks/{task_id}/fail", json={"runner_id": "Runner-01"})
    assert bad_fail.status_code == 400

    # Report failure with reason
    r_fail = await client.post(
        f"/api/runner/tasks/{task_id}/fail",
        json={"runner_id": "Runner-01", "reason": "Farmer produce was wet and unharvested"},
    )
    assert r_fail.status_code == 200
    assert r_fail.json()["task"]["status"] == "FAILED"
    assert r_fail.json()["task"]["failure_reason"] == "Farmer produce was wet and unharvested"

    # Reschedule task
    r_resched = await client.post(
        f"/api/runner/tasks/{task_id}/reschedule",
        json={"new_time_str": "Next Monday at 9am"},
    )
    assert r_resched.status_code == 200
    assert r_resched.json()["task"]["status"] == "PENDING"
    assert r_resched.json()["task"]["scheduled_time_str"] == "Next Monday at 9am"


@pytest.mark.asyncio
async def test_admin_dashboard_orders_and_audit(client: AsyncClient):
    """Verify admin orders listing, filtering, order detail audit with transcript and progression."""
    # Fetch all orders
    r_orders = await client.get("/api/admin/orders")
    assert r_orders.status_code == 200
    data = r_orders.json()
    assert "total" in data
    assert "orders" in data
    assert len(data["orders"]) >= 1

    first_order = data["orders"][0]
    order_id = first_order["id"]

    # Fetch detailed order audit
    r_detail = await client.get(f"/api/admin/orders/{order_id}")
    assert r_detail.status_code == 200
    detail_data = r_detail.json()

    assert detail_data["order"]["id"] == order_id
    assert "progression" in detail_data
    assert len(detail_data["progression"]) >= 5
    # Progression includes OFFER_RECEIVED, ORDER_CONFIRMED, PAYMENT_PENDING, etc.
    assert detail_data["progression"][0]["state"] == "OFFER_RECEIVED"

    # Admin stats
    r_stats = await client.get("/api/admin/stats")
    assert r_stats.status_code == 200
    stats = r_stats.json()
    assert stats["total_orders"] >= 1
    assert stats["total_gmv"] >= 0.0

    # Admin tasks
    r_tasks = await client.get("/api/admin/tasks")
    assert r_tasks.status_code == 200
    assert "tasks" in r_tasks.json()

    # Admin payments
    r_payments = await client.get("/api/admin/payments")
    assert r_payments.status_code == 200
    assert "payments" in r_payments.json()


@pytest.mark.asyncio
async def test_admin_handoff_endpoints_and_farmer_context_sync(client: AsyncClient):
    """Verify human handoff creation in chat reflects in farmer context and admin endpoints."""
    farmer_id = "farmer_presentation_handoff_01"

    # Farmer requests human assistance
    msg_res = await client.post(
        "/agent/message",
        json={"sender_id": farmer_id, "channel": "web_chat", "text": "I want to talk to a human"},
    )
    assert msg_res.status_code == 200
    reply = msg_res.json()["reply"]
    assert "Case ID:" in reply or "case" in reply.lower()

    # Check farmer context reflects handoff case
    ctx_res = await client.get(f"/api/farmer/context/{farmer_id}")
    assert ctx_res.status_code == 200
    ctx_data = ctx_res.json()
    assert ctx_data["handoff_case"] is not None
    case_id = ctx_data["handoff_case"]["case_id"]
    assert ctx_data["handoff_case"]["status"] == "OPEN"
    assert ctx_data["handoff_case"]["reason"] == "FARMER_REQUEST"

    # Check admin list endpoint
    list_res = await client.get("/api/admin/handoffs")
    assert list_res.status_code == 200
    cases = list_res.json()["cases"]
    assert any(c["case_id"] == case_id for c in cases)

    # Check admin single case endpoint
    case_res = await client.get(f"/api/admin/handoffs/{case_id}")
    assert case_res.status_code == 200
    assert case_res.json()["case_id"] == case_id

    # Resolve case via admin
    resolve_res = await client.post(
        f"/api/admin/handoffs/{case_id}/resolve",
        json={"notes": "Resolved during presentation verification"},
    )
    assert resolve_res.status_code == 200
    assert resolve_res.json()["case"]["status"] == "RESOLVED"

    # Stats reflect 0 open handoffs
    stats_res = await client.get("/api/admin/stats")
    assert stats_res.status_code == 200
    assert stats_res.json()["open_handoffs"] == 0
