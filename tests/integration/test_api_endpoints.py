"""Integration tests verifying FastAPI application routes."""

import pytest
from httpx import AsyncClient, ASGITransport
from orca.api.app import app


@pytest.fixture
async def client():
    """Async HTTP client fixture configured for the FastAPI app."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient):
    """Verify /health endpoint returns healthy status and metadata."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "ORCA" in data["app"]
    assert "version" in data


@pytest.mark.asyncio
async def test_produce_endpoint(client: AsyncClient):
    """Verify /produce endpoint returns supported produce."""
    response = await client.get("/produce")
    assert response.status_code == 200
    data = response.json()
    assert "supported_produce" in data
    assert "potato" in data["supported_produce"]
    assert "tomato" in data["supported_produce"]


@pytest.mark.asyncio
async def test_rates_endpoint_found(client: AsyncClient):
    """Verify /rates endpoint returns authoritative rate when produce is supported."""
    response = await client.get("/rates?produce=potato")
    assert response.status_code == 200
    data = response.json()
    assert data["found"] is True
    assert data["rate"]["produce_type"] == "potato"
    assert data["rate"]["rate_per_unit"] == 0.40
    assert data["rate"]["currency"] == "USD"


@pytest.mark.asyncio
async def test_rates_endpoint_not_found(client: AsyncClient):
    """Verify /rates endpoint returns not found when produce is unconfigured."""
    response = await client.get("/rates?produce=unknown_fruit")
    assert response.status_code == 200
    data = response.json()
    assert data["found"] is False


@pytest.mark.asyncio
async def test_agent_message_endpoint(client: AsyncClient):
    """Verify /agent/message processes inbound normalized message."""
    payload = {
        "sender_id": "+1234567890",
        "channel": "whatsapp",
        "text": "I have 20 kg of potatoes available",
    }
    response = await client.post("/agent/message", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "received"
    assert data["sender_id"] == "+1234567890"
    assert data["channel"] == "whatsapp"
