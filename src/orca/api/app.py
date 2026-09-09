"""FastAPI application factory and foundation routes."""

from fastapi import FastAPI
from orca.core.config import settings
from orca.domain.schemas import InboundMessage
from orca.services.pricing import pricing_service


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Global AI Agricultural Procurement Agent API",
    )

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
        """Process normalized inbound conversation message (SRS Section 14)."""
        return {
            "status": "received",
            "sender_id": message.sender_id,
            "channel": message.channel,
            "message": "Message received by agent orchestrator foundation.",
        }

    return app


app = create_app()
