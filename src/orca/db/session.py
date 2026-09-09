"""Database connection and session factory (SQLAlchemy 2.0)."""

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from orca.core.config import settings


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy declarative database models."""
    pass


# Async engine supporting SQLite (local dev/test) and PostgreSQL (production)
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True,
)

async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency injection yield for database session."""
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()
