"""Database repository and persistence operations using SQLAlchemy 2.0 async sessions."""

from datetime import datetime
from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine, async_sessionmaker
from orca.db.session import Base
from orca.db.models import RateModel, OrderModel, PaymentModel, CollectionTaskModel


async def init_db(engine: AsyncEngine) -> None:
    """Create all tables and seed initial baseline rates if needed."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Seed baseline rates if empty using sessionmaker bound to the given engine
    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with session_factory() as session:
        result = await session.execute(select(RateModel))
        existing_rates = result.scalars().all()
        if not existing_rates:
            seed_rates = [
                RateModel(
                    id="rate_potato_001",
                    produce_type="potato",
                    region_code="GLOBAL_DEFAULT",
                    rate_per_unit=0.40,
                    unit="kg",
                    currency="USD",
                    source="CONFIGURED_BASELINE",
                    is_active=True,
                ),
                RateModel(
                    id="rate_tomato_001",
                    produce_type="tomato",
                    region_code="GLOBAL_DEFAULT",
                    rate_per_unit=0.60,
                    unit="kg",
                    currency="USD",
                    source="CONFIGURED_BASELINE",
                    is_active=True,
                ),
                RateModel(
                    id="rate_onion_001",
                    produce_type="onion",
                    region_code="GLOBAL_DEFAULT",
                    rate_per_unit=0.35,
                    unit="kg",
                    currency="USD",
                    source="CONFIGURED_BASELINE",
                    is_active=True,
                ),
            ]
            session.add_all(seed_rates)
            await session.commit()


async def reset_db(engine: AsyncEngine) -> None:
    """Clear all transactional records and reseed baseline rates safely."""
    from sqlalchemy import delete

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with session_factory() as session:
        await session.execute(delete(PaymentModel))
        await session.execute(delete(CollectionTaskModel))
        await session.execute(delete(OrderModel))
        await session.execute(delete(RateModel))

        seed_rates = [
            RateModel(
                id="rate_potato_001",
                produce_type="potato",
                region_code="GLOBAL_DEFAULT",
                rate_per_unit=0.40,
                unit="kg",
                currency="USD",
                source="CONFIGURED_BASELINE",
                is_active=True,
            ),
            RateModel(
                id="rate_tomato_001",
                produce_type="tomato",
                region_code="GLOBAL_DEFAULT",
                rate_per_unit=0.60,
                unit="kg",
                currency="USD",
                source="CONFIGURED_BASELINE",
                is_active=True,
            ),
            RateModel(
                id="rate_onion_001",
                produce_type="onion",
                region_code="GLOBAL_DEFAULT",
                rate_per_unit=0.35,
                unit="kg",
                currency="USD",
                source="CONFIGURED_BASELINE",
                is_active=True,
            ),
        ]
        session.add_all(seed_rates)
        await session.commit()


class OrderRepository:
    """Persistence repository for Orders."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, order_model: OrderModel) -> OrderModel:
        """Persist or update an order."""
        self.session.add(order_model)
        await self.session.commit()
        await self.session.refresh(order_model)
        return order_model

    async def get_by_id(self, order_id: str) -> Optional[OrderModel]:
        """Fetch order by primary key ID."""
        result = await self.session.execute(
            select(OrderModel).where(OrderModel.id == order_id)
        )
        return result.scalars().first()

    async def get_by_conversation_id(self, conversation_id: str) -> Optional[OrderModel]:
        """Fetch order by conversation ID (for idempotency)."""
        result = await self.session.execute(
            select(OrderModel).where(OrderModel.conversation_id == conversation_id)
        )
        return result.scalars().first()

    async def update_status(self, order_id: str, new_status: str) -> Optional[OrderModel]:
        """Update status of existing order."""
        order = await self.get_by_id(order_id)
        if order:
            order.status = new_status
            await self.session.commit()
            await self.session.refresh(order)
        return order

    async def get_all(self, status: Optional[str] = None) -> List[OrderModel]:
        """Fetch all orders, optionally filtered by status, ordered by created_at desc."""
        query = select(OrderModel).order_by(OrderModel.created_at.desc())
        if status:
            query = query.where(OrderModel.status == status)
        result = await self.session.execute(query)
        return list(result.scalars().all())


class RateRepository:
    """Persistence repository for authoritative Rates."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_applicable_rate(
        self, produce_type: str, region_code: str = "GLOBAL_DEFAULT"
    ) -> Optional[RateModel]:
        """Retrieve authoritative rate from database."""
        result = await self.session.execute(
            select(RateModel).where(
                RateModel.produce_type == produce_type.lower(),
                RateModel.region_code == region_code,
                RateModel.is_active == True,
            )
        )
        rate = result.scalars().first()
        if not rate and region_code != "GLOBAL_DEFAULT":
            # Fallback to global default
            result = await self.session.execute(
                select(RateModel).where(
                    RateModel.produce_type == produce_type.lower(),
                    RateModel.region_code == "GLOBAL_DEFAULT",
                    RateModel.is_active == True,
                )
            )
            rate = result.scalars().first()
        return rate

    async def get_supported_produce(self, region_code: str = "GLOBAL_DEFAULT") -> List[str]:
        """Get all supported produce names for a region."""
        result = await self.session.execute(
            select(RateModel.produce_type).where(
                RateModel.region_code == region_code,
                RateModel.is_active == True,
            )
        )
        return sorted(list(set(result.scalars().all())))


class PaymentRepository:
    """Persistence repository for Payments."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, payment_model: PaymentModel) -> PaymentModel:
        """Persist or update a payment."""
        self.session.add(payment_model)
        await self.session.commit()
        await self.session.refresh(payment_model)
        return payment_model

    async def get_by_id(self, payment_id: str) -> Optional[PaymentModel]:
        """Fetch payment by ID."""
        result = await self.session.execute(
            select(PaymentModel).where(PaymentModel.id == payment_id)
        )
        return result.scalars().first()

    async def get_by_order_id(self, order_id: str) -> Optional[PaymentModel]:
        """Fetch payment by order ID (for idempotency)."""
        result = await self.session.execute(
            select(PaymentModel).where(PaymentModel.order_id == order_id)
        )
        return result.scalars().first()

    async def update_status(
        self, payment_id: str, new_status: str, provider_reference: Optional[str] = None
    ) -> Optional[PaymentModel]:
        """Update payment status and provider reference."""
        payment = await self.get_by_id(payment_id)
        if payment:
            payment.status = new_status
            if provider_reference:
                payment.provider_reference = provider_reference
            await self.session.commit()
            await self.session.refresh(payment)
        return payment

    async def get_all(self) -> List[PaymentModel]:
        """Fetch all payments ordered by timestamp desc."""
        result = await self.session.execute(
            select(PaymentModel).order_by(PaymentModel.timestamp.desc())
        )
        return list(result.scalars().all())


class CollectionRepository:
    """Persistence repository for Collection Tasks."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def save(self, task_model: CollectionTaskModel) -> CollectionTaskModel:
        """Persist or update a collection task."""
        self.session.add(task_model)
        await self.session.commit()
        await self.session.refresh(task_model)
        return task_model

    async def get_by_id(self, task_id: str) -> Optional[CollectionTaskModel]:
        """Fetch task by ID."""
        result = await self.session.execute(
            select(CollectionTaskModel).where(CollectionTaskModel.id == task_id)
        )
        return result.scalars().first()

    async def get_by_order_id(self, order_id: str) -> Optional[CollectionTaskModel]:
        """Fetch task by order ID (for idempotency)."""
        result = await self.session.execute(
            select(CollectionTaskModel).where(CollectionTaskModel.order_id == order_id)
        )
        return result.scalars().first()

    async def get_available_tasks(self) -> List[CollectionTaskModel]:
        """Fetch all tasks available for runner pickup (PENDING)."""
        result = await self.session.execute(
            select(CollectionTaskModel).where(CollectionTaskModel.status == "PENDING")
        )
        return list(result.scalars().all())

    async def update_status(
        self,
        task_id: str,
        status: str,
        runner_id: Optional[str] = None,
        assigned_at: Optional[datetime] = None,
        picked_up_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        failure_reason: Optional[str] = None,
        scheduled_datetime: Optional[datetime] = None,
        scheduled_time_str: Optional[str] = None,
    ) -> Optional[CollectionTaskModel]:
        """Update status and timestamps for a collection task."""
        task = await self.get_by_id(task_id)
        if task:
            task.status = status
            if runner_id is not None:
                task.runner_id = runner_id if runner_id != "" else None
            if assigned_at is not None:
                task.assigned_at = assigned_at
            if picked_up_at is not None:
                task.picked_up_at = picked_up_at
            if completed_at is not None:
                task.completed_at = completed_at
            if failure_reason is not None:
                task.failure_reason = failure_reason
            if scheduled_datetime is not None:
                task.scheduled_datetime = scheduled_datetime
            if scheduled_time_str is not None:
                task.scheduled_time_str = scheduled_time_str
            await self.session.commit()
            await self.session.refresh(task)
        return task

    async def get_all(self, status: Optional[str] = None) -> List[CollectionTaskModel]:
        """Fetch all collection tasks, optionally filtered by status."""
        query = select(CollectionTaskModel).order_by(CollectionTaskModel.created_at.desc())
        if status:
            query = query.where(CollectionTaskModel.status == status)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_assigned_tasks(self, runner_id: str) -> List[CollectionTaskModel]:
        """Fetch tasks assigned to a specific runner."""
        result = await self.session.execute(
            select(CollectionTaskModel)
            .where(CollectionTaskModel.runner_id == runner_id)
            .order_by(CollectionTaskModel.created_at.desc())
        )
        return list(result.scalars().all())


