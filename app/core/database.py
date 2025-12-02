from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from app.core.config import get_settings


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


class PostgresDatabase:
    """PostgreSQL database connection manager."""
    
    def __init__(self):
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None
    
    @property
    def engine(self) -> AsyncEngine:
        """Get the database engine, creating it if necessary."""
        if self._engine is None:
            raise RuntimeError("Database not initialized. Call connect() first.")
        return self._engine
    
    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Get the session factory."""
        if self._session_factory is None:
            raise RuntimeError("Database not initialized. Call connect() first.")
        return self._session_factory
    
    async def connect(self, use_pool: bool = True) -> None:
        """
        Initialize database connection and session factory.
        
        Args:
            use_pool: Whether to use connection pooling. Set False for testing.
        """
        settings = get_settings()
        
        pool_config = {}
        if use_pool:
            pool_config = {
                "pool_size": settings.POSTGRES_POOL_SIZE,
                "max_overflow": settings.POSTGRES_MAX_OVERFLOW,
                "pool_pre_ping": True,
            }
        else:
            pool_config = {"poolclass": NullPool}
        
        self._engine = create_async_engine(
            settings.postgres_url,
            echo=settings.DEBUG,
            **pool_config,
        )
        
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    
    async def disconnect(self) -> None:
        """Close database connection and cleanup resources."""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
    
    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Async context manager for database sessions.
        
        Yields:
            AsyncSession: Database session with automatic commit/rollback.
        
        Example:
            async with db.session() as session:
                result = await session.execute(query)
        """
        session = self.session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# Global database instance
postgres_db = PostgresDatabase()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for FastAPI to get database session.
    
    Yields:
        AsyncSession: Database session for request handling.
    """
    async with postgres_db.session() as session:
        yield session
