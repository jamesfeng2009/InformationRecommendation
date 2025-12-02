import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from app.core.clickhouse import clickhouse_client
from app.core.database import postgres_db
from app.core.elasticsearch import elasticsearch_client
from app.core.redis import redis_client

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages all database connections for the application.
    
    Provides centralized startup and shutdown for:
    - PostgreSQL (via asyncpg)
    - Elasticsearch
    - Redis (via redis-py async)
    - ClickHouse
    """
    
    async def connect_all(self) -> None:
        """
        Initialize all database connections.
        
        Should be called during application startup.
        """
        logger.info("Initializing database connections...")
        
        try:
            await postgres_db.connect()
            logger.info("PostgreSQL connection established")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
        
        try:
            await elasticsearch_client.connect()
            logger.info("Elasticsearch connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            raise
        
        try:
            await redis_client.connect()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

        try:
            await clickhouse_client.connect()
            logger.info("ClickHouse connection established")
        except Exception as e:
            logger.error(f"Failed to connect to ClickHouse: {e}")
            raise
        
        logger.info("All database connections initialized successfully")
    
    async def disconnect_all(self) -> None:
        """
        Close all database connections.
        
        Should be called during application shutdown.
        """
        logger.info("Closing database connections...")
        
        try:
            await clickhouse_client.disconnect()
            logger.info("ClickHouse connection closed")
        except Exception as e:
            logger.warning(f"Error closing ClickHouse connection: {e}")
        
        try:
            await redis_client.disconnect()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.warning(f"Error closing Redis connection: {e}")
        
        try:
            await elasticsearch_client.disconnect()
            logger.info("Elasticsearch connection closed")
        except Exception as e:
            logger.warning(f"Error closing Elasticsearch connection: {e}")
        
        try:
            await postgres_db.disconnect()
            logger.info("PostgreSQL connection closed")
        except Exception as e:
            logger.warning(f"Error closing PostgreSQL connection: {e}")
        
        logger.info("All database connections closed")
    
    async def health_check(self) -> dict[str, bool]:
        """
        Check health of all database connections.
        
        Returns:
            dict: Health status for each database.
        """
        return {
            "postgresql": postgres_db._engine is not None,
            "elasticsearch": await elasticsearch_client.health_check(),
            "redis": await redis_client.health_check(),
            "clickhouse": await clickhouse_client.health_check(),
        }


# Global connection manager
connection_manager = ConnectionManager()


@asynccontextmanager
async def lifespan_manager() -> AsyncGenerator[None, None]:
    """
    Async context manager for FastAPI lifespan events.
    
    Example:
        app = FastAPI(lifespan=lifespan_manager)
    """
    await connection_manager.connect_all()
    try:
        yield
    finally:
        await connection_manager.disconnect_all()
