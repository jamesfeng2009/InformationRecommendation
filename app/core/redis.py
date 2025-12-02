from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from redis.asyncio import ConnectionPool, Redis

from app.core.config import get_settings


class RedisClient:
    """Redis connection manager with connection pooling."""
    
    def __init__(self):
        self._pool: ConnectionPool | None = None
        self._client: Redis | None = None
    
    @property
    def client(self) -> Redis:
        """Get the Redis client."""
        if self._client is None:
            raise RuntimeError("Redis not initialized. Call connect() first.")
        return self._client
    
    async def connect(self) -> None:
        """Initialize Redis connection pool and client."""
        settings = get_settings()
        
        self._pool = ConnectionPool.from_url(
            settings.redis_url,
            max_connections=settings.REDIS_POOL_SIZE,
            decode_responses=True,
        )
        
        self._client = Redis(connection_pool=self._pool)
    
    async def disconnect(self) -> None:
        """Close Redis connection and cleanup pool."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._pool is not None:
            await self._pool.disconnect()
            self._pool = None
    
    async def health_check(self) -> bool:
        """
        Check Redis connection health.
        
        Returns:
            bool: True if Redis is reachable, False otherwise.
        """
        try:
            return await self.client.ping()
        except Exception:
            return False

    @asynccontextmanager
    async def get_client(self) -> AsyncGenerator[Redis, None]:
        """
        Async context manager for Redis operations.
        
        Yields:
            Redis: Redis client instance.
        
        Example:
            async with redis.get_client() as client:
                await client.set("key", "value")
                value = await client.get("key")
        """
        yield self.client
    
    async def get(self, key: str) -> Optional[str]:
        """Get value by key."""
        return await self.client.get(key)
    
    async def set(
        self, 
        key: str, 
        value: str, 
        expire: Optional[int] = None
    ) -> bool:
        """
        Set key-value pair with optional expiration.
        
        Args:
            key: Cache key
            value: Value to store
            expire: Expiration time in seconds
        
        Returns:
            bool: True if successful
        """
        return await self.client.set(key, value, ex=expire)
    
    async def delete(self, key: str) -> int:
        """Delete key from cache."""
        return await self.client.delete(key)


# Global Redis instance
redis_client = RedisClient()


async def get_redis_client() -> AsyncGenerator[Redis, None]:
    """
    Dependency for FastAPI to get Redis client.
    
    Yields:
        Redis: Redis client for request handling.
    """
    async with redis_client.get_client() as client:
        yield client
