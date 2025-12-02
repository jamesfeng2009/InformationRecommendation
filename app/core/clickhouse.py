from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

from clickhouse_connect import get_async_client
from clickhouse_connect.driver import AsyncClient

from app.core.config import get_settings


class ClickHouseClient:
    """ClickHouse connection manager."""
    
    def __init__(self):
        self._client: AsyncClient | None = None
    
    @property
    def client(self) -> AsyncClient:
        """Get the ClickHouse client."""
        if self._client is None:
            raise RuntimeError("ClickHouse not initialized. Call connect() first.")
        return self._client
    
    async def connect(self) -> None:
        """Initialize ClickHouse connection."""
        settings = get_settings()
        
        self._client = await get_async_client(
            host=settings.CLICKHOUSE_HOST,
            port=settings.CLICKHOUSE_HTTP_PORT,
            username=settings.CLICKHOUSE_USER,
            password=settings.CLICKHOUSE_PASSWORD,
            database=settings.CLICKHOUSE_DB,
        )
    
    async def disconnect(self) -> None:
        """Close ClickHouse connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
    
    async def health_check(self) -> bool:
        """
        Check ClickHouse connection health.
        
        Returns:
            bool: True if ClickHouse is reachable, False otherwise.
        """
        try:
            result = await self.client.query("SELECT 1")
            return result is not None
        except Exception:
            return False

    @asynccontextmanager
    async def get_client(self) -> AsyncGenerator[AsyncClient, None]:
        """
        Async context manager for ClickHouse operations.
        
        Yields:
            AsyncClient: ClickHouse client instance.
        
        Example:
            async with ch.get_client() as client:
                result = await client.query("SELECT * FROM news_stats")
        """
        yield self.client
    
    async def query(self, sql: str, parameters: Optional[dict] = None) -> Any:
        """
        Execute a query and return results.
        
        Args:
            sql: SQL query string
            parameters: Query parameters
        
        Returns:
            Query result
        """
        return await self.client.query(sql, parameters=parameters)
    
    async def command(self, sql: str, parameters: Optional[dict] = None) -> None:
        """
        Execute a command (INSERT, CREATE, etc.).
        
        Args:
            sql: SQL command string
            parameters: Command parameters
        """
        await self.client.command(sql, parameters=parameters)


# Global ClickHouse instance
clickhouse_client = ClickHouseClient()


async def get_clickhouse_client() -> AsyncGenerator[AsyncClient, None]:
    """
    Dependency for FastAPI to get ClickHouse client.
    
    Yields:
        AsyncClient: ClickHouse client for request handling.
    """
    async with clickhouse_client.get_client() as client:
        yield client
