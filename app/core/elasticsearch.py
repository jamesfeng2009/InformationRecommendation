from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from elasticsearch import AsyncElasticsearch

from app.core.config import get_settings


class ElasticsearchClient:
    """Elasticsearch connection manager."""
    
    def __init__(self):
        self._client: AsyncElasticsearch | None = None
    
    @property
    def client(self) -> AsyncElasticsearch:
        """Get the Elasticsearch client."""
        if self._client is None:
            raise RuntimeError("Elasticsearch not initialized. Call connect() first.")
        return self._client
    
    async def connect(self) -> None:
        """Initialize Elasticsearch connection."""
        settings = get_settings()
        
        auth: Optional[tuple[str, str]] = None
        if settings.ELASTICSEARCH_USER and settings.ELASTICSEARCH_PASSWORD:
            auth = (settings.ELASTICSEARCH_USER, settings.ELASTICSEARCH_PASSWORD)
        
        self._client = AsyncElasticsearch(
            hosts=settings.elasticsearch_hosts,
            basic_auth=auth,
            verify_certs=settings.ELASTICSEARCH_VERIFY_CERTS,
            request_timeout=30,
            max_retries=3,
            retry_on_timeout=True,
        )
    
    async def disconnect(self) -> None:
        """Close Elasticsearch connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None
    
    async def health_check(self) -> bool:
        """
        Check Elasticsearch cluster health.
        
        Returns:
            bool: True if cluster is healthy, False otherwise.
        """
        try:
            health = await self.client.cluster.health()
            return health.get("status") in ("green", "yellow")
        except Exception:
            return False

    @asynccontextmanager
    async def get_client(self) -> AsyncGenerator[AsyncElasticsearch, None]:
        """
        Async context manager for Elasticsearch operations.
        
        Yields:
            AsyncElasticsearch: Elasticsearch client instance.
        
        Example:
            async with es.get_client() as client:
                result = await client.search(index="news", body=query)
        """
        yield self.client


# Global Elasticsearch instance
elasticsearch_client = ElasticsearchClient()


async def get_es_client() -> AsyncGenerator[AsyncElasticsearch, None]:
    """
    Dependency for FastAPI to get Elasticsearch client.
    
    Yields:
        AsyncElasticsearch: Elasticsearch client for request handling.
    """
    async with elasticsearch_client.get_client() as client:
        yield client
