from typing import Optional

from elasticsearch import AsyncElasticsearch
from pydantic_settings import BaseSettings


class ElasticsearchSettings(BaseSettings):
    """Elasticsearch configuration settings."""
    
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_index_prefix: str = "news"
    elasticsearch_timeout: int = 30
    elasticsearch_max_retries: int = 3
    
    class Config:
        env_file = ".env"
        extra = "ignore"


# Global Elasticsearch client instance
_es_client: Optional[AsyncElasticsearch] = None


async def get_elasticsearch_client() -> AsyncElasticsearch:
    """
    Get or create Elasticsearch client instance.
    
    Returns:
        AsyncElasticsearch: Elasticsearch client
    """
    global _es_client
    
    if _es_client is None:
        settings = ElasticsearchSettings()
        _es_client = AsyncElasticsearch(
            hosts=[settings.elasticsearch_url],
            request_timeout=settings.elasticsearch_timeout,
            max_retries=settings.elasticsearch_max_retries,
            retry_on_timeout=True,
        )
    
    return _es_client


async def close_elasticsearch_client():
    """Close Elasticsearch client connection."""
    global _es_client
    
    if _es_client is not None:
        await _es_client.close()
        _es_client = None
