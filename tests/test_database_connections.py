"""
Unit tests for database connection utilities.
Tests connection pooling and error handling.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.core.config import Settings, get_settings
from app.core.database import PostgresDatabase, Base
from app.core.elasticsearch import get_elasticsearch_client, close_elasticsearch_client, ElasticsearchSettings
from app.core.redis import RedisClient
from app.core.clickhouse import ClickHouseClient


class TestSettings:
    """Tests for configuration settings."""
    
    def test_default_settings(self):
        """Test default settings values."""
        settings = Settings()
        assert settings.POSTGRES_HOST == "localhost"
        assert settings.POSTGRES_PORT == 5432
        assert settings.REDIS_PORT == 6379
        assert settings.ELASTICSEARCH_PORT == 9200
    
    def test_postgres_url_format(self):
        """Test PostgreSQL URL is correctly formatted."""
        settings = Settings()
        url = settings.postgres_url
        assert "postgresql+asyncpg://" in url
        assert settings.POSTGRES_HOST in url
        assert str(settings.POSTGRES_PORT) in url
    
    def test_redis_url_without_password(self):
        """Test Redis URL without password."""
        settings = Settings(REDIS_PASSWORD=None)
        url = settings.redis_url
        assert "redis://" in url
        assert "@" not in url
    
    def test_redis_url_with_password(self):
        """Test Redis URL with password."""
        settings = Settings(REDIS_PASSWORD="secret")
        url = settings.redis_url
        assert ":secret@" in url
    
    def test_elasticsearch_hosts(self):
        """Test Elasticsearch hosts list."""
        settings = Settings()
        hosts = settings.elasticsearch_hosts
        assert len(hosts) == 1
        assert "http://localhost:9200" in hosts[0]


class TestPostgresDatabase:
    """Tests for PostgreSQL database connection."""
    
    def test_initial_state(self):
        """Test database starts uninitialized."""
        db = PostgresDatabase()
        assert db._engine is None
        assert db._session_factory is None
    
    def test_engine_raises_before_connect(self):
        """Test accessing engine before connect raises error."""
        db = PostgresDatabase()
        with pytest.raises(RuntimeError, match="Database not initialized"):
            _ = db.engine
    
    def test_session_factory_raises_before_connect(self):
        """Test accessing session factory before connect raises error."""
        db = PostgresDatabase()
        with pytest.raises(RuntimeError, match="Database not initialized"):
            _ = db.session_factory


class TestElasticsearchSettings:
    """Tests for Elasticsearch settings."""
    
    def test_default_settings(self):
        """Test default Elasticsearch settings."""
        settings = ElasticsearchSettings()
        assert settings.elasticsearch_url == "http://localhost:9200"
        assert settings.elasticsearch_index_prefix == "news"
        assert settings.elasticsearch_timeout == 30
        assert settings.elasticsearch_max_retries == 3
    
    @pytest.mark.asyncio
    @patch('app.core.elasticsearch.AsyncElasticsearch')
    async def test_get_elasticsearch_client(self, mock_es):
        """Test getting Elasticsearch client."""
        mock_instance = AsyncMock()
        mock_es.return_value = mock_instance
        
        client = await get_elasticsearch_client()
        assert client is not None
        mock_es.assert_called_once()
        
        await close_elasticsearch_client()


class TestRedisClient:
    """Tests for Redis client."""
    
    def test_initial_state(self):
        """Test client starts uninitialized."""
        client = RedisClient()
        assert client._pool is None
        assert client._client is None
    
    def test_client_raises_before_connect(self):
        """Test accessing client before connect raises error."""
        client = RedisClient()
        with pytest.raises(RuntimeError, match="Redis not initialized"):
            _ = client.client


class TestClickHouseClient:
    """Tests for ClickHouse client."""
    
    def test_initial_state(self):
        """Test client starts uninitialized."""
        client = ClickHouseClient()
        assert client._client is None
    
    def test_client_raises_before_connect(self):
        """Test accessing client before connect raises error."""
        client = ClickHouseClient()
        with pytest.raises(RuntimeError, match="ClickHouse not initialized"):
            _ = client.client
