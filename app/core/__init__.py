# Core module - configuration and database connections
from app.core.config import Settings, get_settings
from app.core.database import Base, PostgresDatabase, postgres_db, get_db_session
from app.core.elasticsearch import ElasticsearchClient, elasticsearch_client, get_es_client
from app.core.redis import RedisClient, redis_client, get_redis_client
from app.core.clickhouse import ClickHouseClient, clickhouse_client, get_clickhouse_client
from app.core.connections import ConnectionManager, connection_manager, lifespan_manager

__all__ = [
    # Config
    "Settings",
    "get_settings",
    # PostgreSQL
    "Base",
    "PostgresDatabase",
    "postgres_db",
    "get_db_session",
    # Elasticsearch
    "ElasticsearchClient",
    "elasticsearch_client",
    "get_es_client",
    # Redis
    "RedisClient",
    "redis_client",
    "get_redis_client",
    # ClickHouse
    "ClickHouseClient",
    "clickhouse_client",
    "get_clickhouse_client",
    # Connection Manager
    "ConnectionManager",
    "connection_manager",
    "lifespan_manager",
]
