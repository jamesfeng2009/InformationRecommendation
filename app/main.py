from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.auth import router as auth_router
from app.api.crawler import router as crawler_router
from app.api.elasticsearch_admin import router as elasticsearch_admin_router
from app.api.news import router as news_router
from app.api.organization import router as organization_router
from app.api.rbac import router as rbac_router
from app.api.recommendation import router as recommendation_router
from app.api.users import router as users_router
from app.core.config import get_settings
from app.core.database import postgres_db
from app.core.elasticsearch import close_elasticsearch_client
from app.core.redis import redis_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    # Startup
    await postgres_db.connect()
    await redis_client.connect()
    yield
    # Shutdown
    await close_elasticsearch_client()
    await redis_client.disconnect()
    await postgres_db.disconnect()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title=settings.APP_NAME,
        description="智能信息推荐系统 API",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    # Include routers
    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(crawler_router, prefix="/api/v1")
    app.include_router(elasticsearch_admin_router)
    app.include_router(news_router)
    app.include_router(organization_router, prefix="/api/v1")
    app.include_router(rbac_router, prefix="/api/v1")
    app.include_router(recommendation_router)
    app.include_router(users_router, prefix="/api/v1")
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}
    
    return app


app = create_app()
