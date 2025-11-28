"""
FastAPI application entry point.
Intelligent Information Recommendation System.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.auth import router as auth_router
from app.api.rbac import router as rbac_router
from app.core.config import get_settings
from app.core.database import postgres_db
from app.core.redis import redis_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    # Startup
    await postgres_db.connect()
    await redis_client.connect()
    yield
    # Shutdown
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
    app.include_router(rbac_router, prefix="/api/v1")
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}
    
    return app


app = create_app()
