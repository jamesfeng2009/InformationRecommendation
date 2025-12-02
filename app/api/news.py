from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from datetime import datetime

from app.core.elasticsearch import get_elasticsearch_client
from app.core.database import get_db_session
from app.services.news_indexing import NewsIndexingService
from app.services.news import NewsService
from sqlalchemy.ext.asyncio import AsyncSession


router = APIRouter(prefix="/api/v1/news", tags=["news"])


class NewsCreate(BaseModel):
    """Schema for creating news."""
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1)
    source_url: str = Field(..., max_length=1000)
    source_name: str = Field(..., max_length=100)
    language: str = Field(default="zh", max_length=10)
    author: Optional[str] = Field(None, max_length=100)
    location: Optional[str] = Field(None, max_length=200)
    category: Optional[str] = Field(None, max_length=50)
    keywords: Optional[list] = None
    summary: Optional[str] = None
    images: Optional[list] = None
    videos: Optional[list] = None
    publish_time: Optional[datetime] = None
    hot_score: float = Field(default=0.0)
    content_hash: Optional[str] = Field(None, max_length=64)


class NewsUpdate(BaseModel):
    """Schema for updating news."""
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    content: Optional[str] = Field(None, min_length=1)
    category: Optional[str] = Field(None, max_length=50)
    keywords: Optional[list] = None
    summary: Optional[str] = None
    translated_title: Optional[str] = Field(None, max_length=500)
    translated_content: Optional[str] = None
    hot_score: Optional[float] = None


class NewsResponse(BaseModel):
    """Schema for news response."""
    id: int
    title: str
    content: str
    source_name: str
    source_url: str
    language: str
    category: Optional[str]
    author: Optional[str]
    location: Optional[str]
    keywords: Optional[list]
    summary: Optional[str]
    publish_time: Optional[datetime]
    crawl_time: datetime
    hot_score: float
    
    class Config:
        from_attributes = True


class BatchIndexRequest(BaseModel):
    """Schema for batch indexing request."""
    limit: Optional[int] = Field(None, gt=0, le=10000)
    offset: int = Field(default=0, ge=0)


class BatchIndexResponse(BaseModel):
    """Schema for batch indexing response."""
    success: int
    errors: int
    total: int
    message: str


async def get_news_service(
    db: AsyncSession = Depends(get_db_session),
    es_client = Depends(get_elasticsearch_client)
) -> NewsService:
    """Dependency to get news service instance."""
    indexing_service = NewsIndexingService(es_client)
    return NewsService(db, indexing_service)


@router.post("", response_model=NewsResponse, status_code=status.HTTP_201_CREATED)
async def create_news(
    news_data: NewsCreate,
    service: NewsService = Depends(get_news_service)
):
    """
    Create a new news item and index it in Elasticsearch.
    
    Requirements: 19.2 - Index news documents on creation
    """
    news = await service.create_news(news_data.model_dump())
    return news


@router.get("/{news_id}", response_model=NewsResponse)
async def get_news(
    news_id: int,
    service: NewsService = Depends(get_news_service)
):
    """Get a news item by ID."""
    news = await service.get_news(news_id)
    if not news:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"News with id {news_id} not found"
        )
    return news


@router.patch("/{news_id}", response_model=NewsResponse)
async def update_news(
    news_id: int,
    update_data: NewsUpdate,
    service: NewsService = Depends(get_news_service)
):
    """
    Update a news item and update its Elasticsearch index.
    
    Requirements: 19.2 - Update index on news modification
    """
    news = await service.update_news(
        news_id,
        update_data.model_dump(exclude_unset=True)
    )
    if not news:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"News with id {news_id} not found"
        )
    return news


@router.delete("/{news_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_news(
    news_id: int,
    service: NewsService = Depends(get_news_service)
):
    """Delete a news item from database and Elasticsearch."""
    success = await service.delete_news(news_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"News with id {news_id} not found"
        )


@router.post("/batch-index", response_model=BatchIndexResponse)
async def batch_index_news(
    request: BatchIndexRequest,
    service: NewsService = Depends(get_news_service)
):
    """
    Batch index news from database to Elasticsearch.
    
    Requirements: 19.2 - Batch indexing for initial data load
    """
    stats = await service.batch_index_news(
        limit=request.limit,
        offset=request.offset
    )
    
    return BatchIndexResponse(
        success=stats["success"],
        errors=stats["errors"],
        total=stats["total"],
        message=f"Indexed {stats['success']} news items successfully"
    )


@router.post("/reindex-all", response_model=BatchIndexResponse)
async def reindex_all_news(
    service: NewsService = Depends(get_news_service)
):
    """
    Reindex all news from database to Elasticsearch.
    
    Requirements: 19.2 - Batch indexing for initial data load
    """
    stats = await service.reindex_all_news()
    
    return BatchIndexResponse(
        success=stats["success"],
        errors=stats["errors"],
        total=stats["total"],
        message=f"Reindexed {stats['success']} news items successfully"
    )
