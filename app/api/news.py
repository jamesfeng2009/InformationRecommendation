from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from datetime import datetime

from app.core.elasticsearch import get_elasticsearch_client
from app.core.database import get_db_session
from app.core.redis import get_redis_client
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


class NewsListResponse(BaseModel):
    """Schema for paginated news list response."""
    items: List[NewsResponse]
    total: int
    page: int
    size: int
    pages: int


class CategoryUpdateRequest(BaseModel):
    """Schema for category update request."""
    category: str = Field(..., min_length=1, max_length=50)


class SearchRequest(BaseModel):
    """Schema for news search request."""
    query: str = Field(..., min_length=1, description="Search query")
    page: int = Field(1, ge=1, description="Page number")
    size: int = Field(20, ge=1, le=100, description="Items per page")
    category: Optional[str] = Field(None, description="Filter by category")
    source_name: Optional[str] = Field(None, description="Filter by source name")
    language: Optional[str] = Field(None, description="Filter by language")
    start_time: Optional[datetime] = Field(None, description="Filter by publish time start")
    end_time: Optional[datetime] = Field(None, description="Filter by publish time end")
    sort_by: str = Field("relevance", description="Sort by relevance or time")


class SearchResultItem(BaseModel):
    """Schema for search result item."""
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
    score: float
    highlight: dict


class SearchResponse(BaseModel):
    """Schema for search response."""
    items: List[SearchResultItem]
    total: int
    page: int
    size: int
    pages: int
    query: str
    took: int


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


class ExportRequest(BaseModel):
    """Schema for news export request."""
    news_ids: List[int] = Field(..., min_items=1, description="List of news IDs to export")
    format: str = Field("json", description="Export format: json, csv, txt")


async def get_news_service(
    db: AsyncSession = Depends(get_db_session),
    es_client = Depends(get_elasticsearch_client),
    redis_client = Depends(get_redis_client)
) -> NewsService:
    """Dependency to get news service instance."""
    indexing_service = NewsIndexingService(es_client)
    return NewsService(db, indexing_service, es_client, redis_client)


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
    """
    Get a news item by ID.
    
    Requirements: 15.1 - Get news detail
    """
    news = await service.get_news(news_id)
    if not news:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"News with id {news_id} not found"
        )
    return news


@router.get("", response_model=NewsListResponse)
async def list_news(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Items per page"),
    category: Optional[str] = Query(None, description="Filter by category"),
    source_name: Optional[str] = Query(None, description="Filter by source name"),
    language: Optional[str] = Query(None, description="Filter by language"),
    start_time: Optional[datetime] = Query(None, description="Filter by publish time start"),
    end_time: Optional[datetime] = Query(None, description="Filter by publish time end"),
    order_by: str = Query("publish_time", description="Order by field"),
    order_desc: bool = Query(True, description="Order in descending order"),
    service: NewsService = Depends(get_news_service)
):
    """
    List news with pagination and filtering.
    
    Requirements: 15.2 - List news with pagination
    """
    result = await service.list_news(
        page=page,
        size=size,
        category=category,
        source_name=source_name,
        language=language,
        start_time=start_time,
        end_time=end_time,
        order_by=order_by,
        order_desc=order_desc
    )
    return result


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


@router.patch("/{news_id}/category", response_model=NewsResponse)
async def update_news_category(
    news_id: int,
    request: CategoryUpdateRequest,
    service: NewsService = Depends(get_news_service)
):
    """
    Update news category.
    
    Requirements: 3.4 - Allow manual modification of news category
    """
    news = await service.update_news_category(news_id, request.category)
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

@router.post("/search", response_model=SearchResponse)
async def search_news(
    request: SearchRequest,
    service: NewsService = Depends(get_news_service)
):
    """
    Search news with full-text search and multi-dimensional filtering.
    
    Requirements: 6.1, 6.2, 6.3, 6.5 - Full-text search with filtering, sorting, and highlighting
    """
    result = await service.search_news(
        query=request.query,
        page=request.page,
        size=request.size,
        category=request.category,
        source_name=request.source_name,
        language=request.language,
        start_time=request.start_time,
        end_time=request.end_time,
        sort_by=request.sort_by
    )
    return result

@router.get("/hot", response_model=List[NewsResponse])
async def get_hot_news(
    limit: int = Query(10, ge=1, le=50, description="Number of hot news items"),
    category: Optional[str] = Query(None, description="Filter by category"),
    hours: int = Query(24, ge=1, le=168, description="Time window in hours"),
    service: NewsService = Depends(get_news_service)
):
    """
    Get hot news list with caching.
    
    Requirements: 4.1, 4.3 - Hot news ranking and caching
    """
    hot_news = await service.get_hot_news(
        limit=limit,
        category=category,
        hours=hours
    )
    return hot_news

@router.get("/{news_id}/related", response_model=List[NewsResponse])
async def get_related_news(
    news_id: int,
    limit: int = Query(5, ge=1, le=20, description="Number of related news items"),
    service: NewsService = Depends(get_news_service)
):
    """
    Get related news by content similarity and geographic relevance.
    
    Requirements: 5.3 - Find similar news by content embedding and geographic relevance
    """
    # First check if the reference news exists
    reference_news = await service.get_news(news_id)
    if not reference_news:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"News with id {news_id} not found"
        )
    
    related_news = await service.get_related_news(news_id, limit)
    return related_news
@router.post("/export")
async def export_news(
    request: ExportRequest,
    service: NewsService = Depends(get_news_service)
):
    """
    Export news items in specified format.
    
    Requirements: 6.6, 15.4 - Support exporting single or batch news items
    """
    try:
        exported_data = await service.export_news(
            news_ids=request.news_ids,
            format=request.format
        )
        
        # Determine content type and filename
        if request.format.lower() == "json":
            media_type = "application/json"
            filename = "news_export.json"
        elif request.format.lower() == "csv":
            media_type = "text/csv"
            filename = "news_export.csv"
        elif request.format.lower() == "txt":
            media_type = "text/plain"
            filename = "news_export.txt"
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported export format: {request.format}"
            )
        
        return Response(
            content=exported_data,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )