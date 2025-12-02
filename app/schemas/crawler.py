"""
Schemas for crawler API endpoints.
Requirements: 1.6
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class NewsSourceResponse(BaseModel):
    """Response schema for news source."""
    id: int
    name: str
    url: str
    category: Optional[str] = None
    status: str
    last_crawl: Optional[datetime] = None
    crawl_config: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class NewsSourceListResponse(BaseModel):
    """Response schema for list of news sources."""
    sources: List[NewsSourceResponse]
    total: int


class CrawlTaskRequest(BaseModel):
    """Request schema for starting a crawl task."""
    source_id: int = Field(..., description="ID of the news source to crawl")
    max_pages: int = Field(default=10, ge=1, le=100, description="Maximum pages to crawl")
    force: bool = Field(default=False, description="Force crawl even if recently crawled")


class CrawlTaskResponse(BaseModel):
    """Response schema for crawl task."""
    task_id: str
    source_id: int
    source_name: str
    status: str
    message: str


class CrawlStatusResponse(BaseModel):
    """Response schema for crawl status."""
    task_id: str
    source_id: int
    source_name: str
    status: str
    progress: float = Field(ge=0.0, le=100.0)
    pages_crawled: int = 0
    articles_found: int = 0
    articles_saved: int = 0
    duplicates_skipped: int = 0
    errors: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class StopCrawlRequest(BaseModel):
    """Request schema for stopping a crawl task."""
    task_id: str = Field(..., description="ID of the crawl task to stop")


class StopCrawlResponse(BaseModel):
    """Response schema for stop crawl."""
    task_id: str
    success: bool
    message: str
