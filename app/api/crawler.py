"""
Crawler API endpoints for managing news crawling operations.
Requirements: 1.6
"""
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.crawler import (
    CrawlStatusResponse,
    CrawlTaskRequest,
    CrawlTaskResponse,
    NewsSourceListResponse,
    NewsSourceResponse,
    StopCrawlRequest,
    StopCrawlResponse,
)
from app.services.crawler import crawler_service

router = APIRouter(prefix="/crawler", tags=["crawler"])


@router.get("/sources", response_model=NewsSourceListResponse)
async def get_sources(db: AsyncSession = Depends(get_db)):
    """
    Get all configured news sources.
    
    Returns list of news sources with their configuration and status.
    Requirements: 1.6
    """
    sources = await crawler_service.get_sources(db)
    return NewsSourceListResponse(
        sources=[NewsSourceResponse.model_validate(s) for s in sources],
        total=len(sources),
    )


@router.post("/start", response_model=CrawlTaskResponse)
async def start_crawl(
    request: CrawlTaskRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Start a crawl task for a news source.
    
    Initiates crawling for the specified news source.
    Requirements: 1.6
    """
    try:
        task = await crawler_service.start_crawl(
            db=db,
            source_id=request.source_id,
            max_pages=request.max_pages,
            force=request.force,
        )
        return CrawlTaskResponse(
            task_id=task.task_id,
            source_id=task.source_id,
            source_name=task.source_name,
            status=task.status.value,
            message=f"Crawl task started for {task.source_name}",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/stop", response_model=StopCrawlResponse)
async def stop_crawl(request: StopCrawlRequest):
    """
    Stop a running crawl task.
    
    Requests the specified crawl task to stop.
    Requirements: 1.6
    """
    success = crawler_service.stop_crawl(request.task_id)
    
    if success:
        return StopCrawlResponse(
            task_id=request.task_id,
            success=True,
            message="Crawl task stop requested",
        )
    else:
        task = crawler_service.get_task_status(request.task_id)
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {request.task_id} not found",
            )
        return StopCrawlResponse(
            task_id=request.task_id,
            success=False,
            message=f"Task is not running (status: {task.status.value})",
        )


@router.get("/status/{task_id}", response_model=CrawlStatusResponse)
async def get_crawl_status(task_id: str):
    """
    Get the status of a crawl task.
    
    Returns detailed status information for the specified task.
    Requirements: 1.6
    """
    task = crawler_service.get_task_status(task_id)
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found",
        )
    
    return CrawlStatusResponse(
        task_id=task.task_id,
        source_id=task.source_id,
        source_name=task.source_name,
        status=task.status.value,
        progress=task.progress,
        pages_crawled=task.pages_crawled,
        articles_found=task.articles_found,
        articles_saved=task.articles_saved,
        duplicates_skipped=task.duplicates_skipped,
        errors=task.errors,
        started_at=task.started_at,
        completed_at=task.completed_at,
        error_message=task.error_message,
    )


@router.get("/status", response_model=List[CrawlStatusResponse])
async def get_all_crawl_status():
    """
    Get status of all crawl tasks.
    
    Returns status information for all tasks.
    Requirements: 1.6
    """
    tasks = crawler_service.get_all_tasks()
    return [
        CrawlStatusResponse(
            task_id=task.task_id,
            source_id=task.source_id,
            source_name=task.source_name,
            status=task.status.value,
            progress=task.progress,
            pages_crawled=task.pages_crawled,
            articles_found=task.articles_found,
            articles_saved=task.articles_saved,
            duplicates_skipped=task.duplicates_skipped,
            errors=task.errors,
            started_at=task.started_at,
            completed_at=task.completed_at,
            error_message=task.error_message,
        )
        for task in tasks
    ]
