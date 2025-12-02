"""
Crawler service for managing news crawling operations.
Requirements: 1.1, 1.6
"""
import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.crawler.base import CrawlerConfig, CrawlerStatus
from app.crawler.cleaner import ContentCleaner
from app.crawler.deduplication import DeduplicationService
from app.crawler.parser import NewsParser
from app.models.news import News
from app.models.system import NewsSource, SensitiveWord


class CrawlTask:
    """Represents a crawl task with its state."""
    
    def __init__(
        self,
        task_id: str,
        source_id: int,
        source_name: str,
        max_pages: int = 10,
    ):
        self.task_id = task_id
        self.source_id = source_id
        self.source_name = source_name
        self.max_pages = max_pages
        self.status = CrawlerStatus.PENDING
        self.progress = 0.0
        self.pages_crawled = 0
        self.articles_found = 0
        self.articles_saved = 0
        self.duplicates_skipped = 0
        self.errors = 0
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error_message: Optional[str] = None
        self._stop_requested = False
    
    def request_stop(self) -> None:
        """Request the task to stop."""
        self._stop_requested = True
    
    @property
    def should_stop(self) -> bool:
        """Check if stop was requested."""
        return self._stop_requested
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "status": self.status.value,
            "progress": self.progress,
            "pages_crawled": self.pages_crawled,
            "articles_found": self.articles_found,
            "articles_saved": self.articles_saved,
            "duplicates_skipped": self.duplicates_skipped,
            "errors": self.errors,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error_message": self.error_message,
        }


class CrawlerService:
    """
    Service for managing news crawling operations.
    Requirements: 1.1, 1.6
    """
    
    def __init__(self):
        self._tasks: Dict[str, CrawlTask] = {}
        self._running_tasks: Dict[int, str] = {}  # source_id -> task_id
        self._dedup_service = DeduplicationService()
        self._cleaner = ContentCleaner()
    
    async def get_sources(self, db: AsyncSession) -> List[NewsSource]:
        """
        Get all configured news sources.
        
        Args:
            db: Database session
            
        Returns:
            List of NewsSource objects
        """
        result = await db.execute(select(NewsSource).order_by(NewsSource.name))
        return list(result.scalars().all())
    
    async def get_source(self, db: AsyncSession, source_id: int) -> Optional[NewsSource]:
        """
        Get a specific news source.
        
        Args:
            db: Database session
            source_id: Source ID
            
        Returns:
            NewsSource or None
        """
        result = await db.execute(
            select(NewsSource).where(NewsSource.id == source_id)
        )
        return result.scalar_one_or_none()
    
    async def _load_sensitive_words(self, db: AsyncSession) -> None:
        """Load sensitive words into the cleaner."""
        result = await db.execute(select(SensitiveWord.word))
        words = {row[0] for row in result.fetchall()}
        self._cleaner.update_sensitive_words(words)
    
    async def start_crawl(
        self,
        db: AsyncSession,
        source_id: int,
        max_pages: int = 10,
        force: bool = False,
    ) -> CrawlTask:
        """
        Start a crawl task for a news source.
        
        Args:
            db: Database session
            source_id: ID of the news source
            max_pages: Maximum pages to crawl
            force: Force crawl even if recently crawled
            
        Returns:
            CrawlTask object
            
        Raises:
            ValueError: If source not found or already crawling
        """
        # Check if source exists
        source = await self.get_source(db, source_id)
        if not source:
            raise ValueError(f"News source {source_id} not found")
        
        # Check if already crawling this source
        if source_id in self._running_tasks and not force:
            existing_task_id = self._running_tasks[source_id]
            existing_task = self._tasks.get(existing_task_id)
            if existing_task and existing_task.status == CrawlerStatus.RUNNING:
                raise ValueError(f"Source {source_id} is already being crawled")
        
        # Load sensitive words
        await self._load_sensitive_words(db)
        
        # Create new task
        task_id = str(uuid.uuid4())
        task = CrawlTask(
            task_id=task_id,
            source_id=source_id,
            source_name=source.name,
            max_pages=max_pages,
        )
        
        self._tasks[task_id] = task
        self._running_tasks[source_id] = task_id
        
        # Start crawl in background (in production, this would use Celery)
        asyncio.create_task(self._run_crawl(db, task, source))
        
        return task
    
    async def _run_crawl(
        self,
        db: AsyncSession,
        task: CrawlTask,
        source: NewsSource,
    ) -> None:
        """
        Run the actual crawl operation.
        
        Args:
            db: Database session
            task: The crawl task
            source: The news source
        """
        task.status = CrawlerStatus.RUNNING
        task.started_at = datetime.utcnow()
        
        try:
            # This is a simplified implementation
            # In production, this would use Scrapy spiders
            parser = NewsParser.for_source(source.url)
            
            # Simulate crawling (placeholder for actual implementation)
            # In production, this would:
            # 1. Fetch the source page
            # 2. Extract article URLs
            # 3. Fetch and parse each article
            # 4. Clean and deduplicate content
            # 5. Save to database
            
            task.progress = 100.0
            task.status = CrawlerStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            
            # Update source last_crawl time
            source.last_crawl = datetime.utcnow()
            await db.commit()
            
        except Exception as e:
            task.status = CrawlerStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.utcnow()
        
        finally:
            # Remove from running tasks
            if source.id in self._running_tasks:
                del self._running_tasks[source.id]
    
    def stop_crawl(self, task_id: str) -> bool:
        """
        Stop a running crawl task.
        
        Args:
            task_id: ID of the task to stop
            
        Returns:
            True if task was found and stop requested
        """
        task = self._tasks.get(task_id)
        if not task:
            return False
        
        if task.status == CrawlerStatus.RUNNING:
            task.request_stop()
            task.status = CrawlerStatus.STOPPED
            task.completed_at = datetime.utcnow()
            
            # Remove from running tasks
            if task.source_id in self._running_tasks:
                del self._running_tasks[task.source_id]
            
            return True
        
        return False
    
    def get_task_status(self, task_id: str) -> Optional[CrawlTask]:
        """
        Get the status of a crawl task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            CrawlTask or None
        """
        return self._tasks.get(task_id)
    
    def get_all_tasks(self) -> List[CrawlTask]:
        """Get all crawl tasks."""
        return list(self._tasks.values())
    
    def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """
        Remove old completed tasks.
        
        Args:
            max_age_hours: Maximum age in hours
            
        Returns:
            Number of tasks removed
        """
        now = datetime.utcnow()
        to_remove = []
        
        for task_id, task in self._tasks.items():
            if task.completed_at:
                age = (now - task.completed_at).total_seconds() / 3600
                if age > max_age_hours:
                    to_remove.append(task_id)
        
        for task_id in to_remove:
            del self._tasks[task_id]
        
        return len(to_remove)


# Global service instance
crawler_service = CrawlerService()
