from typing import Optional, List
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.news import News
from app.services.news_indexing import NewsIndexingService


class NewsService:
    """
    Service for news operations with automatic Elasticsearch indexing.
    Requirements: 19.2
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        indexing_service: NewsIndexingService
    ):
        """
        Initialize news service.
        
        Args:
            db_session: Database session
            indexing_service: Elasticsearch indexing service
        """
        self.db = db_session
        self.indexing_service = indexing_service
    
    async def create_news(self, news_data: dict) -> News:
        """
        Create a new news item and index it in Elasticsearch.
        
        Args:
            news_data: Dictionary with news data
            
        Returns:
            News: Created news instance
            
        Requirements: 19.2 - Index news documents on creation
        """
        # Create news in database
        news = News(**news_data)
        self.db.add(news)
        await self.db.commit()
        await self.db.refresh(news)
        
        # Index in Elasticsearch
        await self.indexing_service.index_news(news)
        
        return news
    
    async def update_news(self, news_id: int, update_data: dict) -> Optional[News]:
        """
        Update a news item and update its index in Elasticsearch.
        
        Args:
            news_id: ID of news to update
            update_data: Dictionary with fields to update
            
        Returns:
            News: Updated news instance or None if not found
            
        Requirements: 19.2 - Update index on news modification
        """
        # Get news from database
        result = await self.db.execute(
            select(News).where(News.id == news_id)
        )
        news = result.scalar_one_or_none()
        
        if not news:
            return None
        
        # Update fields
        for key, value in update_data.items():
            if hasattr(news, key):
                setattr(news, key, value)
        
        news.updated_at = datetime.utcnow()
        await self.db.commit()
        await self.db.refresh(news)
        
        # Update in Elasticsearch
        await self.indexing_service.update_news(news)
        
        return news
    
    async def delete_news(self, news_id: int) -> bool:
        """
        Delete a news item from database and Elasticsearch.
        
        Args:
            news_id: ID of news to delete
            
        Returns:
            bool: True if deleted successfully
        """
        # Get news from database
        result = await self.db.execute(
            select(News).where(News.id == news_id)
        )
        news = result.scalar_one_or_none()
        
        if not news:
            return False
        
        # Delete from database
        await self.db.delete(news)
        await self.db.commit()
        
        # Delete from Elasticsearch
        await self.indexing_service.delete_news(news_id)
        
        return True
    
    async def get_news(self, news_id: int) -> Optional[News]:
        """
        Get a news item by ID.
        
        Args:
            news_id: ID of news to retrieve
            
        Returns:
            News: News instance or None if not found
        """
        result = await self.db.execute(
            select(News).where(News.id == news_id)
        )
        return result.scalar_one_or_none()
    
    async def batch_index_news(
        self,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> dict:
        """
        Batch index news from database to Elasticsearch.
        Useful for initial data load or reindexing.
        
        Args:
            limit: Maximum number of news to index
            offset: Offset for pagination
            
        Returns:
            dict: Indexing statistics
            
        Requirements: 19.2 - Batch indexing for initial data load
        """
        # Query news from database
        query = select(News).offset(offset)
        if limit:
            query = query.limit(limit)
        
        result = await self.db.execute(query)
        news_list = result.scalars().all()
        
        # Batch index
        stats = await self.indexing_service.bulk_index_news(news_list)
        
        return stats
    
    async def reindex_all_news(self, chunk_size: int = 1000) -> dict:
        """
        Reindex all news from database to Elasticsearch in chunks.
        
        Args:
            chunk_size: Number of news items to process per chunk
            
        Returns:
            dict: Total indexing statistics
        """
        total_stats = {"success": 0, "errors": 0, "total": 0}
        offset = 0
        
        while True:
            stats = await self.batch_index_news(limit=chunk_size, offset=offset)
            
            total_stats["success"] += stats["success"]
            total_stats["errors"] += stats["errors"]
            total_stats["total"] += stats["total"]
            
            # Break if no more news
            if stats["total"] < chunk_size:
                break
            
            offset += chunk_size
        
        return total_stats
