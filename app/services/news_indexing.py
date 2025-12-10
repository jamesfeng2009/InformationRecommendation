from typing import List, Optional, Dict, Any
from datetime import datetime

from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk

from app.models.news import News


class NewsIndexingService:
    """
    Service for indexing news documents in Elasticsearch.
    """
    
    def __init__(self, es_client: AsyncElasticsearch, index_name: str = "news"):
        """
        Initialize news indexing service.
        
        Args:
            es_client: AsyncElasticsearch client instance
            index_name: Name of the index or alias to use
        """
        self.es_client = es_client
        self.index_name = index_name
    
    def _news_to_document(self, news: News) -> Dict[str, Any]:
        """
        Convert News model to Elasticsearch document.
        
        Args:
            news: News model instance
            
        Returns:
            dict: Elasticsearch document
        """
        doc = {
            "id": news.id,
            "title": news.title,
            "content": news.content,
            "source_name": news.source_name,
            "source_url": news.source_url,
            "language": news.language,
            "category": news.category,
            "hot_score": news.hot_score,
            "content_hash": news.content_hash,
            "publish_time": news.publish_time.isoformat() if news.publish_time else None,
            "crawl_time": news.crawl_time.isoformat() if news.crawl_time else None,
        }
        
        # Optional fields
        if news.translated_title:
            doc["translated_title"] = news.translated_title
        if news.translated_content:
            doc["translated_content"] = news.translated_content
        if news.summary:
            doc["summary"] = news.summary
        if news.keywords:
            doc["keywords"] = news.keywords
        if news.author:
            doc["author"] = news.author
        if news.location:
            doc["location"] = news.location
        if news.images:
            doc["images"] = news.images
        if news.videos:
            doc["videos"] = news.videos
        
        return doc
    
    async def index_news(self, news: News) -> bool:
        """
        Index a single news document.
        
        Args:
            news: News model instance
            
        Returns:
            bool: True if indexing successful
        """
        doc = self._news_to_document(news)
        
        response = await self.es_client.index(
            index=self.index_name,
            id=news.id,
            document=doc
        )
        
        return response["result"] in ["created", "updated"]
    
    async def update_news(self, news: News) -> bool:
        """
        Update an existing news document in the index.
        
        Args:
            news: News model instance with updated data
            
        Returns:
            bool: True if update successful
        """
        doc = self._news_to_document(news)
        
        try:
            response = await self.es_client.update(
                index=self.index_name,
                id=news.id,
                doc=doc
            )
            return response["result"] == "updated"
        except Exception:
            # If document doesn't exist, create it
            return await self.index_news(news)
    
    async def delete_news(self, news_id: int) -> bool:
        """
        Delete a news document from the index.
        
        Args:
            news_id: ID of news to delete
            
        Returns:
            bool: True if deletion successful
        """
        try:
            response = await self.es_client.delete(
                index=self.index_name,
                id=news_id
            )
            return response["result"] == "deleted"
        except Exception:
            return False
    
    async def bulk_index_news(self, news_list: List[News], chunk_size: int = 500) -> Dict[str, int]:
        """
        Batch index multiple news documents.
        
        Args:
            news_list: List of News model instances
            chunk_size: Number of documents to index per batch
            
        Returns:
            dict: Statistics with success and error counts
        """
        def generate_actions():
            """Generate bulk indexing actions."""
            for news in news_list:
                doc = self._news_to_document(news)
                yield {
                    "_index": self.index_name,
                    "_id": news.id,
                    "_source": doc
                }
        
        success_count = 0
        error_count = 0
        
        async for ok, result in async_bulk(
            self.es_client,
            generate_actions(),
            chunk_size=chunk_size,
            raise_on_error=False
        ):
            if ok:
                success_count += 1
            else:
                error_count += 1
        
        return {
            "success": success_count,
            "errors": error_count,
            "total": len(news_list)
        }
    
    async def exists(self, news_id: int) -> bool:
        """
        Check if a news document exists in the index.
        
        Args:
            news_id: ID of news to check
            
        Returns:
            bool: True if document exists
        """
        return await self.es_client.exists(
            index=self.index_name,
            id=news_id
        )
    
    async def get_document(self, news_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a news document from the index.
        
        Args:
            news_id: ID of news to retrieve
            
        Returns:
            dict: Document source or None if not found
        """
        try:
            response = await self.es_client.get(
                index=self.index_name,
                id=news_id
            )
            return response["_source"]
        except Exception:
            return None
    
    async def count_documents(self) -> int:
        """
        Get total count of documents in the index.
        
        Returns:
            int: Total document count
        """
        response = await self.es_client.count(index=self.index_name)
        return response["count"]
