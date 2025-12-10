from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.news import News
from app.services.news_indexing import NewsIndexingService
from elasticsearch import AsyncElasticsearch
import redis.asyncio as redis


class NewsService:
    
    
    def __init__(
        self,
        db_session: AsyncSession,
        indexing_service: NewsIndexingService,
        es_client: Optional[AsyncElasticsearch] = None,
        redis_client: Optional[redis.Redis] = None
    ):
        """
        Initialize news service.
        
        Args:
            db_session: Database session
            indexing_service: Elasticsearch indexing service
            es_client: Elasticsearch client for search operations
            redis_client: Redis client for caching
        """
        self.db = db_session
        self.indexing_service = indexing_service
        self.es_client = es_client
        self.redis_client = redis_client
    
    async def create_news(self, news_data: dict) -> News:
        """
        Create a new news item and index it in Elasticsearch.
        
        Args:
            news_data: Dictionary with news data
            
        Returns:
            News: Created news instance
            
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
    
    async def list_news(
        self,
        page: int = 1,
        size: int = 20,
        category: Optional[str] = None,
        source_name: Optional[str] = None,
        language: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        order_by: str = "publish_time",
        order_desc: bool = True
    ) -> Dict[str, Any]:
        """
        List news with pagination and filtering.
        
        Args:
            page: Page number (1-based)
            size: Number of items per page
            category: Filter by category
            source_name: Filter by source name
            language: Filter by language
            start_time: Filter by publish time start
            end_time: Filter by publish time end
            order_by: Field to order by (publish_time, crawl_time, hot_score)
            order_desc: Whether to order in descending order
            
        Returns:
            dict: Paginated news list with metadata
        """
        # Build base query
        query = select(News)
        
        # Apply filters
        conditions = []
        if category:
            conditions.append(News.category == category)
        if source_name:
            conditions.append(News.source_name == source_name)
        if language:
            conditions.append(News.language == language)
        if start_time:
            conditions.append(News.publish_time >= start_time)
        if end_time:
            conditions.append(News.publish_time <= end_time)
        
        if conditions:
            query = query.where(and_(*conditions))
        
        # Apply ordering
        order_field = getattr(News, order_by, News.publish_time)
        if order_desc:
            query = query.order_by(order_field.desc())
        else:
            query = query.order_by(order_field.asc())
        
        # Get total count
        count_query = select(func.count(News.id))
        if conditions:
            count_query = count_query.where(and_(*conditions))
        
        total_result = await self.db.execute(count_query)
        total = total_result.scalar()
        
        # Apply pagination
        offset = (page - 1) * size
        query = query.offset(offset).limit(size)
        
        # Execute query
        result = await self.db.execute(query)
        news_list = result.scalars().all()
        
        return {
            "items": news_list,
            "total": total,
            "page": page,
            "size": size,
            "pages": (total + size - 1) // size if total > 0 else 0
        }
    
    async def update_news_category(self, news_id: int, category: str) -> Optional[News]:
        """
        Update news category.
        
        Args:
            news_id: ID of news to update
            category: New category
            
        Returns:
            News: Updated news instance or None if not found
        """
        return await self.update_news(news_id, {"category": category})
    
    async def search_news(
        self,
        query: str,
        page: int = 1,
        size: int = 20,
        category: Optional[str] = None,
        source_name: Optional[str] = None,
        language: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        sort_by: str = "relevance"  # "relevance" or "time"
    ) -> Dict[str, Any]:
        """
        Search news using Elasticsearch with full-text search and filtering.
        
        Args:
            query: Search query string
            page: Page number (1-based)
            size: Number of items per page
            category: Filter by category
            source_name: Filter by source name
            language: Filter by language
            start_time: Filter by publish time start
            end_time: Filter by publish time end
            sort_by: Sort by relevance or time
            
        Returns:
            dict: Search results with metadata and highlighting
        """
        if not self.es_client:
            raise ValueError("Elasticsearch client not configured for search")
        
        # Build search query
        search_body = {
            "query": {
                "bool": {
                    "must": [],
                    "filter": []
                }
            },
            "highlight": {
                "fields": {
                    "title": {"pre_tags": ["<mark>"], "post_tags": ["</mark>"]},
                    "content": {"pre_tags": ["<mark>"], "post_tags": ["</mark>"]},
                    "translated_title": {"pre_tags": ["<mark>"], "post_tags": ["</mark>"]},
                    "translated_content": {"pre_tags": ["<mark>"], "post_tags": ["</mark>"]}
                }
            },
            "from": (page - 1) * size,
            "size": size
        }
        
        # Add text search query
        if query.strip():
            search_body["query"]["bool"]["must"].append({
                "multi_match": {
                    "query": query,
                    "fields": [
                        "title^3",
                        "translated_title^3", 
                        "content",
                        "translated_content",
                        "summary^2"
                    ],
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            })
        else:
            search_body["query"]["bool"]["must"].append({"match_all": {}})
        
        # Add filters
        if category:
            search_body["query"]["bool"]["filter"].append({"term": {"category": category}})
        if source_name:
            search_body["query"]["bool"]["filter"].append({"term": {"source_name": source_name}})
        if language:
            search_body["query"]["bool"]["filter"].append({"term": {"language": language}})
        
        # Add time range filter
        if start_time or end_time:
            time_filter = {"range": {"publish_time": {}}}
            if start_time:
                time_filter["range"]["publish_time"]["gte"] = start_time.isoformat()
            if end_time:
                time_filter["range"]["publish_time"]["lte"] = end_time.isoformat()
            search_body["query"]["bool"]["filter"].append(time_filter)
        
        # Add sorting
        if sort_by == "time":
            search_body["sort"] = [{"publish_time": {"order": "desc"}}]
        # For relevance, use default Elasticsearch scoring
        
        # Execute search
        response = await self.es_client.search(
            index=self.indexing_service.index_name,
            body=search_body
        )
        
        # Process results
        hits = response["hits"]
        items = []
        
        for hit in hits["hits"]:
            source = hit["_source"]
            highlight = hit.get("highlight", {})
            
            # Get news from database for complete data
            news = await self.get_news(source["id"])
            if news:
                news_dict = {
                    "id": news.id,
                    "title": news.title,
                    "content": news.content,
                    "source_name": news.source_name,
                    "source_url": news.source_url,
                    "language": news.language,
                    "category": news.category,
                    "author": news.author,
                    "location": news.location,
                    "keywords": news.keywords,
                    "summary": news.summary,
                    "publish_time": news.publish_time,
                    "crawl_time": news.crawl_time,
                    "hot_score": news.hot_score,
                    "score": hit["_score"],
                    "highlight": highlight
                }
                items.append(news_dict)
        
        return {
            "items": items,
            "total": hits["total"]["value"],
            "page": page,
            "size": size,
            "pages": (hits["total"]["value"] + size - 1) // size if hits["total"]["value"] > 0 else 0,
            "query": query,
            "took": response["took"]
        }
    
    async def calculate_hot_score(self, news_id: int) -> float:
        """
        Calculate hot score for a news item based on user interactions.
        
        Args:
            news_id: ID of news to calculate score for
            
        Returns:
            float: Calculated hot score
        """
        from app.models.news import UserBehavior
        
        # Get behavior counts for the news
        view_count_result = await self.db.execute(
            select(func.count(UserBehavior.id))
            .where(and_(UserBehavior.news_id == news_id, UserBehavior.action == "view"))
        )
        view_count = view_count_result.scalar() or 0
        
        like_count_result = await self.db.execute(
            select(func.count(UserBehavior.id))
            .where(and_(UserBehavior.news_id == news_id, UserBehavior.action == "like"))
        )
        like_count = like_count_result.scalar() or 0
        
        collect_count_result = await self.db.execute(
            select(func.count(UserBehavior.id))
            .where(and_(UserBehavior.news_id == news_id, UserBehavior.action == "collect"))
        )
        collect_count = collect_count_result.scalar() or 0
        
        share_count_result = await self.db.execute(
            select(func.count(UserBehavior.id))
            .where(and_(UserBehavior.news_id == news_id, UserBehavior.action == "share"))
        )
        share_count = share_count_result.scalar() or 0
        
        # Calculate weighted hot score
        # Views: 1 point, Likes: 3 points, Collects: 5 points, Shares: 10 points
        hot_score = (view_count * 1.0 + 
                    like_count * 3.0 + 
                    collect_count * 5.0 + 
                    share_count * 10.0)
        
        # Apply time decay (newer news get higher scores)
        news = await self.get_news(news_id)
        if news and news.publish_time:
            hours_since_publish = (datetime.utcnow() - news.publish_time).total_seconds() / 3600
            # Decay factor: score decreases by 50% every 24 hours
            decay_factor = 0.5 ** (hours_since_publish / 24)
            hot_score *= decay_factor
        
        return hot_score
    
    async def update_hot_scores(self, limit: int = 1000) -> int:
        """
        Update hot scores for recent news items.
        
        Args:
            limit: Maximum number of news items to update
            
        Returns:
            int: Number of news items updated
        """
        # Get recent news (last 7 days)
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        
        result = await self.db.execute(
            select(News.id)
            .where(News.publish_time >= seven_days_ago)
            .order_by(News.publish_time.desc())
            .limit(limit)
        )
        news_ids = result.scalars().all()
        
        updated_count = 0
        for news_id in news_ids:
            hot_score = await self.calculate_hot_score(news_id)
            await self.update_news(news_id, {"hot_score": hot_score})
            updated_count += 1
        
        return updated_count
    
    async def get_hot_news(
        self,
        limit: int = 10,
        category: Optional[str] = None,
        hours: int = 24
    ) -> List[News]:
        """
        Get hot news list, cached in Redis.
        
        Args:
            limit: Number of hot news items to return
            category: Filter by category (optional)
            hours: Time window in hours for hot news
            
        Returns:
            List[News]: List of hot news items
        """
        # Build cache key
        cache_key = f"hot_news:{category or 'all'}:{hours}h:{limit}"
        
        # Try to get from cache first
        if self.redis_client:
            try:
                cached_ids = await self.redis_client.lrange(cache_key, 0, -1)
                if cached_ids:
                    # Get news from database
                    news_ids = [int(id_bytes.decode()) for id_bytes in cached_ids]
                    result = await self.db.execute(
                        select(News).where(News.id.in_(news_ids))
                    )
                    news_dict = {news.id: news for news in result.scalars().all()}
                    # Return in the same order as cached
                    return [news_dict[news_id] for news_id in news_ids if news_id in news_dict]
            except Exception:
                pass  # Fall back to database query
        
        # Query from database
        time_threshold = datetime.utcnow() - timedelta(hours=hours)
        
        query = select(News).where(News.publish_time >= time_threshold)
        
        if category:
            query = query.where(News.category == category)
        
        query = query.order_by(News.hot_score.desc()).limit(limit)
        
        result = await self.db.execute(query)
        hot_news = result.scalars().all()
        
        # Cache the results
        if self.redis_client and hot_news:
            try:
                news_ids = [str(news.id) for news in hot_news]
                await self.redis_client.delete(cache_key)
                await self.redis_client.lpush(cache_key, *news_ids)
                await self.redis_client.expire(cache_key, 300)  # Cache for 5 minutes
            except Exception:
                pass  # Continue without caching
        
        return hot_news
    
    async def get_related_news(
        self,
        news_id: int,
        limit: int = 5
    ) -> List[News]:
        """
        Find related news by content similarity and geographic relevance.
        
        Args:
            news_id: ID of the reference news
            limit: Number of related news items to return
            
        Returns:
            List[News]: List of related news items
        """
        if not self.es_client:
            raise ValueError("Elasticsearch client not configured for similarity search")
        
        # Get the reference news
        reference_news = await self.get_news(news_id)
        if not reference_news:
            return []
        
        # Build More Like This query
        search_body = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "more_like_this": {
                                "fields": ["title", "content", "translated_title", "translated_content"],
                                "like": [
                                    {
                                        "_index": self.indexing_service.index_name,
                                        "_id": news_id
                                    }
                                ],
                                "min_term_freq": 1,
                                "max_query_terms": 25,
                                "min_doc_freq": 1
                            }
                        }
                    ],
                    "must_not": [
                        {"term": {"id": news_id}}  # Exclude the reference news itself
                    ],
                    "should": []
                }
            },
            "size": limit * 2  # Get more candidates for filtering
        }
        
        # Add geographic relevance boost if location is available
        if reference_news.location:
            search_body["query"]["bool"]["should"].append({
                "term": {
                    "location": {
                        "value": reference_news.location,
                        "boost": 2.0
                    }
                }
            })
        
        # Add category relevance boost
        if reference_news.category:
            search_body["query"]["bool"]["should"].append({
                "term": {
                    "category": {
                        "value": reference_news.category,
                        "boost": 1.5
                    }
                }
            })
        
        # Add source diversity (prefer different sources)
        search_body["query"]["bool"]["should"].append({
            "bool": {
                "must_not": [
                    {"term": {"source_name": reference_news.source_name}}
                ],
                "boost": 1.2
            }
        })
        
        try:
            # Execute search
            response = await self.es_client.search(
                index=self.indexing_service.index_name,
                body=search_body
            )
            
            # Get news IDs from results
            news_ids = []
            for hit in response["hits"]["hits"]:
                news_ids.append(hit["_source"]["id"])
            
            if not news_ids:
                return []
            
            # Get full news objects from database
            result = await self.db.execute(
                select(News).where(News.id.in_(news_ids[:limit]))
            )
            related_news = result.scalars().all()
            
            # Sort by the original Elasticsearch relevance order
            news_dict = {news.id: news for news in related_news}
            ordered_news = []
            for news_id in news_ids[:limit]:
                if news_id in news_dict:
                    ordered_news.append(news_dict[news_id])
            
            return ordered_news
            
        except Exception as e:
            # Fallback to simple category-based recommendation
            return await self._get_related_news_fallback(reference_news, limit)
    
    async def _get_related_news_fallback(
        self,
        reference_news: News,
        limit: int
    ) -> List[News]:
        """
        Fallback method for related news when Elasticsearch is not available.
        
        Args:
            reference_news: Reference news object
            limit: Number of related news items to return
            
        Returns:
            List[News]: List of related news items
        """
        conditions = [News.id != reference_news.id]
        
        # Prefer same category
        if reference_news.category:
            conditions.append(News.category == reference_news.category)
        
        # Prefer same location
        if reference_news.location:
            conditions.append(News.location == reference_news.location)
        
        query = (
            select(News)
            .where(and_(*conditions))
            .order_by(News.hot_score.desc(), News.publish_time.desc())
            .limit(limit)
        )
        
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def export_news(
        self,
        news_ids: List[int],
        format: str = "json"
    ) -> bytes:
        """
        Export news items in specified format.
        
        Args:
            news_ids: List of news IDs to export
            format: Export format (json, csv, txt)
            
        Returns:
            bytes: Exported data
        """
        # Get news items
        result = await self.db.execute(
            select(News).where(News.id.in_(news_ids))
        )
        news_list = result.scalars().all()
        
        if format.lower() == "json":
            import json
            data = []
            for news in news_list:
                data.append({
                    "id": news.id,
                    "title": news.title,
                    "content": news.content,
                    "source_name": news.source_name,
                    "source_url": news.source_url,
                    "language": news.language,
                    "category": news.category,
                    "author": news.author,
                    "location": news.location,
                    "keywords": news.keywords,
                    "summary": news.summary,
                    "publish_time": news.publish_time.isoformat() if news.publish_time else None,
                    "crawl_time": news.crawl_time.isoformat(),
                    "hot_score": news.hot_score
                })
            return json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8')
        
        elif format.lower() == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow([
                "ID", "Title", "Content", "Source Name", "Source URL", 
                "Language", "Category", "Author", "Location", "Keywords",
                "Summary", "Publish Time", "Crawl Time", "Hot Score"
            ])
            
            # Write data
            for news in news_list:
                writer.writerow([
                    news.id,
                    news.title,
                    news.content,
                    news.source_name,
                    news.source_url,
                    news.language,
                    news.category,
                    news.author,
                    news.location,
                    str(news.keywords) if news.keywords else "",
                    news.summary,
                    news.publish_time.isoformat() if news.publish_time else "",
                    news.crawl_time.isoformat(),
                    news.hot_score
                ])
            
            return output.getvalue().encode('utf-8')
        
        elif format.lower() == "txt":
            lines = []
            for news in news_list:
                lines.append(f"ID: {news.id}")
                lines.append(f"Title: {news.title}")
                lines.append(f"Source: {news.source_name}")
                lines.append(f"Category: {news.category or 'N/A'}")
                lines.append(f"Author: {news.author or 'N/A'}")
                lines.append(f"Location: {news.location or 'N/A'}")
                lines.append(f"Publish Time: {news.publish_time or 'N/A'}")
                lines.append(f"Summary: {news.summary or 'N/A'}")
                lines.append(f"Content:\n{news.content}")
                lines.append("-" * 80)
                lines.append("")
            
            return "\n".join(lines).encode('utf-8')
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
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
