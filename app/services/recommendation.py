import asyncio
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.redis import get_redis
from app.models.news import News, UserBehavior, UserInterest
from app.models.user import User
from app.services.user_profile import UserProfileService, UserProfile


class RecommendationServiceError(Exception):
    """Base exception for recommendation service errors."""
    pass


class RecommendationService:
    """
    Personalized news recommendation service.
    
    Implements hybrid recommendation combining:
    1. Content-based filtering using news embeddings and categories
    2. Collaborative filtering based on similar users
    3. Popularity-based recommendations for cold start
    """
    
    def __init__(self, db: AsyncSession, user_profile_service: UserProfileService):
        self.db = db
        self.user_profile_service = user_profile_service
        self.redis = get_redis()
        
        # Recommendation parameters
        self.content_weight = 0.4
        self.collaborative_weight = 0.3
        self.popularity_weight = 0.3
        self.min_interactions_for_cf = 5  # Minimum interactions for collaborative filtering
        self.cache_ttl = 3600  # 1 hour cache TTL
        
    # ==================== Main Recommendation Methods ====================
    
    async def get_personalized_feed(
        self,
        user_id: int,
        page: int = 1,
        size: int = 20,
        exclude_read: bool = True,
    ) -> List[Dict]:
        """
        Get personalized news feed for user.
        
        Args:
            user_id: User ID to generate recommendations for.
            page: Page number (1-based).
            size: Number of items per page.
            exclude_read: Whether to exclude already read news.
        
        Returns:
            List of recommended news items with scores.
        """
        # Check cache first
        cache_key = f"recommend:feed:{user_id}:{page}:{size}:{exclude_read}"
        cached = await self.redis.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Get user profile
        try:
            user_profile = await self.user_profile_service.build_user_profile(user_id)
        except Exception:
            # Fallback to popular news for users without profile
            return await self.get_popular_news(page, size)
        
        # Get candidate news (recent news not yet read)
        candidates = await self._get_candidate_news(
            user_id, 
            days_back=7, 
            limit=size * 10,  # Get more candidates for better filtering
            exclude_read=exclude_read
        )
        
        if not candidates:
            return await self.get_popular_news(page, size)
        
        # Generate hybrid recommendations
        recommendations = await self._generate_hybrid_recommendations(
            user_profile, candidates, user_id
        )
        
        # Paginate results
        start_idx = (page - 1) * size
        end_idx = start_idx + size
        result = recommendations[start_idx:end_idx]
        
        # Cache results
        await self.redis.setex(cache_key, self.cache_ttl, json.dumps(result))
        
        return result

    async def get_latest_feed(
        self,
        page: int = 1,
        size: int = 20,
        category: Optional[str] = None,
    ) -> List[Dict]:
        """
        Get latest news feed (chronological order).
        
        Args:
            page: Page number (1-based).
            size: Number of items per page.
            category: Optional category filter.
        
        Returns:
            List of latest news items.
        """
        # Build query
        stmt = select(News).order_by(desc(News.publish_time))
        
        if category:
            stmt = stmt.where(News.category == category)
        
        # Apply pagination
        offset = (page - 1) * size
        stmt = stmt.offset(offset).limit(size)
        
        result = await self.db.execute(stmt)
        news_items = result.scalars().all()
        
        return [self._format_news_item(news) for news in news_items]

    async def get_popular_news(
        self,
        page: int = 1,
        size: int = 20,
        time_window_hours: int = 24,
    ) -> List[Dict]:
        """
        Get popular news based on hot score and recent interactions.
        
        Args:
            page: Page number (1-based).
            size: Number of items per page.
            time_window_hours: Time window for popularity calculation.
        
        Returns:
            List of popular news items.
        """
        # Check cache first
        cache_key = f"recommend:popular:{page}:{size}:{time_window_hours}"
        cached = await self.redis.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Get popular news based on hot score and recent activity
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        
        stmt = (
            select(News)
            .where(News.publish_time >= cutoff_time)
            .order_by(desc(News.hot_score), desc(News.publish_time))
            .offset((page - 1) * size)
            .limit(size)
        )
        
        result = await self.db.execute(stmt)
        news_items = result.scalars().all()
        
        formatted_items = [self._format_news_item(news) for news in news_items]
        
        # Cache results
        await self.redis.setex(cache_key, self.cache_ttl, json.dumps(formatted_items))
        
        return formatted_items

    async def get_topic_hot_news(
        self,
        user_id: int,
        limit: int = 10,
        time_window_hours: int = 24,
    ) -> List[Dict]:
        """
        Get topic hot news filtered by user keywords and topic keywords.
        
        Combines user's explicit interests and keywords with trending topics
        to surface relevant hot news.
        
        Args:
            user_id: User ID to get personalized topic news for.
            limit: Maximum number of news items to return.
            time_window_hours: Time window for hot news calculation.
        
        Returns:
            List of topic hot news items ranked by relevance and recency.
        """
        # Check cache first
        cache_key = f"recommend:topic_hot:{user_id}:{limit}:{time_window_hours}"
        cached = await self.redis.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Get user profile for keywords and interests
        try:
            user_profile = await self.user_profile_service.build_user_profile(user_id)
        except Exception:
            # Fallback to general hot news
            return await self.get_popular_news(page=1, size=limit, time_window_hours=time_window_hours)
        
        # Get user's keywords from profile and explicit settings
        user_keywords = set()
        
        # Add keywords from user profile
        if user_profile.preferred_keywords:
            user_keywords.update(user_profile.preferred_keywords)
        
        # Add explicit interest tags
        if user_profile.explicit_interests:
            user_keywords.update(user_profile.explicit_interests)
        
        # Get user's manual keywords from User model
        user = await self._get_user(user_id)
        if user and user.keywords:
            # Parse user keywords (assuming comma-separated)
            manual_keywords = [kw.strip() for kw in user.keywords.split(',') if kw.strip()]
            user_keywords.update(manual_keywords)
        
        if not user_keywords:
            # No keywords available, return general hot news
            return await self.get_popular_news(page=1, size=limit, time_window_hours=time_window_hours)
        
        # Get hot news within time window
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        
        # Build query to find news matching user keywords
        stmt = (
            select(News)
            .where(
                News.publish_time >= cutoff_time,
                News.hot_score > 0  # Only news with some popularity
            )
            .order_by(desc(News.hot_score), desc(News.publish_time))
            .limit(limit * 3)  # Get more candidates for filtering
        )
        
        result = await self.db.execute(stmt)
        candidate_news = result.scalars().all()
        
        # Score and filter news based on keyword relevance
        scored_news = []
        
        for news in candidate_news:
            relevance_score = self._calculate_topic_relevance(news, user_keywords)
            
            if relevance_score > 0:  # Only include relevant news
                scored_news.append({
                    **self._format_news_item(news),
                    "relevance_score": relevance_score,
                    "topic_match_keywords": self._get_matching_keywords(news, user_keywords),
                })
        
        # Sort by combined score (relevance + hot score + recency)
        scored_news.sort(
            key=lambda x: self._calculate_topic_hot_score(x, time_window_hours),
            reverse=True
        )
        
        # Take top results
        result_news = scored_news[:limit]
        
        # Cache results
        await self.redis.setex(cache_key, self.cache_ttl, json.dumps(result_news))
        
        return result_news

    def _calculate_topic_relevance(self, news: News, user_keywords: Set[str]) -> float:
        """
        Calculate how relevant a news item is to user's topic keywords.
        
        Args:
            news: News item to score.
            user_keywords: Set of user's keywords and interests.
        
        Returns:
            Relevance score between 0.0 and 1.0.
        """
        score = 0.0
        matches = 0
        
        # Check title for keyword matches (higher weight)
        if news.title:
            title_lower = news.title.lower()
            for keyword in user_keywords:
                if keyword.lower() in title_lower:
                    score += 0.3
                    matches += 1
        
        # Check category match
        if news.category and news.category in user_keywords:
            score += 0.4
            matches += 1
        
        # Check news keywords
        if news.keywords:
            for news_keyword in news.keywords:
                if isinstance(news_keyword, str) and news_keyword in user_keywords:
                    score += 0.2
                    matches += 1
        
        # Check content for keyword matches (lower weight, first 500 chars)
        if news.content:
            content_sample = news.content[:500].lower()
            for keyword in user_keywords:
                if keyword.lower() in content_sample:
                    score += 0.1
                    matches += 1
        
        # Normalize score based on number of matches
        if matches > 0:
            # Boost score for multiple matches but cap at 1.0
            score = min(1.0, score + (matches - 1) * 0.05)
        
        return score

    def _get_matching_keywords(self, news: News, user_keywords: Set[str]) -> List[str]:
        """Get list of user keywords that match this news item."""
        matching = []
        
        # Check title
        if news.title:
            title_lower = news.title.lower()
            for keyword in user_keywords:
                if keyword.lower() in title_lower:
                    matching.append(keyword)
        
        # Check category
        if news.category and news.category in user_keywords:
            matching.append(news.category)
        
        # Check news keywords
        if news.keywords:
            for news_keyword in news.keywords:
                if isinstance(news_keyword, str) and news_keyword in user_keywords:
                    matching.append(news_keyword)
        
        return list(set(matching))  # Remove duplicates

    def _calculate_topic_hot_score(self, news_item: Dict, time_window_hours: int) -> float:
        """
        Calculate combined score for topic hot news ranking.
        
        Combines relevance, hot score, and recency.
        """
        relevance_score = news_item.get("relevance_score", 0.0)
        hot_score = news_item.get("hot_score", 0.0)
        
        # Calculate recency score
        publish_time_str = news_item.get("publish_time")
        if publish_time_str:
            try:
                publish_time = datetime.fromisoformat(publish_time_str.replace('Z', '+00:00'))
                hours_old = (datetime.utcnow() - publish_time.replace(tzinfo=None)).total_seconds() / 3600
                recency_score = max(0.0, 1.0 - (hours_old / time_window_hours))
            except Exception:
                recency_score = 0.0
        else:
            recency_score = 0.0
        
        # Weighted combination
        # Relevance is most important for topic news
        combined_score = (
            0.5 * relevance_score +
            0.3 * (hot_score / 10.0 if hot_score else 0.0) +  # Normalize hot score
            0.2 * recency_score
        )
        
        return combined_score

    async def _get_user(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        stmt = select(User).where(User.id == user_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    # ==================== Hybrid Recommendation Engine ====================
    
    async def _generate_hybrid_recommendations(
        self,
        user_profile: UserProfile,
        candidates: List[News],
        user_id: int,
    ) -> List[Dict]:
        """Generate hybrid recommendations combining multiple approaches."""
        
        # Get scores from different approaches
        content_scores = await self._content_based_scoring(user_profile, candidates)
        collaborative_scores = await self._collaborative_filtering_scoring(user_id, candidates)
        popularity_scores = self._popularity_based_scoring(candidates)
        
        # Combine scores with weights
        final_scores = {}
        for news_id in set(content_scores.keys()) | set(collaborative_scores.keys()) | set(popularity_scores.keys()):
            content_score = content_scores.get(news_id, 0.0)
            collaborative_score = collaborative_scores.get(news_id, 0.0)
            popularity_score = popularity_scores.get(news_id, 0.0)
            
            # Weighted combination
            final_score = (
                self.content_weight * content_score +
                self.collaborative_weight * collaborative_score +
                self.popularity_weight * popularity_score
            )
            
            final_scores[news_id] = final_score
        
        # Sort by final score and format results
        sorted_candidates = sorted(
            candidates,
            key=lambda news: final_scores.get(news.id, 0.0),
            reverse=True
        )
        
        return [
            {
                **self._format_news_item(news),
                "recommendation_score": final_scores.get(news.id, 0.0),
                "content_score": content_scores.get(news.id, 0.0),
                "collaborative_score": collaborative_scores.get(news.id, 0.0),
                "popularity_score": popularity_scores.get(news.id, 0.0),
            }
            for news in sorted_candidates
        ]

    async def _content_based_scoring(
        self,
        user_profile: UserProfile,
        candidates: List[News],
    ) -> Dict[int, float]:
        """
        Score news items based on content similarity to user interests.
        
        Uses category affinity and keyword matching.
        """
        scores = {}
        
        for news in candidates:
            score = 0.0
            
            # Category affinity
            if news.category:
                category_affinity = self.user_profile_service.get_category_affinity(
                    user_profile, news.category
                )
                score += category_affinity * 0.6
            
            # Keyword matching
            if news.keywords:
                keyword_match = self.user_profile_service.is_interested_in_keywords(
                    user_profile, news.keywords, threshold=0.1
                )
                if keyword_match:
                    score += 0.4
            
            # Title/content keyword matching (simple approach)
            if user_profile.preferred_keywords:
                title_content = f"{news.title} {news.content[:500]}".lower()
                keyword_matches = sum(
                    1 for keyword in user_profile.preferred_keywords
                    if keyword.lower() in title_content
                )
                if keyword_matches > 0:
                    score += min(0.3, keyword_matches * 0.1)
            
            scores[news.id] = min(1.0, score)  # Cap at 1.0
        
        return scores

    async def _collaborative_filtering_scoring(
        self,
        user_id: int,
        candidates: List[News],
    ) -> Dict[int, float]:
        """
        Score news items based on collaborative filtering.
        
        Finds users with similar behavior and recommends news they liked.
        """
        scores = {}
        
        # Get user's behavior history
        user_behaviors = await self._get_user_behavior_history(user_id)
        
        if len(user_behaviors) < self.min_interactions_for_cf:
            # Not enough data for collaborative filtering
            return scores
        
        # Find similar users
        similar_users = await self._find_similar_users(user_id, user_behaviors)
        
        if not similar_users:
            return scores
        
        # Get recommendations from similar users
        candidate_ids = [news.id for news in candidates]
        
        for similar_user_id, similarity_score in similar_users[:10]:  # Top 10 similar users
            similar_user_likes = await self._get_user_liked_news(similar_user_id, candidate_ids)
            
            for news_id in similar_user_likes:
                if news_id not in scores:
                    scores[news_id] = 0.0
                scores[news_id] += similarity_score * 0.1  # Weight by similarity
        
        # Normalize scores
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v / max_score for k, v in scores.items()}
        
        return scores

    def _popularity_based_scoring(self, candidates: List[News]) -> Dict[int, float]:
        """
        Score news items based on popularity metrics.
        
        Uses hot_score and recency.
        """
        scores = {}
        
        if not candidates:
            return scores
        
        # Get max hot score for normalization
        max_hot_score = max((news.hot_score or 0.0) for news in candidates)
        if max_hot_score == 0:
            max_hot_score = 1.0
        
        # Calculate recency scores
        now = datetime.utcnow()
        
        for news in candidates:
            # Hot score component (normalized)
            hot_score = (news.hot_score or 0.0) / max_hot_score
            
            # Recency component (decay over time)
            if news.publish_time:
                hours_old = (now - news.publish_time).total_seconds() / 3600
                recency_score = math.exp(-hours_old / 24)  # Exponential decay over 24 hours
            else:
                recency_score = 0.0
            
            # Combine hot score and recency
            final_score = 0.7 * hot_score + 0.3 * recency_score
            scores[news.id] = final_score
        
        return scores

    # ==================== Helper Methods ====================
    
    async def _get_candidate_news(
        self,
        user_id: int,
        days_back: int = 7,
        limit: int = 200,
        exclude_read: bool = True,
    ) -> List[News]:
        """Get candidate news items for recommendation."""
        
        # Get recent news
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        stmt = (
            select(News)
            .where(News.publish_time >= cutoff_date)
            .order_by(desc(News.publish_time))
            .limit(limit)
        )
        
        # Exclude already read news if requested
        if exclude_read:
            read_news_subquery = (
                select(UserBehavior.news_id)
                .where(
                    UserBehavior.user_id == user_id,
                    UserBehavior.action == "view"
                )
            )
            stmt = stmt.where(~News.id.in_(read_news_subquery))
        
        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def _get_user_behavior_history(
        self,
        user_id: int,
        days_back: int = 30,
    ) -> List[Tuple[int, str]]:
        """Get user's behavior history as (news_id, action) tuples."""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        stmt = (
            select(UserBehavior.news_id, UserBehavior.action)
            .where(
                UserBehavior.user_id == user_id,
                UserBehavior.created_at >= cutoff_date,
            )
            .order_by(desc(UserBehavior.created_at))
        )
        
        result = await self.db.execute(stmt)
        return result.all()

    async def _find_similar_users(
        self,
        user_id: int,
        user_behaviors: List[Tuple[int, str]],
        limit: int = 20,
    ) -> List[Tuple[int, float]]:
        """
        Find users with similar behavior patterns.
        
        Returns list of (user_id, similarity_score) tuples.
        """
        # Get news items the user interacted with
        user_news_ids = set(news_id for news_id, action in user_behaviors if action in ["like", "collect"])
        
        if not user_news_ids:
            return []
        
        # Find other users who interacted with the same news
        stmt = (
            select(UserBehavior.user_id, func.count(UserBehavior.news_id).label("common_count"))
            .where(
                UserBehavior.news_id.in_(user_news_ids),
                UserBehavior.user_id != user_id,
                UserBehavior.action.in_(["like", "collect"]),
            )
            .group_by(UserBehavior.user_id)
            .having(func.count(UserBehavior.news_id) >= 2)  # At least 2 common interactions
            .order_by(desc("common_count"))
            .limit(limit)
        )
        
        result = await self.db.execute(stmt)
        similar_users_data = result.all()
        
        # Calculate similarity scores (Jaccard similarity)
        similar_users = []
        for other_user_id, common_count in similar_users_data:
            # Get other user's liked news
            other_user_stmt = (
                select(UserBehavior.news_id)
                .where(
                    UserBehavior.user_id == other_user_id,
                    UserBehavior.action.in_(["like", "collect"]),
                )
            )
            other_result = await self.db.execute(other_user_stmt)
            other_news_ids = set(news_id for news_id, in other_result.all())
            
            # Calculate Jaccard similarity
            intersection = len(user_news_ids & other_news_ids)
            union = len(user_news_ids | other_news_ids)
            
            if union > 0:
                similarity = intersection / union
                similar_users.append((other_user_id, similarity))
        
        # Sort by similarity score
        similar_users.sort(key=lambda x: x[1], reverse=True)
        return similar_users

    async def _get_user_liked_news(
        self,
        user_id: int,
        candidate_news_ids: List[int],
    ) -> List[int]:
        """Get news items that a user liked from the candidate list."""
        
        stmt = (
            select(UserBehavior.news_id)
            .where(
                UserBehavior.user_id == user_id,
                UserBehavior.news_id.in_(candidate_news_ids),
                UserBehavior.action.in_(["like", "collect"]),
            )
        )
        
        result = await self.db.execute(stmt)
        return [news_id for news_id, in result.all()]

    def _format_news_item(self, news: News) -> Dict:
        """Format news item for API response."""
        return {
            "id": news.id,
            "title": news.title,
            "summary": news.summary,
            "category": news.category,
            "source_name": news.source_name,
            "author": news.author,
            "location": news.location,
            "publish_time": news.publish_time.isoformat() if news.publish_time else None,
            "hot_score": news.hot_score,
            "keywords": news.keywords or [],
            "images": news.images or [],
        }

    # ==================== Cache Management ====================
    
    async def invalidate_user_cache(self, user_id: int):
        """Invalidate cached recommendations for a user."""
        pattern = f"recommend:feed:{user_id}:*"
        keys = await self.redis.keys(pattern)
        if keys:
            await self.redis.delete(*keys)

    async def warm_cache_for_active_users(self, limit: int = 100):
        """Pre-generate recommendations for active users."""
        # Get recently active users
        cutoff_date = datetime.utcnow() - timedelta(days=7)
        stmt = (
            select(UserBehavior.user_id)
            .where(UserBehavior.created_at >= cutoff_date)
            .group_by(UserBehavior.user_id)
            .order_by(desc(func.count(UserBehavior.id)))
            .limit(limit)
        )
        
        result = await self.db.execute(stmt)
        active_user_ids = [user_id for user_id, in result.all()]
        
        # Generate recommendations for each active user
        for user_id in active_user_ids:
            try:
                await self.get_personalized_feed(user_id, page=1, size=20)
            except Exception:
                # Skip users with errors
                continue