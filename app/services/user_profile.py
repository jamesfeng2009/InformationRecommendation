from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.news import News, UserBehavior, UserInterest
from app.models.user import User


class UserProfileServiceError(Exception):
    """Base exception for user profile service errors."""
    pass


class UserNotFoundError(UserProfileServiceError):
    """User not found."""
    pass


@dataclass
class InterestScore:
    """Interest score for a category or keyword."""
    tag: str
    score: float
    source: str  # 'behavior' or 'explicit'
    last_updated: datetime


@dataclass
class UserProfile:
    """
    User profile containing aggregated interests and preferences.
    """
    user_id: int
    explicit_interests: List[str]  # User-selected interest tags
    behavioral_interests: Dict[str, float]  # Category/keyword -> score from behavior
    combined_interests: Dict[str, InterestScore]  # All interests with scores
    preferred_categories: List[str]  # Top categories from behavior
    preferred_keywords: List[str]  # Top keywords from behavior
    activity_level: float  # User activity score (0.0 to 1.0)
    last_updated: datetime


@dataclass
class BehaviorAnalysis:
    """Analysis of user behavior patterns."""
    total_interactions: int
    category_scores: Dict[str, float]
    keyword_scores: Dict[str, float]
    source_preferences: Dict[str, float]
    interaction_patterns: Dict[str, int]  # action -> count
    recent_activity_score: float


class UserProfileService:
    """
    Service for building and managing user profiles based on behavior and explicit interests.
    
    Aggregates user interests from:
    1. User behavior (views, likes, collects, shares)
    2. Explicit interest tags selected by user
    """
    
    # Predefined interest tag library
    PREDEFINED_TAGS = [
        # Political/Military
        "政治", "军事", "国防", "外交", "国际关系", "安全", "反恐",
        # Economic
        "经济", "金融", "贸易", "投资", "股市", "货币", "产业",
        # Technology
        "科技", "人工智能", "互联网", "5G", "区块链", "新能源", "航空航天",
        # Social
        "社会", "教育", "医疗", "环保", "文化", "体育", "娱乐",
        # Regional
        "中国", "美国", "欧洲", "亚太", "中东", "非洲", "拉美",
        # Specific Topics
        "网络安全", "气候变化", "疫情", "能源", "交通", "房地产", "农业"
    ]
    
    # Behavior scoring weights
    BEHAVIOR_WEIGHTS = {
        "view": 1.0,
        "like": 3.0,
        "collect": 5.0,
        "share": 4.0,
        "dislike": -2.0,
    }
    
    def __init__(self, db: AsyncSession):
        self.db = db

    # ==================== Profile Building ====================
    
    async def build_user_profile(
        self,
        user_id: int,
        days_back: int = 30,
        min_interactions: int = 5,
    ) -> UserProfile:
        """
        Build comprehensive user profile from behavior and explicit interests.
        
        Args:
            user_id: User ID to build profile for.
            days_back: Number of days to look back for behavior analysis.
            min_interactions: Minimum interactions needed for behavioral scoring.
        
        Returns:
            UserProfile with aggregated interests and preferences.
        
        Raises:
            UserNotFoundError: If user doesn't exist.
        """
        # Verify user exists
        user = await self._get_user(user_id)
        if not user:
            raise UserNotFoundError(f"User with ID {user_id} not found")
        
        # Get explicit interests
        explicit_interests = await self.get_user_explicit_interests(user_id)
        
        # Analyze behavior
        behavior_analysis = await self.analyze_user_behavior(
            user_id, days_back, min_interactions
        )
        
        # Combine interests with scores
        combined_interests = self._combine_interests(
            explicit_interests, behavior_analysis
        )
        
        # Extract top preferences
        preferred_categories = self._get_top_items(
            behavior_analysis.category_scores, limit=5
        )
        preferred_keywords = self._get_top_items(
            behavior_analysis.keyword_scores, limit=10
        )
        
        return UserProfile(
            user_id=user_id,
            explicit_interests=explicit_interests,
            behavioral_interests=behavior_analysis.category_scores,
            combined_interests=combined_interests,
            preferred_categories=preferred_categories,
            preferred_keywords=preferred_keywords,
            activity_level=behavior_analysis.recent_activity_score,
            last_updated=datetime.utcnow(),
        )

    async def analyze_user_behavior(
        self,
        user_id: int,
        days_back: int = 30,
        min_interactions: int = 5,
    ) -> BehaviorAnalysis:
        """
        Analyze user behavior patterns to extract interests.
        
        Args:
            user_id: User ID to analyze.
            days_back: Number of days to look back.
            min_interactions: Minimum interactions for scoring.
        
        Returns:
            BehaviorAnalysis with scored interests from behavior.
        """
        # Get user behaviors within time window
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        stmt = (
            select(UserBehavior, News)
            .join(News, UserBehavior.news_id == News.id)
            .where(
                UserBehavior.user_id == user_id,
                UserBehavior.created_at >= cutoff_date,
            )
            .order_by(UserBehavior.created_at.desc())
        )
        
        result = await self.db.execute(stmt)
        behavior_news_pairs = await result.all()
        
        if len(behavior_news_pairs) < min_interactions:
            # Not enough data for meaningful analysis
            return BehaviorAnalysis(
                total_interactions=len(behavior_news_pairs),
                category_scores={},
                keyword_scores={},
                source_preferences={},
                interaction_patterns={},
                recent_activity_score=0.0,
            )
        
        # Analyze patterns
        category_scores = defaultdict(float)
        keyword_scores = defaultdict(float)
        source_scores = defaultdict(float)
        interaction_counts = Counter()
        
        for behavior, news in behavior_news_pairs:
            weight = self.BEHAVIOR_WEIGHTS.get(behavior.action, 0.0)
            
            # Score categories
            if news.category:
                category_scores[news.category] += weight
            
            # Score keywords
            if news.keywords:
                for keyword in news.keywords:
                    if isinstance(keyword, str) and len(keyword.strip()) > 1:
                        keyword_scores[keyword.strip()] += weight * 0.5
            
            # Score sources
            if news.source_name:
                source_scores[news.source_name] += weight * 0.3
            
            # Count interaction types
            interaction_counts[behavior.action] += 1
        
        # Normalize scores
        total_interactions = len(behavior_news_pairs)
        category_scores = {k: v / total_interactions for k, v in category_scores.items()}
        keyword_scores = {k: v / total_interactions for k, v in keyword_scores.items()}
        source_scores = {k: v / total_interactions for k, v in source_scores.items()}
        
        # Calculate activity level (recent activity vs historical)
        recent_cutoff = datetime.utcnow() - timedelta(days=7)
        recent_behaviors = [
            b for b, n in behavior_news_pairs 
            if b.created_at >= recent_cutoff
        ]
        activity_level = min(1.0, len(recent_behaviors) / max(1, days_back / 7))
        
        return BehaviorAnalysis(
            total_interactions=total_interactions,
            category_scores=dict(category_scores),
            keyword_scores=dict(keyword_scores),
            source_preferences=dict(source_scores),
            interaction_patterns=dict(interaction_counts),
            recent_activity_score=activity_level,
        )

    # ==================== Explicit Interest Management ====================
    
    async def get_user_explicit_interests(self, user_id: int) -> List[str]:
        """Get user's explicitly selected interest tags."""
        stmt = (
            select(UserInterest.tag)
            .where(UserInterest.user_id == user_id)
            .order_by(UserInterest.tag)
        )
        result = await self.db.execute(stmt)
        rows = await result.all()
        return [tag for tag, in rows]

    async def update_user_interests(
        self,
        user_id: int,
        interest_tags: List[str],
    ) -> List[str]:
        """
        Update user's explicit interest tags.
        
        Args:
            user_id: User ID.
            interest_tags: List of interest tags to set.
        
        Returns:
            Updated list of interest tags.
        
        Raises:
            UserNotFoundError: If user doesn't exist.
            UserProfileServiceError: If invalid tags provided.
        """
        # Verify user exists
        user = await self._get_user(user_id)
        if not user:
            raise UserNotFoundError(f"User with ID {user_id} not found")
        
        # Validate tags against predefined library
        invalid_tags = [tag for tag in interest_tags if tag not in self.PREDEFINED_TAGS]
        if invalid_tags:
            raise UserProfileServiceError(
                f"Invalid interest tags: {invalid_tags}. "
                f"Must be from predefined library."
            )
        
        # Remove existing interests
        delete_stmt = select(UserInterest).where(UserInterest.user_id == user_id)
        existing_interests = await self.db.execute(delete_stmt)
        interests_to_delete = await existing_interests.scalars()
        for interest in interests_to_delete:
            await self.db.delete(interest)
        
        # Add new interests
        for tag in set(interest_tags):  # Remove duplicates
            interest = UserInterest(user_id=user_id, tag=tag)
            self.db.add(interest)
        
        await self.db.flush()
        return sorted(set(interest_tags))

    async def add_user_interest(self, user_id: int, tag: str) -> bool:
        """
        Add a single interest tag to user's profile.
        
        Args:
            user_id: User ID.
            tag: Interest tag to add.
        
        Returns:
            True if added (False if already exists).
        
        Raises:
            UserNotFoundError: If user doesn't exist.
            UserProfileServiceError: If invalid tag.
        """
        if tag not in self.PREDEFINED_TAGS:
            raise UserProfileServiceError(
                f"Invalid interest tag: {tag}. Must be from predefined library."
            )
        
        # Check if already exists
        stmt = select(UserInterest).where(
            UserInterest.user_id == user_id,
            UserInterest.tag == tag,
        )
        existing = await self.db.execute(stmt)
        if await existing.scalar_one_or_none():
            return False
        
        # Add new interest
        interest = UserInterest(user_id=user_id, tag=tag)
        self.db.add(interest)
        await self.db.flush()
        return True

    async def remove_user_interest(self, user_id: int, tag: str) -> bool:
        """
        Remove an interest tag from user's profile.
        
        Args:
            user_id: User ID.
            tag: Interest tag to remove.
        
        Returns:
            True if removed (False if didn't exist).
        """
        stmt = select(UserInterest).where(
            UserInterest.user_id == user_id,
            UserInterest.tag == tag,
        )
        result = await self.db.execute(stmt)
        interest = await result.scalar_one_or_none()
        
        if interest:
            await self.db.delete(interest)
            await self.db.flush()
            return True
        return False

    # ==================== Profile Utilities ====================
    
    def get_predefined_tags(self) -> List[str]:
        """Get the predefined interest tag library."""
        return self.PREDEFINED_TAGS.copy()

    def get_interest_score(self, profile: UserProfile, tag: str) -> float:
        """
        Get the combined interest score for a specific tag.
        
        Args:
            profile: User profile.
            tag: Interest tag or keyword.
        
        Returns:
            Combined score (0.0 to 1.0+).
        """
        if tag in profile.combined_interests:
            return profile.combined_interests[tag].score
        return 0.0

    def get_category_affinity(self, profile: UserProfile, category: str) -> float:
        """
        Get user's affinity for a news category.
        
        Args:
            profile: User profile.
            category: News category.
        
        Returns:
            Affinity score (0.0 to 1.0+).
        """
        # Check behavioral interests
        behavioral_score = profile.behavioral_interests.get(category, 0.0)
        
        # Check if category matches explicit interests
        explicit_bonus = 0.0
        if category in profile.explicit_interests:
            explicit_bonus = 0.5
        
        return behavioral_score + explicit_bonus

    def is_interested_in_keywords(
        self,
        profile: UserProfile,
        keywords: List[str],
        threshold: float = 0.1,
    ) -> bool:
        """
        Check if user is interested in any of the given keywords.
        
        Args:
            profile: User profile.
            keywords: List of keywords to check.
            threshold: Minimum interest score threshold.
        
        Returns:
            True if user shows interest in any keyword.
        """
        for keyword in keywords:
            if self.get_interest_score(profile, keyword) >= threshold:
                return True
        return False

    # ==================== Helper Methods ====================
    
    def _combine_interests(
        self,
        explicit_interests: List[str],
        behavior_analysis: BehaviorAnalysis,
    ) -> Dict[str, InterestScore]:
        """Combine explicit and behavioral interests with scores."""
        combined = {}
        now = datetime.utcnow()
        
        # Add explicit interests with high base score
        for tag in explicit_interests:
            combined[tag] = InterestScore(
                tag=tag,
                score=1.0,  # High score for explicit interests
                source="explicit",
                last_updated=now,
            )
        
        # Add behavioral interests
        for tag, score in behavior_analysis.category_scores.items():
            if tag in combined:
                # Boost existing explicit interest
                combined[tag].score += score * 0.5
            else:
                # Add new behavioral interest
                combined[tag] = InterestScore(
                    tag=tag,
                    score=score,
                    source="behavior",
                    last_updated=now,
                )
        
        # Add top keywords from behavior
        top_keywords = self._get_top_items(behavior_analysis.keyword_scores, limit=20)
        for keyword in top_keywords:
            score = behavior_analysis.keyword_scores[keyword]
            if keyword not in combined and score > 0.1:  # Only significant keywords
                combined[keyword] = InterestScore(
                    tag=keyword,
                    score=score,
                    source="behavior",
                    last_updated=now,
                )
        
        return combined

    def _get_top_items(self, scores: Dict[str, float], limit: int) -> List[str]:
        """Get top items by score."""
        if not scores:
            return []
        
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, score in sorted_items[:limit] if score > 0]

    async def _get_user(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        stmt = select(User).where(User.id == user_id)
        result = await self.db.execute(stmt)
        return await result.scalar_one_or_none()