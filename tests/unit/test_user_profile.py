import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from app.services.user_profile import (
    UserProfileService,
    UserProfile,
    BehaviorAnalysis,
    InterestScore,
    UserNotFoundError,
    UserProfileServiceError,
)
from app.models.user import User
from app.models.news import News, UserBehavior, UserInterest


class TestUserProfileService:
    """Unit tests for UserProfileService."""

    @pytest.fixture
    def mock_db(self):
        """Mock database session."""
        return AsyncMock()

    @pytest.fixture
    def service(self, mock_db):
        """UserProfileService instance with mocked database."""
        return UserProfileService(mock_db)

    @pytest.fixture
    def sample_user(self):
        """Sample user for testing."""
        return User(
            id=1,
            username="testuser",
            name="Test User",
            account="testuser",
            status="enabled",
        )

    @pytest.fixture
    def sample_news(self):
        """Sample news items for testing."""
        return [
            News(
                id=1,
                title="Military News",
                content="Military content",
                category="军事",
                keywords=["国防", "安全"],
                source_name="军事网",
            ),
            News(
                id=2,
                title="Tech News",
                content="Technology content",
                category="科技",
                keywords=["人工智能", "5G"],
                source_name="科技日报",
            ),
        ]

    @pytest.fixture
    def sample_behaviors(self, sample_news):
        """Sample user behaviors for testing."""
        now = datetime.utcnow()
        return [
            (
                UserBehavior(
                    user_id=1,
                    news_id=1,
                    action="like",
                    created_at=now - timedelta(days=1),
                ),
                sample_news[0],
            ),
            (
                UserBehavior(
                    user_id=1,
                    news_id=2,
                    action="view",
                    created_at=now - timedelta(days=2),
                ),
                sample_news[1],
            ),
            (
                UserBehavior(
                    user_id=1,
                    news_id=1,
                    action="collect",
                    created_at=now - timedelta(days=3),
                ),
                sample_news[0],
            ),
        ]

    # ==================== Profile Building Tests ====================

    @pytest.mark.asyncio
    async def test_build_user_profile_success(
        self, service, mock_db, sample_user, sample_behaviors
    ):
        """Test successful user profile building."""
        # Mock database queries
        mock_result_user = AsyncMock()
        mock_result_user.scalar_one_or_none.return_value = sample_user
        
        mock_result_behaviors = AsyncMock()
        mock_result_behaviors.all = AsyncMock(return_value=sample_behaviors)
        
        mock_result_interests = AsyncMock()
        mock_result_interests.all = AsyncMock(return_value=[("军事",), ("科技",)])
        
        mock_db.execute.side_effect = [
            mock_result_user,  # _get_user call
            mock_result_interests,  # get_user_explicit_interests call
            mock_result_behaviors,  # analyze_user_behavior call
        ]

        profile = await service.build_user_profile(1)

        assert profile.user_id == 1
        assert isinstance(profile.explicit_interests, list)
        assert isinstance(profile.behavioral_interests, dict)
        assert isinstance(profile.combined_interests, dict)
        assert isinstance(profile.activity_level, float)
        assert 0.0 <= profile.activity_level <= 1.0

    @pytest.mark.asyncio
    async def test_build_user_profile_user_not_found(self, service, mock_db):
        """Test profile building with non-existent user."""
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        with pytest.raises(UserNotFoundError):
            await service.build_user_profile(999)

    @pytest.mark.asyncio
    async def test_analyze_user_behavior_sufficient_data(
        self, service, mock_db, sample_behaviors
    ):
        """Test behavior analysis with sufficient interaction data."""
        mock_result = AsyncMock()
        mock_result.all = AsyncMock(return_value=sample_behaviors)
        mock_db.execute.return_value = mock_result

        analysis = await service.analyze_user_behavior(1, days_back=30, min_interactions=2)

        assert analysis.total_interactions == 3
        assert "军事" in analysis.category_scores
        assert "科技" in analysis.category_scores
        assert analysis.category_scores["军事"] > analysis.category_scores["科技"]  # More interactions
        assert isinstance(analysis.keyword_scores, dict)
        assert isinstance(analysis.interaction_patterns, dict)
        assert 0.0 <= analysis.recent_activity_score <= 1.0

    @pytest.mark.asyncio
    async def test_analyze_user_behavior_insufficient_data(self, service, mock_db):
        """Test behavior analysis with insufficient interaction data."""
        mock_result = AsyncMock()
        mock_result.all = AsyncMock(return_value=[])  # No behaviors
        mock_db.execute.return_value = mock_result

        analysis = await service.analyze_user_behavior(1, days_back=30, min_interactions=5)

        assert analysis.total_interactions == 0
        assert analysis.category_scores == {}
        assert analysis.keyword_scores == {}
        assert analysis.recent_activity_score == 0.0

    # ==================== Explicit Interest Tests ====================

    @pytest.mark.asyncio
    async def test_get_user_explicit_interests(self, service, mock_db):
        """Test getting user's explicit interests."""
        mock_result = AsyncMock()
        mock_result.all = AsyncMock(return_value=[("军事",), ("科技",), ("经济",)])
        mock_db.execute.return_value = mock_result

        interests = await service.get_user_explicit_interests(1)

        assert interests == ["军事", "科技", "经济"]

    @pytest.mark.asyncio
    async def test_update_user_interests_success(self, service, mock_db, sample_user):
        """Test successful interest update."""
        # Mock _get_user call
        mock_result_user = AsyncMock()
        mock_result_user.scalar_one_or_none.return_value = sample_user
        
        # Mock existing interests query
        mock_result_interests = AsyncMock()
        mock_result_interests.scalars = AsyncMock(return_value=[])  # No existing interests
        
        mock_db.execute.side_effect = [mock_result_user, mock_result_interests]
        mock_db.delete = AsyncMock()

        new_interests = ["军事", "科技", "经济"]
        result = await service.update_user_interests(1, new_interests)

        assert result == new_interests
        assert mock_db.add.call_count == 3  # Three interests added

    @pytest.mark.asyncio
    async def test_update_user_interests_invalid_tags(self, service, mock_db, sample_user):
        """Test interest update with invalid tags."""
        mock_db.execute.return_value.scalar_one_or_none.return_value = sample_user

        invalid_interests = ["军事", "invalid_tag", "科技"]
        
        with pytest.raises(UserProfileServiceError) as exc_info:
            await service.update_user_interests(1, invalid_interests)
        
        assert "Invalid interest tags" in str(exc_info.value)
        assert "invalid_tag" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_add_user_interest_success(self, service, mock_db):
        """Test adding a single interest."""
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = AsyncMock(return_value=None)  # Doesn't exist
        mock_db.execute.return_value = mock_result
        mock_db.flush = AsyncMock()

        result = await service.add_user_interest(1, "军事")

        assert result is True
        mock_db.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_user_interest_already_exists(self, service, mock_db):
        """Test adding an interest that already exists."""
        mock_interest = UserInterest(user_id=1, tag="军事")
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = AsyncMock(return_value=mock_interest)
        mock_db.execute.return_value = mock_result

        result = await service.add_user_interest(1, "军事")

        assert result is False
        mock_db.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_user_interest_invalid_tag(self, service, mock_db):
        """Test adding an invalid interest tag."""
        with pytest.raises(UserProfileServiceError) as exc_info:
            await service.add_user_interest(1, "invalid_tag")
        
        assert "Invalid interest tag" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_remove_user_interest_success(self, service, mock_db):
        """Test removing an existing interest."""
        mock_interest = UserInterest(user_id=1, tag="军事")
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = AsyncMock(return_value=mock_interest)
        mock_db.execute.return_value = mock_result
        mock_db.delete = AsyncMock()
        mock_db.flush = AsyncMock()

        result = await service.remove_user_interest(1, "军事")

        assert result is True
        mock_db.delete.assert_called_once_with(mock_interest)

    @pytest.mark.asyncio
    async def test_remove_user_interest_not_exists(self, service, mock_db):
        """Test removing a non-existent interest."""
        mock_result = AsyncMock()
        mock_result.scalar_one_or_none = AsyncMock(return_value=None)
        mock_db.execute.return_value = mock_result
        mock_db.delete = AsyncMock()

        result = await service.remove_user_interest(1, "军事")

        assert result is False
        mock_db.delete.assert_not_called()

    # ==================== Utility Tests ====================

    def test_get_predefined_tags(self, service):
        """Test getting predefined tag library."""
        tags = service.get_predefined_tags()
        
        assert isinstance(tags, list)
        assert len(tags) > 0
        assert "军事" in tags
        assert "科技" in tags
        assert "经济" in tags

    def test_get_interest_score(self, service):
        """Test getting interest score from profile."""
        profile = UserProfile(
            user_id=1,
            explicit_interests=["军事"],
            behavioral_interests={"科技": 0.5},
            combined_interests={
                "军事": InterestScore("军事", 1.0, "explicit", datetime.utcnow()),
                "科技": InterestScore("科技", 0.5, "behavior", datetime.utcnow()),
            },
            preferred_categories=["军事"],
            preferred_keywords=["国防"],
            activity_level=0.8,
            last_updated=datetime.utcnow(),
        )

        assert service.get_interest_score(profile, "军事") == 1.0
        assert service.get_interest_score(profile, "科技") == 0.5
        assert service.get_interest_score(profile, "经济") == 0.0

    def test_get_category_affinity(self, service):
        """Test getting category affinity from profile."""
        profile = UserProfile(
            user_id=1,
            explicit_interests=["军事"],
            behavioral_interests={"军事": 0.3, "科技": 0.5},
            combined_interests={},
            preferred_categories=["军事", "科技"],
            preferred_keywords=[],
            activity_level=0.8,
            last_updated=datetime.utcnow(),
        )

        # Explicit + behavioral
        assert service.get_category_affinity(profile, "军事") == 0.8  # 0.3 + 0.5 bonus
        # Behavioral only
        assert service.get_category_affinity(profile, "科技") == 0.5
        # Neither
        assert service.get_category_affinity(profile, "经济") == 0.0

    def test_is_interested_in_keywords(self, service):
        """Test checking keyword interest."""
        profile = UserProfile(
            user_id=1,
            explicit_interests=[],
            behavioral_interests={},
            combined_interests={
                "国防": InterestScore("国防", 0.8, "behavior", datetime.utcnow()),
                "安全": InterestScore("安全", 0.05, "behavior", datetime.utcnow()),
            },
            preferred_categories=[],
            preferred_keywords=["国防"],
            activity_level=0.8,
            last_updated=datetime.utcnow(),
        )

        assert service.is_interested_in_keywords(profile, ["国防", "军事"], threshold=0.1)
        assert not service.is_interested_in_keywords(profile, ["安全"], threshold=0.1)
        assert not service.is_interested_in_keywords(profile, ["经济"], threshold=0.1)

    def test_combine_interests(self, service):
        """Test combining explicit and behavioral interests."""
        explicit_interests = ["军事", "科技"]
        behavior_analysis = BehaviorAnalysis(
            total_interactions=10,
            category_scores={"军事": 0.6, "经济": 0.3},
            keyword_scores={"国防": 0.4, "AI": 0.2},
            source_preferences={},
            interaction_patterns={},
            recent_activity_score=0.8,
        )

        combined = service._combine_interests(explicit_interests, behavior_analysis)

        # Explicit interests should have high base score
        assert combined["军事"].source == "explicit"
        assert combined["科技"].source == "explicit"
        
        # Behavioral interests should be added
        assert combined["经济"].source == "behavior"
        assert combined["国防"].source == "behavior"
        
        # Military should have boosted score (explicit + behavioral)
        assert combined["军事"].score > 1.0

    def test_get_top_items(self, service):
        """Test getting top items by score."""
        scores = {"军事": 0.8, "科技": 0.6, "经济": 0.4, "政治": 0.2}
        
        top_2 = service._get_top_items(scores, limit=2)
        assert top_2 == ["军事", "科技"]
        
        top_all = service._get_top_items(scores, limit=10)
        assert len(top_all) == 4
        assert top_all[0] == "军事"  # Highest score first
        
        empty_result = service._get_top_items({}, limit=5)
        assert empty_result == []