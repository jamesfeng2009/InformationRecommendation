"""
Unit tests for SQLAlchemy data models.
Tests model relationships, constraints, and validations.
Requirements: 23.1
"""
import pytest
from datetime import datetime

from app.models.user import User, Department
from app.models.rbac import Role, Permission, UserRole, RolePermission
from app.models.news import News, Collection, UserBehavior, UserInterest
from app.models.topic import Topic, TopicShare, TopicNews
from app.models.system import SensitiveWord, LoginLog, OperationLog, NewsSource


class TestDepartmentModel:
    """Tests for Department model."""

    def test_department_creation(self):
        """Test creating a department with required fields."""
        dept = Department(
            id=1,
            name="Engineering",
            parent_id=None,
            manager="John Doe",
            contact="john@example.com",
            description="Engineering department",
            sort_order=1
        )
        assert dept.name == "Engineering"
        assert dept.parent_id is None
        assert dept.manager == "John Doe"

    def test_department_tree_relationship(self):
        """Test department parent-child relationship."""
        root = Department(id=1, name="Root", parent_id=None)
        child = Department(id=2, name="Child", parent_id=1)
        child.parent = root
        
        assert child.parent == root
        assert child.get_depth() == 1

    def test_department_get_depth_root(self):
        """Test get_depth returns 0 for root department."""
        root = Department(id=1, name="Root", parent_id=None)
        assert root.get_depth() == 0

    def test_department_get_depth_nested(self):
        """Test get_depth for nested departments."""
        root = Department(id=1, name="Root", parent_id=None)
        level1 = Department(id=2, name="Level1", parent_id=1)
        level1.parent = root
        level2 = Department(id=3, name="Level2", parent_id=2)
        level2.parent = level1
        
        assert root.get_depth() == 0
        assert level1.get_depth() == 1
        assert level2.get_depth() == 2


class TestUserModel:
    """Tests for User model."""

    def test_user_creation(self):
        """Test creating a user with required fields."""
        user = User(
            id=1,
            username="testuser",
            password_hash="hashed_password",
            name="Test User",
            account="testuser",
            department_id=1,
            status="enabled"
        )
        assert user.username == "testuser"
        assert user.status == "enabled"

    def test_user_status_can_be_set(self):
        """Test user status can be set explicitly."""
        user = User(
            id=1,
            username="testuser",
            password_hash="hashed_password",
            name="Test User",
            account="testuser",
            status="disabled"
        )
        assert user.status == "disabled"

    def test_user_keywords_optional(self):
        """Test user keywords field is optional."""
        user = User(
            id=1,
            username="testuser",
            password_hash="hashed_password",
            name="Test User",
            account="testuser"
        )
        assert user.keywords is None


class TestRoleModel:
    """Tests for Role model."""

    def test_role_creation(self):
        """Test creating a role."""
        role = Role(
            id=1,
            name="admin",
            description="Administrator role"
        )
        assert role.name == "admin"
        assert role.description == "Administrator role"


class TestPermissionModel:
    """Tests for Permission model."""

    def test_permission_creation(self):
        """Test creating a permission."""
        permission = Permission(
            id=1,
            resource="users",
            action="read",
            description="Read users"
        )
        assert permission.resource == "users"
        assert permission.action == "read"


class TestUserRoleModel:
    """Tests for UserRole junction model."""

    def test_user_role_creation(self):
        """Test creating a user-role association."""
        user_role = UserRole(
            id=1,
            user_id=1,
            role_id=1
        )
        assert user_role.user_id == 1
        assert user_role.role_id == 1


class TestRolePermissionModel:
    """Tests for RolePermission junction model."""

    def test_role_permission_creation(self):
        """Test creating a role-permission association."""
        role_permission = RolePermission(
            id=1,
            role_id=1,
            permission_id=1
        )
        assert role_permission.role_id == 1
        assert role_permission.permission_id == 1


class TestNewsModel:
    """Tests for News model."""

    def test_news_creation(self):
        """Test creating a news article."""
        news = News(
            id=1,
            title="Test News",
            content="This is test content",
            source_url="https://example.com/news/1",
            source_name="Example News",
            language="zh"
        )
        assert news.title == "Test News"
        assert news.language == "zh"

    def test_news_hot_score_can_be_set(self):
        """Test news hot score can be set explicitly."""
        news = News(
            id=1,
            title="Test News",
            content="Content",
            source_url="https://example.com",
            source_name="Example",
            hot_score=5.5
        )
        assert news.hot_score == 5.5

    def test_news_optional_fields(self):
        """Test news optional fields."""
        news = News(
            id=1,
            title="Test News",
            content="Content",
            source_url="https://example.com",
            source_name="Example"
        )
        assert news.translated_title is None
        assert news.translated_content is None
        assert news.author is None
        assert news.location is None
        assert news.category is None
        assert news.keywords is None
        assert news.summary is None
        assert news.images is None
        assert news.videos is None


class TestCollectionModel:
    """Tests for Collection model."""

    def test_collection_creation(self):
        """Test creating a collection."""
        collection = Collection(
            id=1,
            user_id=1,
            news_id=1
        )
        assert collection.user_id == 1
        assert collection.news_id == 1


class TestUserBehaviorModel:
    """Tests for UserBehavior model."""

    def test_user_behavior_creation(self):
        """Test creating a user behavior record."""
        behavior = UserBehavior(
            id=1,
            user_id=1,
            news_id=1,
            action="view"
        )
        assert behavior.action == "view"

    def test_user_behavior_valid_actions(self):
        """Test valid action types."""
        valid_actions = ["view", "like", "collect", "dislike", "share"]
        for action in valid_actions:
            behavior = UserBehavior(
                id=1,
                user_id=1,
                news_id=1,
                action=action
            )
            assert behavior.action == action


class TestUserInterestModel:
    """Tests for UserInterest model."""

    def test_user_interest_creation(self):
        """Test creating a user interest."""
        interest = UserInterest(
            id=1,
            user_id=1,
            tag="technology"
        )
        assert interest.tag == "technology"


class TestTopicModel:
    """Tests for Topic model."""

    def test_topic_creation(self):
        """Test creating a topic."""
        topic = Topic(
            id=1,
            name="Test Topic",
            description="A test topic",
            creator_id=1,
            filter_conditions={"category": "military"},
            status="active"
        )
        assert topic.name == "Test Topic"
        assert topic.status == "active"
        assert topic.filter_conditions == {"category": "military"}

    def test_topic_status_can_be_set(self):
        """Test topic status can be set explicitly."""
        topic = Topic(
            id=1,
            name="Test Topic",
            creator_id=1,
            status="paused"
        )
        assert topic.status == "paused"


class TestTopicShareModel:
    """Tests for TopicShare model."""

    def test_topic_share_creation(self):
        """Test creating a topic share."""
        share = TopicShare(
            id=1,
            topic_id=1,
            user_id=2
        )
        assert share.topic_id == 1
        assert share.user_id == 2


class TestTopicNewsModel:
    """Tests for TopicNews junction model."""

    def test_topic_news_creation(self):
        """Test creating a topic-news association."""
        topic_news = TopicNews(
            id=1,
            topic_id=1,
            news_id=1,
            event_time=datetime(2025, 1, 1, 12, 0, 0)
        )
        assert topic_news.topic_id == 1
        assert topic_news.news_id == 1
        assert topic_news.event_time is not None


class TestSensitiveWordModel:
    """Tests for SensitiveWord model."""

    def test_sensitive_word_creation(self):
        """Test creating a sensitive word."""
        word = SensitiveWord(
            id=1,
            word="badword",
            category="profanity"
        )
        assert word.word == "badword"
        assert word.category == "profanity"


class TestLoginLogModel:
    """Tests for LoginLog model."""

    def test_login_log_creation(self):
        """Test creating a login log."""
        log = LoginLog(
            id=1,
            user_id=1,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            success=True
        )
        assert log.ip_address == "192.168.1.1"
        assert log.success is True


class TestOperationLogModel:
    """Tests for OperationLog model."""

    def test_operation_log_creation(self):
        """Test creating an operation log."""
        log = OperationLog(
            id=1,
            user_id=1,
            operation="create_user",
            target_type="user",
            target_id=2,
            details={"username": "newuser"}
        )
        assert log.operation == "create_user"
        assert log.details == {"username": "newuser"}


class TestNewsSourceModel:
    """Tests for NewsSource model."""

    def test_news_source_creation(self):
        """Test creating a news source."""
        source = NewsSource(
            id=1,
            name="Example News",
            url="https://example.com",
            category="domestic",
            crawl_config={"interval": 3600},
            status="active"
        )
        assert source.name == "Example News"
        assert source.status == "active"

    def test_news_source_status_can_be_set(self):
        """Test news source status can be set explicitly."""
        source = NewsSource(
            id=1,
            name="Example News",
            url="https://example.com",
            status="inactive"
        )
        assert source.status == "inactive"
