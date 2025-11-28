"""
SQLAlchemy models for the Intelligent Information Recommendation System.
"""
from app.models.user import User, Department
from app.models.rbac import Role, Permission, UserRole, RolePermission
from app.models.news import News, Collection, UserBehavior, UserInterest
from app.models.topic import Topic, TopicShare, TopicNews
from app.models.system import SensitiveWord, LoginLog, OperationLog, NewsSource

__all__ = [
    # User and Organization
    "User",
    "Department",
    # RBAC
    "Role",
    "Permission",
    "UserRole",
    "RolePermission",
    # News
    "News",
    "Collection",
    "UserBehavior",
    "UserInterest",
    # Topic
    "Topic",
    "TopicShare",
    "TopicNews",
    # System
    "SensitiveWord",
    "LoginLog",
    "OperationLog",
    "NewsSource",
]
