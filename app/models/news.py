from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.user import User
    from app.models.topic import TopicNews


class News(Base):
    """
    News article model with all metadata.
    Requirements: 15.1, 15.2
    """
    __tablename__ = "news"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    original_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    language: Mapped[str] = mapped_column(String(10), nullable=False, default="zh")
    translated_title: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    translated_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_url: Mapped[str] = mapped_column(String(1000), nullable=False)
    source_name: Mapped[str] = mapped_column(String(100), nullable=False)
    author: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    location: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    category: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    keywords: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    images: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    videos: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    publish_time: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    crawl_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    hot_score: Mapped[float] = mapped_column(Float, default=0.0)
    content_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    collections: Mapped[List["Collection"]] = relationship(
        "Collection", back_populates="news", cascade="all, delete-orphan"
    )
    behaviors: Mapped[List["UserBehavior"]] = relationship(
        "UserBehavior", back_populates="news", cascade="all, delete-orphan"
    )
    topic_news: Mapped[List["TopicNews"]] = relationship(
        "TopicNews", back_populates="news", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_news_title", "title"),
        Index("ix_news_category", "category"),
        Index("ix_news_source_name", "source_name"),
        Index("ix_news_language", "language"),
        Index("ix_news_publish_time", "publish_time"),
        Index("ix_news_crawl_time", "crawl_time"),
        Index("ix_news_hot_score", "hot_score"),
        Index("ix_news_content_hash", "content_hash"),
    )


class Collection(Base):
    """
    User news collection (favorites) model.
    Requirements: 5.1
    """
    __tablename__ = "collections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    news_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("news.id", ondelete="CASCADE"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="collections")
    news: Mapped["News"] = relationship("News", back_populates="collections")

    __table_args__ = (
        UniqueConstraint("user_id", "news_id", name="uq_collections_user_news"),
        Index("ix_collections_user_id", "user_id"),
        Index("ix_collections_news_id", "news_id"),
    )


class UserBehavior(Base):
    """
    User behavior tracking model for recording interactions.
    Requirements: 5.1
    """
    __tablename__ = "user_behaviors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    news_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("news.id", ondelete="CASCADE"), nullable=False
    )
    action: Mapped[str] = mapped_column(String(20), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="behaviors")
    news: Mapped["News"] = relationship("News", back_populates="behaviors")

    __table_args__ = (
        CheckConstraint(
            "action IN ('view', 'like', 'collect', 'dislike', 'share')",
            name="ck_user_behaviors_action"
        ),
        Index("ix_user_behaviors_user_id", "user_id"),
        Index("ix_user_behaviors_news_id", "news_id"),
        Index("ix_user_behaviors_action", "action"),
        Index("ix_user_behaviors_created_at", "created_at"),
    )


class UserInterest(Base):
    """
    User interest tags model.
    Requirements: 5.1
    """
    __tablename__ = "user_interests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    tag: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="interests")

    __table_args__ = (
        UniqueConstraint("user_id", "tag", name="uq_user_interests_user_tag"),
        Index("ix_user_interests_user_id", "user_id"),
        Index("ix_user_interests_tag", "tag"),
    )
