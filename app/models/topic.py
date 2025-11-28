"""
Topic and sharing models for news topic management.
Requirements: 8.1, 9.1, 9.2
"""
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import (
    CheckConstraint,
    DateTime,
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
    from app.models.news import News


class Topic(Base):
    """
    Topic model for tracking news events over time.
    Requirements: 8.1
    """
    __tablename__ = "topics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    creator_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    filter_conditions: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="active", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    creator: Mapped["User"] = relationship("User", back_populates="topics")
    shares: Mapped[List["TopicShare"]] = relationship(
        "TopicShare", back_populates="topic", cascade="all, delete-orphan"
    )
    topic_news: Mapped[List["TopicNews"]] = relationship(
        "TopicNews", back_populates="topic", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'paused')",
            name="ck_topics_status"
        ),
        Index("ix_topics_creator_id", "creator_id"),
        Index("ix_topics_status", "status"),
        Index("ix_topics_name", "name"),
    )


class TopicShare(Base):
    """
    Topic sharing model for sharing topics with other users.
    Requirements: 9.1, 9.2
    """
    __tablename__ = "topic_shares"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    topic_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("topics.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    shared_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    topic: Mapped["Topic"] = relationship("Topic", back_populates="shares")
    user: Mapped["User"] = relationship("User", back_populates="shared_topics")

    __table_args__ = (
        UniqueConstraint("topic_id", "user_id", name="uq_topic_shares_topic_user"),
        Index("ix_topic_shares_topic_id", "topic_id"),
        Index("ix_topic_shares_user_id", "user_id"),
    )


class TopicNews(Base):
    """
    Junction table for Topic-News many-to-many relationship.
    Requirements: 8.1
    """
    __tablename__ = "topic_news"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    topic_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("topics.id", ondelete="CASCADE"), nullable=False
    )
    news_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("news.id", ondelete="CASCADE"), nullable=False
    )
    event_time: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    added_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    topic: Mapped["Topic"] = relationship("Topic", back_populates="topic_news")
    news: Mapped["News"] = relationship("News", back_populates="topic_news")

    __table_args__ = (
        UniqueConstraint("topic_id", "news_id", name="uq_topic_news_topic_news"),
        Index("ix_topic_news_topic_id", "topic_id"),
        Index("ix_topic_news_news_id", "news_id"),
        Index("ix_topic_news_event_time", "event_time"),
    )
