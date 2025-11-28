"""
System models for logging, sensitive words, and news sources.
Requirements: 13.1, 13.2, 14.1
"""
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.user import User


class SensitiveWord(Base):
    """
    Sensitive word model for content filtering.
    Requirements: 14.1
    """
    __tablename__ = "sensitive_words"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    word: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    category: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("ix_sensitive_words_word", "word"),
        Index("ix_sensitive_words_category", "category"),
    )


class LoginLog(Base):
    """
    Login log model for tracking user authentication.
    Requirements: 13.1
    """
    __tablename__ = "login_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    ip_address: Mapped[str] = mapped_column(String(45), nullable=False)
    user_agent: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    login_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    success: Mapped[bool] = mapped_column(default=True)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="login_logs")

    __table_args__ = (
        Index("ix_login_logs_user_id", "user_id"),
        Index("ix_login_logs_login_time", "login_time"),
        Index("ix_login_logs_ip_address", "ip_address"),
    )


class OperationLog(Base):
    """
    Operation log model for tracking admin operations.
    Requirements: 13.2
    """
    __tablename__ = "operation_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    operation: Mapped[str] = mapped_column(String(100), nullable=False)
    target_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    target_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    details: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="operation_logs")

    __table_args__ = (
        Index("ix_operation_logs_user_id", "user_id"),
        Index("ix_operation_logs_operation", "operation"),
        Index("ix_operation_logs_target_type", "target_type"),
        Index("ix_operation_logs_created_at", "created_at"),
    )


class NewsSource(Base):
    """
    News source configuration model.
    Requirements: 1.1
    """
    __tablename__ = "news_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    url: Mapped[str] = mapped_column(String(500), nullable=False)
    category: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    crawl_config: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="active", nullable=False)
    last_crawl: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'inactive', 'error')",
            name="ck_news_sources_status"
        ),
        Index("ix_news_sources_name", "name"),
        Index("ix_news_sources_status", "status"),
        Index("ix_news_sources_category", "category"),
    )
