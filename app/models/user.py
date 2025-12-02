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
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.rbac import UserRole
    from app.models.news import Collection, UserBehavior, UserInterest
    from app.models.topic import Topic, TopicShare
    from app.models.system import LoginLog, OperationLog


class Department(Base):
    """
    Department model for organization tree structure.
    Supports up to 3 levels: root -> level 1 -> level 2
    Requirements: 11.1, 11.2
    """
    __tablename__ = "departments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    parent_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("departments.id", ondelete="RESTRICT"), nullable=True
    )
    manager: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    contact: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Self-referential relationship for tree structure
    parent: Mapped[Optional["Department"]] = relationship(
        "Department",
        remote_side=[id],
        back_populates="children",
    )
    children: Mapped[List["Department"]] = relationship(
        "Department",
        back_populates="parent",
        cascade="all, delete-orphan",
    )
    
    # Users in this department
    users: Mapped[List["User"]] = relationship(
        "User", back_populates="department", passive_deletes=True
    )

    __table_args__ = (
        Index("ix_departments_parent_id", "parent_id"),
        Index("ix_departments_name", "name"),
    )

    def get_depth(self) -> int:
        """Calculate the depth of this department in the tree."""
        depth = 0
        current = self
        while current.parent is not None:
            depth += 1
            current = current.parent
        return depth


class User(Base):
    """
    User model for authentication and user management.
    Requirements: 10.1, 10.4
    """
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    account: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    department_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("departments.id", ondelete="SET NULL"), nullable=True
    )
    status: Mapped[str] = mapped_column(
        String(20), default="enabled", nullable=False
    )
    keywords: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    last_login: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    department: Mapped[Optional["Department"]] = relationship(
        "Department", back_populates="users"
    )
    roles: Mapped[List["UserRole"]] = relationship(
        "UserRole", back_populates="user", cascade="all, delete-orphan"
    )
    interests: Mapped[List["UserInterest"]] = relationship(
        "UserInterest", back_populates="user", cascade="all, delete-orphan"
    )
    behaviors: Mapped[List["UserBehavior"]] = relationship(
        "UserBehavior", back_populates="user", cascade="all, delete-orphan"
    )
    collections: Mapped[List["Collection"]] = relationship(
        "Collection", back_populates="user", cascade="all, delete-orphan"
    )
    topics: Mapped[List["Topic"]] = relationship(
        "Topic", back_populates="creator", cascade="all, delete-orphan"
    )
    shared_topics: Mapped[List["TopicShare"]] = relationship(
        "TopicShare", back_populates="user", cascade="all, delete-orphan"
    )
    login_logs: Mapped[List["LoginLog"]] = relationship(
        "LoginLog", back_populates="user", cascade="all, delete-orphan"
    )
    operation_logs: Mapped[List["OperationLog"]] = relationship(
        "OperationLog", back_populates="user", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('enabled', 'disabled')",
            name="ck_users_status"
        ),
        Index("ix_users_username", "username"),
        Index("ix_users_account", "account"),
        Index("ix_users_department_id", "department_id"),
        Index("ix_users_status", "status"),
    )
