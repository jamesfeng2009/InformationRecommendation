from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ==================== User Schemas ====================

class UserCreate(BaseModel):
    """Schema for creating a new user."""
    username: str = Field(..., min_length=1, max_length=50, description="Unique username")
    password: str = Field(..., min_length=8, description="User password")
    name: str = Field(..., min_length=1, max_length=100, description="Display name")
    account: Optional[str] = Field(None, max_length=100, description="Login account (defaults to username)")
    department_id: Optional[int] = Field(None, description="Department ID")
    status: str = Field(default="enabled", pattern="^(enabled|disabled)$", description="User status")
    keywords: Optional[str] = Field(None, description="User keywords for recommendation")


class UserUpdate(BaseModel):
    """Schema for updating a user."""
    username: Optional[str] = Field(None, min_length=1, max_length=50, description="New username")
    name: Optional[str] = Field(None, min_length=1, max_length=100, description="New display name")
    account: Optional[str] = Field(None, max_length=100, description="New login account")
    department_id: Optional[int] = Field(None, description="New department ID (-1 to clear)")
    status: Optional[str] = Field(None, pattern="^(enabled|disabled)$", description="New status")
    keywords: Optional[str] = Field(None, description="New keywords")


class UserResponse(BaseModel):
    """User response schema."""
    id: int
    username: str
    name: str
    account: str
    department_id: Optional[int] = None
    department_name: Optional[str] = None
    status: str
    keywords: Optional[str] = None
    last_login: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class UserListResponse(BaseModel):
    """Paginated user list response."""
    items: List[UserResponse]
    total: int
    page: int
    size: int
    pages: int


# ==================== Batch Operations ====================

class BatchUpdateRequest(BaseModel):
    """Schema for batch updating users."""
    user_ids: List[int] = Field(..., min_length=1, description="List of user IDs to update")
    department_id: Optional[int] = Field(None, description="New department ID for all users")
    status: Optional[str] = Field(None, pattern="^(enabled|disabled)$", description="New status for all users")
    keywords: Optional[str] = Field(None, description="New keywords for all users")


class BatchUpdateResponse(BaseModel):
    """Batch update result."""
    updated_count: int
    user_ids: List[int]
    message: str


# ==================== Password Operations ====================

class ResetPasswordRequest(BaseModel):
    """Schema for admin password reset."""
    new_password: str = Field(..., min_length=8, description="New password")


class ChangePasswordRequest(BaseModel):
    """Schema for user password change."""
    old_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password")


class PasswordResponse(BaseModel):
    """Password operation result."""
    success: bool
    message: str


# ==================== Filter Schemas ====================

class UserFilterParams(BaseModel):
    """Query parameters for filtering users."""
    name: Optional[str] = Field(None, description="Filter by name (partial match)")
    account: Optional[str] = Field(None, description="Filter by account (partial match)")
    department_id: Optional[int] = Field(None, description="Filter by department ID")
    status: Optional[str] = Field(None, pattern="^(enabled|disabled)$", description="Filter by status")
    keywords: Optional[str] = Field(None, description="Filter by keywords (partial match)")
