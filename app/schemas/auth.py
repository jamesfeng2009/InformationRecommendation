from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    """Login request schema."""
    username: str = Field(..., min_length=1, max_length=100, description="Username or account")
    password: str = Field(..., min_length=1, description="User password")


class TokenResponse(BaseModel):
    """Token response schema."""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Access token expiration in seconds")


class RefreshRequest(BaseModel):
    """Token refresh request schema."""
    refresh_token: str = Field(..., description="Valid refresh token")


class UserResponse(BaseModel):
    """Current user response schema."""
    id: int
    username: str
    name: str
    account: str
    department_id: Optional[int] = None
    status: str
    keywords: Optional[str] = None
    last_login: Optional[datetime] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class MessageResponse(BaseModel):
    """Generic message response."""
    message: str
    success: bool = True
