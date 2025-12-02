from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ==================== Role Schemas ====================

class RoleCreate(BaseModel):
    """Schema for creating a new role."""
    name: str = Field(..., min_length=1, max_length=50, description="Unique role name")
    description: Optional[str] = Field(None, max_length=500, description="Role description")


class RoleUpdate(BaseModel):
    """Schema for updating a role."""
    name: Optional[str] = Field(None, min_length=1, max_length=50, description="New role name")
    description: Optional[str] = Field(None, max_length=500, description="New description")


class RoleResponse(BaseModel):
    """Role response schema."""
    id: int
    name: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class RoleWithPermissionsResponse(RoleResponse):
    """Role response with permissions included."""
    permissions: List["PermissionResponse"] = []


# ==================== Permission Schemas ====================

class PermissionCreate(BaseModel):
    """Schema for creating a new permission."""
    resource: str = Field(..., min_length=1, max_length=100, description="Resource name")
    action: str = Field(..., min_length=1, max_length=50, description="Action name")
    description: Optional[str] = Field(None, max_length=500, description="Permission description")


class PermissionResponse(BaseModel):
    """Permission response schema."""
    id: int
    resource: str
    action: str
    description: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


# ==================== Assignment Schemas ====================

class PermissionAssignment(BaseModel):
    """Schema for assigning permissions to a role."""
    permission_ids: List[int] = Field(..., description="List of permission IDs to assign")


class RoleAssignment(BaseModel):
    """Schema for assigning roles to a user."""
    role_ids: List[int] = Field(..., description="List of role IDs to assign")


class UserRoleResponse(BaseModel):
    """User's roles response schema."""
    user_id: int
    roles: List[RoleResponse]


class UserPermissionsResponse(BaseModel):
    """User's permissions response schema."""
    user_id: int
    permissions: List[PermissionResponse]
    permission_strings: List[str] = Field(
        ..., description="Permissions in 'resource:action' format"
    )


# ==================== Permission Check Schemas ====================

class PermissionCheckRequest(BaseModel):
    """Schema for checking a permission."""
    resource: str = Field(..., description="Resource name")
    action: str = Field(..., description="Action name")


class PermissionCheckResponse(BaseModel):
    """Permission check result."""
    has_permission: bool
    resource: str
    action: str


class MenuPermissionsResponse(BaseModel):
    """User's accessible menus response."""
    user_id: int
    menus: List[str] = Field(..., description="List of menu resources user can access")


# Update forward references
RoleWithPermissionsResponse.model_rebuild()
