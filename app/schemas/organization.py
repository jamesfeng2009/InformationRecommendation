"""
Organization management schemas for request/response validation.
Requirements: 11.1, 11.2, 11.3, 11.4
"""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ==================== Department Schemas ====================

class DepartmentCreate(BaseModel):
    """Schema for creating a new department."""
    name: str = Field(..., min_length=1, max_length=100, description="Department name")
    parent_id: Optional[int] = Field(None, description="Parent department ID (None for root)")
    manager: Optional[str] = Field(None, max_length=100, description="Responsible person")
    contact: Optional[str] = Field(None, max_length=100, description="Contact information")
    description: Optional[str] = Field(None, description="Department description")
    sort_order: int = Field(default=0, description="Sort order for display")


class DepartmentUpdate(BaseModel):
    """Schema for updating a department."""
    name: Optional[str] = Field(None, min_length=1, max_length=100, description="New department name")
    manager: Optional[str] = Field(None, max_length=100, description="New responsible person")
    contact: Optional[str] = Field(None, max_length=100, description="New contact information")
    description: Optional[str] = Field(None, description="New department description")
    sort_order: Optional[int] = Field(None, description="New sort order")


class DepartmentMove(BaseModel):
    """Schema for moving a department to a new parent."""
    new_parent_id: Optional[int] = Field(None, description="New parent department ID (None for root)")


class DepartmentResponse(BaseModel):
    """Department response schema."""
    id: int
    name: str
    parent_id: Optional[int] = None
    manager: Optional[str] = None
    contact: Optional[str] = None
    description: Optional[str] = None
    sort_order: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class DepartmentTreeNode(BaseModel):
    """Department tree node with nested children."""
    id: int
    name: str
    parent_id: Optional[int] = None
    manager: Optional[str] = None
    contact: Optional[str] = None
    description: Optional[str] = None
    sort_order: int
    depth: int
    user_count: int
    children: List["DepartmentTreeNode"] = []
    
    class Config:
        from_attributes = True


# Enable self-referencing model
DepartmentTreeNode.model_rebuild()


class DepartmentTreeResponse(BaseModel):
    """Response containing the department tree."""
    tree: List[DepartmentTreeNode]
    total_departments: int


class DepartmentDeleteResponse(BaseModel):
    """Response for department deletion."""
    success: bool
    message: str
