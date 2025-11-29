"""
Organization management API endpoints.
Requirements: 11.1, 11.2, 11.3, 11.4

Endpoints:
- POST /api/v1/departments - Create department
- GET /api/v1/departments/tree - Get department tree
- PUT /api/v1/departments/{id} - Update department
- DELETE /api/v1/departments/{id} - Delete department
- POST /api/v1/departments/{id}/move - Move department to new parent
"""
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.schemas.organization import (
    DepartmentCreate,
    DepartmentDeleteResponse,
    DepartmentMove,
    DepartmentResponse,
    DepartmentTreeNode,
    DepartmentTreeResponse,
    DepartmentUpdate,
)
from app.services.organization import (
    CircularReferenceError,
    DepartmentDepthExceededError,
    DepartmentHasChildrenError,
    DepartmentHasUsersError,
    DepartmentNotFoundError,
    OrganizationService,
    OrganizationServiceError,
)

router = APIRouter(prefix="/departments", tags=["Organization Management"])


async def get_organization_service(
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> OrganizationService:
    """Dependency to get OrganizationService instance."""
    return OrganizationService(db)


def _department_to_response(department) -> DepartmentResponse:
    """Convert Department model to DepartmentResponse schema."""
    return DepartmentResponse(
        id=department.id,
        name=department.name,
        parent_id=department.parent_id,
        manager=department.manager,
        contact=department.contact,
        description=department.description,
        sort_order=department.sort_order,
        created_at=department.created_at,
        updated_at=department.updated_at,
    )


def _node_to_tree_response(node) -> DepartmentTreeNode:
    """Convert DepartmentNode to DepartmentTreeNode schema."""
    return DepartmentTreeNode(
        id=node.id,
        name=node.name,
        parent_id=node.parent_id,
        manager=node.manager,
        contact=node.contact,
        description=node.description,
        sort_order=node.sort_order,
        depth=node.depth,
        user_count=node.user_count,
        children=[_node_to_tree_response(child) for child in node.children],
    )


def _count_nodes(nodes) -> int:
    """Count total nodes in tree."""
    count = len(nodes)
    for node in nodes:
        count += _count_nodes(node.children)
    return count


@router.post("", response_model=DepartmentResponse, status_code=status.HTTP_201_CREATED)
async def create_department(
    request: DepartmentCreate,
    org_service: Annotated[OrganizationService, Depends(get_organization_service)],
) -> DepartmentResponse:
    """
    Create a new department.
    
    - **name**: Department name
    - **parent_id**: Parent department ID (None for root)
    - **manager**: Responsible person (optional)
    - **contact**: Contact information (optional)
    - **description**: Department description (optional)
    - **sort_order**: Sort order for display
    
    Note: Maximum tree depth is 3 levels (root -> level 1 -> level 2).
    
    Requirements: 11.1, 11.2
    """
    try:
        department = await org_service.create_department(
            name=request.name,
            parent_id=request.parent_id,
            manager=request.manager,
            contact=request.contact,
            description=request.description,
            sort_order=request.sort_order,
        )
        return _department_to_response(department)
    except DepartmentNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except DepartmentDepthExceededError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": str(e),
                "code": "DEPTH_EXCEEDED",
                "max_levels": e.max_depth + 1,
            },
        )


@router.get("/tree", response_model=DepartmentTreeResponse)
async def get_department_tree(
    org_service: Annotated[OrganizationService, Depends(get_organization_service)],
) -> DepartmentTreeResponse:
    """
    Get the complete department tree structure.
    
    Returns a hierarchical tree of all departments with nested children.
    Each node includes user count and depth information.
    
    Requirements: 11.1
    """
    tree = await org_service.get_department_tree()
    tree_nodes = [_node_to_tree_response(node) for node in tree]
    total = _count_nodes(tree)
    
    return DepartmentTreeResponse(
        tree=tree_nodes,
        total_departments=total,
    )


@router.get("/{department_id}", response_model=DepartmentResponse)
async def get_department(
    department_id: int,
    org_service: Annotated[OrganizationService, Depends(get_organization_service)],
) -> DepartmentResponse:
    """
    Get department by ID.
    
    Requirements: 11.1
    """
    department = await org_service.get_department(department_id)
    if not department:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Department with ID {department_id} not found",
        )
    return _department_to_response(department)


@router.put("/{department_id}", response_model=DepartmentResponse)
async def update_department(
    department_id: int,
    request: DepartmentUpdate,
    org_service: Annotated[OrganizationService, Depends(get_organization_service)],
) -> DepartmentResponse:
    """
    Update a department's information.
    
    Note: To change parent (move department), use the /move endpoint.
    
    Requirements: 11.2
    """
    try:
        department = await org_service.update_department(
            department_id=department_id,
            name=request.name,
            manager=request.manager,
            contact=request.contact,
            description=request.description,
            sort_order=request.sort_order,
        )
        return _department_to_response(department)
    except DepartmentNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.delete("/{department_id}", response_model=DepartmentDeleteResponse)
async def delete_department(
    department_id: int,
    org_service: Annotated[OrganizationService, Depends(get_organization_service)],
) -> DepartmentDeleteResponse:
    """
    Delete a department.
    
    A department can only be deleted if:
    - It has no users assigned to it
    - It has no child departments
    
    Requirements: 11.3 - Property 20: Department Deletion Protection
    """
    try:
        await org_service.delete_department(department_id)
        return DepartmentDeleteResponse(
            success=True,
            message=f"Department {department_id} deleted successfully",
        )
    except DepartmentNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except DepartmentHasUsersError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": str(e),
                "code": "HAS_USERS",
                "department_id": e.department_id,
                "user_count": e.user_count,
            },
        )
    except DepartmentHasChildrenError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": str(e),
                "code": "HAS_CHILDREN",
                "department_id": e.department_id,
                "children_count": e.children_count,
            },
        )


@router.post("/{department_id}/move", response_model=DepartmentResponse)
async def move_department(
    department_id: int,
    request: DepartmentMove,
    org_service: Annotated[OrganizationService, Depends(get_organization_service)],
) -> DepartmentResponse:
    """
    Move a department to a new parent (change hierarchy).
    
    - **new_parent_id**: New parent department ID (None to make it a root)
    
    Constraints:
    - Cannot create circular references
    - Maximum tree depth is 3 levels
    
    Requirements: 11.4
    """
    try:
        department = await org_service.move_department(
            department_id=department_id,
            new_parent_id=request.new_parent_id,
        )
        return _department_to_response(department)
    except DepartmentNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except CircularReferenceError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": str(e),
                "code": "CIRCULAR_REFERENCE",
            },
        )
    except DepartmentDepthExceededError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": str(e),
                "code": "DEPTH_EXCEEDED",
                "max_levels": e.max_depth + 1,
            },
        )
