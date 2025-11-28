"""
RBAC API endpoints for role and permission management.
Requirements: 12.1, 12.4

Endpoints:
- Role management: CRUD operations for roles
- Permission management: CRUD operations for permissions
- Assignment: Assign permissions to roles, roles to users
- Permission checking: Check user permissions
"""
from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.core.permissions import (
    CurrentUser,
    PermissionChecker,
    RBACServiceDep,
    get_current_user,
    get_rbac_service,
)
from app.models.user import User
from app.schemas.rbac import (
    MenuPermissionsResponse,
    PermissionAssignment,
    PermissionCheckRequest,
    PermissionCheckResponse,
    PermissionCreate,
    PermissionResponse,
    RoleAssignment,
    RoleCreate,
    RoleResponse,
    RoleUpdate,
    RoleWithPermissionsResponse,
    UserPermissionsResponse,
    UserRoleResponse,
)
from app.services.rbac import (
    PermissionExistsError,
    PermissionNotFoundError,
    RBACService,
    RoleExistsError,
    RoleNotFoundError,
)

router = APIRouter(prefix="/rbac", tags=["RBAC"])


# ==================== Role Management Endpoints ====================

@router.post("/roles", response_model=RoleResponse, status_code=status.HTTP_201_CREATED)
async def create_role(
    request: RoleCreate,
    current_user: Annotated[User, Depends(PermissionChecker("roles", "create"))],
    rbac_service: RBACServiceDep,
) -> RoleResponse:
    """
    Create a new role.
    
    Requires: roles:create permission
    """
    try:
        role = await rbac_service.create_role(request.name, request.description)
        return RoleResponse.model_validate(role)
    except RoleExistsError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.get("/roles", response_model=List[RoleResponse])
async def list_roles(
    current_user: Annotated[User, Depends(PermissionChecker("roles", "read"))],
    rbac_service: RBACServiceDep,
) -> List[RoleResponse]:
    """
    List all roles.
    
    Requires: roles:read permission
    """
    roles = await rbac_service.list_roles()
    return [RoleResponse.model_validate(r) for r in roles]


@router.get("/roles/{role_id}", response_model=RoleWithPermissionsResponse)
async def get_role(
    role_id: int,
    current_user: Annotated[User, Depends(PermissionChecker("roles", "read"))],
    rbac_service: RBACServiceDep,
) -> RoleWithPermissionsResponse:
    """
    Get role details with permissions.
    
    Requires: roles:read permission
    """
    role = await rbac_service.get_role_with_permissions(role_id)
    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Role with ID {role_id} not found"
        )
    
    permissions = [
        PermissionResponse.model_validate(rp.permission)
        for rp in role.role_permissions
    ]
    
    return RoleWithPermissionsResponse(
        id=role.id,
        name=role.name,
        description=role.description,
        created_at=role.created_at,
        updated_at=role.updated_at,
        permissions=permissions,
    )


@router.put("/roles/{role_id}", response_model=RoleResponse)
async def update_role(
    role_id: int,
    request: RoleUpdate,
    current_user: Annotated[User, Depends(PermissionChecker("roles", "update"))],
    rbac_service: RBACServiceDep,
) -> RoleResponse:
    """
    Update a role.
    
    Requires: roles:update permission
    """
    try:
        role = await rbac_service.update_role(role_id, request.name, request.description)
        return RoleResponse.model_validate(role)
    except RoleNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except RoleExistsError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.delete("/roles/{role_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_role(
    role_id: int,
    current_user: Annotated[User, Depends(PermissionChecker("roles", "delete"))],
    rbac_service: RBACServiceDep,
) -> None:
    """
    Delete a role.
    
    Requires: roles:delete permission
    """
    deleted = await rbac_service.delete_role(role_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Role with ID {role_id} not found"
        )


# ==================== Permission Management Endpoints ====================

@router.post("/permissions", response_model=PermissionResponse, status_code=status.HTTP_201_CREATED)
async def create_permission(
    request: PermissionCreate,
    current_user: Annotated[User, Depends(PermissionChecker("permissions", "create"))],
    rbac_service: RBACServiceDep,
) -> PermissionResponse:
    """
    Create a new permission.
    
    Requires: permissions:create permission
    """
    try:
        permission = await rbac_service.create_permission(
            request.resource, request.action, request.description
        )
        return PermissionResponse.model_validate(permission)
    except PermissionExistsError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.get("/permissions", response_model=List[PermissionResponse])
async def list_permissions(
    current_user: Annotated[User, Depends(PermissionChecker("permissions", "read"))],
    rbac_service: RBACServiceDep,
    resource: str = Query(None, description="Filter by resource name"),
) -> List[PermissionResponse]:
    """
    List all permissions, optionally filtered by resource.
    
    Requires: permissions:read permission
    """
    permissions = await rbac_service.list_permissions(resource)
    return [PermissionResponse.model_validate(p) for p in permissions]


@router.delete("/permissions/{permission_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_permission(
    permission_id: int,
    current_user: Annotated[User, Depends(PermissionChecker("permissions", "delete"))],
    rbac_service: RBACServiceDep,
) -> None:
    """
    Delete a permission.
    
    Requires: permissions:delete permission
    """
    deleted = await rbac_service.delete_permission(permission_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Permission with ID {permission_id} not found"
        )


# ==================== Permission Assignment Endpoints ====================

@router.post("/roles/{role_id}/permissions", response_model=dict)
async def assign_permissions_to_role(
    role_id: int,
    request: PermissionAssignment,
    current_user: Annotated[User, Depends(PermissionChecker("roles", "update"))],
    rbac_service: RBACServiceDep,
) -> dict:
    """
    Assign permissions to a role.
    
    Requires: roles:update permission
    """
    try:
        count = await rbac_service.assign_permissions_to_role(role_id, request.permission_ids)
        return {"assigned": count, "message": f"Assigned {count} permissions to role"}
    except RoleNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.put("/roles/{role_id}/permissions", response_model=dict)
async def set_role_permissions(
    role_id: int,
    request: PermissionAssignment,
    current_user: Annotated[User, Depends(PermissionChecker("roles", "update"))],
    rbac_service: RBACServiceDep,
) -> dict:
    """
    Set exact permissions for a role (replaces existing).
    
    Requires: roles:update permission
    """
    try:
        await rbac_service.set_role_permissions(role_id, request.permission_ids)
        return {"message": "Role permissions updated successfully"}
    except RoleNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.delete("/roles/{role_id}/permissions/{permission_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_permission_from_role(
    role_id: int,
    permission_id: int,
    current_user: Annotated[User, Depends(PermissionChecker("roles", "update"))],
    rbac_service: RBACServiceDep,
) -> None:
    """
    Revoke a permission from a role.
    
    Requires: roles:update permission
    """
    revoked = await rbac_service.revoke_permission_from_role(role_id, permission_id)
    if not revoked:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Permission not assigned to this role"
        )


@router.get("/roles/{role_id}/permissions", response_model=List[PermissionResponse])
async def get_role_permissions(
    role_id: int,
    current_user: Annotated[User, Depends(PermissionChecker("roles", "read"))],
    rbac_service: RBACServiceDep,
) -> List[PermissionResponse]:
    """
    Get all permissions assigned to a role.
    
    Requires: roles:read permission
    """
    role = await rbac_service.get_role(role_id)
    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Role with ID {role_id} not found"
        )
    
    permissions = await rbac_service.get_role_permissions(role_id)
    return [PermissionResponse.model_validate(p) for p in permissions]


# ==================== User-Role Assignment Endpoints ====================

@router.post("/users/{user_id}/roles", response_model=dict)
async def assign_roles_to_user(
    user_id: int,
    request: RoleAssignment,
    current_user: Annotated[User, Depends(PermissionChecker("users", "update"))],
    rbac_service: RBACServiceDep,
) -> dict:
    """
    Assign roles to a user.
    
    Requires: users:update permission
    """
    count = await rbac_service.assign_roles_to_user(user_id, request.role_ids)
    return {"assigned": count, "message": f"Assigned {count} roles to user"}


@router.put("/users/{user_id}/roles", response_model=dict)
async def set_user_roles(
    user_id: int,
    request: RoleAssignment,
    current_user: Annotated[User, Depends(PermissionChecker("users", "update"))],
    rbac_service: RBACServiceDep,
) -> dict:
    """
    Set exact roles for a user (replaces existing).
    
    Requires: users:update permission
    """
    await rbac_service.set_user_roles(user_id, request.role_ids)
    return {"message": "User roles updated successfully"}


@router.delete("/users/{user_id}/roles/{role_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_role_from_user(
    user_id: int,
    role_id: int,
    current_user: Annotated[User, Depends(PermissionChecker("users", "update"))],
    rbac_service: RBACServiceDep,
) -> None:
    """
    Revoke a role from a user.
    
    Requires: users:update permission
    """
    revoked = await rbac_service.revoke_role_from_user(user_id, role_id)
    if not revoked:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Role not assigned to this user"
        )


@router.get("/users/{user_id}/roles", response_model=UserRoleResponse)
async def get_user_roles(
    user_id: int,
    current_user: Annotated[User, Depends(PermissionChecker("users", "read"))],
    rbac_service: RBACServiceDep,
) -> UserRoleResponse:
    """
    Get all roles assigned to a user.
    
    Requires: users:read permission
    """
    roles = await rbac_service.get_user_roles(user_id)
    return UserRoleResponse(
        user_id=user_id,
        roles=[RoleResponse.model_validate(r) for r in roles]
    )


@router.get("/users/{user_id}/permissions", response_model=UserPermissionsResponse)
async def get_user_permissions(
    user_id: int,
    current_user: Annotated[User, Depends(PermissionChecker("users", "read"))],
    rbac_service: RBACServiceDep,
) -> UserPermissionsResponse:
    """
    Get all permissions for a user through their roles.
    
    Requires: users:read permission
    """
    permissions = await rbac_service.get_user_permissions(user_id)
    perm_set = await rbac_service.get_user_permission_set(user_id)
    
    return UserPermissionsResponse(
        user_id=user_id,
        permissions=[PermissionResponse.model_validate(p) for p in permissions],
        permission_strings=sorted(list(perm_set))
    )


# ==================== Permission Checking Endpoints ====================

@router.post("/check-permission", response_model=PermissionCheckResponse)
async def check_permission(
    request: PermissionCheckRequest,
    current_user: CurrentUser,
    rbac_service: RBACServiceDep,
) -> PermissionCheckResponse:
    """
    Check if current user has a specific permission.
    
    No special permission required - users can check their own permissions.
    """
    has_permission = await rbac_service.check_permission(
        current_user.id, request.resource, request.action
    )
    return PermissionCheckResponse(
        has_permission=has_permission,
        resource=request.resource,
        action=request.action
    )


@router.get("/my-permissions", response_model=UserPermissionsResponse)
async def get_my_permissions(
    current_user: CurrentUser,
    rbac_service: RBACServiceDep,
) -> UserPermissionsResponse:
    """
    Get current user's permissions.
    
    No special permission required - users can view their own permissions.
    """
    permissions = await rbac_service.get_user_permissions(current_user.id)
    perm_set = await rbac_service.get_user_permission_set(current_user.id)
    
    return UserPermissionsResponse(
        user_id=current_user.id,
        permissions=[PermissionResponse.model_validate(p) for p in permissions],
        permission_strings=sorted(list(perm_set))
    )


@router.get("/my-menus", response_model=MenuPermissionsResponse)
async def get_my_menus(
    current_user: CurrentUser,
    rbac_service: RBACServiceDep,
) -> MenuPermissionsResponse:
    """
    Get menus accessible to current user.
    
    No special permission required - users can view their own menu access.
    
    Requirements: 12.2 - Menu permission filtering
    """
    menus = await rbac_service.get_user_menu_permissions(current_user.id)
    return MenuPermissionsResponse(
        user_id=current_user.id,
        menus=menus
    )
