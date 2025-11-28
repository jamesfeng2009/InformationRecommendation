"""
User management API endpoints.
Requirements: 10.2, 10.3, 10.4, 10.5

Endpoints:
- POST /api/v1/users - Create user
- GET /api/v1/users - List users with filters
- GET /api/v1/users/{id} - Get user by ID
- PUT /api/v1/users/{id} - Update user
- DELETE /api/v1/users/{id} - Delete user
- POST /api/v1/users/batch-update - Batch update users
- POST /api/v1/users/{id}/reset-password - Reset user password
"""
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.schemas.user import (
    BatchUpdateRequest,
    BatchUpdateResponse,
    ChangePasswordRequest,
    PasswordResponse,
    ResetPasswordRequest,
    UserCreate,
    UserListResponse,
    UserResponse,
    UserUpdate,
)
from app.services.user import (
    BatchUpdateData,
    BatchUpdateError,
    DepartmentNotFoundError,
    PasswordPolicyError,
    UserExistsError,
    UserFilter,
    UserNotFoundError,
    UserService,
    UserServiceError,
)

router = APIRouter(prefix="/users", tags=["User Management"])


async def get_user_service(
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> UserService:
    """Dependency to get UserService instance."""
    return UserService(db)


def _user_to_response(user) -> UserResponse:
    """Convert User model to UserResponse schema."""
    return UserResponse(
        id=user.id,
        username=user.username,
        name=user.name,
        account=user.account,
        department_id=user.department_id,
        department_name=user.department.name if user.department else None,
        status=user.status,
        keywords=user.keywords,
        last_login=user.last_login,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )



@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    request: UserCreate,
    user_service: Annotated[UserService, Depends(get_user_service)],
) -> UserResponse:
    """
    Create a new user.
    
    - **username**: Unique username
    - **password**: User password (must meet policy requirements)
    - **name**: Display name
    - **account**: Login account (optional, defaults to username)
    - **department_id**: Department ID (optional)
    - **status**: User status (enabled/disabled)
    - **keywords**: User keywords for recommendation
    
    Requirements: 10.2
    """
    try:
        user = await user_service.create_user(
            username=request.username,
            password=request.password,
            name=request.name,
            account=request.account,
            department_id=request.department_id,
            status=request.status,
            keywords=request.keywords,
        )
        # Reload with department relation
        user = await user_service.get_user_with_relations(user.id)
        return _user_to_response(user)
    except UserExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )
    except DepartmentNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except PasswordPolicyError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Password policy violation", "errors": e.errors},
        )


@router.get("", response_model=UserListResponse)
async def list_users(
    user_service: Annotated[UserService, Depends(get_user_service)],
    name: Optional[str] = Query(None, description="Filter by name (partial match)"),
    account: Optional[str] = Query(None, description="Filter by account (partial match)"),
    department_id: Optional[int] = Query(None, description="Filter by department ID"),
    user_status: Optional[str] = Query(None, alias="status", description="Filter by status"),
    keywords: Optional[str] = Query(None, description="Filter by keywords (partial match)"),
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Page size"),
) -> UserListResponse:
    """
    List users with optional filters and pagination.
    
    Requirements: 10.4
    """
    filters = UserFilter(
        name=name,
        account=account,
        department_id=department_id,
        status=user_status,
        keywords=keywords,
    )
    
    result = await user_service.list_users(filters=filters, page=page, size=size)
    
    return UserListResponse(
        items=[_user_to_response(u) for u in result.items],
        total=result.total,
        page=result.page,
        size=result.size,
        pages=result.pages,
    )


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    user_service: Annotated[UserService, Depends(get_user_service)],
) -> UserResponse:
    """
    Get user by ID.
    
    Requirements: 10.4
    """
    user = await user_service.get_user_with_relations(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found",
        )
    return _user_to_response(user)


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    request: UserUpdate,
    user_service: Annotated[UserService, Depends(get_user_service)],
) -> UserResponse:
    """
    Update a user.
    
    Requirements: 10.3
    """
    try:
        user = await user_service.update_user(
            user_id=user_id,
            username=request.username,
            name=request.name,
            account=request.account,
            department_id=request.department_id,
            status=request.status,
            keywords=request.keywords,
        )
        # Reload with department relation
        user = await user_service.get_user_with_relations(user.id)
        return _user_to_response(user)
    except UserNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except UserExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )
    except DepartmentNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except UserServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: int,
    user_service: Annotated[UserService, Depends(get_user_service)],
) -> None:
    """
    Delete a user.
    
    Requirements: 10.2
    """
    try:
        await user_service.delete_user(user_id)
    except UserNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.post("/batch-update", response_model=BatchUpdateResponse)
async def batch_update_users(
    request: BatchUpdateRequest,
    user_service: Annotated[UserService, Depends(get_user_service)],
) -> BatchUpdateResponse:
    """
    Batch update multiple users atomically.
    
    Either all users are updated, or none are (rollback on error).
    
    Requirements: 10.3 - Batch update atomicity (Property 18)
    """
    try:
        data = BatchUpdateData(
            department_id=request.department_id,
            status=request.status,
            keywords=request.keywords,
        )
        updated_users = await user_service.batch_update_users(
            user_ids=request.user_ids,
            data=data,
        )
        return BatchUpdateResponse(
            updated_count=len(updated_users),
            user_ids=[u.id for u in updated_users],
            message=f"Successfully updated {len(updated_users)} users",
        )
    except BatchUpdateError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": str(e),
                "failed_user_ids": e.failed_user_ids,
            },
        )
    except DepartmentNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except UserServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/{user_id}/reset-password", response_model=PasswordResponse)
async def reset_password(
    user_id: int,
    request: ResetPasswordRequest,
    user_service: Annotated[UserService, Depends(get_user_service)],
) -> PasswordResponse:
    """
    Reset a user's password (admin operation).
    
    Requirements: 10.5
    """
    try:
        await user_service.reset_password(user_id, request.new_password)
        return PasswordResponse(
            success=True,
            message="Password reset successfully",
        )
    except UserNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except PasswordPolicyError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Password policy violation", "errors": e.errors},
        )


@router.post("/{user_id}/change-password", response_model=PasswordResponse)
async def change_password(
    user_id: int,
    request: ChangePasswordRequest,
    user_service: Annotated[UserService, Depends(get_user_service)],
) -> PasswordResponse:
    """
    Change a user's password (requires old password verification).
    
    Requirements: 16.5
    """
    try:
        await user_service.change_password(
            user_id=user_id,
            old_password=request.old_password,
            new_password=request.new_password,
        )
        return PasswordResponse(
            success=True,
            message="Password changed successfully",
        )
    except UserNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except UserServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except PasswordPolicyError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "Password policy violation", "errors": e.errors},
        )
