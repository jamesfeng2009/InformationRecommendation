from functools import wraps
from typing import Annotated, Callable, List, Optional, Tuple

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.core.redis import get_redis_client
from app.core.security import TokenData, TokenError, verify_token
from app.models.user import User
from app.services.auth import AuthService
from app.services.rbac import RBACService


# Security scheme for bearer token
security = HTTPBearer(auto_error=False)


class PermissionDeniedError(Exception):
    """User does not have required permission."""
    pass


async def get_current_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
) -> str:
    """Extract token from Authorization header."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


async def get_token_data(
    token: Annotated[str, Depends(get_current_token)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
    redis: Annotated[Redis, Depends(get_redis_client)],
) -> TokenData:
    """
    Verify token and return token data.
    
    Raises:
        HTTPException: If token is invalid or expired.
    """
    auth_service = AuthService(db, redis)
    try:
        return await auth_service.verify_access_token(token)
    except TokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    token_data: Annotated[TokenData, Depends(get_token_data)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> User:
    """
    Get current authenticated user from token.
    
    Returns:
        User: Current authenticated user.
    
    Raises:
        HTTPException: If user not found or disabled.
    """
    from sqlalchemy import select
    
    stmt = select(User).where(User.id == token_data.user_id)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if user.status != "enabled":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )
    
    return user


async def get_rbac_service(
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> RBACService:
    """Dependency to get RBACService instance."""
    return RBACService(db)


class PermissionChecker:
    """
    Permission checker dependency for FastAPI endpoints.
    
    Usage:
        @router.get("/users")
        async def list_users(
            user: Annotated[User, Depends(PermissionChecker("users", "read"))]
        ):
            ...
    
    Requirements: 12.2 - Permission checking middleware
    """
    
    def __init__(
        self,
        resource: str,
        action: str,
        require_all: bool = True,
    ):
        """
        Initialize permission checker.
        
        Args:
            resource: Resource name to check permission for.
            action: Action name to check permission for.
            require_all: If True, require all permissions (for multiple checks).
        """
        self.resource = resource
        self.action = action
        self.require_all = require_all
    
    async def __call__(
        self,
        current_user: Annotated[User, Depends(get_current_user)],
        rbac_service: Annotated[RBACService, Depends(get_rbac_service)],
    ) -> User:
        """
        Check if current user has required permission.
        
        Returns:
            User: Current user if permission granted.
        
        Raises:
            HTTPException: If permission denied.
        """
        has_permission = await rbac_service.check_permission(
            current_user.id, self.resource, self.action
        )
        
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: {self.resource}:{self.action}",
            )
        
        return current_user


class MultiPermissionChecker:
    """
    Check multiple permissions for an endpoint.
    
    Usage:
        @router.delete("/users/{id}")
        async def delete_user(
            user: Annotated[User, Depends(MultiPermissionChecker(
                [("users", "delete"), ("admin", "manage")],
                require_all=False  # User needs ANY of these permissions
            ))]
        ):
            ...
    """
    
    def __init__(
        self,
        permissions: List[Tuple[str, str]],
        require_all: bool = False,
    ):
        """
        Initialize multi-permission checker.
        
        Args:
            permissions: List of (resource, action) tuples.
            require_all: If True, require all permissions. If False, require any.
        """
        self.permissions = permissions
        self.require_all = require_all
    
    async def __call__(
        self,
        current_user: Annotated[User, Depends(get_current_user)],
        rbac_service: Annotated[RBACService, Depends(get_rbac_service)],
    ) -> User:
        """
        Check if current user has required permissions.
        
        Returns:
            User: Current user if permission granted.
        
        Raises:
            HTTPException: If permission denied.
        """
        if self.require_all:
            has_permission = await rbac_service.check_all_permissions(
                current_user.id, self.permissions
            )
        else:
            has_permission = await rbac_service.check_any_permission(
                current_user.id, self.permissions
            )
        
        if not has_permission:
            perm_str = ", ".join(f"{r}:{a}" for r, a in self.permissions)
            mode = "all" if self.require_all else "any"
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: requires {mode} of [{perm_str}]",
            )
        
        return current_user


def require_permission(resource: str, action: str):
    """
    Decorator to require permission for an endpoint function.
    
    Usage:
        @router.get("/users")
        @require_permission("users", "read")
        async def list_users(current_user: User = Depends(get_current_user)):
            ...
    
    Note: This decorator is for documentation purposes. 
    Use PermissionChecker dependency for actual permission checking.
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        
        # Add permission metadata to function
        if not hasattr(wrapper, "_required_permissions"):
            wrapper._required_permissions = []
        wrapper._required_permissions.append((resource, action))
        
        return wrapper
    return decorator


async def get_user_accessible_menus(
    current_user: Annotated[User, Depends(get_current_user)],
    rbac_service: Annotated[RBACService, Depends(get_rbac_service)],
) -> List[str]:
    """
    Get list of menu resources the current user can access.
    
    Returns:
        List of menu resource names.
    
    Requirements: 12.2 - Menu permission filtering
    """
    return await rbac_service.get_user_menu_permissions(current_user.id)


async def get_user_permissions(
    current_user: Annotated[User, Depends(get_current_user)],
    rbac_service: Annotated[RBACService, Depends(get_rbac_service)],
) -> List[str]:
    """
    Get all permissions for the current user as strings.
    
    Returns:
        List of permission strings in "resource:action" format.
    """
    perm_set = await rbac_service.get_user_permission_set(current_user.id)
    return sorted(list(perm_set))


class DataPermissionFilter:
    """
    Data permission filter for controlling data access scope.
    
    This filter can be used to restrict query results based on user's
    data permission level (e.g., own data, department data, all data).
    
    Requirements: 12.3 - Data permission filtering
    """
    
    # Data permission levels
    LEVEL_OWN = "own"           # User can only access their own data
    LEVEL_DEPARTMENT = "dept"   # User can access department data
    LEVEL_ALL = "all"           # User can access all data
    
    def __init__(self, resource: str):
        """
        Initialize data permission filter.
        
        Args:
            resource: Resource name to filter data for.
        """
        self.resource = resource
    
    async def get_data_scope(
        self,
        user: User,
        rbac_service: RBACService,
    ) -> Tuple[str, Optional[int]]:
        """
        Determine data access scope for a user.
        
        Args:
            user: Current user.
            rbac_service: RBAC service instance.
        
        Returns:
            Tuple of (scope_level, department_id).
            - ("all", None) - User can access all data
            - ("dept", dept_id) - User can access department data
            - ("own", None) - User can only access own data
        """
        # Check for all data access
        if await rbac_service.check_permission(user.id, self.resource, "read_all"):
            return (self.LEVEL_ALL, None)
        
        # Check for department data access
        if await rbac_service.check_permission(user.id, self.resource, "read_dept"):
            return (self.LEVEL_DEPARTMENT, user.department_id)
        
        # Default to own data only
        return (self.LEVEL_OWN, None)
    
    def apply_filter(
        self,
        query,
        scope: str,
        user_id: int,
        department_id: Optional[int],
        user_id_column,
        department_id_column=None,
    ):
        """
        Apply data permission filter to a SQLAlchemy query.
        
        Args:
            query: SQLAlchemy select statement.
            scope: Data scope level.
            user_id: Current user's ID.
            department_id: Current user's department ID.
            user_id_column: Column to filter by user ID.
            department_id_column: Column to filter by department ID (optional).
        
        Returns:
            Filtered query.
        """
        if scope == self.LEVEL_ALL:
            return query
        
        if scope == self.LEVEL_DEPARTMENT and department_id_column is not None:
            return query.where(department_id_column == department_id)
        
        # Default: own data only
        return query.where(user_id_column == user_id)


# Convenience type aliases for dependency injection
CurrentUser = Annotated[User, Depends(get_current_user)]
TokenDataDep = Annotated[TokenData, Depends(get_token_data)]
RBACServiceDep = Annotated[RBACService, Depends(get_rbac_service)]
AccessibleMenus = Annotated[List[str], Depends(get_user_accessible_menus)]
UserPermissions = Annotated[List[str], Depends(get_user_permissions)]
