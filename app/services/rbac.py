from typing import List, Optional, Set

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.rbac import Permission, Role, RolePermission, UserRole
from app.models.user import User


class RBACError(Exception):
    """Base exception for RBAC-related errors."""
    pass


class RoleNotFoundError(RBACError):
    """Role not found."""
    pass


class PermissionNotFoundError(RBACError):
    """Permission not found."""
    pass


class RoleExistsError(RBACError):
    """Role already exists."""
    pass


class PermissionExistsError(RBACError):
    """Permission already exists."""
    pass


class RBACService:
    
    def __init__(self, db: AsyncSession):
        self.db = db

    # ==================== Role CRUD Operations ====================
    
    async def create_role(self, name: str, description: Optional[str] = None) -> Role:
        """
        Create a new role.
        
        Args:
            name: Unique role name.
            description: Optional role description.
        
        Returns:
            Created Role instance.
        
        Raises:
            RoleExistsError: If role with same name exists.
        """
        # Check if role exists
        existing = await self.get_role_by_name(name)
        if existing:
            raise RoleExistsError(f"Role '{name}' already exists")
        
        role = Role(name=name, description=description)
        self.db.add(role)
        await self.db.flush()
        await self.db.refresh(role)
        return role
    
    async def get_role(self, role_id: int) -> Optional[Role]:
        """Get role by ID."""
        stmt = select(Role).where(Role.id == role_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_role_by_name(self, name: str) -> Optional[Role]:
        """Get role by name."""
        stmt = select(Role).where(Role.name == name)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_role_with_permissions(self, role_id: int) -> Optional[Role]:
        """Get role with its permissions loaded."""
        stmt = (
            select(Role)
            .where(Role.id == role_id)
            .options(selectinload(Role.role_permissions).selectinload(RolePermission.permission))
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def list_roles(self) -> List[Role]:
        """List all roles."""
        stmt = select(Role).order_by(Role.name)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
    
    async def update_role(
        self, role_id: int, name: Optional[str] = None, description: Optional[str] = None
    ) -> Role:
        """
        Update role information.

        Args:
            role_id: Role ID to update.
            name: New role name (optional).
            description: New description (optional).

        Returns:
            Updated Role instance.

        Raises:
            RoleNotFoundError: If role not found.
            RoleExistsError: If new name conflicts with existing role.
        """
        role = await self.get_role(role_id)
        if not role:
            raise RoleNotFoundError(f"Role with ID {role_id} not found")
        
        if name and name != role.name:
            existing = await self.get_role_by_name(name)
            if existing:
                raise RoleExistsError(f"Role '{name}' already exists")
            role.name = name
        
        if description is not None:
            role.description = description
        
        await self.db.flush()
        await self.db.refresh(role)
        return role
    
    async def delete_role(self, role_id: int) -> bool:
        """
        Delete a role and all its associations.
        
        Args:
            role_id: Role ID to delete.
        
        Returns:
            True if deleted, False if not found.
        """
        role = await self.get_role(role_id)
        if not role:
            return False
        
        await self.db.delete(role)
        await self.db.flush()
        return True

    # ==================== Permission CRUD Operations ====================
    
    async def create_permission(
        self, resource: str, action: str, description: Optional[str] = None
    ) -> Permission:
        """
        Create a new permission.
        
        Args:
            resource: Resource name (e.g., "users", "news", "topics").
            action: Action name (e.g., "read", "write", "delete").
            description: Optional permission description.
        
        Returns:
            Created Permission instance.
        
        Raises:
            PermissionExistsError: If permission with same resource/action exists.
        """
        existing = await self.get_permission_by_resource_action(resource, action)
        if existing:
            raise PermissionExistsError(f"Permission '{resource}:{action}' already exists")
        
        permission = Permission(resource=resource, action=action, description=description)
        self.db.add(permission)
        await self.db.flush()
        await self.db.refresh(permission)
        return permission
    
    async def get_permission(self, permission_id: int) -> Optional[Permission]:
        """Get permission by ID."""
        stmt = select(Permission).where(Permission.id == permission_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_permission_by_resource_action(
        self, resource: str, action: str
    ) -> Optional[Permission]:
        """Get permission by resource and action."""
        stmt = select(Permission).where(
            Permission.resource == resource,
            Permission.action == action
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def list_permissions(self, resource: Optional[str] = None) -> List[Permission]:
        """
        List permissions, optionally filtered by resource.
        
        Args:
            resource: Optional resource filter.
        
        Returns:
            List of Permission instances.
        """
        stmt = select(Permission).order_by(Permission.resource, Permission.action)
        if resource:
            stmt = stmt.where(Permission.resource == resource)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
    
    async def delete_permission(self, permission_id: int) -> bool:
        """
        Delete a permission and all its role associations.
        
        Args:
            permission_id: Permission ID to delete.
        
        Returns:
            True if deleted, False if not found.
        """
        permission = await self.get_permission(permission_id)
        if not permission:
            return False
        
        await self.db.delete(permission)
        await self.db.flush()
        return True

    # ==================== Permission Assignment to Roles ====================
    
    async def assign_permission_to_role(self, role_id: int, permission_id: int) -> bool:
        """
        Assign a permission to a role.
        
        Args:
            role_id: Role ID.
            permission_id: Permission ID.
        
        Returns:
            True if assigned, False if already assigned.
        
        Raises:
            RoleNotFoundError: If role not found.
            PermissionNotFoundError: If permission not found.
        """
        role = await self.get_role(role_id)
        if not role:
            raise RoleNotFoundError(f"Role with ID {role_id} not found")
        
        permission = await self.get_permission(permission_id)
        if not permission:
            raise PermissionNotFoundError(f"Permission with ID {permission_id} not found")
        
        # Check if already assigned
        stmt = select(RolePermission).where(
            RolePermission.role_id == role_id,
            RolePermission.permission_id == permission_id
        )
        result = await self.db.execute(stmt)
        if result.scalar_one_or_none():
            return False
        
        role_permission = RolePermission(role_id=role_id, permission_id=permission_id)
        self.db.add(role_permission)
        await self.db.flush()
        return True
    
    async def assign_permissions_to_role(
        self, role_id: int, permission_ids: List[int]
    ) -> int:
        """
        Assign multiple permissions to a role.
        
        Args:
            role_id: Role ID.
            permission_ids: List of permission IDs.
        
        Returns:
            Number of permissions assigned.
        
        Raises:
            RoleNotFoundError: If role not found.
        """
        role = await self.get_role(role_id)
        if not role:
            raise RoleNotFoundError(f"Role with ID {role_id} not found")
        
        count = 0
        for perm_id in permission_ids:
            try:
                if await self.assign_permission_to_role(role_id, perm_id):
                    count += 1
            except PermissionNotFoundError:
                continue
        
        return count
    
    async def revoke_permission_from_role(self, role_id: int, permission_id: int) -> bool:
        """
        Revoke a permission from a role.
        
        Args:
            role_id: Role ID.
            permission_id: Permission ID.
        
        Returns:
            True if revoked, False if not assigned.
        """
        stmt = delete(RolePermission).where(
            RolePermission.role_id == role_id,
            RolePermission.permission_id == permission_id
        )
        result = await self.db.execute(stmt)
        await self.db.flush()
        return result.rowcount > 0
    
    async def set_role_permissions(self, role_id: int, permission_ids: List[int]) -> None:
        """
        Set exact permissions for a role (replaces existing).
        
        Args:
            role_id: Role ID.
            permission_ids: List of permission IDs to set.
        
        Raises:
            RoleNotFoundError: If role not found.
        """
        role = await self.get_role(role_id)
        if not role:
            raise RoleNotFoundError(f"Role with ID {role_id} not found")
        
        # Remove all existing permissions
        stmt = delete(RolePermission).where(RolePermission.role_id == role_id)
        await self.db.execute(stmt)
        
        # Add new permissions
        for perm_id in permission_ids:
            permission = await self.get_permission(perm_id)
            if permission:
                role_permission = RolePermission(role_id=role_id, permission_id=perm_id)
                self.db.add(role_permission)
        
        await self.db.flush()
    
    async def get_role_permissions(self, role_id: int) -> List[Permission]:
        """
        Get all permissions assigned to a role.
        
        Args:
            role_id: Role ID.
        
        Returns:
            List of Permission instances.
        """
        stmt = (
            select(Permission)
            .join(RolePermission, RolePermission.permission_id == Permission.id)
            .where(RolePermission.role_id == role_id)
            .order_by(Permission.resource, Permission.action)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    # ==================== User-Role Binding ====================
    
    async def assign_role_to_user(self, user_id: int, role_id: int) -> bool:
        """
        Assign a role to a user.
        
        Args:
            user_id: User ID.
            role_id: Role ID.
        
        Returns:
            True if assigned, False if already assigned.
        
        Raises:
            RoleNotFoundError: If role not found.
        """
        role = await self.get_role(role_id)
        if not role:
            raise RoleNotFoundError(f"Role with ID {role_id} not found")
        
        # Check if already assigned
        stmt = select(UserRole).where(
            UserRole.user_id == user_id,
            UserRole.role_id == role_id
        )
        result = await self.db.execute(stmt)
        if result.scalar_one_or_none():
            return False
        
        user_role = UserRole(user_id=user_id, role_id=role_id)
        self.db.add(user_role)
        await self.db.flush()
        return True
    
    async def assign_roles_to_user(self, user_id: int, role_ids: List[int]) -> int:
        """
        Assign multiple roles to a user.
        
        Args:
            user_id: User ID.
            role_ids: List of role IDs.
        
        Returns:
            Number of roles assigned.
        """
        count = 0
        for role_id in role_ids:
            try:
                if await self.assign_role_to_user(user_id, role_id):
                    count += 1
            except RoleNotFoundError:
                continue
        return count
    
    async def revoke_role_from_user(self, user_id: int, role_id: int) -> bool:
        """
        Revoke a role from a user.
        
        Args:
            user_id: User ID.
            role_id: Role ID.
        
        Returns:
            True if revoked, False if not assigned.
        """
        stmt = delete(UserRole).where(
            UserRole.user_id == user_id,
            UserRole.role_id == role_id
        )
        result = await self.db.execute(stmt)
        await self.db.flush()
        return result.rowcount > 0
    
    async def set_user_roles(self, user_id: int, role_ids: List[int]) -> None:
        """
        Set exact roles for a user (replaces existing).
        
        Args:
            user_id: User ID.
            role_ids: List of role IDs to set.
        """
        # Remove all existing roles
        stmt = delete(UserRole).where(UserRole.user_id == user_id)
        await self.db.execute(stmt)
        
        # Add new roles
        for role_id in role_ids:
            role = await self.get_role(role_id)
            if role:
                user_role = UserRole(user_id=user_id, role_id=role_id)
                self.db.add(user_role)
        
        await self.db.flush()
    
    async def get_user_roles(self, user_id: int) -> List[Role]:
        """
        Get all roles assigned to a user.
        
        Args:
            user_id: User ID.
        
        Returns:
            List of Role instances.
        """
        stmt = (
            select(Role)
            .join(UserRole, UserRole.role_id == Role.id)
            .where(UserRole.user_id == user_id)
            .order_by(Role.name)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    # ==================== Permission Checking ====================
    
    async def get_user_permissions(self, user_id: int) -> List[Permission]:
        """
        Get all permissions for a user through their roles.
        
        Args:
            user_id: User ID.
        
        Returns:
            List of unique Permission instances.
        """
        stmt = (
            select(Permission)
            .distinct()
            .join(RolePermission, RolePermission.permission_id == Permission.id)
            .join(UserRole, UserRole.role_id == RolePermission.role_id)
            .where(UserRole.user_id == user_id)
            .order_by(Permission.resource, Permission.action)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
    
    async def get_user_permission_set(self, user_id: int) -> Set[str]:
        """
        Get all permissions for a user as a set of "resource:action" strings.
        
        Args:
            user_id: User ID.
        
        Returns:
            Set of permission strings in "resource:action" format.
        """
        permissions = await self.get_user_permissions(user_id)
        return {f"{p.resource}:{p.action}" for p in permissions}
    
    async def check_permission(
        self, user_id: int, resource: str, action: str
    ) -> bool:
        """
        Check if a user has a specific permission.
        
        A user has a permission if at least one of their assigned roles
        has that permission granted.
        
        Args:
            user_id: User ID.
            resource: Resource name.
            action: Action name.
        
        Returns:
            True if user has permission, False otherwise.
        
        Requirements: 12.1, 12.2 - RBAC Permission Enforcement
        """
        stmt = (
            select(RolePermission.id)
            .join(UserRole, UserRole.role_id == RolePermission.role_id)
            .join(Permission, Permission.id == RolePermission.permission_id)
            .where(
                UserRole.user_id == user_id,
                Permission.resource == resource,
                Permission.action == action
            )
            .limit(1)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none() is not None
    
    async def check_any_permission(
        self, user_id: int, permissions: List[tuple[str, str]]
    ) -> bool:
        """
        Check if a user has any of the specified permissions.
        
        Args:
            user_id: User ID.
            permissions: List of (resource, action) tuples.
        
        Returns:
            True if user has at least one permission, False otherwise.
        """
        for resource, action in permissions:
            if await self.check_permission(user_id, resource, action):
                return True
        return False
    
    async def check_all_permissions(
        self, user_id: int, permissions: List[tuple[str, str]]
    ) -> bool:
        """
        Check if a user has all of the specified permissions.
        
        Args:
            user_id: User ID.
            permissions: List of (resource, action) tuples.
        
        Returns:
            True if user has all permissions, False otherwise.
        """
        for resource, action in permissions:
            if not await self.check_permission(user_id, resource, action):
                return False
        return True
    
    # ==================== Menu Permission Filtering ====================
    
    async def get_user_menu_permissions(self, user_id: int) -> List[str]:
        """
        Get menu resources that a user can access.
        
        Args:
            user_id: User ID.
        
        Returns:
            List of menu resource names the user can view.
        
        Requirements: 12.2 - Menu permission filtering
        """
        stmt = (
            select(Permission.resource)
            .distinct()
            .join(RolePermission, RolePermission.permission_id == Permission.id)
            .join(UserRole, UserRole.role_id == RolePermission.role_id)
            .where(
                UserRole.user_id == user_id,
                Permission.action == "view"
            )
            .order_by(Permission.resource)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
    
    async def can_view_menu(self, user_id: int, menu_resource: str) -> bool:
        """
        Check if a user can view a specific menu.
        
        Args:
            user_id: User ID.
            menu_resource: Menu resource name.
        
        Returns:
            True if user can view the menu, False otherwise.
        """
        return await self.check_permission(user_id, menu_resource, "view")
