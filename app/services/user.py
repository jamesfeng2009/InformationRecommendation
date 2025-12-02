from dataclasses import dataclass
from typing import List, Optional

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.security import hash_password, validate_password_policy
from app.models.user import Department, User


class UserServiceError(Exception):
    """Base exception for user service errors."""
    pass


class UserNotFoundError(UserServiceError):
    """User not found."""
    pass


class UserExistsError(UserServiceError):
    """User already exists."""
    pass


class DepartmentNotFoundError(UserServiceError):
    """Department not found."""
    pass


class PasswordPolicyError(UserServiceError):
    """Password does not meet policy requirements."""
    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Password policy violation: {', '.join(errors)}")


class BatchUpdateError(UserServiceError):
    """Batch update operation failed."""
    def __init__(self, message: str, failed_user_ids: List[int]):
        self.failed_user_ids = failed_user_ids
        super().__init__(message)


@dataclass
class UserFilter:
    """Filter criteria for user queries."""
    name: Optional[str] = None
    account: Optional[str] = None
    department_id: Optional[int] = None
    status: Optional[str] = None
    keywords: Optional[str] = None


@dataclass
class PagedResult:
    """Paginated result container."""
    items: List[User]
    total: int
    page: int
    size: int
    pages: int


@dataclass
class BatchUpdateData:
    """Data for batch user updates."""
    department_id: Optional[int] = None
    status: Optional[str] = None
    role_ids: Optional[List[int]] = None
    keywords: Optional[str] = None


class UserService:
    """
    User management service handling CRUD operations.
    Requirements: 10.2, 10.3, 10.4
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db

    # ==================== Create Operations ====================
    
    async def create_user(
        self,
        username: str,
        password: str,
        name: str,
        account: Optional[str] = None,
        department_id: Optional[int] = None,
        status: str = "enabled",
        keywords: Optional[str] = None,
    ) -> User:
        """
        Create a new user with validation.
        
        Args:
            username: Unique username.
            password: Plain text password (will be hashed).
            name: User's display name.
            account: Login account (defaults to pinyin of name if not provided).
            department_id: Optional department ID.
            status: User status ('enabled' or 'disabled').
            keywords: Optional user keywords.
        
        Returns:
            Created User instance.
        
        Raises:
            UserExistsError: If username or account already exists.
            DepartmentNotFoundError: If department_id is invalid.
            PasswordPolicyError: If password doesn't meet policy.
        
        Requirements: 10.2
        """
        # Validate password policy
        validation = validate_password_policy(password)
        if not validation.is_valid:
            raise PasswordPolicyError(validation.errors)
        
        # Check if username exists
        existing = await self.get_user_by_username(username)
        if existing:
            raise UserExistsError(f"Username '{username}' already exists")
        
        # Generate account from name if not provided
        if account is None:
            account = username  # Default to username
        
        # Check if account exists
        existing_account = await self.get_user_by_account(account)
        if existing_account:
            raise UserExistsError(f"Account '{account}' already exists")
        
        # Validate department if provided
        if department_id is not None:
            dept = await self._get_department(department_id)
            if not dept:
                raise DepartmentNotFoundError(f"Department with ID {department_id} not found")
        
        # Create user
        user = User(
            username=username,
            password_hash=hash_password(password),
            name=name,
            account=account,
            department_id=department_id,
            status=status,
            keywords=keywords,
        )
        
        self.db.add(user)
        await self.db.flush()
        await self.db.refresh(user)
        return user

    # ==================== Read Operations ====================
    
    async def get_user(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        stmt = select(User).where(User.id == user_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_user_with_relations(self, user_id: int) -> Optional[User]:
        """Get user with department and roles loaded."""
        stmt = (
            select(User)
            .where(User.id == user_id)
            .options(
                selectinload(User.department),
                selectinload(User.roles),
            )
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        stmt = select(User).where(User.username == username)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_user_by_account(self, account: str) -> Optional[User]:
        """Get user by account."""
        stmt = select(User).where(User.account == account)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()


    async def list_users(
        self,
        filters: Optional[UserFilter] = None,
        page: int = 1,
        size: int = 20,
    ) -> PagedResult:
        """
        List users with optional filters and pagination.
        
        Args:
            filters: Optional filter criteria.
            page: Page number (1-indexed).
            size: Page size.
        
        Returns:
            PagedResult with users and pagination info.
        
        Requirements: 10.4
        """
        # Base query
        stmt = select(User).options(selectinload(User.department))
        count_stmt = select(func.count(User.id))
        
        # Apply filters
        if filters:
            if filters.name:
                stmt = stmt.where(User.name.ilike(f"%{filters.name}%"))
                count_stmt = count_stmt.where(User.name.ilike(f"%{filters.name}%"))
            
            if filters.account:
                stmt = stmt.where(User.account.ilike(f"%{filters.account}%"))
                count_stmt = count_stmt.where(User.account.ilike(f"%{filters.account}%"))
            
            if filters.department_id is not None:
                stmt = stmt.where(User.department_id == filters.department_id)
                count_stmt = count_stmt.where(User.department_id == filters.department_id)
            
            if filters.status:
                stmt = stmt.where(User.status == filters.status)
                count_stmt = count_stmt.where(User.status == filters.status)
            
            if filters.keywords:
                stmt = stmt.where(User.keywords.ilike(f"%{filters.keywords}%"))
                count_stmt = count_stmt.where(User.keywords.ilike(f"%{filters.keywords}%"))
        
        # Get total count
        total_result = await self.db.execute(count_stmt)
        total = total_result.scalar() or 0
        
        # Apply pagination
        offset = (page - 1) * size
        stmt = stmt.order_by(User.id).offset(offset).limit(size)
        
        # Execute query
        result = await self.db.execute(stmt)
        users = list(result.scalars().all())
        
        # Calculate total pages
        pages = (total + size - 1) // size if size > 0 else 0
        
        return PagedResult(
            items=users,
            total=total,
            page=page,
            size=size,
            pages=pages,
        )

    # ==================== Update Operations ====================
    
    async def update_user(
        self,
        user_id: int,
        username: Optional[str] = None,
        name: Optional[str] = None,
        account: Optional[str] = None,
        department_id: Optional[int] = None,
        status: Optional[str] = None,
        keywords: Optional[str] = None,
    ) -> User:
        """
        Update a single user.
        
        Args:
            user_id: User ID to update.
            username: New username (optional).
            name: New display name (optional).
            account: New account (optional).
            department_id: New department ID (optional, use -1 to clear).
            status: New status (optional).
            keywords: New keywords (optional).
        
        Returns:
            Updated User instance.
        
        Raises:
            UserNotFoundError: If user not found.
            UserExistsError: If new username/account conflicts.
            DepartmentNotFoundError: If department_id is invalid.
        
        Requirements: 10.3
        """
        user = await self.get_user(user_id)
        if not user:
            raise UserNotFoundError(f"User with ID {user_id} not found")
        
        # Update username if provided
        if username is not None and username != user.username:
            existing = await self.get_user_by_username(username)
            if existing:
                raise UserExistsError(f"Username '{username}' already exists")
            user.username = username
        
        # Update name if provided
        if name is not None:
            user.name = name
        
        # Update account if provided
        if account is not None and account != user.account:
            existing = await self.get_user_by_account(account)
            if existing:
                raise UserExistsError(f"Account '{account}' already exists")
            user.account = account
        
        # Update department if provided
        if department_id is not None:
            if department_id == -1:
                user.department_id = None
            else:
                dept = await self._get_department(department_id)
                if not dept:
                    raise DepartmentNotFoundError(f"Department with ID {department_id} not found")
                user.department_id = department_id
        
        # Update status if provided
        if status is not None:
            if status not in ("enabled", "disabled"):
                raise UserServiceError(f"Invalid status: {status}")
            user.status = status
        
        # Update keywords if provided
        if keywords is not None:
            user.keywords = keywords
        
        await self.db.flush()
        await self.db.refresh(user)
        return user

    async def batch_update_users(
        self,
        user_ids: List[int],
        data: BatchUpdateData,
    ) -> List[User]:
        """
        Batch update multiple users atomically.
        
        Either all users are updated successfully, or none are updated (rollback).
        
        Args:
            user_ids: List of user IDs to update.
            data: BatchUpdateData with fields to update.
        
        Returns:
            List of updated User instances.
        
        Raises:
            BatchUpdateError: If any user update fails (all changes rolled back).
            DepartmentNotFoundError: If department_id is invalid.
        
        Requirements: 10.3 - Batch update atomicity (Property 18)
        """
        if not user_ids:
            return []
        
        # Validate department if provided
        if data.department_id is not None:
            dept = await self._get_department(data.department_id)
            if not dept:
                raise DepartmentNotFoundError(f"Department with ID {data.department_id} not found")
        
        # Validate status if provided
        if data.status is not None and data.status not in ("enabled", "disabled"):
            raise UserServiceError(f"Invalid status: {data.status}")
        
        # Fetch all users first to validate they exist
        stmt = select(User).where(User.id.in_(user_ids))
        result = await self.db.execute(stmt)
        users = list(result.scalars().all())
        
        # Check if all users were found
        found_ids = {u.id for u in users}
        missing_ids = set(user_ids) - found_ids
        if missing_ids:
            raise BatchUpdateError(
                f"Users not found: {missing_ids}",
                failed_user_ids=list(missing_ids)
            )
        
        # Apply updates to all users
        updated_users = []
        for user in users:
            if data.department_id is not None:
                user.department_id = data.department_id
            if data.status is not None:
                user.status = data.status
            if data.keywords is not None:
                user.keywords = data.keywords
            updated_users.append(user)
        
        # Flush all changes (will be committed by caller or rolled back on error)
        await self.db.flush()
        
        # Refresh all users to get updated values
        for user in updated_users:
            await self.db.refresh(user)
        
        return updated_users

    # ==================== Delete Operations ====================
    
    async def delete_user(self, user_id: int) -> bool:
        """
        Delete a user by ID.
        
        Args:
            user_id: User ID to delete.
        
        Returns:
            True if user was deleted.
        
        Raises:
            UserNotFoundError: If user not found.
        
        Requirements: 10.2
        """
        user = await self.get_user(user_id)
        if not user:
            raise UserNotFoundError(f"User with ID {user_id} not found")
        
        await self.db.delete(user)
        await self.db.flush()
        return True

    # ==================== Password Operations ====================
    
    async def reset_password(self, user_id: int, new_password: str) -> bool:
        """
        Reset a user's password (admin operation).
        
        Args:
            user_id: User ID whose password to reset.
            new_password: New plain text password.
        
        Returns:
            True if password was reset.
        
        Raises:
            UserNotFoundError: If user not found.
            PasswordPolicyError: If new password doesn't meet policy.
        
        Requirements: 10.5
        """
        # Validate password policy
        validation = validate_password_policy(new_password)
        if not validation.is_valid:
            raise PasswordPolicyError(validation.errors)
        
        user = await self.get_user(user_id)
        if not user:
            raise UserNotFoundError(f"User with ID {user_id} not found")
        
        user.password_hash = hash_password(new_password)
        await self.db.flush()
        return True

    async def change_password(
        self,
        user_id: int,
        old_password: str,
        new_password: str,
    ) -> bool:
        """
        Change a user's password (user operation, requires old password).
        
        Args:
            user_id: User ID.
            old_password: Current password for verification.
            new_password: New plain text password.
        
        Returns:
            True if password was changed.
        
        Raises:
            UserNotFoundError: If user not found.
            UserServiceError: If old password is incorrect.
            PasswordPolicyError: If new password doesn't meet policy.
        
        Requirements: 16.5
        """
        from app.core.security import verify_password
        
        user = await self.get_user(user_id)
        if not user:
            raise UserNotFoundError(f"User with ID {user_id} not found")
        
        # Verify old password
        if not verify_password(old_password, user.password_hash):
            raise UserServiceError("Current password is incorrect")
        
        # Validate new password policy
        validation = validate_password_policy(new_password)
        if not validation.is_valid:
            raise PasswordPolicyError(validation.errors)
        
        user.password_hash = hash_password(new_password)
        await self.db.flush()
        return True

    # ==================== Helper Methods ====================
    
    async def _get_department(self, department_id: int) -> Optional[Department]:
        """Get department by ID."""
        stmt = select(Department).where(Department.id == department_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
