"""
Organization management service for department CRUD operations.
Requirements: 11.1, 11.2, 11.3, 11.4

Provides:
- Department CRUD operations (create, read, update, delete)
- Department tree query
- Move department (change parent)
- Depth validation for 3-level tree structure
"""
from dataclasses import dataclass
from typing import List, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.user import Department, User


class OrganizationServiceError(Exception):
    """Base exception for organization service errors."""
    pass


class DepartmentNotFoundError(OrganizationServiceError):
    """Department not found."""
    pass


class DepartmentExistsError(OrganizationServiceError):
    """Department with same name already exists."""
    pass


class DepartmentHasUsersError(OrganizationServiceError):
    """Department has users and cannot be deleted."""
    def __init__(self, department_id: int, user_count: int):
        self.department_id = department_id
        self.user_count = user_count
        super().__init__(
            f"Department {department_id} has {user_count} users. "
            "Transfer users before deletion."
        )


class DepartmentHasChildrenError(OrganizationServiceError):
    """Department has child departments and cannot be deleted."""
    def __init__(self, department_id: int, children_count: int):
        self.department_id = department_id
        self.children_count = children_count
        super().__init__(
            f"Department {department_id} has {children_count} child departments. "
            "Delete children first."
        )


class DepartmentDepthExceededError(OrganizationServiceError):
    """Department tree depth would exceed the maximum allowed (3 levels)."""
    def __init__(self, max_depth: int = 2):
        self.max_depth = max_depth
        super().__init__(
            f"Department tree depth would exceed maximum of {max_depth + 1} levels "
            "(root -> level 1 -> level 2)"
        )


class CircularReferenceError(OrganizationServiceError):
    """Moving department would create a circular reference."""
    pass


@dataclass
class DepartmentNode:
    """Department node for tree representation."""
    id: int
    name: str
    parent_id: Optional[int]
    manager: Optional[str]
    contact: Optional[str]
    description: Optional[str]
    sort_order: int
    depth: int
    user_count: int
    children: List["DepartmentNode"]


class OrganizationService:
    """
    Organization management service handling department CRUD operations.
    Requirements: 11.1, 11.2, 11.3, 11.4
    """
    
    MAX_DEPTH = 2  # 3 levels: root (0) -> level 1 (1) -> level 2 (2)
    
    def __init__(self, db: AsyncSession):
        self.db = db

    # ==================== Create Operations ====================
    
    async def create_department(
        self,
        name: str,
        parent_id: Optional[int] = None,
        manager: Optional[str] = None,
        contact: Optional[str] = None,
        description: Optional[str] = None,
        sort_order: int = 0,
    ) -> Department:
        """
        Create a new department with ID generation.
        
        Args:
            name: Department name.
            parent_id: Parent department ID (None for root).
            manager: Responsible person.
            contact: Contact information.
            description: Department description.
            sort_order: Sort order for display.
        
        Returns:
            Created Department instance.
        
        Raises:
            DepartmentNotFoundError: If parent_id is invalid.
            DepartmentDepthExceededError: If adding would exceed 3 levels.
        
        Requirements: 11.1, 11.2
        """
        # Validate parent and check depth
        parent_depth = 0
        if parent_id is not None:
            parent = await self.get_department_with_parent_chain(parent_id)
            if not parent:
                raise DepartmentNotFoundError(f"Parent department with ID {parent_id} not found")
            
            # Calculate parent's depth
            parent_depth = await self._calculate_depth(parent)
            
            # Check if adding a child would exceed max depth
            if parent_depth >= self.MAX_DEPTH:
                raise DepartmentDepthExceededError(self.MAX_DEPTH)
        
        # Create department
        department = Department(
            name=name,
            parent_id=parent_id,
            manager=manager,
            contact=contact,
            description=description,
            sort_order=sort_order,
        )
        
        self.db.add(department)
        await self.db.flush()
        await self.db.refresh(department)
        return department

    # ==================== Read Operations ====================
    
    async def get_department(self, department_id: int) -> Optional[Department]:
        """Get department by ID."""
        stmt = select(Department).where(Department.id == department_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_department_with_parent_chain(self, department_id: int) -> Optional[Department]:
        """Get department with parent chain loaded for depth calculation."""
        stmt = (
            select(Department)
            .where(Department.id == department_id)
            .options(selectinload(Department.parent))
        )
        result = await self.db.execute(stmt)
        dept = result.scalar_one_or_none()
        
        if dept and dept.parent_id:
            # Recursively load parent chain
            await self._load_parent_chain(dept)
        
        return dept
    
    async def _load_parent_chain(self, department: Department) -> None:
        """Recursively load the parent chain for a department."""
        if department.parent_id and department.parent:
            parent_stmt = (
                select(Department)
                .where(Department.id == department.parent_id)
                .options(selectinload(Department.parent))
            )
            result = await self.db.execute(parent_stmt)
            parent = result.scalar_one_or_none()
            if parent:
                department.parent = parent
                await self._load_parent_chain(parent)
    
    async def get_department_with_children(self, department_id: int) -> Optional[Department]:
        """Get department with children loaded."""
        stmt = (
            select(Department)
            .where(Department.id == department_id)
            .options(selectinload(Department.children))
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_department_tree(self) -> List[DepartmentNode]:
        """
        Get the complete department tree structure.
        
        Returns:
            List of root DepartmentNode objects with nested children.
        
        Requirements: 11.1
        """
        # Get all departments
        stmt = select(Department).order_by(Department.sort_order, Department.id)
        result = await self.db.execute(stmt)
        all_departments = list(result.scalars().all())
        
        # Get user counts per department
        user_counts = await self._get_user_counts()
        
        # Build tree structure
        dept_map = {}
        root_nodes = []
        
        # First pass: create all nodes
        for dept in all_departments:
            depth = await self._calculate_depth_from_list(dept, all_departments)
            node = DepartmentNode(
                id=dept.id,
                name=dept.name,
                parent_id=dept.parent_id,
                manager=dept.manager,
                contact=dept.contact,
                description=dept.description,
                sort_order=dept.sort_order,
                depth=depth,
                user_count=user_counts.get(dept.id, 0),
                children=[],
            )
            dept_map[dept.id] = node
        
        # Second pass: build tree
        for dept in all_departments:
            node = dept_map[dept.id]
            if dept.parent_id is None:
                root_nodes.append(node)
            elif dept.parent_id in dept_map:
                dept_map[dept.parent_id].children.append(node)
        
        return root_nodes
    
    async def _get_user_counts(self) -> dict[int, int]:
        """Get user count per department."""
        stmt = (
            select(User.department_id, func.count(User.id))
            .where(User.department_id.isnot(None))
            .group_by(User.department_id)
        )
        result = await self.db.execute(stmt)
        return {row[0]: row[1] for row in result.all()}
    
    async def _calculate_depth(self, department: Department) -> int:
        """Calculate the depth of a department by traversing parent chain."""
        depth = 0
        current = department
        while current.parent is not None:
            depth += 1
            current = current.parent
        return depth
    
    def _calculate_depth_from_list(
        self, department: Department, all_departments: List[Department]
    ) -> int:
        """Calculate depth using a list of all departments."""
        dept_map = {d.id: d for d in all_departments}
        depth = 0
        current_id = department.parent_id
        while current_id is not None:
            depth += 1
            parent = dept_map.get(current_id)
            if parent:
                current_id = parent.parent_id
            else:
                break
        return depth

    # ==================== Update Operations ====================
    
    async def update_department(
        self,
        department_id: int,
        name: Optional[str] = None,
        manager: Optional[str] = None,
        contact: Optional[str] = None,
        description: Optional[str] = None,
        sort_order: Optional[int] = None,
    ) -> Department:
        """
        Update a department's information.
        
        Args:
            department_id: Department ID to update.
            name: New department name (optional).
            manager: New manager (optional).
            contact: New contact info (optional).
            description: New description (optional).
            sort_order: New sort order (optional).
        
        Returns:
            Updated Department instance.
        
        Raises:
            DepartmentNotFoundError: If department not found.
        
        Requirements: 11.2
        """
        department = await self.get_department(department_id)
        if not department:
            raise DepartmentNotFoundError(f"Department with ID {department_id} not found")
        
        if name is not None:
            department.name = name
        if manager is not None:
            department.manager = manager
        if contact is not None:
            department.contact = contact
        if description is not None:
            department.description = description
        if sort_order is not None:
            department.sort_order = sort_order
        
        await self.db.flush()
        await self.db.refresh(department)
        return department

    async def move_department(
        self,
        department_id: int,
        new_parent_id: Optional[int],
    ) -> Department:
        """
        Move a department to a new parent (change hierarchy).
        
        Args:
            department_id: Department ID to move.
            new_parent_id: New parent department ID (None for root).
        
        Returns:
            Updated Department instance.
        
        Raises:
            DepartmentNotFoundError: If department or new parent not found.
            DepartmentDepthExceededError: If move would exceed 3 levels.
            CircularReferenceError: If move would create circular reference.
        
        Requirements: 11.4
        """
        department = await self.get_department_with_children(department_id)
        if not department:
            raise DepartmentNotFoundError(f"Department with ID {department_id} not found")
        
        # If moving to same parent, no-op
        if department.parent_id == new_parent_id:
            return department
        
        # Validate new parent
        new_parent_depth = -1  # -1 so that department at root has depth 0
        if new_parent_id is not None:
            new_parent = await self.get_department_with_parent_chain(new_parent_id)
            if not new_parent:
                raise DepartmentNotFoundError(f"New parent department with ID {new_parent_id} not found")
            
            # Check for circular reference
            if await self._would_create_cycle(department_id, new_parent_id):
                raise CircularReferenceError(
                    f"Moving department {department_id} under {new_parent_id} would create a circular reference"
                )
            
            new_parent_depth = await self._calculate_depth(new_parent)
        
        # Calculate the subtree depth of the department being moved
        subtree_depth = await self._get_subtree_depth(department_id)
        
        # Check if the move would exceed max depth
        # New depth of department = new_parent_depth + 1
        # Max depth in subtree after move = new_parent_depth + 1 + subtree_depth
        new_max_depth = new_parent_depth + 1 + subtree_depth
        if new_max_depth > self.MAX_DEPTH:
            raise DepartmentDepthExceededError(self.MAX_DEPTH)
        
        # Perform the move
        department.parent_id = new_parent_id
        await self.db.flush()
        await self.db.refresh(department)
        return department
    
    async def _would_create_cycle(self, department_id: int, new_parent_id: int) -> bool:
        """Check if moving department under new_parent would create a cycle."""
        # A cycle would occur if new_parent is a descendant of department
        current_id = new_parent_id
        while current_id is not None:
            if current_id == department_id:
                return True
            dept = await self.get_department(current_id)
            if dept:
                current_id = dept.parent_id
            else:
                break
        return False
    
    async def _get_subtree_depth(self, department_id: int) -> int:
        """Get the maximum depth of the subtree rooted at department_id."""
        # Get all departments
        stmt = select(Department)
        result = await self.db.execute(stmt)
        all_departments = list(result.scalars().all())
        
        # Build adjacency list
        children_map: dict[int, List[int]] = {}
        for dept in all_departments:
            if dept.parent_id is not None:
                if dept.parent_id not in children_map:
                    children_map[dept.parent_id] = []
                children_map[dept.parent_id].append(dept.id)
        
        # BFS/DFS to find max depth
        def get_max_depth(node_id: int, current_depth: int) -> int:
            children = children_map.get(node_id, [])
            if not children:
                return current_depth
            return max(get_max_depth(child_id, current_depth + 1) for child_id in children)
        
        return get_max_depth(department_id, 0)

    # ==================== Delete Operations ====================
    
    async def delete_department(self, department_id: int) -> bool:
        """
        Delete a department by ID.
        
        Args:
            department_id: Department ID to delete.
        
        Returns:
            True if department was deleted.
        
        Raises:
            DepartmentNotFoundError: If department not found.
            DepartmentHasUsersError: If department has users.
            DepartmentHasChildrenError: If department has child departments.
        
        Requirements: 11.3 - Property 20: Department Deletion Protection
        """
        department = await self.get_department_with_children(department_id)
        if not department:
            raise DepartmentNotFoundError(f"Department with ID {department_id} not found")
        
        # Check for users in department
        user_count = await self._count_users_in_department(department_id)
        if user_count > 0:
            raise DepartmentHasUsersError(department_id, user_count)
        
        # Check for child departments
        children_count = len(department.children) if department.children else 0
        if children_count > 0:
            raise DepartmentHasChildrenError(department_id, children_count)
        
        await self.db.delete(department)
        await self.db.flush()
        return True
    
    async def _count_users_in_department(self, department_id: int) -> int:
        """Count users in a department."""
        stmt = (
            select(func.count(User.id))
            .where(User.department_id == department_id)
        )
        result = await self.db.execute(stmt)
        return result.scalar() or 0

    # ==================== Validation Helpers ====================
    
    async def validate_depth_for_new_child(self, parent_id: Optional[int]) -> bool:
        """
        Validate if a new child can be added under the given parent.
        
        Args:
            parent_id: Parent department ID (None for root).
        
        Returns:
            True if a child can be added, False otherwise.
        """
        if parent_id is None:
            return True
        
        parent = await self.get_department_with_parent_chain(parent_id)
        if not parent:
            return False
        
        parent_depth = await self._calculate_depth(parent)
        return parent_depth < self.MAX_DEPTH
