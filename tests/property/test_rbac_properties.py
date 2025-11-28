"""
Property-based tests for RBAC permission enforcement.

**Feature: intelligent-recommendation-system, Property 21: RBAC Permission Enforcement**
**Validates: Requirements 12.1, 12.2**

A user SHALL have access to a resource if and only if at least one of their
assigned roles has the required permission for that resource.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import pytest
from hypothesis import given, strategies as st, settings, assume


# ==================== Pure Domain Model for Testing ====================
# We test the RBAC logic using a pure in-memory model to avoid database dependencies

@dataclass
class MockPermission:
    """Test permission model."""
    id: int
    resource: str
    action: str
    
    def __hash__(self):
        return hash((self.resource, self.action))
    
    def __eq__(self, other):
        if not isinstance(other, MockPermission):
            return False
        return self.resource == other.resource and self.action == other.action


@dataclass
class MockRole:
    """Test role model."""
    id: int
    name: str
    permissions: Set[MockPermission] = field(default_factory=set)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if not isinstance(other, MockRole):
            return False
        return self.id == other.id


@dataclass
class MockUser:
    """Test user model."""
    id: int
    username: str
    roles: Set[MockRole] = field(default_factory=set)


class RBACModel:
    """
    Pure in-memory RBAC model for property testing.
    
    This model implements the same logic as the actual RBACService
    but without database dependencies, allowing us to test the
    correctness properties in isolation.
    """
    
    def __init__(self):
        self.users: Dict[int, MockUser] = {}
        self.roles: Dict[int, MockRole] = {}
        self.permissions: Dict[int, MockPermission] = {}
        self._next_id = 1
    
    def _get_next_id(self) -> int:
        id = self._next_id
        self._next_id += 1
        return id

    def create_permission(self, resource: str, action: str) -> MockPermission:
        """Create a new permission."""
        perm_id = self._get_next_id()
        perm = MockPermission(id=perm_id, resource=resource, action=action)
        self.permissions[perm_id] = perm
        return perm
    
    def create_role(self, name: str) -> MockRole:
        """Create a new role."""
        role_id = self._get_next_id()
        role = MockRole(id=role_id, name=name)
        self.roles[role_id] = role
        return role
    
    def create_user(self, username: str) -> MockUser:
        """Create a new user."""
        user_id = self._get_next_id()
        user = MockUser(id=user_id, username=username)
        self.users[user_id] = user
        return user
    
    def assign_permission_to_role(self, role: MockRole, permission: MockPermission) -> None:
        """Assign a permission to a role."""
        role.permissions.add(permission)
    
    def assign_role_to_user(self, user: MockUser, role: MockRole) -> None:
        """Assign a role to a user."""
        user.roles.add(role)
    
    def revoke_role_from_user(self, user: MockUser, role: MockRole) -> None:
        """Revoke a role from a user."""
        user.roles.discard(role)
    
    def revoke_permission_from_role(self, role: MockRole, permission: MockPermission) -> None:
        """Revoke a permission from a role."""
        role.permissions.discard(permission)
    
    def get_user_permissions(self, user: MockUser) -> Set[MockPermission]:
        """Get all permissions for a user through their roles."""
        permissions = set()
        for role in user.roles:
            permissions.update(role.permissions)
        return permissions
    
    def check_permission(self, user: MockUser, resource: str, action: str) -> bool:
        """
        Check if a user has a specific permission.
        
        A user has a permission if at least one of their assigned roles
        has that permission granted.
        
        This is the core RBAC enforcement logic being tested.
        """
        for role in user.roles:
            for perm in role.permissions:
                if perm.resource == resource and perm.action == action:
                    return True
        return False
    
    def user_has_role_with_permission(
        self, user: MockUser, resource: str, action: str
    ) -> bool:
        """
        Alternative implementation to verify check_permission.
        
        Returns True if user has at least one role that has the permission.
        """
        target_perm = MockPermission(id=0, resource=resource, action=action)
        for role in user.roles:
            if target_perm in role.permissions:
                return True
        return False


# ==================== Hypothesis Strategies ====================

# Strategy for generating resource names
resource_strategy = st.sampled_from([
    "users", "roles", "permissions", "news", "topics", 
    "departments", "system", "reports", "settings"
])

# Strategy for generating action names
action_strategy = st.sampled_from([
    "create", "read", "update", "delete", "view", 
    "manage", "export", "import", "share"
])

# Strategy for generating permission tuples
permission_tuple_strategy = st.tuples(resource_strategy, action_strategy)

# Strategy for generating role names
role_name_strategy = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz_",
    min_size=1,
    max_size=20
).map(lambda x: f"role_{x}")

# Strategy for generating username
username_strategy = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789_",
    min_size=1,
    max_size=20
).map(lambda x: f"user_{x}")


# ==================== Property Tests ====================

class TestRBACPermissionEnforcement:
    """
    Property tests for RBAC permission enforcement.
    
    **Feature: intelligent-recommendation-system, Property 21: RBAC Permission Enforcement**
    **Validates: Requirements 12.1, 12.2**
    
    Core property: A user SHALL have access to a resource if and only if
    at least one of their assigned roles has the required permission.
    """

    @settings(max_examples=100, deadline=None)
    @given(
        role_name=role_name_strategy,
        username=username_strategy,
        resource=resource_strategy,
        action=action_strategy
    )
    def test_user_with_role_permission_has_access(
        self, role_name: str, username: str, resource: str, action: str
    ):
        """
        **Feature: intelligent-recommendation-system, Property 21: RBAC Permission Enforcement**
        **Validates: Requirements 12.1, 12.2**
        
        For any user with a role that has a permission,
        check_permission SHALL return True for that resource:action.
        """
        # Arrange
        model = RBACModel()
        permission = model.create_permission(resource, action)
        role = model.create_role(role_name)
        user = model.create_user(username)
        
        model.assign_permission_to_role(role, permission)
        model.assign_role_to_user(user, role)
        
        # Act
        has_permission = model.check_permission(user, resource, action)
        
        # Assert
        assert has_permission is True

    @settings(max_examples=100, deadline=None)
    @given(
        username=username_strategy,
        resource=resource_strategy,
        action=action_strategy
    )
    def test_user_without_roles_has_no_access(
        self, username: str, resource: str, action: str
    ):
        """
        **Feature: intelligent-recommendation-system, Property 21: RBAC Permission Enforcement**
        **Validates: Requirements 12.1, 12.2**
        
        For any user without any roles,
        check_permission SHALL return False for any resource:action.
        """
        # Arrange
        model = RBACModel()
        user = model.create_user(username)
        
        # Act
        has_permission = model.check_permission(user, resource, action)
        
        # Assert
        assert has_permission is False


    @settings(max_examples=100, deadline=None)
    @given(
        role_name=role_name_strategy,
        username=username_strategy,
        granted_resource=resource_strategy,
        granted_action=action_strategy,
        requested_resource=resource_strategy,
        requested_action=action_strategy
    )
    def test_permission_check_is_specific(
        self,
        role_name: str,
        username: str,
        granted_resource: str,
        granted_action: str,
        requested_resource: str,
        requested_action: str
    ):
        """
        **Feature: intelligent-recommendation-system, Property 21: RBAC Permission Enforcement**
        **Validates: Requirements 12.1, 12.2**
        
        For any user with a specific permission granted,
        check_permission SHALL return True only for that exact resource:action,
        and False for different resource:action combinations.
        """
        # Arrange
        model = RBACModel()
        permission = model.create_permission(granted_resource, granted_action)
        role = model.create_role(role_name)
        user = model.create_user(username)
        
        model.assign_permission_to_role(role, permission)
        model.assign_role_to_user(user, role)
        
        # Act
        has_permission = model.check_permission(user, requested_resource, requested_action)
        
        # Assert
        expected = (granted_resource == requested_resource and granted_action == requested_action)
        assert has_permission == expected

    @settings(max_examples=100, deadline=None)
    @given(
        role_names=st.lists(role_name_strategy, min_size=1, max_size=5, unique=True),
        username=username_strategy,
        permissions=st.lists(permission_tuple_strategy, min_size=1, max_size=10, unique=True)
    )
    def test_multiple_roles_union_permissions(
        self,
        role_names: List[str],
        username: str,
        permissions: List[Tuple[str, str]]
    ):
        """
        **Feature: intelligent-recommendation-system, Property 21: RBAC Permission Enforcement**
        **Validates: Requirements 12.1, 12.2**
        
        For any user with multiple roles, the effective permissions
        SHALL be the union of all permissions from all assigned roles.
        """
        # Arrange
        model = RBACModel()
        user = model.create_user(username)
        
        # Distribute permissions across roles
        all_granted_perms = set()
        for i, role_name in enumerate(role_names):
            role = model.create_role(role_name)
            model.assign_role_to_user(user, role)
            
            # Assign some permissions to this role
            for j, (resource, action) in enumerate(permissions):
                if (i + j) % len(role_names) == i:  # Distribute permissions
                    perm = model.create_permission(resource, action)
                    model.assign_permission_to_role(role, perm)
                    all_granted_perms.add((resource, action))
        
        # Act & Assert - Check all granted permissions
        for resource, action in all_granted_perms:
            assert model.check_permission(user, resource, action) is True

    @settings(max_examples=100, deadline=None)
    @given(
        role_name=role_name_strategy,
        username=username_strategy,
        resource=resource_strategy,
        action=action_strategy
    )
    def test_revoke_role_removes_access(
        self,
        role_name: str,
        username: str,
        resource: str,
        action: str
    ):
        """
        **Feature: intelligent-recommendation-system, Property 21: RBAC Permission Enforcement**
        **Validates: Requirements 12.1, 12.2**
        
        For any user, revoking a role SHALL remove access to permissions
        that were only granted through that role.
        """
        # Arrange
        model = RBACModel()
        permission = model.create_permission(resource, action)
        role = model.create_role(role_name)
        user = model.create_user(username)
        
        model.assign_permission_to_role(role, permission)
        model.assign_role_to_user(user, role)
        
        # Verify initial access
        assert model.check_permission(user, resource, action) is True
        
        # Act - Revoke the role
        model.revoke_role_from_user(user, role)
        
        # Assert - Access should be removed
        assert model.check_permission(user, resource, action) is False


    @settings(max_examples=100, deadline=None)
    @given(
        role_names=st.lists(role_name_strategy, min_size=2, max_size=3, unique=True),
        username=username_strategy,
        resource=resource_strategy,
        action=action_strategy
    )
    def test_overlapping_permissions_still_work_after_partial_revoke(
        self,
        role_names: List[str],
        username: str,
        resource: str,
        action: str
    ):
        """
        **Feature: intelligent-recommendation-system, Property 21: RBAC Permission Enforcement**
        **Validates: Requirements 12.1, 12.2**
        
        For any user with the same permission granted through multiple roles,
        revoking one role SHALL NOT remove access if another role still grants it.
        """
        assume(len(role_names) >= 2)
        
        # Arrange
        model = RBACModel()
        user = model.create_user(username)
        roles = []
        
        # Create multiple roles with the same permission
        for role_name in role_names:
            role = model.create_role(role_name)
            perm = model.create_permission(resource, action)
            model.assign_permission_to_role(role, perm)
            model.assign_role_to_user(user, role)
            roles.append(role)
        
        # Verify initial access
        assert model.check_permission(user, resource, action) is True
        
        # Act - Revoke one role
        model.revoke_role_from_user(user, roles[0])
        
        # Assert - Access should still exist through other roles
        assert model.check_permission(user, resource, action) is True

    @settings(max_examples=100, deadline=None)
    @given(
        role_name=role_name_strategy,
        username=username_strategy,
        resource=resource_strategy,
        action=action_strategy
    )
    def test_revoke_permission_from_role_removes_access(
        self,
        role_name: str,
        username: str,
        resource: str,
        action: str
    ):
        """
        **Feature: intelligent-recommendation-system, Property 21: RBAC Permission Enforcement**
        **Validates: Requirements 12.1, 12.2**
        
        For any role, revoking a permission SHALL remove access for all users
        who only have that permission through that role.
        """
        # Arrange
        model = RBACModel()
        permission = model.create_permission(resource, action)
        role = model.create_role(role_name)
        user = model.create_user(username)
        
        model.assign_permission_to_role(role, permission)
        model.assign_role_to_user(user, role)
        
        # Verify initial access
        assert model.check_permission(user, resource, action) is True
        
        # Act - Revoke the permission from the role
        model.revoke_permission_from_role(role, permission)
        
        # Assert - Access should be removed
        assert model.check_permission(user, resource, action) is False

    @settings(max_examples=100, deadline=None)
    @given(
        role_name=role_name_strategy,
        username=username_strategy,
        resource=resource_strategy,
        action=action_strategy
    )
    def test_check_permission_matches_alternative_implementation(
        self,
        role_name: str,
        username: str,
        resource: str,
        action: str
    ):
        """
        **Feature: intelligent-recommendation-system, Property 21: RBAC Permission Enforcement**
        **Validates: Requirements 12.1, 12.2**
        
        The check_permission implementation SHALL produce the same result
        as the alternative user_has_role_with_permission implementation.
        """
        # Arrange
        model = RBACModel()
        permission = model.create_permission(resource, action)
        role = model.create_role(role_name)
        user = model.create_user(username)
        
        model.assign_permission_to_role(role, permission)
        model.assign_role_to_user(user, role)
        
        # Act
        result1 = model.check_permission(user, resource, action)
        result2 = model.user_has_role_with_permission(user, resource, action)
        
        # Assert
        assert result1 == result2

    @settings(max_examples=100, deadline=None)
    @given(
        role_name=role_name_strategy,
        username=username_strategy,
        permissions=st.lists(permission_tuple_strategy, min_size=1, max_size=10, unique=True)
    )
    def test_get_user_permissions_returns_all_granted(
        self,
        role_name: str,
        username: str,
        permissions: List[Tuple[str, str]]
    ):
        """
        **Feature: intelligent-recommendation-system, Property 21: RBAC Permission Enforcement**
        **Validates: Requirements 12.1, 12.2**
        
        For any user, get_user_permissions SHALL return all permissions
        granted through their roles.
        """
        # Arrange
        model = RBACModel()
        role = model.create_role(role_name)
        user = model.create_user(username)
        model.assign_role_to_user(user, role)
        
        created_perms = []
        for resource, action in permissions:
            perm = model.create_permission(resource, action)
            model.assign_permission_to_role(role, perm)
            created_perms.append(perm)
        
        # Act
        user_perms = model.get_user_permissions(user)
        
        # Assert - All created permissions should be in user's permissions
        for perm in created_perms:
            assert perm in user_perms
        
        # Assert - User permissions should not exceed what was granted
        assert len(user_perms) == len(created_perms)


    @settings(max_examples=100, deadline=None)
    @given(
        role_name=role_name_strategy,
        username=username_strategy,
        resource=resource_strategy,
        action=action_strategy
    )
    def test_permission_check_is_idempotent(
        self,
        role_name: str,
        username: str,
        resource: str,
        action: str
    ):
        """
        **Feature: intelligent-recommendation-system, Property 21: RBAC Permission Enforcement**
        **Validates: Requirements 12.1, 12.2**
        
        For any user and permission, calling check_permission multiple times
        SHALL return the same result (idempotence).
        """
        # Arrange
        model = RBACModel()
        permission = model.create_permission(resource, action)
        role = model.create_role(role_name)
        user = model.create_user(username)
        
        model.assign_permission_to_role(role, permission)
        model.assign_role_to_user(user, role)
        
        # Act - Call multiple times
        result1 = model.check_permission(user, resource, action)
        result2 = model.check_permission(user, resource, action)
        result3 = model.check_permission(user, resource, action)
        
        # Assert - All results should be the same
        assert result1 == result2 == result3 == True

    @settings(max_examples=100, deadline=None)
    @given(
        role_name=role_name_strategy,
        username=username_strategy,
        resource=resource_strategy,
        action=action_strategy
    )
    def test_double_assignment_is_idempotent(
        self,
        role_name: str,
        username: str,
        resource: str,
        action: str
    ):
        """
        **Feature: intelligent-recommendation-system, Property 21: RBAC Permission Enforcement**
        **Validates: Requirements 12.1, 12.2**
        
        Assigning the same role to a user twice SHALL have the same effect
        as assigning it once (idempotence).
        """
        # Arrange
        model = RBACModel()
        permission = model.create_permission(resource, action)
        role = model.create_role(role_name)
        user = model.create_user(username)
        
        model.assign_permission_to_role(role, permission)
        
        # Act - Assign role twice
        model.assign_role_to_user(user, role)
        model.assign_role_to_user(user, role)
        
        # Assert - User should have exactly one instance of the role
        assert len(user.roles) == 1
        assert model.check_permission(user, resource, action) is True


class TestRBACEdgeCases:
    """
    Edge case tests for RBAC permission enforcement.
    
    **Feature: intelligent-recommendation-system, Property 21: RBAC Permission Enforcement**
    **Validates: Requirements 12.1, 12.2**
    """

    def test_empty_role_grants_no_permissions(self):
        """
        **Feature: intelligent-recommendation-system, Property 21: RBAC Permission Enforcement**
        **Validates: Requirements 12.1, 12.2**
        
        A role with no permissions SHALL grant no access.
        """
        # Arrange
        model = RBACModel()
        role = model.create_role("empty_role")
        user = model.create_user("test_user")
        model.assign_role_to_user(user, role)
        
        # Act & Assert
        assert model.check_permission(user, "users", "read") is False
        assert model.check_permission(user, "news", "write") is False
        assert len(model.get_user_permissions(user)) == 0

    def test_user_with_no_roles_has_empty_permissions(self):
        """
        **Feature: intelligent-recommendation-system, Property 21: RBAC Permission Enforcement**
        **Validates: Requirements 12.1, 12.2**
        
        A user with no roles SHALL have an empty permission set.
        """
        # Arrange
        model = RBACModel()
        user = model.create_user("test_user")
        
        # Act
        permissions = model.get_user_permissions(user)
        
        # Assert
        assert len(permissions) == 0

    def test_permission_check_with_similar_but_different_values(self):
        """
        **Feature: intelligent-recommendation-system, Property 21: RBAC Permission Enforcement**
        **Validates: Requirements 12.1, 12.2**
        
        Permission check SHALL be exact - similar but different values
        SHALL NOT match.
        """
        # Arrange
        model = RBACModel()
        permission = model.create_permission("users", "read")
        role = model.create_role("test_role")
        user = model.create_user("test_user")
        
        model.assign_permission_to_role(role, permission)
        model.assign_role_to_user(user, role)
        
        # Act & Assert - Exact match works
        assert model.check_permission(user, "users", "read") is True
        
        # Act & Assert - Similar but different values don't match
        assert model.check_permission(user, "users", "write") is False
        assert model.check_permission(user, "user", "read") is False
        assert model.check_permission(user, "Users", "read") is False
        assert model.check_permission(user, "users", "Read") is False
