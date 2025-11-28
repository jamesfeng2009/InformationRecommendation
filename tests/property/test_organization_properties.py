"""
Property-based tests for organization tree structure.

**Feature: intelligent-recommendation-system, Property 19: Organization Tree Depth Constraint**
**Validates: Requirements 11.1**
"""
import pytest
from hypothesis import given, strategies as st, settings, assume

from app.models.user import Department


# Strategy for generating valid department names
department_name_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P")),
    min_size=1,
    max_size=50
).filter(lambda x: x.strip() != "")


# Strategy for generating a department at a specific level
def department_at_level(level: int) -> Department:
    """Create a department at a specific level in the tree."""
    if level == 0:
        return Department(id=1, name="Root", parent_id=None)
    
    parent = department_at_level(level - 1)
    dept = Department(id=level + 1, name=f"Level{level}", parent_id=parent.id)
    dept.parent = parent
    return dept


class TestOrganizationTreeDepth:
    """
    Property tests for organization tree depth constraint.
    
    **Feature: intelligent-recommendation-system, Property 19: Organization Tree Depth Constraint**
    **Validates: Requirements 11.1**
    """

    @settings(max_examples=100)
    @given(depth=st.integers(min_value=0, max_value=2))
    def test_valid_tree_depth_within_limit(self, depth: int):
        """
        **Feature: intelligent-recommendation-system, Property 19: Organization Tree Depth Constraint**
        **Validates: Requirements 11.1**
        
        For any organization tree with depth 0, 1, or 2 (up to 3 levels),
        the tree depth SHALL not exceed 3 levels (root -> level 1 -> level 2).
        """
        # Arrange: Create a department at the given depth
        dept = department_at_level(depth)
        
        # Act: Calculate the depth
        calculated_depth = dept.get_depth()
        
        # Assert: Depth should match and be within limit
        assert calculated_depth == depth
        assert calculated_depth <= 2  # Max 3 levels means max depth of 2 (0-indexed)

    @settings(max_examples=100)
    @given(depth=st.integers(min_value=3, max_value=10))
    def test_tree_depth_exceeds_limit(self, depth: int):
        """
        **Feature: intelligent-recommendation-system, Property 19: Organization Tree Depth Constraint**
        **Validates: Requirements 11.1**
        
        For any organization tree with depth > 2 (more than 3 levels),
        the tree depth SHALL exceed the allowed limit.
        """
        # Arrange: Create a department at an invalid depth
        dept = department_at_level(depth)
        
        # Act: Calculate the depth
        calculated_depth = dept.get_depth()
        
        # Assert: Depth should exceed the limit
        assert calculated_depth == depth
        assert calculated_depth > 2  # Exceeds the 3-level limit

    @settings(max_examples=100)
    @given(
        names=st.lists(
            department_name_strategy,
            min_size=1,
            max_size=3
        )
    )
    def test_tree_chain_depth_matches_chain_length(self, names: list[str]):
        """
        **Feature: intelligent-recommendation-system, Property 19: Organization Tree Depth Constraint**
        **Validates: Requirements 11.1**
        
        For any chain of departments, the depth of the last department
        SHALL equal the number of ancestors (chain length - 1).
        """
        # Arrange: Build a chain of departments
        departments = []
        for i, name in enumerate(names):
            parent = departments[-1] if departments else None
            dept = Department(
                id=i + 1,
                name=name,
                parent_id=parent.id if parent else None
            )
            dept.parent = parent
            departments.append(dept)
        
        # Act: Get the depth of the last department
        last_dept = departments[-1]
        calculated_depth = last_dept.get_depth()
        
        # Assert: Depth should equal chain length - 1
        expected_depth = len(names) - 1
        assert calculated_depth == expected_depth

    def test_root_department_has_zero_depth(self):
        """
        **Feature: intelligent-recommendation-system, Property 19: Organization Tree Depth Constraint**
        **Validates: Requirements 11.1**
        
        A root department (no parent) SHALL have depth 0.
        """
        # Arrange
        root = Department(id=1, name="Root", parent_id=None)
        
        # Act
        depth = root.get_depth()
        
        # Assert
        assert depth == 0

    def test_max_valid_depth_is_two(self):
        """
        **Feature: intelligent-recommendation-system, Property 19: Organization Tree Depth Constraint**
        **Validates: Requirements 11.1**
        
        The maximum valid depth for a 3-level tree is 2 (root=0, level1=1, level2=2).
        """
        # Arrange: Create a 3-level tree
        root = Department(id=1, name="Root", parent_id=None)
        level1 = Department(id=2, name="Level1", parent_id=1)
        level1.parent = root
        level2 = Department(id=3, name="Level2", parent_id=2)
        level2.parent = level1
        
        # Act
        root_depth = root.get_depth()
        level1_depth = level1.get_depth()
        level2_depth = level2.get_depth()
        
        # Assert
        assert root_depth == 0
        assert level1_depth == 1
        assert level2_depth == 2
        # All are within the 3-level limit
        assert all(d <= 2 for d in [root_depth, level1_depth, level2_depth])


def is_valid_tree_depth(department: Department, max_depth: int = 2) -> bool:
    """
    Helper function to validate if a department's depth is within the allowed limit.
    
    Args:
        department: The department to check
        max_depth: Maximum allowed depth (default 2 for 3 levels)
    
    Returns:
        True if the department's depth is within the limit
    """
    return department.get_depth() <= max_depth
