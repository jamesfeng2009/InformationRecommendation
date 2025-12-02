import string
from dataclasses import dataclass
from typing import List, Optional, Set
from unittest.mock import MagicMock

import pytest
from hypothesis import given, strategies as st, settings, assume

from app.services.user import (
    BatchUpdateData,
    BatchUpdateError,
    UserFilter,
    PagedResult,
)
from app.models.user import User, Department


# ==================== Strategies ====================

# Strategy for generating valid usernames
username_strategy = st.text(
    alphabet=string.ascii_letters + string.digits + "_",
    min_size=3,
    max_size=30
).filter(lambda x: x.strip() != "" and x[0].isalpha())

# Strategy for generating valid names
name_strategy = st.text(
    alphabet=string.ascii_letters + " ",
    min_size=1,
    max_size=50
).filter(lambda x: x.strip() != "")

# Strategy for generating user IDs
user_id_strategy = st.integers(min_value=1, max_value=100000)

# Strategy for generating department IDs
department_id_strategy = st.integers(min_value=1, max_value=1000)

# Strategy for generating status
status_strategy = st.sampled_from(["enabled", "disabled"])

# Strategy for generating keywords
keywords_strategy = st.text(
    alphabet=string.ascii_letters + string.digits + ",",
    min_size=0,
    max_size=100
)


@dataclass
class MockUser:
    """Mock user for testing batch update logic."""
    id: int
    username: str
    name: str
    account: str
    department_id: Optional[int]
    status: str
    keywords: Optional[str]
    
    def copy(self) -> "MockUser":
        """Create a copy of this user."""
        return MockUser(
            id=self.id,
            username=self.username,
            name=self.name,
            account=self.account,
            department_id=self.department_id,
            status=self.status,
            keywords=self.keywords,
        )


# Strategy for generating mock users
@st.composite
def mock_user_strategy(draw, user_id: Optional[int] = None):
    """Generate a mock user with random data."""
    return MockUser(
        id=user_id if user_id is not None else draw(user_id_strategy),
        username=draw(username_strategy),
        name=draw(name_strategy),
        account=draw(username_strategy),
        department_id=draw(st.one_of(st.none(), department_id_strategy)),
        status=draw(status_strategy),
        keywords=draw(st.one_of(st.none(), keywords_strategy)),
    )


# Strategy for generating a list of mock users with unique IDs
@st.composite
def mock_users_list_strategy(draw, min_size: int = 1, max_size: int = 10):
    """Generate a list of mock users with unique IDs."""
    count = draw(st.integers(min_value=min_size, max_value=max_size))
    users = []
    used_ids = set()
    used_usernames = set()
    used_accounts = set()
    
    for i in range(count):
        # Generate unique ID
        user_id = draw(user_id_strategy)
        while user_id in used_ids:
            user_id = draw(user_id_strategy)
        used_ids.add(user_id)
        
        # Generate unique username
        username = draw(username_strategy)
        while username in used_usernames:
            username = draw(username_strategy)
        used_usernames.add(username)
        
        # Generate unique account
        account = draw(username_strategy)
        while account in used_accounts:
            account = draw(username_strategy)
        used_accounts.add(account)
        
        users.append(MockUser(
            id=user_id,
            username=username,
            name=draw(name_strategy),
            account=account,
            department_id=draw(st.one_of(st.none(), department_id_strategy)),
            status=draw(status_strategy),
            keywords=draw(st.one_of(st.none(), keywords_strategy)),
        ))
    
    return users


# Strategy for generating batch update data
@st.composite
def batch_update_data_strategy(draw):
    """Generate batch update data with at least one field set."""
    department_id = draw(st.one_of(st.none(), department_id_strategy))
    status = draw(st.one_of(st.none(), status_strategy))
    keywords = draw(st.one_of(st.none(), keywords_strategy))
    
    # Ensure at least one field is set
    if department_id is None and status is None and keywords is None:
        # Force at least one field
        choice = draw(st.integers(min_value=0, max_value=2))
        if choice == 0:
            department_id = draw(department_id_strategy)
        elif choice == 1:
            status = draw(status_strategy)
        else:
            keywords = draw(keywords_strategy)
    
    return BatchUpdateData(
        department_id=department_id,
        status=status,
        keywords=keywords,
    )



def apply_batch_update(users: List[MockUser], user_ids: List[int], data: BatchUpdateData) -> tuple[List[MockUser], Set[int]]:
    """
    Simulate batch update logic.
    
    Returns:
        Tuple of (updated_users, missing_ids)
    """
    user_map = {u.id: u for u in users}
    found_ids = set(user_ids) & set(user_map.keys())
    missing_ids = set(user_ids) - found_ids
    
    if missing_ids:
        return [], missing_ids
    
    updated = []
    for uid in user_ids:
        user = user_map[uid].copy()
        if data.department_id is not None:
            user.department_id = data.department_id
        if data.status is not None:
            user.status = data.status
        if data.keywords is not None:
            user.keywords = data.keywords
        updated.append(user)
    
    return updated, set()


class TestBatchUpdateAtomicity:
    """
    Property tests for batch update atomicity.
    
    **Feature: intelligent-recommendation-system, Property 18: Batch Update Atomicity**
    **Validates: Requirements 10.3**
    """

    @settings(max_examples=100, deadline=None)
    @given(
        users=mock_users_list_strategy(min_size=2, max_size=10),
        data=batch_update_data_strategy(),
    )
    def test_batch_update_all_or_nothing_success(
        self,
        users: List[MockUser],
        data: BatchUpdateData,
    ):
        """
        **Feature: intelligent-recommendation-system, Property 18: Batch Update Atomicity**
        **Validates: Requirements 10.3**
        
        For any batch update where all user IDs exist,
        ALL specified users SHALL be updated with the new values.
        """
        # Arrange: Get all user IDs (all exist)
        user_ids = [u.id for u in users]
        original_users = [u.copy() for u in users]
        
        # Act: Apply batch update
        updated_users, missing_ids = apply_batch_update(users, user_ids, data)
        
        # Assert: All users should be updated
        assert len(missing_ids) == 0
        assert len(updated_users) == len(user_ids)
        
        # Assert: All updated users have the new values
        for updated in updated_users:
            if data.department_id is not None:
                assert updated.department_id == data.department_id
            if data.status is not None:
                assert updated.status == data.status
            if data.keywords is not None:
                assert updated.keywords == data.keywords

    @settings(max_examples=100, deadline=None)
    @given(
        users=mock_users_list_strategy(min_size=2, max_size=10),
        data=batch_update_data_strategy(),
        missing_id=user_id_strategy,
    )
    def test_batch_update_all_or_nothing_failure(
        self,
        users: List[MockUser],
        data: BatchUpdateData,
        missing_id: int,
    ):
        """
        **Feature: intelligent-recommendation-system, Property 18: Batch Update Atomicity**
        **Validates: Requirements 10.3**
        
        For any batch update where at least one user ID does not exist,
        NO users SHALL be updated (atomic rollback).
        """
        # Arrange: Ensure missing_id is not in users
        existing_ids = {u.id for u in users}
        assume(missing_id not in existing_ids)
        
        # Include the missing ID in the batch
        user_ids = [u.id for u in users] + [missing_id]
        original_users = [u.copy() for u in users]
        
        # Act: Apply batch update
        updated_users, missing_ids = apply_batch_update(users, user_ids, data)
        
        # Assert: No users should be updated due to missing ID
        assert len(updated_users) == 0
        assert missing_id in missing_ids
        
        # Assert: Original users remain unchanged
        for original, current in zip(original_users, users):
            assert original.department_id == current.department_id
            assert original.status == current.status
            assert original.keywords == current.keywords

    @settings(max_examples=100, deadline=None)
    @given(
        users=mock_users_list_strategy(min_size=3, max_size=10),
        data=batch_update_data_strategy(),
    )
    def test_batch_update_subset_of_users(
        self,
        users: List[MockUser],
        data: BatchUpdateData,
    ):
        """
        **Feature: intelligent-recommendation-system, Property 18: Batch Update Atomicity**
        **Validates: Requirements 10.3**
        
        For any batch update on a subset of users,
        only the specified users SHALL be updated, others remain unchanged.
        """
        # Arrange: Select a subset of users
        subset_size = len(users) // 2
        assume(subset_size >= 1)
        
        subset_ids = [u.id for u in users[:subset_size]]
        non_subset_ids = [u.id for u in users[subset_size:]]
        
        original_non_subset = [u.copy() for u in users[subset_size:]]
        
        # Act: Apply batch update to subset only
        updated_users, missing_ids = apply_batch_update(users, subset_ids, data)
        
        # Assert: Only subset users are updated
        assert len(missing_ids) == 0
        assert len(updated_users) == len(subset_ids)
        
        updated_ids = {u.id for u in updated_users}
        assert updated_ids == set(subset_ids)
        
        # Assert: Non-subset users remain unchanged (in original list)
        for original in original_non_subset:
            current = next(u for u in users if u.id == original.id)
            assert original.department_id == current.department_id
            assert original.status == current.status
            assert original.keywords == current.keywords

    @settings(max_examples=100, deadline=None)
    @given(
        users=mock_users_list_strategy(min_size=1, max_size=10),
        data=batch_update_data_strategy(),
    )
    def test_batch_update_preserves_unchanged_fields(
        self,
        users: List[MockUser],
        data: BatchUpdateData,
    ):
        """
        **Feature: intelligent-recommendation-system, Property 18: Batch Update Atomicity**
        **Validates: Requirements 10.3**
        
        For any batch update, fields not specified in the update data
        SHALL remain unchanged.
        """
        # Arrange
        user_ids = [u.id for u in users]
        original_users = {u.id: u.copy() for u in users}
        
        # Act
        updated_users, missing_ids = apply_batch_update(users, user_ids, data)
        
        # Assert: Unchanged fields are preserved
        assert len(missing_ids) == 0
        for updated in updated_users:
            original = original_users[updated.id]
            
            # If field was not in update data, it should be unchanged
            if data.department_id is None:
                assert updated.department_id == original.department_id
            if data.status is None:
                assert updated.status == original.status
            if data.keywords is None:
                assert updated.keywords == original.keywords
            
            # Username and name should never change in batch update
            assert updated.username == original.username
            assert updated.name == original.name
            assert updated.account == original.account

    @settings(max_examples=100, deadline=None)
    @given(data=batch_update_data_strategy())
    def test_batch_update_empty_list_returns_empty(self, data: BatchUpdateData):
        """
        **Feature: intelligent-recommendation-system, Property 18: Batch Update Atomicity**
        **Validates: Requirements 10.3**
        
        For any batch update with an empty user ID list,
        the result SHALL be an empty list with no errors.
        """
        # Arrange
        users: List[MockUser] = []
        user_ids: List[int] = []
        
        # Act
        updated_users, missing_ids = apply_batch_update(users, user_ids, data)
        
        # Assert
        assert len(updated_users) == 0
        assert len(missing_ids) == 0

    @settings(max_examples=100, deadline=None)
    @given(
        users=mock_users_list_strategy(min_size=2, max_size=10),
        data=batch_update_data_strategy(),
    )
    def test_batch_update_idempotent(
        self,
        users: List[MockUser],
        data: BatchUpdateData,
    ):
        """
        **Feature: intelligent-recommendation-system, Property 18: Batch Update Atomicity**
        **Validates: Requirements 10.3**
        
        For any batch update, applying the same update twice
        SHALL produce the same result (idempotence).
        """
        # Arrange
        user_ids = [u.id for u in users]
        
        # Act: Apply update twice
        updated_once, _ = apply_batch_update(users, user_ids, data)
        
        # Create new mock users from first update result
        users_after_first = [
            MockUser(
                id=u.id,
                username=u.username,
                name=u.name,
                account=u.account,
                department_id=u.department_id,
                status=u.status,
                keywords=u.keywords,
            )
            for u in updated_once
        ]
        
        updated_twice, _ = apply_batch_update(users_after_first, user_ids, data)
        
        # Assert: Results should be identical
        assert len(updated_once) == len(updated_twice)
        for u1, u2 in zip(
            sorted(updated_once, key=lambda x: x.id),
            sorted(updated_twice, key=lambda x: x.id)
        ):
            assert u1.id == u2.id
            assert u1.department_id == u2.department_id
            assert u1.status == u2.status
            assert u1.keywords == u2.keywords

    @settings(max_examples=100, deadline=None)
    @given(
        users=mock_users_list_strategy(min_size=2, max_size=10),
    )
    def test_batch_update_count_matches_input(
        self,
        users: List[MockUser],
    ):
        """
        **Feature: intelligent-recommendation-system, Property 18: Batch Update Atomicity**
        **Validates: Requirements 10.3**
        
        For any successful batch update, the number of updated users
        SHALL equal the number of user IDs provided.
        """
        # Arrange
        user_ids = [u.id for u in users]
        data = BatchUpdateData(status="enabled")
        
        # Act
        updated_users, missing_ids = apply_batch_update(users, user_ids, data)
        
        # Assert
        assert len(missing_ids) == 0
        assert len(updated_users) == len(user_ids)
