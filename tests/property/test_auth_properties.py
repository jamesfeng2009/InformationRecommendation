"""
Property-based tests for authentication.

**Feature: intelligent-recommendation-system, Property 17: Authentication Correctness**
**Validates: Requirements 10.1**

**Feature: intelligent-recommendation-system, Property 24: Password Policy Compliance**
**Validates: Requirements 21.1**
"""
import string

import pytest
from hypothesis import given, strategies as st, settings, assume

from app.core.security import (
    hash_password,
    verify_password,
    validate_password_policy,
    is_password_valid,
    create_access_token,
    create_refresh_token,
    create_token_pair,
    decode_token,
    verify_token,
    TokenExpiredError,
    TokenInvalidError,
)


# Strategies for generating test data
valid_password_strategy = st.text(
    alphabet=string.ascii_letters + string.digits + "!@#$%^&*()_+-=",
    min_size=8,
    max_size=50
).filter(
    lambda p: (
        any(c.isupper() for c in p) and
        any(c.islower() for c in p) and
        any(c.isdigit() for c in p) and
        any(c in "!@#$%^&*()_+-=" for c in p)
    )
)

invalid_password_too_short = st.text(
    alphabet=string.ascii_letters + string.digits + "!@#$%^&*",
    min_size=1,
    max_size=7
)

username_strategy = st.text(
    alphabet=string.ascii_letters + string.digits + "_",
    min_size=1,
    max_size=50
).filter(lambda x: x.strip() != "")

user_id_strategy = st.integers(min_value=1, max_value=1000000)


class TestAuthenticationCorrectness:
    """
    Property tests for authentication correctness.
    
    **Feature: intelligent-recommendation-system, Property 17: Authentication Correctness**
    **Validates: Requirements 10.1**
    """

    @settings(max_examples=100, deadline=None)
    @given(password=valid_password_strategy)
    def test_password_hash_verify_roundtrip(self, password: str):
        """
        **Feature: intelligent-recommendation-system, Property 17: Authentication Correctness**
        **Validates: Requirements 10.1**
        
        For any valid password, hashing and then verifying with the same password
        SHALL return True.
        """
        # Arrange & Act
        hashed = hash_password(password)
        result = verify_password(password, hashed)
        
        # Assert
        assert result is True

    @settings(max_examples=100, deadline=None)
    @given(
        password=valid_password_strategy,
        wrong_password=valid_password_strategy
    )
    def test_wrong_password_fails_verification(self, password: str, wrong_password: str):
        """
        **Feature: intelligent-recommendation-system, Property 17: Authentication Correctness**
        **Validates: Requirements 10.1**
        
        For any valid password, verifying with a different password
        SHALL return False.
        """
        # Skip if passwords happen to be the same
        assume(password != wrong_password)
        
        # Arrange
        hashed = hash_password(password)
        
        # Act
        result = verify_password(wrong_password, hashed)
        
        # Assert
        assert result is False

    @settings(max_examples=100)
    @given(
        user_id=user_id_strategy,
        username=username_strategy
    )
    def test_access_token_roundtrip(self, user_id: int, username: str):
        """
        **Feature: intelligent-recommendation-system, Property 17: Authentication Correctness**
        **Validates: Requirements 10.1**
        
        For any valid user_id and username, creating an access token and decoding it
        SHALL return the same user_id and username.
        """
        # Arrange & Act
        token = create_access_token(user_id, username)
        decoded = decode_token(token)
        
        # Assert
        assert decoded.user_id == user_id
        assert decoded.username == username
        assert decoded.token_type == "access"

    @settings(max_examples=100)
    @given(
        user_id=user_id_strategy,
        username=username_strategy,
        jti=st.text(min_size=1, max_size=36).filter(lambda x: x.strip() != "")
    )
    def test_refresh_token_roundtrip(self, user_id: int, username: str, jti: str):
        """
        **Feature: intelligent-recommendation-system, Property 17: Authentication Correctness**
        **Validates: Requirements 10.1**
        
        For any valid user_id, username, and jti, creating a refresh token and decoding it
        SHALL return the same values.
        """
        # Arrange & Act
        token = create_refresh_token(user_id, username, jti=jti)
        decoded = decode_token(token)
        
        # Assert
        assert decoded.user_id == user_id
        assert decoded.username == username
        assert decoded.token_type == "refresh"
        assert decoded.jti == jti

    @settings(max_examples=100)
    @given(
        user_id=user_id_strategy,
        username=username_strategy
    )
    def test_token_pair_creates_both_tokens(self, user_id: int, username: str):
        """
        **Feature: intelligent-recommendation-system, Property 17: Authentication Correctness**
        **Validates: Requirements 10.1**
        
        For any valid user_id and username, creating a token pair
        SHALL produce valid access and refresh tokens with correct types.
        """
        # Arrange & Act
        tokens = create_token_pair(user_id, username)
        access_data = decode_token(tokens.access_token)
        refresh_data = decode_token(tokens.refresh_token)
        
        # Assert
        assert access_data.token_type == "access"
        assert refresh_data.token_type == "refresh"
        assert access_data.user_id == user_id
        assert refresh_data.user_id == user_id
        assert tokens.token_type == "bearer"
        assert tokens.expires_in > 0

    @settings(max_examples=100)
    @given(
        user_id=user_id_strategy,
        username=username_strategy
    )
    def test_verify_token_type_enforcement(self, user_id: int, username: str):
        """
        **Feature: intelligent-recommendation-system, Property 17: Authentication Correctness**
        **Validates: Requirements 10.1**
        
        For any token, verifying with the wrong expected type
        SHALL raise TokenInvalidError.
        """
        # Arrange
        access_token = create_access_token(user_id, username)
        refresh_token = create_refresh_token(user_id, username)
        
        # Act & Assert - Access token verified as refresh should fail
        with pytest.raises(TokenInvalidError):
            verify_token(access_token, expected_type="refresh")
        
        # Act & Assert - Refresh token verified as access should fail
        with pytest.raises(TokenInvalidError):
            verify_token(refresh_token, expected_type="access")

    def test_invalid_token_raises_error(self):
        """
        **Feature: intelligent-recommendation-system, Property 17: Authentication Correctness**
        **Validates: Requirements 10.1**
        
        An invalid/malformed token SHALL raise TokenInvalidError.
        """
        # Arrange
        invalid_token = "not.a.valid.jwt.token"
        
        # Act & Assert
        with pytest.raises(TokenInvalidError):
            decode_token(invalid_token)

    @settings(max_examples=100, deadline=None)
    @given(password=valid_password_strategy)
    def test_hash_produces_different_output_each_time(self, password: str):
        """
        **Feature: intelligent-recommendation-system, Property 17: Authentication Correctness**
        **Validates: Requirements 10.1**
        
        For any password, hashing twice SHALL produce different hashes
        (due to random salt), but both SHALL verify correctly.
        """
        # Arrange & Act
        hash1 = hash_password(password)
        hash2 = hash_password(password)
        
        # Assert - Different hashes due to salt
        assert hash1 != hash2
        
        # Assert - Both verify correctly
        assert verify_password(password, hash1) is True
        assert verify_password(password, hash2) is True



class TestPasswordPolicyCompliance:
    """
    Property tests for password policy compliance.
    
    **Feature: intelligent-recommendation-system, Property 24: Password Policy Compliance**
    **Validates: Requirements 21.1**
    """

    @settings(max_examples=100)
    @given(password=valid_password_strategy)
    def test_valid_password_passes_policy(self, password: str):
        """
        **Feature: intelligent-recommendation-system, Property 24: Password Policy Compliance**
        **Validates: Requirements 21.1**
        
        For any password meeting all requirements (length, uppercase, lowercase, digit, special),
        the password policy validation SHALL return is_valid=True with no errors.
        """
        # Act
        result = validate_password_policy(password)
        
        # Assert
        assert result.is_valid is True
        assert len(result.errors) == 0

    @settings(max_examples=100)
    @given(password=invalid_password_too_short)
    def test_short_password_fails_policy(self, password: str):
        """
        **Feature: intelligent-recommendation-system, Property 24: Password Policy Compliance**
        **Validates: Requirements 21.1**
        
        For any password shorter than minimum length,
        the password policy validation SHALL return is_valid=False.
        """
        # Act
        result = validate_password_policy(password, min_length=8)
        
        # Assert
        assert result.is_valid is False
        assert any("at least 8 characters" in err for err in result.errors)

    @settings(max_examples=100)
    @given(
        password=st.text(
            alphabet=string.ascii_lowercase + string.digits + "!@#$%",
            min_size=8,
            max_size=50
        ).filter(lambda p: not any(c.isupper() for c in p))
    )
    def test_password_without_uppercase_fails(self, password: str):
        """
        **Feature: intelligent-recommendation-system, Property 24: Password Policy Compliance**
        **Validates: Requirements 21.1**
        
        For any password without uppercase letters,
        the password policy validation SHALL return is_valid=False when uppercase is required.
        """
        # Act
        result = validate_password_policy(
            password,
            require_uppercase=True,
            require_lowercase=False,
            require_digit=False,
            require_special=False
        )
        
        # Assert
        assert result.is_valid is False
        assert any("uppercase" in err.lower() for err in result.errors)

    @settings(max_examples=100)
    @given(
        password=st.text(
            alphabet=string.ascii_uppercase + string.digits + "!@#$%",
            min_size=8,
            max_size=50
        ).filter(lambda p: not any(c.islower() for c in p))
    )
    def test_password_without_lowercase_fails(self, password: str):
        """
        **Feature: intelligent-recommendation-system, Property 24: Password Policy Compliance**
        **Validates: Requirements 21.1**
        
        For any password without lowercase letters,
        the password policy validation SHALL return is_valid=False when lowercase is required.
        """
        # Act
        result = validate_password_policy(
            password,
            require_uppercase=False,
            require_lowercase=True,
            require_digit=False,
            require_special=False
        )
        
        # Assert
        assert result.is_valid is False
        assert any("lowercase" in err.lower() for err in result.errors)

    @settings(max_examples=100)
    @given(
        password=st.text(
            alphabet=string.ascii_letters + "!@#$%",
            min_size=8,
            max_size=50
        ).filter(lambda p: not any(c.isdigit() for c in p))
    )
    def test_password_without_digit_fails(self, password: str):
        """
        **Feature: intelligent-recommendation-system, Property 24: Password Policy Compliance**
        **Validates: Requirements 21.1**
        
        For any password without digits,
        the password policy validation SHALL return is_valid=False when digit is required.
        """
        # Act
        result = validate_password_policy(
            password,
            require_uppercase=False,
            require_lowercase=False,
            require_digit=True,
            require_special=False
        )
        
        # Assert
        assert result.is_valid is False
        assert any("digit" in err.lower() for err in result.errors)

    @settings(max_examples=100)
    @given(
        password=st.text(
            alphabet=string.ascii_letters + string.digits,
            min_size=8,
            max_size=50
        )
    )
    def test_password_without_special_char_fails(self, password: str):
        """
        **Feature: intelligent-recommendation-system, Property 24: Password Policy Compliance**
        **Validates: Requirements 21.1**
        
        For any password without special characters,
        the password policy validation SHALL return is_valid=False when special char is required.
        """
        # Act
        result = validate_password_policy(
            password,
            require_uppercase=False,
            require_lowercase=False,
            require_digit=False,
            require_special=True
        )
        
        # Assert
        assert result.is_valid is False
        assert any("special" in err.lower() for err in result.errors)

    @settings(max_examples=100)
    @given(password=valid_password_strategy)
    def test_is_password_valid_matches_validate_result(self, password: str):
        """
        **Feature: intelligent-recommendation-system, Property 24: Password Policy Compliance**
        **Validates: Requirements 21.1**
        
        For any password, is_password_valid() SHALL return the same result
        as validate_password_policy().is_valid.
        """
        # Act
        quick_result = is_password_valid(password)
        detailed_result = validate_password_policy(password)
        
        # Assert
        assert quick_result == detailed_result.is_valid

    @settings(max_examples=100)
    @given(
        min_length=st.integers(min_value=1, max_value=20),
        password_length=st.integers(min_value=1, max_value=30)
    )
    def test_length_validation_is_consistent(self, min_length: int, password_length: int):
        """
        **Feature: intelligent-recommendation-system, Property 24: Password Policy Compliance**
        **Validates: Requirements 21.1**
        
        For any min_length and password_length combination,
        the validation SHALL correctly identify if password meets length requirement.
        """
        # Arrange
        password = "A" * password_length
        
        # Act
        result = validate_password_policy(
            password,
            min_length=min_length,
            require_uppercase=False,
            require_lowercase=False,
            require_digit=False,
            require_special=False
        )
        
        # Assert
        if password_length >= min_length:
            assert result.is_valid is True
        else:
            assert result.is_valid is False
            assert any(f"at least {min_length} characters" in err for err in result.errors)

    def test_empty_password_fails_all_checks(self):
        """
        **Feature: intelligent-recommendation-system, Property 24: Password Policy Compliance**
        **Validates: Requirements 21.1**
        
        An empty password SHALL fail validation.
        """
        # Act
        result = validate_password_policy("")
        
        # Assert
        assert result.is_valid is False
        assert len(result.errors) > 0
