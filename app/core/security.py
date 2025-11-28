"""
Security utilities for password hashing, verification, policy validation, and JWT tokens.
Requirements: 10.1, 18.1, 21.1
"""
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import bcrypt
import jwt

from app.core.config import get_settings


@dataclass
class PasswordValidationResult:
    """Result of password policy validation."""
    is_valid: bool
    errors: List[str]


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.
    
    Args:
        password: Plain text password to hash.
    
    Returns:
        str: Bcrypt hashed password.
    """
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.
    
    Args:
        plain_password: Plain text password to verify.
        hashed_password: Bcrypt hashed password to compare against.
    
    Returns:
        bool: True if password matches, False otherwise.
    """
    try:
        return bcrypt.checkpw(
            plain_password.encode('utf-8'),
            hashed_password.encode('utf-8')
        )
    except (ValueError, TypeError):
        return False


def validate_password_policy(
    password: str,
    min_length: Optional[int] = None,
    require_uppercase: Optional[bool] = None,
    require_lowercase: Optional[bool] = None,
    require_digit: Optional[bool] = None,
    require_special: Optional[bool] = None,
) -> PasswordValidationResult:
    """
    Validate password against security policy.
    
    Args:
        password: Password to validate.
        min_length: Minimum password length (default from settings).
        require_uppercase: Require at least one uppercase letter.
        require_lowercase: Require at least one lowercase letter.
        require_digit: Require at least one digit.
        require_special: Require at least one special character.
    
    Returns:
        PasswordValidationResult: Validation result with errors if any.
    
    Requirements: 21.1 - Password policy compliance
    """
    settings = get_settings()
    
    # Use settings defaults if not specified
    min_length = min_length if min_length is not None else settings.PASSWORD_MIN_LENGTH
    require_uppercase = require_uppercase if require_uppercase is not None else settings.PASSWORD_REQUIRE_UPPERCASE
    require_lowercase = require_lowercase if require_lowercase is not None else settings.PASSWORD_REQUIRE_LOWERCASE
    require_digit = require_digit if require_digit is not None else settings.PASSWORD_REQUIRE_DIGIT
    require_special = require_special if require_special is not None else settings.PASSWORD_REQUIRE_SPECIAL
    
    errors: List[str] = []
    
    # Check minimum length
    if len(password) < min_length:
        errors.append(f"Password must be at least {min_length} characters long")
    
    # Check for uppercase letter
    if require_uppercase and not re.search(r'[A-Z]', password):
        errors.append("Password must contain at least one uppercase letter")
    
    # Check for lowercase letter
    if require_lowercase and not re.search(r'[a-z]', password):
        errors.append("Password must contain at least one lowercase letter")
    
    # Check for digit
    if require_digit and not re.search(r'\d', password):
        errors.append("Password must contain at least one digit")
    
    # Check for special character
    if require_special and not re.search(r'[!@#$%^&*(),.?":{}|<>_\-+=\[\]\\;\'`~]', password):
        errors.append("Password must contain at least one special character")
    
    return PasswordValidationResult(
        is_valid=len(errors) == 0,
        errors=errors
    )


def is_password_valid(password: str) -> bool:
    """
    Quick check if password meets policy requirements.
    
    Args:
        password: Password to validate.
    
    Returns:
        bool: True if password is valid, False otherwise.
    """
    result = validate_password_policy(password)
    return result.is_valid


# JWT Token Management
# Requirements: 10.1, 18.1

@dataclass
class TokenData:
    """Decoded token data."""
    user_id: int
    username: str
    token_type: str  # "access" or "refresh"
    exp: datetime
    iat: datetime
    jti: Optional[str] = None  # JWT ID for token tracking


@dataclass
class TokenPair:
    """Access and refresh token pair."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 0  # seconds until access token expires


class TokenError(Exception):
    """Base exception for token-related errors."""
    pass


class TokenExpiredError(TokenError):
    """Token has expired."""
    pass


class TokenInvalidError(TokenError):
    """Token is invalid or malformed."""
    pass


def create_access_token(
    user_id: int,
    username: str,
    expires_delta: Optional[timedelta] = None,
    additional_claims: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Create a JWT access token.
    
    Args:
        user_id: User's database ID.
        username: User's username.
        expires_delta: Custom expiration time (default from settings).
        additional_claims: Additional claims to include in token.
    
    Returns:
        str: Encoded JWT access token.
    """
    settings = get_settings()
    
    if expires_delta is None:
        expires_delta = timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    
    now = datetime.now(timezone.utc)
    expire = now + expires_delta
    
    payload = {
        "sub": str(user_id),
        "username": username,
        "type": "access",
        "exp": expire,
        "iat": now,
    }
    
    if additional_claims:
        payload.update(additional_claims)
    
    return jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )


def create_refresh_token(
    user_id: int,
    username: str,
    expires_delta: Optional[timedelta] = None,
    jti: Optional[str] = None,
) -> str:
    """
    Create a JWT refresh token.
    
    Args:
        user_id: User's database ID.
        username: User's username.
        expires_delta: Custom expiration time (default from settings).
        jti: JWT ID for token tracking/revocation.
    
    Returns:
        str: Encoded JWT refresh token.
    """
    settings = get_settings()
    
    if expires_delta is None:
        expires_delta = timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    
    now = datetime.now(timezone.utc)
    expire = now + expires_delta
    
    payload = {
        "sub": str(user_id),
        "username": username,
        "type": "refresh",
        "exp": expire,
        "iat": now,
    }
    
    if jti:
        payload["jti"] = jti
    
    return jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )


def create_token_pair(
    user_id: int,
    username: str,
    jti: Optional[str] = None,
) -> TokenPair:
    """
    Create both access and refresh tokens.
    
    Args:
        user_id: User's database ID.
        username: User's username.
        jti: JWT ID for refresh token tracking.
    
    Returns:
        TokenPair: Access and refresh tokens.
    """
    settings = get_settings()
    
    access_token = create_access_token(user_id, username)
    refresh_token = create_refresh_token(user_id, username, jti=jti)
    
    return TokenPair(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


def decode_token(token: str) -> TokenData:
    """
    Decode and validate a JWT token.
    
    Args:
        token: JWT token string.
    
    Returns:
        TokenData: Decoded token data.
    
    Raises:
        TokenExpiredError: If token has expired.
        TokenInvalidError: If token is invalid or malformed.
    """
    settings = get_settings()
    
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        
        return TokenData(
            user_id=int(payload["sub"]),
            username=payload["username"],
            token_type=payload.get("type", "access"),
            exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
            iat=datetime.fromtimestamp(payload["iat"], tz=timezone.utc),
            jti=payload.get("jti"),
        )
    except jwt.ExpiredSignatureError:
        raise TokenExpiredError("Token has expired")
    except jwt.InvalidTokenError as e:
        raise TokenInvalidError(f"Invalid token: {str(e)}")


def verify_token(token: str, expected_type: str = "access") -> TokenData:
    """
    Verify a JWT token and check its type.
    
    Args:
        token: JWT token string.
        expected_type: Expected token type ("access" or "refresh").
    
    Returns:
        TokenData: Decoded token data.
    
    Raises:
        TokenExpiredError: If token has expired.
        TokenInvalidError: If token is invalid or wrong type.
    """
    token_data = decode_token(token)
    
    if token_data.token_type != expected_type:
        raise TokenInvalidError(f"Expected {expected_type} token, got {token_data.token_type}")
    
    return token_data
