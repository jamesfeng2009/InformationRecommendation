import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.security import (
    TokenData,
    TokenError,
    TokenExpiredError,
    TokenInvalidError,
    TokenPair,
    create_access_token,
    create_refresh_token,
    create_token_pair,
    hash_password,
    verify_password,
    verify_token,
)
from app.models.user import User


class AuthenticationError(Exception):
    """Authentication failed."""
    pass


class SessionExpiredError(Exception):
    """Session has expired or been revoked."""
    pass


class AuthService:
    """
    Authentication service handling login, logout, and token management.
    Uses Redis for session storage and token blacklisting.
    """
    
    # Redis key prefixes
    SESSION_PREFIX = "session:"
    BLACKLIST_PREFIX = "blacklist:"
    USER_SESSIONS_PREFIX = "user_sessions:"
    
    def __init__(self, db: AsyncSession, redis: Redis):
        self.db = db
        self.redis = redis
        self.settings = get_settings()
    
    async def authenticate(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate user with username and password.
        
        Args:
            username: User's username or account.
            password: Plain text password.
        
        Returns:
            User if authentication successful, None otherwise.
        """
        # Query user by username or account
        stmt = select(User).where(
            (User.username == username) | (User.account == username)
        )
        result = await self.db.execute(stmt)
        user = result.scalar_one_or_none()
        
        if user is None:
            return None
        
        # Check if user is enabled
        if user.status != "enabled":
            return None
        
        # Verify password
        if not verify_password(password, user.password_hash):
            return None
        
        return user
    
    async def login(self, username: str, password: str) -> TokenPair:
        """
        Authenticate user and create session.
        
        Args:
            username: User's username or account.
            password: Plain text password.
        
        Returns:
            TokenPair with access and refresh tokens.
        
        Raises:
            AuthenticationError: If authentication fails.
        """
        user = await self.authenticate(username, password)
        
        if user is None:
            raise AuthenticationError("Invalid username or password")
        
        # Generate unique session ID for refresh token
        session_id = str(uuid.uuid4())
        
        # Create token pair
        tokens = create_token_pair(user.id, user.username, jti=session_id)
        
        # Store session in Redis
        await self._store_session(user.id, session_id, tokens)
        
        # Update last login time
        user.last_login = datetime.now(timezone.utc)
        await self.db.commit()
        
        return tokens
    
    async def logout(self, token: str) -> bool:
        """
        Logout user and invalidate session.
        
        Args:
            token: Access or refresh token.
        
        Returns:
            True if logout successful.
        """
        try:
            token_data = verify_token(token, expected_type="access")
        except TokenError:
            # Try as refresh token
            try:
                token_data = verify_token(token, expected_type="refresh")
            except TokenError:
                return False
        
        # Blacklist the token
        await self._blacklist_token(token, token_data)
        
        # If it's a refresh token with jti, remove the session
        if token_data.jti:
            await self._remove_session(token_data.user_id, token_data.jti)
        
        return True
    
    async def refresh_tokens(self, refresh_token: str) -> TokenPair:
        """
        Refresh access token using refresh token.
        
        Args:
            refresh_token: Valid refresh token.
        
        Returns:
            New TokenPair with fresh access and refresh tokens.
        
        Raises:
            TokenExpiredError: If refresh token has expired.
            TokenInvalidError: If refresh token is invalid.
            SessionExpiredError: If session has been revoked.
        """
        # Verify refresh token
        token_data = verify_token(refresh_token, expected_type="refresh")
        
        # Check if token is blacklisted
        if await self._is_blacklisted(refresh_token):
            raise TokenInvalidError("Token has been revoked")
        
        # Check if session exists
        if token_data.jti:
            session_exists = await self._session_exists(token_data.user_id, token_data.jti)
            if not session_exists:
                raise SessionExpiredError("Session has expired or been revoked")
        
        # Blacklist old refresh token
        await self._blacklist_token(refresh_token, token_data)
        
        # Generate new session ID
        new_session_id = str(uuid.uuid4())
        
        # Create new token pair
        tokens = create_token_pair(token_data.user_id, token_data.username, jti=new_session_id)
        
        # Update session in Redis
        if token_data.jti:
            await self._remove_session(token_data.user_id, token_data.jti)
        await self._store_session(token_data.user_id, new_session_id, tokens)
        
        return tokens
    
    async def verify_access_token(self, token: str) -> TokenData:
        """
        Verify access token and check if it's valid.
        
        Args:
            token: Access token to verify.
        
        Returns:
            TokenData if valid.
        
        Raises:
            TokenExpiredError: If token has expired.
            TokenInvalidError: If token is invalid or blacklisted.
        """
        token_data = verify_token(token, expected_type="access")
        
        # Check if token is blacklisted
        if await self._is_blacklisted(token):
            raise TokenInvalidError("Token has been revoked")
        
        return token_data
    
    async def get_current_user(self, token: str) -> Optional[User]:
        """
        Get current user from access token.
        
        Args:
            token: Access token.
        
        Returns:
            User if token is valid, None otherwise.
        """
        try:
            token_data = await self.verify_access_token(token)
        except TokenError:
            return None
        
        stmt = select(User).where(User.id == token_data.user_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def logout_all_sessions(self, user_id: int) -> int:
        """
        Logout user from all sessions.
        
        Args:
            user_id: User's database ID.
        
        Returns:
            Number of sessions invalidated.
        """
        sessions_key = f"{self.USER_SESSIONS_PREFIX}{user_id}"
        session_ids = await self.redis.smembers(sessions_key)
        
        count = 0
        for session_id in session_ids:
            session_key = f"{self.SESSION_PREFIX}{user_id}:{session_id}"
            await self.redis.delete(session_key)
            count += 1
        
        await self.redis.delete(sessions_key)
        return count
    
    # Private helper methods
    
    async def _store_session(self, user_id: int, session_id: str, tokens: TokenPair) -> None:
        """Store session data in Redis."""
        session_key = f"{self.SESSION_PREFIX}{user_id}:{session_id}"
        sessions_key = f"{self.USER_SESSIONS_PREFIX}{user_id}"
        
        session_data = {
            "user_id": user_id,
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        
        # Store session with expiration matching refresh token
        expire_seconds = self.settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
        await self.redis.set(session_key, json.dumps(session_data), ex=expire_seconds)
        
        # Track user's sessions
        await self.redis.sadd(sessions_key, session_id)
        await self.redis.expire(sessions_key, expire_seconds)
    
    async def _remove_session(self, user_id: int, session_id: str) -> None:
        """Remove session from Redis."""
        session_key = f"{self.SESSION_PREFIX}{user_id}:{session_id}"
        sessions_key = f"{self.USER_SESSIONS_PREFIX}{user_id}"
        
        await self.redis.delete(session_key)
        await self.redis.srem(sessions_key, session_id)
    
    async def _session_exists(self, user_id: int, session_id: str) -> bool:
        """Check if session exists in Redis."""
        session_key = f"{self.SESSION_PREFIX}{user_id}:{session_id}"
        return await self.redis.exists(session_key) > 0
    
    async def _blacklist_token(self, token: str, token_data: TokenData) -> None:
        """Add token to blacklist."""
        blacklist_key = f"{self.BLACKLIST_PREFIX}{hash(token)}"
        
        # Calculate remaining TTL
        now = datetime.now(timezone.utc)
        ttl = int((token_data.exp - now).total_seconds())
        
        if ttl > 0:
            await self.redis.set(blacklist_key, "1", ex=ttl)
    
    async def _is_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted."""
        blacklist_key = f"{self.BLACKLIST_PREFIX}{hash(token)}"
        return await self.redis.exists(blacklist_key) > 0
