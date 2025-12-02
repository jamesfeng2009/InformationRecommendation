from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.core.redis import get_redis_client
from app.core.security import TokenError, TokenExpiredError, TokenInvalidError
from app.schemas.auth import (
    LoginRequest,
    MessageResponse,
    RefreshRequest,
    TokenResponse,
    UserResponse,
)
from app.services.auth import AuthenticationError, AuthService, SessionExpiredError

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Security scheme for bearer token
security = HTTPBearer(auto_error=False)


async def get_auth_service(
    db: Annotated[AsyncSession, Depends(get_db_session)],
    redis: Annotated[Redis, Depends(get_redis_client)],
) -> AuthService:
    """Dependency to get AuthService instance."""
    return AuthService(db, redis)


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


@router.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> TokenResponse:
    """
    Authenticate user and return access/refresh tokens.
    
    - **username**: Username or account
    - **password**: User password
    """
    try:
        tokens = await auth_service.login(request.username, request.password)
        return TokenResponse(
            access_token=tokens.access_token,
            refresh_token=tokens.refresh_token,
            token_type=tokens.token_type,
            expires_in=tokens.expires_in,
        )
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/logout", response_model=MessageResponse)
async def logout(
    token: Annotated[str, Depends(get_current_token)],
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> MessageResponse:
    """
    Logout user and invalidate current session.
    
    Requires valid access token in Authorization header.
    """
    success = await auth_service.logout(token)
    if success:
        return MessageResponse(message="Successfully logged out", success=True)
    return MessageResponse(message="Logout failed", success=False)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_tokens(
    request: RefreshRequest,
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> TokenResponse:
    """
    Refresh access token using refresh token.
    
    - **refresh_token**: Valid refresh token from login
    """
    try:
        tokens = await auth_service.refresh_tokens(request.refresh_token)
        return TokenResponse(
            access_token=tokens.access_token,
            refresh_token=tokens.refresh_token,
            token_type=tokens.token_type,
            expires_in=tokens.expires_in,
        )
    except TokenExpiredError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except (TokenInvalidError, SessionExpiredError) as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user(
    token: Annotated[str, Depends(get_current_token)],
    auth_service: Annotated[AuthService, Depends(get_auth_service)],
) -> UserResponse:
    """
    Get current authenticated user information.
    
    Requires valid access token in Authorization header.
    """
    try:
        user = await auth_service.get_current_user(token)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or token invalid",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return UserResponse.model_validate(user)
    except TokenExpiredError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except TokenInvalidError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
