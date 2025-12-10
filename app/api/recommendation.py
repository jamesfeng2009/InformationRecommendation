from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.permissions import require_permission
from app.models.user import User
from app.services.auth import get_current_user
from app.services.recommendation import RecommendationService
from app.services.user_profile import UserProfileService, UserNotFoundError


router = APIRouter(prefix="/api/v1/recommend", tags=["recommendation"])


# ==================== Request/Response Models ====================

class InteractionRequest(BaseModel):
    """Request model for recording user interactions."""
    news_id: int = Field(..., description="ID of the news item")
    action: str = Field(..., description="Action type: view, like, collect, dislike, share")


class InterestUpdateRequest(BaseModel):
    """Request model for updating user interests."""
    interests: List[str] = Field(..., description="List of interest tags")


class NewsRecommendation(BaseModel):
    """Response model for news recommendations."""
    id: int
    title: str
    summary: Optional[str]
    category: Optional[str]
    source_name: str
    author: Optional[str]
    location: Optional[str]
    publish_time: Optional[str]
    hot_score: float
    keywords: List[str]
    images: List[str]
    recommendation_score: Optional[float] = None
    content_score: Optional[float] = None
    collaborative_score: Optional[float] = None
    popularity_score: Optional[float] = None
    relevance_score: Optional[float] = None
    topic_match_keywords: Optional[List[str]] = None


class RecommendationResponse(BaseModel):
    """Response model for recommendation endpoints."""
    items: List[NewsRecommendation]
    page: int
    size: int
    total: Optional[int] = None


# ==================== Dependency Injection ====================

async def get_recommendation_service(db: AsyncSession = Depends(get_db)) -> RecommendationService:
    """Get recommendation service instance."""
    user_profile_service = UserProfileService(db)
    return RecommendationService(db, user_profile_service)


async def get_user_profile_service(db: AsyncSession = Depends(get_db)) -> UserProfileService:
    """Get user profile service instance."""
    return UserProfileService(db)


# ==================== API Endpoints ====================

@router.get("/feed", response_model=RecommendationResponse)
async def get_personalized_feed(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Items per page"),
    exclude_read: bool = Query(True, description="Exclude already read news"),
    current_user: User = Depends(get_current_user),
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
):
    """
    Get personalized news feed for the current user.
    
    Returns news recommendations based on user's interests, behavior, and preferences.
    Uses hybrid recommendation combining content-based, collaborative filtering,
    and popularity-based approaches.
    """
    try:
        items = await recommendation_service.get_personalized_feed(
            user_id=current_user.id,
            page=page,
            size=size,
            exclude_read=exclude_read,
        )
        
        return RecommendationResponse(
            items=[NewsRecommendation(**item) for item in items],
            page=page,
            size=size,
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get personalized feed: {str(e)}")


@router.get("/latest", response_model=RecommendationResponse)
async def get_latest_feed(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Items per page"),
    category: Optional[str] = Query(None, description="Filter by category"),
    current_user: User = Depends(get_current_user),
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
):
    """
    Get latest news feed in chronological order.
    
    Returns the most recent news items, optionally filtered by category.
    """
    try:
        items = await recommendation_service.get_latest_feed(
            page=page,
            size=size,
            category=category,
        )
        
        return RecommendationResponse(
            items=[NewsRecommendation(**item) for item in items],
            page=page,
            size=size,
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get latest feed: {str(e)}")


@router.get("/topic-hot", response_model=RecommendationResponse)
async def get_topic_hot_news(
    limit: int = Query(10, ge=1, le=50, description="Maximum number of items"),
    time_window_hours: int = Query(24, ge=1, le=168, description="Time window in hours"),
    current_user: User = Depends(get_current_user),
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
):
    """
    Get topic hot news filtered by user keywords and interests.
    
    Returns trending news that matches the user's topics of interest,
    combining user keywords, explicit interests, and hot news ranking.
    """
    try:
        items = await recommendation_service.get_topic_hot_news(
            user_id=current_user.id,
            limit=limit,
            time_window_hours=time_window_hours,
        )
        
        return RecommendationResponse(
            items=[NewsRecommendation(**item) for item in items],
            page=1,
            size=limit,
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get topic hot news: {str(e)}")


@router.post("/interaction")
async def record_interaction(
    request: InteractionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Record user interaction with news item.
    
    Records user behavior (view, like, collect, dislike, share) for
    improving future recommendations.
    """
    from app.models.news import UserBehavior
    
    # Validate action type
    valid_actions = {"view", "like", "collect", "dislike", "share"}
    if request.action not in valid_actions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action. Must be one of: {', '.join(valid_actions)}"
        )
    
    try:
        # Check if news exists
        from app.models.news import News
        from sqlalchemy import select
        
        news_stmt = select(News).where(News.id == request.news_id)
        news_result = await db.execute(news_stmt)
        news = news_result.scalar_one_or_none()
        
        if not news:
            raise HTTPException(status_code=404, detail="News item not found")
        
        # Record the interaction
        behavior = UserBehavior(
            user_id=current_user.id,
            news_id=request.news_id,
            action=request.action,
        )
        
        db.add(behavior)
        await db.commit()
        
        # Invalidate user's recommendation cache
        recommendation_service = await get_recommendation_service(db)
        await recommendation_service.invalidate_user_cache(current_user.id)
        
        return {"message": "Interaction recorded successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to record interaction: {str(e)}")


@router.put("/interests")
async def update_user_interests(
    request: InterestUpdateRequest,
    current_user: User = Depends(get_current_user),
    user_profile_service: UserProfileService = Depends(get_user_profile_service),
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
):
    """
    Update user's explicit interest tags.
    
    Updates the user's interest preferences which will be used for
    personalized recommendations.
    """
    try:
        # Update interests
        updated_interests = await user_profile_service.update_user_interests(
            user_id=current_user.id,
            interest_tags=request.interests,
        )
        
        # Invalidate user's recommendation cache
        await recommendation_service.invalidate_user_cache(current_user.id)
        
        return {
            "message": "Interests updated successfully",
            "interests": updated_interests,
        }
    
    except UserNotFoundError:
        raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/interests")
async def get_user_interests(
    current_user: User = Depends(get_current_user),
    user_profile_service: UserProfileService = Depends(get_user_profile_service),
):
    """
    Get user's current interest tags and available predefined tags.
    
    Returns both the user's selected interests and the full library
    of available interest tags.
    """
    try:
        # Get user's current interests
        current_interests = await user_profile_service.get_user_explicit_interests(current_user.id)
        
        # Get predefined tag library
        predefined_tags = user_profile_service.get_predefined_tags()
        
        return {
            "current_interests": current_interests,
            "predefined_tags": predefined_tags,
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get interests: {str(e)}")


@router.get("/profile")
async def get_user_profile(
    current_user: User = Depends(get_current_user),
    user_profile_service: UserProfileService = Depends(get_user_profile_service),
):
    """
    Get user's complete recommendation profile.
    
    Returns detailed information about the user's interests, preferences,
    and behavior patterns used for recommendations.
    """
    try:
        profile = await user_profile_service.build_user_profile(current_user.id)
        
        return {
            "user_id": profile.user_id,
            "explicit_interests": profile.explicit_interests,
            "preferred_categories": profile.preferred_categories,
            "preferred_keywords": profile.preferred_keywords,
            "activity_level": profile.activity_level,
            "last_updated": profile.last_updated.isoformat(),
            "total_interests": len(profile.combined_interests),
        }
    
    except UserNotFoundError:
        raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user profile: {str(e)}")


# ==================== Admin Endpoints ====================

@router.post("/cache/warm")
@require_permission("system", "admin")
async def warm_recommendation_cache(
    limit: int = Query(100, ge=1, le=1000, description="Number of active users to warm cache for"),
    current_user: User = Depends(get_current_user),
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
):
    """
    Warm recommendation cache for active users.
    
    Pre-generates recommendations for recently active users to improve
    response times. Admin only.
    """
    try:
        await recommendation_service.warm_cache_for_active_users(limit=limit)
        
        return {"message": f"Cache warmed for up to {limit} active users"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to warm cache: {str(e)}")


@router.delete("/cache/{user_id}")
@require_permission("system", "admin")
async def invalidate_user_recommendation_cache(
    user_id: int,
    current_user: User = Depends(get_current_user),
    recommendation_service: RecommendationService = Depends(get_recommendation_service),
):
    """
    Invalidate recommendation cache for a specific user.
    
    Forces regeneration of recommendations for the specified user.
    Admin only.
    """
    try:
        await recommendation_service.invalidate_user_cache(user_id)
        
        return {"message": f"Cache invalidated for user {user_id}"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to invalidate cache: {str(e)}")