import warnings
warnings.filterwarnings("ignore")

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock

from hypothesis import given, strategies as st, settings, assume
from hypothesis.strategies import composite

from app.services.news import NewsService
from app.models.news import News


# =============================================================================
# Strategies for generating test data
# =============================================================================

@composite
def news_data(draw):
    """Generate realistic news data for testing."""
    return {
        "id": draw(st.integers(min_value=1, max_value=10000)),
        "title": draw(st.text(min_size=10, max_size=200, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs")))),
        "content": draw(st.text(min_size=50, max_size=2000, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs", "Po")))),
        "source_name": draw(st.sampled_from(["BBC", "CNN", "Reuters", "人民网", "新华网"])),
        "source_url": draw(st.text(min_size=10, max_size=100)),
        "language": draw(st.sampled_from(["zh", "en", "ru", "ja", "ko"])),
        "category": draw(st.sampled_from(["政治", "军事", "经济", "科技", "体育", None])),
        "author": draw(st.one_of(st.none(), st.text(min_size=2, max_size=50))),
        "location": draw(st.one_of(st.none(), st.text(min_size=2, max_size=100))),
        "keywords": draw(st.one_of(st.none(), st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=10))),
        "summary": draw(st.one_of(st.none(), st.text(min_size=20, max_size=300))),
        "publish_time": draw(st.one_of(st.none(), st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2025, 12, 31)))),
        "crawl_time": draw(st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2025, 12, 31))),
        "hot_score": draw(st.floats(min_value=0.0, max_value=1000.0))
    }

@composite
def search_query(draw):
    """Generate search queries."""
    return draw(st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs"))))

@composite
def search_filters(draw):
    """Generate search filter combinations."""
    return {
        "category": draw(st.one_of(st.none(), st.sampled_from(["政治", "军事", "经济", "科技", "体育"]))),
        "source_name": draw(st.one_of(st.none(), st.sampled_from(["BBC", "CNN", "Reuters", "人民网", "新华网"]))),
        "language": draw(st.one_of(st.none(), st.sampled_from(["zh", "en", "ru", "ja", "ko"]))),
        "start_time": draw(st.one_of(st.none(), st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2024, 12, 31)))),
        "end_time": draw(st.one_of(st.none(), st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2025, 12, 31))))
    }


# =============================================================================
# Mock helpers
# =============================================================================

def create_mock_news_service():
    """Create a mock news service for testing."""
    mock_db = AsyncMock()
    mock_indexing_service = MagicMock()
    mock_es_client = AsyncMock()
    mock_redis_client = AsyncMock()
    
    service = NewsService(
        db_session=mock_db,
        indexing_service=mock_indexing_service,
        es_client=mock_es_client,
        redis_client=mock_redis_client
    )
    
    return service, mock_es_client

def create_mock_search_response(query: str, news_items: List[dict], filters: dict):
    """Create a mock Elasticsearch search response."""
    hits = []
    for news in news_items:
        # Check if news matches filters
        if filters.get("category") and news.get("category") != filters["category"]:
            continue
        if filters.get("source_name") and news.get("source_name") != filters["source_name"]:
            continue
        if filters.get("language") and news.get("language") != filters["language"]:
            continue
        if filters.get("start_time") and news.get("publish_time") and news["publish_time"] < filters["start_time"]:
            continue
        if filters.get("end_time") and news.get("publish_time") and news["publish_time"] > filters["end_time"]:
            continue
        
        # Check if news contains query terms (simplified)
        query_lower = query.lower()
        title_lower = news.get("title", "").lower()
        content_lower = news.get("content", "").lower()
        
        if query_lower in title_lower or query_lower in content_lower:
            hits.append({
                "_source": news,
                "_score": 1.0,
                "highlight": {
                    "title": [news.get("title", "")],
                    "content": [news.get("content", "")[:200]]
                }
            })
    
    return {
        "hits": {
            "hits": hits,
            "total": {"value": len(hits)}
        },
        "took": 10
    }


# =============================================================================
# Property Tests
# =============================================================================

@settings(max_examples=100, deadline=None)
@given(
    query=search_query(),
    news_list=st.lists(news_data(), min_size=1, max_size=20),
    filters=search_filters()
)
def test_search_result_relevance(query, news_list, filters):
    """
    **Feature: intelligent-recommendation-system, Property 11: Search Result Relevance**
    **Validates: Requirements 6.1**
    
    For any search query, all returned results SHALL contain the search keywords 
    in either title or content (with stemming/tokenization applied).
    """
    # Skip empty queries or queries with only whitespace
    assume(query.strip())
    
    # Ensure time filters are valid
    if filters["start_time"] and filters["end_time"]:
        assume(filters["start_time"] <= filters["end_time"])
    
    async def run_test():
        service, mock_es_client = create_mock_news_service()
        
        # Mock the get_news method to return news objects
        async def mock_get_news(news_id):
            for news_data in news_list:
                if news_data["id"] == news_id:
                    news = News(**news_data)
                    return news
            return None
        
        service.get_news = mock_get_news
        
        # Mock Elasticsearch response
        mock_response = create_mock_search_response(query, news_list, filters)
        mock_es_client.search.return_value = mock_response
        
        # Execute search
        result = await service.search_news(
            query=query,
            page=1,
            size=20,
            category=filters["category"],
            source_name=filters["source_name"],
            language=filters["language"],
            start_time=filters["start_time"],
            end_time=filters["end_time"],
            sort_by="relevance"
        )
        
        # Verify all results contain the query terms
        query_terms = query.lower().split()
        for item in result["items"]:
            title_lower = item["title"].lower()
            content_lower = item["content"].lower()
            
            # At least one query term should be found in title or content
            found_term = False
            for term in query_terms:
                if term in title_lower or term in content_lower:
                    found_term = True
                    break
            
            assert found_term, f"Search result does not contain query terms. Query: '{query}', Title: '{item['title'][:100]}...'"
    
    # Run the async test
    asyncio.run(run_test())


@settings(max_examples=100, deadline=None)
@given(
    query=search_query(),
    news_list=st.lists(news_data(), min_size=5, max_size=20),
    filters=search_filters()
)
def test_search_filter_correctness(query, news_list, filters):
    """
    **Feature: intelligent-recommendation-system, Property 12: Search Filter Correctness**
    **Validates: Requirements 6.2**
    
    For any search with filters (time range, category, source), all returned results 
    SHALL satisfy ALL specified filter conditions.
    """
    # Skip empty queries
    assume(query.strip())
    
    # Ensure time filters are valid
    if filters["start_time"] and filters["end_time"]:
        assume(filters["start_time"] <= filters["end_time"])
    
    # Skip if no filters are applied
    has_filters = any([
        filters["category"],
        filters["source_name"], 
        filters["language"],
        filters["start_time"],
        filters["end_time"]
    ])
    assume(has_filters)
    
    async def run_test():
        service, mock_es_client = create_mock_news_service()
        
        # Mock the get_news method
        async def mock_get_news(news_id):
            for news_data in news_list:
                if news_data["id"] == news_id:
                    news = News(**news_data)
                    return news
            return None
        
        service.get_news = mock_get_news
        
        # Mock Elasticsearch response
        mock_response = create_mock_search_response(query, news_list, filters)
        mock_es_client.search.return_value = mock_response
        
        # Execute search
        result = await service.search_news(
            query=query,
            page=1,
            size=20,
            category=filters["category"],
            source_name=filters["source_name"],
            language=filters["language"],
            start_time=filters["start_time"],
            end_time=filters["end_time"],
            sort_by="relevance"
        )
        
        # Verify all results satisfy filter conditions
        for item in result["items"]:
            # Category filter
            if filters["category"]:
                assert item["category"] == filters["category"], \
                    f"Result category '{item['category']}' does not match filter '{filters['category']}'"
            
            # Source name filter
            if filters["source_name"]:
                assert item["source_name"] == filters["source_name"], \
                    f"Result source '{item['source_name']}' does not match filter '{filters['source_name']}'"
            
            # Language filter
            if filters["language"]:
                assert item["language"] == filters["language"], \
                    f"Result language '{item['language']}' does not match filter '{filters['language']}'"
            
            # Time range filters
            if filters["start_time"] and item["publish_time"]:
                assert item["publish_time"] >= filters["start_time"], \
                    f"Result publish time {item['publish_time']} is before start time {filters['start_time']}"
            
            if filters["end_time"] and item["publish_time"]:
                assert item["publish_time"] <= filters["end_time"], \
                    f"Result publish time {item['publish_time']} is after end time {filters['end_time']}"
    
    # Run the async test
    asyncio.run(run_test())


@settings(max_examples=100, deadline=None)
@given(
    query=search_query(),
    news_list=st.lists(news_data(), min_size=3, max_size=15),
    sort_by=st.sampled_from(["relevance", "time"])
)
def test_search_sort_consistency(query, news_list, sort_by):
    """
    **Feature: intelligent-recommendation-system, Property 13: Search Sort Consistency**
    **Validates: Requirements 6.3**
    
    For any search results sorted by time, the results SHALL be in strictly descending 
    order by publish_time. For relevance sort, each result's relevance score SHALL be 
    >= the next result's score.
    """
    # Skip empty queries
    assume(query.strip())
    
    # Ensure all news have publish_time for time sorting
    if sort_by == "time":
        assume(all(news.get("publish_time") is not None for news in news_list))
    
    async def run_test():
        service, mock_es_client = create_mock_news_service()
        
        # Mock the get_news method
        async def mock_get_news(news_id):
            for news_data in news_list:
                if news_data["id"] == news_id:
                    news = News(**news_data)
                    return news
            return None
        
        service.get_news = mock_get_news
        
        # Create mock response with proper sorting
        filtered_news = []
        for news in news_list:
            query_lower = query.lower()
            title_lower = news.get("title", "").lower()
            content_lower = news.get("content", "").lower()
            
            if query_lower in title_lower or query_lower in content_lower:
                filtered_news.append(news)
        
        # Sort the filtered news based on sort_by parameter
        if sort_by == "time":
            filtered_news.sort(key=lambda x: x.get("publish_time", datetime.min), reverse=True)
        
        hits = []
        for i, news in enumerate(filtered_news):
            score = len(filtered_news) - i if sort_by == "relevance" else 1.0  # Descending scores for relevance
            hits.append({
                "_source": news,
                "_score": score,
                "highlight": {}
            })
        
        mock_response = {
            "hits": {
                "hits": hits,
                "total": {"value": len(hits)}
            },
            "took": 10
        }
        
        mock_es_client.search.return_value = mock_response
        
        # Execute search
        result = await service.search_news(
            query=query,
            page=1,
            size=20,
            sort_by=sort_by
        )
        
        # Verify sorting consistency
        items = result["items"]
        if len(items) > 1:
            if sort_by == "time":
                # Check time ordering (descending)
                for i in range(len(items) - 1):
                    current_time = items[i]["publish_time"]
                    next_time = items[i + 1]["publish_time"]
                    
                    if current_time and next_time:
                        assert current_time >= next_time, \
                            f"Time sort order violated: {current_time} should be >= {next_time}"
            
            else:  # relevance
                # Check score ordering (descending)
                for i in range(len(items) - 1):
                    current_score = items[i]["score"]
                    next_score = items[i + 1]["score"]
                    
                    assert current_score >= next_score, \
                        f"Relevance sort order violated: {current_score} should be >= {next_score}"
    
    # Run the async test
    asyncio.run(run_test())