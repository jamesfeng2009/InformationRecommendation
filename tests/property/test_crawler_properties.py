"""
Property-based tests for news crawler.

**Feature: intelligent-recommendation-system, Property 1: News Extraction Completeness**
**Validates: Requirements 1.2**

**Feature: intelligent-recommendation-system, Property 2: Content Cleaning Preservation**
**Validates: Requirements 1.3**

**Feature: intelligent-recommendation-system, Property 3: Deduplication Idempotence**
**Validates: Requirements 1.5**
"""
import re
from datetime import datetime
from typing import List, Optional

import pytest
from hypothesis import given, strategies as st, settings, assume

from app.crawler.parser import NewsParser, ParsedNews, ParserConfig
from app.crawler.cleaner import ContentCleaner, CleanerConfig
from app.crawler.deduplication import DeduplicationService, DeduplicationConfig


# =============================================================================
# Strategies for generating test data
# =============================================================================

# Strategy for generating valid HTML news pages
def html_news_page_strategy():
    """Generate valid HTML news pages with required fields."""
    return st.builds(
        _build_html_page,
        title=st.text(min_size=5, max_size=200).filter(lambda x: x.strip() != ""),
        content=st.text(min_size=50, max_size=2000).filter(lambda x: x.strip() != ""),
        publish_time=st.one_of(
            st.none(),
            st.datetimes(
                min_value=datetime(2020, 1, 1),
                max_value=datetime(2025, 12, 31)
            )
        ),
        author=st.one_of(st.none(), st.text(min_size=2, max_size=50).filter(lambda x: x.strip() != "")),
        images=st.lists(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-./", min_size=5, max_size=50)
            .filter(lambda x: x.strip() != "" and "\n" not in x and " " not in x),
            max_size=5
        ),
        videos=st.lists(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-./", min_size=5, max_size=50)
            .filter(lambda x: x.strip() != "" and "\n" not in x and " " not in x),
            max_size=3
        ),
    )


def _build_html_page(
    title: str,
    content: str,
    publish_time: Optional[datetime],
    author: Optional[str],
    images: List[str],
    videos: List[str],
) -> dict:
    """Build an HTML page with the given content."""
    # Escape HTML special characters in content
    safe_title = title.replace("<", "&lt;").replace(">", "&gt;").replace("&", "&amp;")
    safe_content = content.replace("<", "&lt;").replace(">", "&gt;").replace("&", "&amp;")
    
    time_html = ""
    if publish_time:
        time_html = f'<time datetime="{publish_time.isoformat()}">{publish_time.strftime("%Y-%m-%d %H:%M")}</time>'
    
    author_html = ""
    if author:
        safe_author = author.replace("<", "&lt;").replace(">", "&gt;")
        author_html = f'<span class="author">{safe_author}</span>'
    
    images_html = "\n".join(f'<img src="{img}" />' for img in images)
    videos_html = "\n".join(f'<video><source src="{vid}" /></video>' for vid in videos)
    
    # Build paragraphs from content
    paragraphs = safe_content.split("\n")
    content_html = "\n".join(f"<p>{p}</p>" for p in paragraphs if p.strip())
    if not content_html:
        content_html = f"<p>{safe_content}</p>"
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head><title>{safe_title}</title></head>
    <body>
        <article>
            <h1 class="title">{safe_title}</h1>
            {author_html}
            {time_html}
            <div class="article-content">
                {content_html}
                {images_html}
                {videos_html}
            </div>
        </article>
    </body>
    </html>
    """
    
    return {
        "html": html,
        "expected_title": title,
        "expected_content": content,
        "expected_publish_time": publish_time,
        "expected_author": author,
        "expected_images": images,
        "expected_videos": videos,
    }


# Strategy for content with noise
def content_with_noise_strategy():
    """Generate content with various types of noise."""
    return st.builds(
        _build_content_with_noise,
        clean_content=st.text(min_size=100, max_size=1000).filter(lambda x: x.strip() != ""),
        ad_links=st.lists(
            st.sampled_from([
                "https://ad.example.com/click",
                "https://track.doubleclick.net/ad",
                "https://googlesyndication.com/banner",
            ]),
            max_size=3
        ),
        nav_elements=st.lists(
            st.sampled_from([
                "首页", "返回", "更多", "分享到", "点赞",
                "Home", "Back", "More", "Share", "Like",
            ]),
            max_size=3
        ),
    )


def _build_content_with_noise(
    clean_content: str,
    ad_links: List[str],
    nav_elements: List[str],
) -> dict:
    """Build content with noise elements."""
    noisy_content = clean_content
    
    # Add ad links
    for link in ad_links:
        noisy_content += f"\n{link}"
    
    # Add navigation elements (as short lines)
    for nav in nav_elements:
        noisy_content += f"\n{nav}"
    
    return {
        "noisy_content": noisy_content,
        "clean_content": clean_content,
        "ad_links": ad_links,
        "nav_elements": nav_elements,
    }


# Strategy for news items for deduplication
def news_item_strategy():
    """Generate news items for deduplication testing."""
    return st.builds(
        lambda title, content: {"title": title, "content": content},
        title=st.text(min_size=10, max_size=200).filter(lambda x: x.strip() != ""),
        content=st.text(min_size=100, max_size=2000).filter(lambda x: x.strip() != ""),
    )


# =============================================================================
# Property 1: News Extraction Completeness
# =============================================================================

class TestNewsExtractionCompleteness:
    """
    Property tests for news extraction completeness.
    
    **Feature: intelligent-recommendation-system, Property 1: News Extraction Completeness**
    **Validates: Requirements 1.2**
    """

    @settings(max_examples=100, deadline=None)
    @given(page_data=html_news_page_strategy())
    def test_parser_extracts_required_fields(self, page_data: dict):
        """
        **Feature: intelligent-recommendation-system, Property 1: News Extraction Completeness**
        **Validates: Requirements 1.2**
        
        For any valid HTML news page from a supported source, the parser SHALL extract
        all required fields (title, content) as non-empty values.
        """
        # Arrange
        parser = NewsParser()
        html = page_data["html"]
        url = "https://example.com/news/article"
        
        # Act
        result = parser.parse(html, url)
        
        # Assert - Required fields must be non-empty
        assert result.title != "", "Title must be non-empty"
        assert result.content != "", "Content must be non-empty"
        assert result.source_url == url

    @settings(max_examples=100, deadline=None)
    @given(page_data=html_news_page_strategy())
    def test_parser_extracts_optional_fields_as_valid_types(self, page_data: dict):
        """
        **Feature: intelligent-recommendation-system, Property 1: News Extraction Completeness**
        **Validates: Requirements 1.2**
        
        For any valid HTML news page, optional fields (images, videos, author, location)
        SHALL be valid types (list for images/videos, string or None for author/location).
        """
        # Arrange
        parser = NewsParser()
        html = page_data["html"]
        url = "https://example.com/news/article"
        
        # Act
        result = parser.parse(html, url)
        
        # Assert - Optional fields have correct types
        assert isinstance(result.images, list), "Images must be a list"
        assert isinstance(result.videos, list), "Videos must be a list"
        assert result.author is None or isinstance(result.author, str), "Author must be string or None"
        assert result.location is None or isinstance(result.location, str), "Location must be string or None"
        assert result.publish_time is None or isinstance(result.publish_time, datetime), "Publish time must be datetime or None"

    @settings(max_examples=100, deadline=None)
    @given(page_data=html_news_page_strategy())
    def test_parsed_news_is_valid(self, page_data: dict):
        """
        **Feature: intelligent-recommendation-system, Property 1: News Extraction Completeness**
        **Validates: Requirements 1.2**
        
        For any valid HTML news page with title and content, the parsed result
        SHALL pass the is_valid() check.
        """
        # Arrange
        parser = NewsParser()
        html = page_data["html"]
        url = "https://example.com/news/article"
        
        # Act
        result = parser.parse(html, url)
        
        # Assert
        assert result.is_valid(), "Parsed news with title and content should be valid"

    @settings(max_examples=50, deadline=None)
    @given(page_data=html_news_page_strategy())
    def test_parser_extracts_images_from_content(self, page_data: dict):
        """
        **Feature: intelligent-recommendation-system, Property 1: News Extraction Completeness**
        **Validates: Requirements 1.2**
        
        For any HTML page with images in the article content, the parser SHALL
        extract image URLs as a list.
        """
        # Arrange
        parser = NewsParser()
        html = page_data["html"]
        url = "https://example.com/news/article"
        expected_images = page_data["expected_images"]
        
        # Act
        result = parser.parse(html, url)
        
        # Assert - All expected images should be found (as absolute URLs)
        for img in expected_images:
            # The parser converts to absolute URLs
            found = any(img in extracted_img for extracted_img in result.images)
            assert found or len(expected_images) == 0, f"Image {img} should be extracted"


# =============================================================================
# Property 2: Content Cleaning Preservation
# =============================================================================

class TestContentCleaningPreservation:
    """
    Property tests for content cleaning preservation.
    
    **Feature: intelligent-recommendation-system, Property 2: Content Cleaning Preservation**
    **Validates: Requirements 1.3**
    """

    @settings(max_examples=100, deadline=None)
    @given(data=content_with_noise_strategy())
    def test_cleaner_removes_ad_links(self, data: dict):
        """
        **Feature: intelligent-recommendation-system, Property 2: Content Cleaning Preservation**
        **Validates: Requirements 1.3**
        
        For any content containing advertisement links, the cleaner SHALL remove
        those links while preserving the semantic content.
        """
        # Arrange
        cleaner = ContentCleaner()
        noisy_content = data["noisy_content"]
        ad_links = data["ad_links"]
        
        # Act
        cleaned = cleaner.clean(noisy_content)
        
        # Assert - Ad links should be removed
        for link in ad_links:
            assert link not in cleaned, f"Ad link {link} should be removed"

    @settings(max_examples=100, deadline=None)
    @given(data=content_with_noise_strategy())
    def test_cleaner_preserves_semantic_content(self, data: dict):
        """
        **Feature: intelligent-recommendation-system, Property 2: Content Cleaning Preservation**
        **Validates: Requirements 1.3**
        
        For any content with noise, the cleaned content SHALL preserve the
        semantic meaning (core words from original content should remain).
        """
        # Arrange
        cleaner = ContentCleaner()
        noisy_content = data["noisy_content"]
        clean_content = data["clean_content"]
        
        # Act
        cleaned = cleaner.clean(noisy_content)
        
        # Assert - Core content words should be preserved
        # Extract significant words (longer than 3 chars) from clean content
        clean_words = set(w.lower() for w in re.findall(r'\w+', clean_content) if len(w) > 3)
        cleaned_words = set(w.lower() for w in re.findall(r'\w+', cleaned) if len(w) > 3)
        
        # Most significant words should be preserved
        if clean_words:
            preserved_ratio = len(clean_words & cleaned_words) / len(clean_words)
            assert preserved_ratio >= 0.8, f"At least 80% of content words should be preserved, got {preserved_ratio:.2%}"

    @settings(max_examples=100, deadline=None)
    @given(
        content=st.text(min_size=100, max_size=1000).filter(lambda x: x.strip() != ""),
        sensitive_words=st.lists(
            st.text(min_size=2, max_size=20).filter(lambda x: x.strip() != ""),
            min_size=1,
            max_size=5
        )
    )
    def test_cleaner_filters_sensitive_words(self, content: str, sensitive_words: List[str]):
        """
        **Feature: intelligent-recommendation-system, Property 2: Content Cleaning Preservation**
        **Validates: Requirements 1.3**
        
        For any content containing sensitive words, the cleaner SHALL replace
        those words with asterisks.
        """
        # Arrange
        cleaner = ContentCleaner(sensitive_words=set(sensitive_words))
        
        # Add sensitive words to content
        content_with_sensitive = content
        for word in sensitive_words:
            content_with_sensitive += f" {word} "
        
        # Act
        cleaned = cleaner.clean(content_with_sensitive)
        
        # Assert - Sensitive words should be replaced
        for word in sensitive_words:
            if word in content_with_sensitive:
                assert word not in cleaned, f"Sensitive word '{word}' should be filtered"

    @settings(max_examples=100, deadline=None)
    @given(content=st.text(min_size=50, max_size=500).filter(lambda x: x.strip() != ""))
    def test_cleaner_normalizes_whitespace(self, content: str):
        """
        **Feature: intelligent-recommendation-system, Property 2: Content Cleaning Preservation**
        **Validates: Requirements 1.3**
        
        For any content, the cleaner SHALL normalize whitespace without
        creating excessive consecutive whitespace.
        """
        # Arrange
        cleaner = ContentCleaner()
        # Add excessive whitespace
        content_with_whitespace = content.replace(" ", "    ")
        content_with_whitespace += "\n\n\n\n\n"
        
        # Act
        cleaned = cleaner.clean(content_with_whitespace)
        
        # Assert - No excessive whitespace
        assert "    " not in cleaned, "Should not have 4+ consecutive spaces"
        assert "\n\n\n" not in cleaned, "Should not have 3+ consecutive newlines"

    @settings(max_examples=100, deadline=None)
    @given(content=st.text(min_size=100, max_size=500).filter(lambda x: x.strip() != ""))
    def test_cleaning_is_idempotent(self, content: str):
        """
        **Feature: intelligent-recommendation-system, Property 2: Content Cleaning Preservation**
        **Validates: Requirements 1.3**
        
        For any content, cleaning twice SHALL produce the same result as cleaning once.
        """
        # Arrange
        cleaner = ContentCleaner()
        
        # Act
        cleaned_once = cleaner.clean(content)
        cleaned_twice = cleaner.clean(cleaned_once)
        
        # Assert
        assert cleaned_once == cleaned_twice, "Cleaning should be idempotent"


# =============================================================================
# Property 3: Deduplication Idempotence
# =============================================================================

class TestDeduplicationIdempotence:
    """
    Property tests for deduplication idempotence.
    
    **Feature: intelligent-recommendation-system, Property 3: Deduplication Idempotence**
    **Validates: Requirements 1.5**
    """

    @settings(max_examples=100, deadline=None)
    @given(items=st.lists(news_item_strategy(), min_size=1, max_size=20))
    def test_deduplication_is_idempotent(self, items: List[dict]):
        """
        **Feature: intelligent-recommendation-system, Property 3: Deduplication Idempotence**
        **Validates: Requirements 1.5**
        
        For any set of news items, applying deduplication twice SHALL produce
        the same result as applying it once.
        """
        # Arrange
        service1 = DeduplicationService()
        service2 = DeduplicationService()
        
        # Add IDs to items
        for i, item in enumerate(items):
            item["id"] = i
        
        # Act - First deduplication
        result1 = service1.deduplicate_batch(items.copy())
        
        # Act - Second deduplication on the result
        result2 = service2.deduplicate_batch(result1.copy())
        
        # Assert - Same number of items
        assert len(result1) == len(result2), "Deduplication should be idempotent"
        
        # Assert - Same content hashes
        hashes1 = {item.get("content_hash") for item in result1}
        hashes2 = {item.get("content_hash") for item in result2}
        assert hashes1 == hashes2, "Content hashes should be the same"

    @settings(max_examples=100, deadline=None)
    @given(items=st.lists(news_item_strategy(), min_size=2, max_size=10))
    def test_no_duplicates_in_result(self, items: List[dict]):
        """
        **Feature: intelligent-recommendation-system, Property 3: Deduplication Idempotence**
        **Validates: Requirements 1.5**
        
        For any set of news items after deduplication, no two items SHALL have
        content similarity above the threshold.
        """
        # Arrange
        config = DeduplicationConfig(similarity_threshold=0.8)
        service = DeduplicationService(config)
        
        # Add IDs to items
        for i, item in enumerate(items):
            item["id"] = i
        
        # Act
        result = service.deduplicate_batch(items.copy())
        
        # Assert - No two items should be too similar
        for i, item1 in enumerate(result):
            for j, item2 in enumerate(result):
                if i >= j:
                    continue
                
                similarity = service.compute_similarity(
                    item1["content"],
                    item2["content"]
                )
                assert similarity < config.similarity_threshold, \
                    f"Items {i} and {j} have similarity {similarity:.2f} >= threshold {config.similarity_threshold}"

    @settings(max_examples=100, deadline=None)
    @given(
        title=st.text(min_size=10, max_size=100).filter(lambda x: x.strip() != ""),
        content=st.text(min_size=100, max_size=500).filter(lambda x: x.strip() != "")
    )
    def test_exact_duplicate_detection(self, title: str, content: str):
        """
        **Feature: intelligent-recommendation-system, Property 3: Deduplication Idempotence**
        **Validates: Requirements 1.5**
        
        For any news item, adding the exact same item twice SHALL be detected
        as a duplicate.
        """
        # Arrange
        service = DeduplicationService()
        
        # Act - Add first item
        is_dup1, hash1, _, _ = service.is_duplicate(title, content, "item1")
        service.add_content(title, content, "item1")
        
        # Act - Try to add same item again
        is_dup2, hash2, _, _ = service.is_duplicate(title, content, "item2")
        
        # Assert
        assert is_dup1 is False, "First item should not be a duplicate"
        assert is_dup2 is True, "Second identical item should be a duplicate"
        assert hash1 == hash2, "Same content should have same hash"

    @settings(max_examples=100, deadline=None)
    @given(
        title=st.text(min_size=10, max_size=100).filter(lambda x: x.strip() != ""),
        content=st.text(min_size=100, max_size=500).filter(lambda x: x.strip() != "")
    )
    def test_hash_consistency(self, title: str, content: str):
        """
        **Feature: intelligent-recommendation-system, Property 3: Deduplication Idempotence**
        **Validates: Requirements 1.5**
        
        For any news item, computing the hash multiple times SHALL produce
        the same result.
        """
        # Arrange
        service = DeduplicationService()
        
        # Act
        hash1 = service.compute_hash(title, content)
        hash2 = service.compute_hash(title, content)
        hash3 = service.compute_hash(title, content)
        
        # Assert
        assert hash1 == hash2 == hash3, "Hash should be consistent"

    @settings(max_examples=100, deadline=None)
    @given(
        title1=st.text(min_size=10, max_size=100).filter(lambda x: x.strip() != ""),
        content1=st.text(min_size=100, max_size=500).filter(lambda x: x.strip() != ""),
        title2=st.text(min_size=10, max_size=100).filter(lambda x: x.strip() != ""),
        content2=st.text(min_size=100, max_size=500).filter(lambda x: x.strip() != "")
    )
    def test_different_content_different_hash(self, title1: str, content1: str, title2: str, content2: str):
        """
        **Feature: intelligent-recommendation-system, Property 3: Deduplication Idempotence**
        **Validates: Requirements 1.5**
        
        For any two different news items, they SHALL have different hashes
        (with high probability).
        """
        # Skip if content is the same
        assume(content1.strip().lower() != content2.strip().lower())
        assume(title1.strip().lower() != title2.strip().lower())
        
        # Arrange
        service = DeduplicationService()
        
        # Act
        hash1 = service.compute_hash(title1, content1)
        hash2 = service.compute_hash(title2, content2)
        
        # Assert
        assert hash1 != hash2, "Different content should have different hashes"

    @settings(max_examples=50, deadline=None)
    @given(
        base_content=st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz ",
            min_size=200,
            max_size=500
        ).filter(lambda x: len(x.split()) >= 20)
    )
    def test_similar_content_detected(self, base_content: str):
        """
        **Feature: intelligent-recommendation-system, Property 3: Deduplication Idempotence**
        **Validates: Requirements 1.5**
        
        For any content with sufficient words, a slightly modified version
        SHALL have measurable similarity.
        """
        # Arrange
        config = DeduplicationConfig(similarity_threshold=0.8)
        service = DeduplicationService(config)
        
        # Create slightly modified content (change ~10% of words)
        words = base_content.split()
        assume(len(words) >= 20)  # Ensure enough words for meaningful comparison
        
        # Modify a few words (every 10th word)
        modified_words = words.copy()
        for i in range(0, len(words), 10):
            if i < len(modified_words):
                modified_words[i] = "modified"
        modified_content = " ".join(modified_words)
        
        # Act
        similarity = service.compute_similarity(base_content, modified_content)
        
        # Assert - Similar content should have measurable similarity
        # With 10% modification, we expect at least 0.3 similarity
        assert similarity >= 0.3, f"Similar content should have similarity >= 0.3, got {similarity:.2f}"
