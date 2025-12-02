"""
News page parser for extracting article content from HTML.
Requirements: 1.2
"""
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Tag


@dataclass
class ParsedNews:
    """Parsed news article data."""
    title: str
    content: str
    publish_time: Optional[datetime] = None
    author: Optional[str] = None
    location: Optional[str] = None
    images: List[str] = field(default_factory=list)
    videos: List[str] = field(default_factory=list)
    source_url: str = ""
    source_name: str = ""
    
    def is_valid(self) -> bool:
        """Check if the parsed news has required fields."""
        return bool(self.title and self.content)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "content": self.content,
            "publish_time": self.publish_time.isoformat() if self.publish_time else None,
            "author": self.author,
            "location": self.location,
            "images": self.images,
            "videos": self.videos,
            "source_url": self.source_url,
            "source_name": self.source_name,
        }


@dataclass
class ParserConfig:
    """Configuration for a specific news source parser."""
    source_name: str
    title_selectors: List[str] = field(default_factory=lambda: [
        "h1.title", "h1.article-title", "h1.news-title",
        ".article-header h1", ".post-title", "h1",
    ])
    content_selectors: List[str] = field(default_factory=lambda: [
        ".article-content", ".article-body", ".news-content",
        ".post-content", ".entry-content", "article",
    ])
    author_selectors: List[str] = field(default_factory=lambda: [
        ".author", ".article-author", ".byline",
        "[rel='author']", ".writer",
    ])
    time_selectors: List[str] = field(default_factory=lambda: [
        "time", ".publish-time", ".article-time",
        ".date", ".datetime", "[datetime]",
    ])
    location_selectors: List[str] = field(default_factory=lambda: [
        ".location", ".article-location", ".dateline",
    ])
    image_selectors: List[str] = field(default_factory=lambda: [
        ".article-content img", ".article-body img",
        "article img", ".post-content img",
    ])
    video_selectors: List[str] = field(default_factory=lambda: [
        "video source", "iframe[src*='youtube']",
        "iframe[src*='youku']", "iframe[src*='bilibili']",
    ])
    # Elements to remove before parsing
    remove_selectors: List[str] = field(default_factory=lambda: [
        "script", "style", "nav", "header", "footer",
        ".advertisement", ".ad", ".sidebar", ".related",
        ".comments", ".share", ".social",
    ])
    # Time format patterns
    time_formats: List[str] = field(default_factory=lambda: [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d",
        "%Y年%m月%d日 %H:%M:%S",
        "%Y年%m月%d日 %H:%M",
        "%Y年%m月%d日",
    ])


class NewsParser:
    """
    HTML parser for extracting news article content.
    Requirements: 1.2
    """
    
    # Default parser configurations for supported sources
    SOURCE_CONFIGS: Dict[str, ParserConfig] = {
        "default": ParserConfig(source_name="default"),
        "people.com.cn": ParserConfig(
            source_name="人民网",
            title_selectors=["h1.title", "h1", ".article-title"],
            content_selectors=[".article", ".text", "#rwb_zw"],
        ),
        "xinhuanet.com": ParserConfig(
            source_name="新华网",
            title_selectors=["#title", "h1", ".article-title"],
            content_selectors=["#detail", ".article", "#p-detail"],
        ),
        "gmw.cn": ParserConfig(
            source_name="光明网",
            title_selectors=["h1", ".article-title"],
            content_selectors=[".article-content", ".u-mainText"],
        ),
        "cnr.cn": ParserConfig(
            source_name="央广网",
            title_selectors=["h1", ".article-title"],
            content_selectors=[".article-content", ".TRS_Editor"],
        ),
        "chinadaily.com.cn": ParserConfig(
            source_name="中国日报",
            title_selectors=["h1", ".article-title"],
            content_selectors=["#Content", ".article-content"],
        ),
        "chinanews.com": ParserConfig(
            source_name="中国新闻网",
            title_selectors=["h1", ".content_title"],
            content_selectors=[".left_zw", ".content_desc"],
        ),
        "81.cn": ParserConfig(
            source_name="中国军网",
            title_selectors=["h1", ".article-title"],
            content_selectors=[".article-content", ".TRS_Editor"],
        ),
        "bbc.com": ParserConfig(
            source_name="BBC",
            title_selectors=["h1", "[data-component='headline-block']"],
            content_selectors=["article", "[data-component='text-block']"],
        ),
        "nytimes.com": ParserConfig(
            source_name="纽约时报",
            title_selectors=["h1", ".headline"],
            content_selectors=["article", ".story-body"],
        ),
        "reuters.com": ParserConfig(
            source_name="路透社",
            title_selectors=["h1", ".article-header__title"],
            content_selectors=["article", ".article-body__content"],
        ),
    }
    
    def __init__(self, config: Optional[ParserConfig] = None):
        self.config = config or ParserConfig(source_name="default")
    
    @classmethod
    def for_source(cls, url: str) -> "NewsParser":
        """
        Create a parser configured for a specific news source.
        
        Args:
            url: The news source URL
            
        Returns:
            Configured NewsParser instance
        """
        domain = urlparse(url).netloc.lower()
        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]
        
        # Find matching config
        for source_domain, config in cls.SOURCE_CONFIGS.items():
            if source_domain in domain:
                return cls(config)
        
        return cls(cls.SOURCE_CONFIGS["default"])
    
    def parse(self, html: str, url: str) -> ParsedNews:
        """
        Parse HTML content to extract news article data.
        
        Args:
            html: The HTML content
            url: The source URL
            
        Returns:
            ParsedNews object with extracted data
        """
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove unwanted elements
        self._remove_elements(soup)
        
        # Extract fields
        title = self._extract_title(soup)
        content = self._extract_content(soup)
        author = self._extract_author(soup)
        publish_time = self._extract_time(soup)
        location = self._extract_location(soup)
        images = self._extract_images(soup, url)
        videos = self._extract_videos(soup, url)
        
        return ParsedNews(
            title=title,
            content=content,
            publish_time=publish_time,
            author=author,
            location=location,
            images=images,
            videos=videos,
            source_url=url,
            source_name=self.config.source_name,
        )
    
    def _remove_elements(self, soup: BeautifulSoup) -> None:
        """Remove unwanted elements from the soup."""
        for selector in self.config.remove_selectors:
            for element in soup.select(selector):
                element.decompose()
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract article title."""
        for selector in self.config.title_selectors:
            element = soup.select_one(selector)
            if element:
                title = element.get_text(strip=True)
                if title:
                    return title
        return ""
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract article content."""
        for selector in self.config.content_selectors:
            element = soup.select_one(selector)
            if element:
                # Get text with paragraph separation
                paragraphs = []
                for p in element.find_all(["p", "div"], recursive=True):
                    text = p.get_text(strip=True)
                    if text and len(text) > 10:  # Filter out short fragments
                        paragraphs.append(text)
                
                if paragraphs:
                    return "\n\n".join(paragraphs)
                
                # Fallback to full text
                content = element.get_text(separator="\n", strip=True)
                if content:
                    return content
        return ""
    
    def _extract_author(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract article author."""
        for selector in self.config.author_selectors:
            element = soup.select_one(selector)
            if element:
                author = element.get_text(strip=True)
                if author:
                    # Clean up common prefixes
                    author = re.sub(r"^(作者|记者|编辑|By|Author)[：:]\s*", "", author)
                    return author[:100]  # Limit length
        return None
    
    def _extract_time(self, soup: BeautifulSoup) -> Optional[datetime]:
        """Extract publish time."""
        for selector in self.config.time_selectors:
            element = soup.select_one(selector)
            if element:
                # Try datetime attribute first
                datetime_attr = element.get("datetime")
                if datetime_attr:
                    parsed = self._parse_datetime(datetime_attr)
                    if parsed:
                        return parsed
                
                # Try text content
                time_text = element.get_text(strip=True)
                if time_text:
                    parsed = self._parse_datetime(time_text)
                    if parsed:
                        return parsed
        
        # Try to find time in meta tags
        for meta in soup.find_all("meta"):
            prop = meta.get("property", "") or meta.get("name", "")
            if "time" in prop.lower() or "date" in prop.lower():
                content = meta.get("content", "")
                if content:
                    parsed = self._parse_datetime(content)
                    if parsed:
                        return parsed
        
        return None
    
    def _parse_datetime(self, text: str) -> Optional[datetime]:
        """Parse datetime from text."""
        if not text:
            return None
        
        # Clean up text
        text = text.strip()
        
        # Try ISO format first
        try:
            # Handle ISO format with timezone
            if "T" in text:
                text = text.replace("Z", "+00:00")
                if "+" in text or text.endswith("Z"):
                    from datetime import timezone
                    return datetime.fromisoformat(text.replace("Z", "+00:00"))
                return datetime.fromisoformat(text)
        except ValueError:
            pass
        
        # Try configured formats
        for fmt in self.config.time_formats:
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        
        # Try to extract date pattern
        date_pattern = r"(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})"
        match = re.search(date_pattern, text)
        if match:
            try:
                year, month, day = map(int, match.groups())
                return datetime(year, month, day)
            except ValueError:
                pass
        
        return None
    
    def _extract_location(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract article location/dateline."""
        for selector in self.config.location_selectors:
            element = soup.select_one(selector)
            if element:
                location = element.get_text(strip=True)
                if location:
                    return location[:200]
        
        # Try to extract from content beginning (common pattern: "北京1月1日电")
        content_elem = None
        for selector in self.config.content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                break
        
        if content_elem:
            first_text = content_elem.get_text()[:100]
            location_pattern = r"^([^，。]+?)(电|讯|报道)"
            match = re.match(location_pattern, first_text.strip())
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract image URLs from article."""
        images = []
        seen = set()
        
        for selector in self.config.image_selectors:
            for img in soup.select(selector):
                src = img.get("src") or img.get("data-src")
                if src:
                    # Convert to absolute URL
                    abs_url = urljoin(base_url, src)
                    if abs_url not in seen:
                        seen.add(abs_url)
                        images.append(abs_url)
        
        return images
    
    def _extract_videos(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract video URLs from article."""
        videos = []
        seen = set()
        
        for selector in self.config.video_selectors:
            for elem in soup.select(selector):
                src = elem.get("src")
                if src:
                    abs_url = urljoin(base_url, src)
                    if abs_url not in seen:
                        seen.add(abs_url)
                        videos.append(abs_url)
        
        return videos
