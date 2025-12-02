import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Pattern


@dataclass
class CleanerConfig:
    """Configuration for content cleaning."""
    # Patterns for advertisement links
    ad_patterns: List[str] = field(default_factory=lambda: [
        r"https?://[^\s]*?(ad|ads|advert|advertisement|banner|click|track)[^\s]*",
        r"https?://[^\s]*?(doubleclick|googlesyndication|googleadservices)[^\s]*",
        r"https?://[^\s]*?(taobao|tmall|jd|amazon|affiliate)[^\s]*",
    ])
    
    # Patterns for navigation text
    nav_patterns: List[str] = field(default_factory=lambda: [
        r"(首页|返回|上一页|下一页|更多|查看更多|阅读更多)",
        r"(Home|Back|Previous|Next|More|Read More)",
        r"(分享到|转发|评论|点赞|收藏)",
        r"(Share|Comment|Like|Bookmark)",
        r"(版权所有|Copyright|All Rights Reserved)",
        r"(关注我们|Follow Us|订阅|Subscribe)",
    ])
    
    # Patterns for noise content
    noise_patterns: List[str] = field(default_factory=lambda: [
        r"\[.*?广告.*?\]",
        r"【.*?推广.*?】",
        r"点击.*?了解更多",
        r"扫码.*?关注",
        r"责任编辑[：:]\s*\S+",
        r"来源[：:]\s*\S+\s*$",
    ])
    
    # Minimum content length after cleaning
    min_content_length: int = 50
    
    # Maximum consecutive whitespace
    max_consecutive_whitespace: int = 2


class ContentCleaner:
    """
    Cleans news content by removing ads, navigation, and sensitive words.
    Requirements: 1.3, 14.2
    """
    
    def __init__(
        self,
        config: Optional[CleanerConfig] = None,
        sensitive_words: Optional[Set[str]] = None,
    ):
        self.config = config or CleanerConfig()
        self.sensitive_words = sensitive_words or set()
        
        # Compile patterns for efficiency
        self._ad_patterns: List[Pattern] = [
            re.compile(p, re.IGNORECASE) for p in self.config.ad_patterns
        ]
        self._nav_patterns: List[Pattern] = [
            re.compile(p, re.IGNORECASE) for p in self.config.nav_patterns
        ]
        self._noise_patterns: List[Pattern] = [
            re.compile(p, re.IGNORECASE | re.MULTILINE) 
            for p in self.config.noise_patterns
        ]
    
    def update_sensitive_words(self, words: Set[str]) -> None:
        """Update the sensitive word list."""
        self.sensitive_words = words
    
    def add_sensitive_words(self, words: Set[str]) -> None:
        """Add words to the sensitive word list."""
        self.sensitive_words.update(words)
    
    def clean(self, content: str) -> str:
        """
        Clean content by removing ads, navigation, and noise.
        
        Args:
            content: The raw content to clean
            
        Returns:
            Cleaned content
        """
        if not content:
            return ""
        
        # Remove advertisement links
        content = self._remove_ad_links(content)
        
        # Remove navigation elements
        content = self._remove_navigation(content)
        
        # Remove noise patterns
        content = self._remove_noise(content)
        
        # Apply sensitive word filtering
        content = self._filter_sensitive_words(content)
        
        # Normalize whitespace
        content = self._normalize_whitespace(content)
        
        return content.strip()
    
    def clean_title(self, title: str) -> str:
        """
        Clean a news title.
        
        Args:
            title: The raw title
            
        Returns:
            Cleaned title
        """
        if not title:
            return ""
        
        # Remove common title noise
        title = re.sub(r"\s*[-_|]\s*[^-_|]+$", "", title)  # Remove site name suffix
        title = re.sub(r"^\s*[【\[].+?[】\]]\s*", "", title)  # Remove prefix tags
        
        # Apply sensitive word filtering
        title = self._filter_sensitive_words(title)
        
        return title.strip()
    
    def _remove_ad_links(self, content: str) -> str:
        """Remove advertisement links from content."""
        for pattern in self._ad_patterns:
            content = pattern.sub("", content)
        return content
    
    def _remove_navigation(self, content: str) -> str:
        """Remove navigation text from content."""
        lines = content.split("\n")
        cleaned_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # Skip empty lines
            if not line_stripped:
                cleaned_lines.append(line)
                continue
            
            # Check if line is primarily navigation
            is_nav = False
            for pattern in self._nav_patterns:
                if pattern.search(line_stripped):
                    # Only remove if the line is short (likely just navigation)
                    if len(line_stripped) < 50:
                        is_nav = True
                        break
            
            if not is_nav:
                cleaned_lines.append(line)
        
        return "\n".join(cleaned_lines)
    
    def _remove_noise(self, content: str) -> str:
        """Remove noise patterns from content."""
        for pattern in self._noise_patterns:
            content = pattern.sub("", content)
        return content
    
    def _filter_sensitive_words(self, content: str) -> str:
        """
        Filter sensitive words from content.
        Replaces sensitive words with asterisks.
        
        Args:
            content: The content to filter
            
        Returns:
            Filtered content
        """
        if not self.sensitive_words:
            return content
        
        for word in self.sensitive_words:
            if word in content:
                # Replace with asterisks of same length
                replacement = "*" * len(word)
                content = content.replace(word, replacement)
        
        return content
    
    def _normalize_whitespace(self, content: str) -> str:
        """Normalize whitespace in content."""
        # Replace multiple spaces with single space
        content = re.sub(r"[ \t]+", " ", content)
        
        # Replace multiple newlines with max allowed
        max_newlines = "\n" * self.config.max_consecutive_whitespace
        content = re.sub(r"\n{3,}", max_newlines, content)
        
        return content
    
    def contains_sensitive_word(self, content: str) -> bool:
        """
        Check if content contains any sensitive words.
        
        Args:
            content: The content to check
            
        Returns:
            True if sensitive words found
        """
        if not self.sensitive_words:
            return False
        
        for word in self.sensitive_words:
            if word in content:
                return True
        
        return False
    
    def get_sensitive_words_found(self, content: str) -> List[str]:
        """
        Get list of sensitive words found in content.
        
        Args:
            content: The content to check
            
        Returns:
            List of sensitive words found
        """
        found = []
        for word in self.sensitive_words:
            if word in content:
                found.append(word)
        return found
    
    def is_valid_content(self, content: str) -> bool:
        """
        Check if content is valid after cleaning.
        
        Args:
            content: The cleaned content
            
        Returns:
            True if content meets minimum requirements
        """
        if not content:
            return False
        
        # Check minimum length
        if len(content.strip()) < self.config.min_content_length:
            return False
        
        return True
