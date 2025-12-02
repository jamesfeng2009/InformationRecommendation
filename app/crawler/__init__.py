"""
News crawler module for collecting news from various sources.
Requirements: 1.1, 1.2, 1.3, 1.5, 1.6, 1.7
"""
from app.crawler.base import BaseCrawler, CrawlerConfig, CrawlResult, CrawlerStatus
from app.crawler.parser import NewsParser, ParsedNews, ParserConfig
from app.crawler.cleaner import ContentCleaner, CleanerConfig
from app.crawler.deduplication import DeduplicationService, DeduplicationConfig

__all__ = [
    "BaseCrawler",
    "CrawlerConfig",
    "CrawlResult",
    "CrawlerStatus",
    "NewsParser",
    "ParsedNews",
    "ParserConfig",
    "ContentCleaner",
    "CleanerConfig",
    "DeduplicationService",
    "DeduplicationConfig",
]
