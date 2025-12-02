"""
Base crawler framework with rate limiting, user-agent rotation, and proxy support.
Requirements: 1.1, 1.7
"""
import asyncio
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx


class CrawlerStatus(str, Enum):
    """Crawler task status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class CrawlerConfig:
    """Configuration for crawler behavior."""
    # Rate limiting
    requests_per_second: float = 1.0
    min_delay_seconds: float = 0.5
    max_delay_seconds: float = 2.0
    
    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: float = 5.0
    
    # Timeout settings
    request_timeout_seconds: float = 30.0
    connect_timeout_seconds: float = 10.0
    
    # Proxy settings
    proxies: List[str] = field(default_factory=list)
    rotate_proxies: bool = True
    
    # User agent rotation
    user_agents: List[str] = field(default_factory=lambda: [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ])
    rotate_user_agents: bool = True
    
    # Concurrent requests
    max_concurrent_requests: int = 5


@dataclass
class CrawlResult:
    """Result of a single crawl operation."""
    url: str
    success: bool
    status_code: Optional[int] = None
    content: Optional[str] = None
    error: Optional[str] = None
    crawl_time: datetime = field(default_factory=datetime.utcnow)
    response_time_ms: Optional[float] = None


class RateLimiter:
    """Token bucket rate limiter for controlling request frequency."""
    
    def __init__(self, requests_per_second: float):
        self.requests_per_second = requests_per_second
        self.tokens = requests_per_second
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens = min(
                self.requests_per_second,
                self.tokens + elapsed * self.requests_per_second
            )
            self.last_update = now
            
            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.requests_per_second
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


class ProxyRotator:
    """Rotates through a list of proxies."""
    
    def __init__(self, proxies: List[str]):
        self.proxies = proxies
        self._index = 0
        self._lock = asyncio.Lock()
    
    async def get_proxy(self) -> Optional[str]:
        """Get the next proxy in rotation."""
        if not self.proxies:
            return None
        
        async with self._lock:
            proxy = self.proxies[self._index]
            self._index = (self._index + 1) % len(self.proxies)
            return proxy


class UserAgentRotator:
    """Rotates through a list of user agents."""
    
    def __init__(self, user_agents: List[str], rotate: bool = True):
        self.user_agents = user_agents
        self.rotate = rotate
        self._index = 0
    
    def get_user_agent(self) -> str:
        """Get a user agent string."""
        if not self.user_agents:
            return "Mozilla/5.0 (compatible; NewsCrawler/1.0)"
        
        if self.rotate:
            return random.choice(self.user_agents)
        
        ua = self.user_agents[self._index]
        self._index = (self._index + 1) % len(self.user_agents)
        return ua


class BaseCrawler(ABC):
    """
    Base crawler class with rate limiting, user-agent rotation, and proxy support.
    Requirements: 1.1, 1.7
    """
    
    def __init__(self, config: Optional[CrawlerConfig] = None):
        self.config = config or CrawlerConfig()
        self.rate_limiter = RateLimiter(self.config.requests_per_second)
        self.proxy_rotator = ProxyRotator(self.config.proxies)
        self.ua_rotator = UserAgentRotator(
            self.config.user_agents,
            self.config.rotate_user_agents
        )
        self._status = CrawlerStatus.PENDING
        self._stop_requested = False
        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
    
    @property
    def status(self) -> CrawlerStatus:
        """Get current crawler status."""
        return self._status
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def start(self) -> None:
        """Initialize the crawler."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=self.config.connect_timeout_seconds,
                read=self.config.request_timeout_seconds,
                write=self.config.request_timeout_seconds,
                pool=self.config.request_timeout_seconds,
            ),
            follow_redirects=True,
        )
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        self._status = CrawlerStatus.RUNNING
        self._stop_requested = False
    
    async def close(self) -> None:
        """Close the crawler and release resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._status = CrawlerStatus.STOPPED
    
    def request_stop(self) -> None:
        """Request the crawler to stop."""
        self._stop_requested = True
    
    async def _get_headers(self) -> Dict[str, str]:
        """Get request headers with rotated user agent."""
        return {
            "User-Agent": self.ua_rotator.get_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }
    
    async def _get_proxy_config(self) -> Optional[str]:
        """Get proxy configuration for request."""
        if self.config.rotate_proxies:
            return await self.proxy_rotator.get_proxy()
        return self.config.proxies[0] if self.config.proxies else None
    
    async def fetch(self, url: str) -> CrawlResult:
        """
        Fetch a URL with rate limiting, retries, and error handling.
        
        Args:
            url: The URL to fetch
            
        Returns:
            CrawlResult with the response or error information
        """
        if not self._client:
            raise RuntimeError("Crawler not started. Use 'async with' or call start()")
        
        if self._stop_requested:
            return CrawlResult(
                url=url,
                success=False,
                error="Crawler stop requested"
            )
        
        async with self._semaphore:
            for attempt in range(self.config.max_retries):
                try:
                    # Apply rate limiting
                    await self.rate_limiter.acquire()
                    
                    # Add random delay
                    delay = random.uniform(
                        self.config.min_delay_seconds,
                        self.config.max_delay_seconds
                    )
                    await asyncio.sleep(delay)
                    
                    # Get headers and proxy
                    headers = await self._get_headers()
                    proxy = await self._get_proxy_config()
                    
                    # Make request
                    start_time = time.monotonic()
                    
                    if proxy:
                        # Create a new client with proxy for this request
                        async with httpx.AsyncClient(
                            proxies=proxy,
                            timeout=self._client.timeout,
                            follow_redirects=True,
                        ) as proxy_client:
                            response = await proxy_client.get(url, headers=headers)
                    else:
                        response = await self._client.get(url, headers=headers)
                    
                    response_time = (time.monotonic() - start_time) * 1000
                    
                    if response.status_code == 200:
                        return CrawlResult(
                            url=url,
                            success=True,
                            status_code=response.status_code,
                            content=response.text,
                            response_time_ms=response_time,
                        )
                    else:
                        # Non-200 status, might retry
                        if attempt < self.config.max_retries - 1:
                            await asyncio.sleep(self.config.retry_delay_seconds)
                            continue
                        
                        return CrawlResult(
                            url=url,
                            success=False,
                            status_code=response.status_code,
                            error=f"HTTP {response.status_code}",
                            response_time_ms=response_time,
                        )
                
                except httpx.TimeoutException as e:
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(self.config.retry_delay_seconds)
                        continue
                    return CrawlResult(
                        url=url,
                        success=False,
                        error=f"Timeout: {str(e)}",
                    )
                
                except httpx.RequestError as e:
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(self.config.retry_delay_seconds)
                        continue
                    return CrawlResult(
                        url=url,
                        success=False,
                        error=f"Request error: {str(e)}",
                    )
                
                except Exception as e:
                    return CrawlResult(
                        url=url,
                        success=False,
                        error=f"Unexpected error: {str(e)}",
                    )
        
        return CrawlResult(
            url=url,
            success=False,
            error="Max retries exceeded",
        )
    
    async def crawl_urls(self, urls: List[str]) -> List[CrawlResult]:
        """
        Crawl multiple URLs concurrently.
        
        Args:
            urls: List of URLs to crawl
            
        Returns:
            List of CrawlResults
        """
        tasks = [self.fetch(url) for url in urls]
        return await asyncio.gather(*tasks)
    
    @abstractmethod
    async def extract_urls(self, base_url: str) -> List[str]:
        """
        Extract news article URLs from a source page.
        Must be implemented by subclasses.
        
        Args:
            base_url: The source page URL
            
        Returns:
            List of article URLs
        """
        pass
    
    @abstractmethod
    async def parse_article(self, url: str, content: str) -> Dict[str, Any]:
        """
        Parse article content from HTML.
        Must be implemented by subclasses.
        
        Args:
            url: The article URL
            content: The HTML content
            
        Returns:
            Parsed article data
        """
        pass
