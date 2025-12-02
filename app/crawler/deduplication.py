"""
Deduplication service for detecting and filtering duplicate news content.
Requirements: 1.5
"""
import hashlib
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class DeduplicationConfig:
    """Configuration for deduplication."""
    # Similarity threshold (0.0 to 1.0)
    similarity_threshold: float = 0.8
    
    # Minimum content length for similarity comparison
    min_content_length: int = 100
    
    # Number of shingles (n-grams) for similarity
    shingle_size: int = 3
    
    # Use title in hash calculation
    include_title_in_hash: bool = True


class DeduplicationService:
    """
    Service for detecting and filtering duplicate news content.
    Uses content hashing and similarity comparison.
    Requirements: 1.5
    """
    
    def __init__(self, config: Optional[DeduplicationConfig] = None):
        self.config = config or DeduplicationConfig()
        self._seen_hashes: Set[str] = set()
        self._content_shingles: Dict[str, Set[int]] = {}
    
    def clear(self) -> None:
        """Clear all stored hashes and shingles."""
        self._seen_hashes.clear()
        self._content_shingles.clear()
    
    def compute_hash(self, title: str, content: str) -> str:
        """
        Compute a content hash for deduplication.
        
        Args:
            title: The news title
            content: The news content
            
        Returns:
            SHA-256 hash string
        """
        # Normalize content
        normalized_content = self._normalize_text(content)
        
        if self.config.include_title_in_hash:
            normalized_title = self._normalize_text(title)
            text_to_hash = f"{normalized_title}|{normalized_content}"
        else:
            text_to_hash = normalized_content
        
        return hashlib.sha256(text_to_hash.encode("utf-8")).hexdigest()
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        
        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        
        return text.strip()
    
    def _get_shingles(self, text: str) -> Set[int]:
        """
        Generate shingles (n-grams) from text.
        
        Args:
            text: The text to process
            
        Returns:
            Set of shingle hashes
        """
        normalized = self._normalize_text(text)
        words = normalized.split()
        
        if len(words) < self.config.shingle_size:
            # For short text, use character n-grams
            shingles = set()
            for i in range(len(normalized) - self.config.shingle_size + 1):
                shingle = normalized[i:i + self.config.shingle_size]
                shingles.add(hash(shingle))
            return shingles
        
        # Use word n-grams
        shingles = set()
        for i in range(len(words) - self.config.shingle_size + 1):
            shingle = " ".join(words[i:i + self.config.shingle_size])
            shingles.add(hash(shingle))
        
        return shingles
    
    def compute_similarity(self, content1: str, content2: str) -> float:
        """
        Compute Jaccard similarity between two texts.
        
        Args:
            content1: First text
            content2: Second text
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        shingles1 = self._get_shingles(content1)
        shingles2 = self._get_shingles(content2)
        
        if not shingles1 or not shingles2:
            return 0.0
        
        intersection = len(shingles1 & shingles2)
        union = len(shingles1 | shingles2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def is_duplicate_by_hash(self, title: str, content: str) -> Tuple[bool, str]:
        """
        Check if content is a duplicate using hash comparison.
        
        Args:
            title: The news title
            content: The news content
            
        Returns:
            Tuple of (is_duplicate, content_hash)
        """
        content_hash = self.compute_hash(title, content)
        is_dup = content_hash in self._seen_hashes
        return is_dup, content_hash
    
    def is_duplicate_by_similarity(
        self,
        content: str,
        content_id: Optional[str] = None,
    ) -> Tuple[bool, Optional[str], float]:
        """
        Check if content is similar to any stored content.
        
        Args:
            content: The news content
            content_id: Optional ID for the content
            
        Returns:
            Tuple of (is_duplicate, similar_content_id, similarity_score)
        """
        if len(content) < self.config.min_content_length:
            return False, None, 0.0
        
        new_shingles = self._get_shingles(content)
        
        max_similarity = 0.0
        most_similar_id = None
        
        for stored_id, stored_shingles in self._content_shingles.items():
            if not stored_shingles:
                continue
            
            intersection = len(new_shingles & stored_shingles)
            union = len(new_shingles | stored_shingles)
            
            if union > 0:
                similarity = intersection / union
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_id = stored_id
        
        is_dup = max_similarity >= self.config.similarity_threshold
        return is_dup, most_similar_id, max_similarity
    
    def is_duplicate(
        self,
        title: str,
        content: str,
        content_id: Optional[str] = None,
        check_similarity: bool = True,
    ) -> Tuple[bool, str, Optional[str], float]:
        """
        Check if content is a duplicate using both hash and similarity.
        
        Args:
            title: The news title
            content: The news content
            content_id: Optional ID for the content
            check_similarity: Whether to check similarity (slower)
            
        Returns:
            Tuple of (is_duplicate, content_hash, similar_id, similarity_score)
        """
        # First check by hash (fast)
        is_hash_dup, content_hash = self.is_duplicate_by_hash(title, content)
        
        if is_hash_dup:
            return True, content_hash, None, 1.0
        
        # Then check by similarity (slower)
        if check_similarity:
            is_sim_dup, similar_id, similarity = self.is_duplicate_by_similarity(
                content, content_id
            )
            if is_sim_dup:
                return True, content_hash, similar_id, similarity
        
        return False, content_hash, None, 0.0
    
    def add_content(
        self,
        title: str,
        content: str,
        content_id: str,
    ) -> str:
        """
        Add content to the deduplication index.
        
        Args:
            title: The news title
            content: The news content
            content_id: Unique ID for the content
            
        Returns:
            The content hash
        """
        content_hash = self.compute_hash(title, content)
        self._seen_hashes.add(content_hash)
        
        if len(content) >= self.config.min_content_length:
            self._content_shingles[content_id] = self._get_shingles(content)
        
        return content_hash
    
    def remove_content(self, content_id: str, content_hash: Optional[str] = None) -> None:
        """
        Remove content from the deduplication index.
        
        Args:
            content_id: The content ID to remove
            content_hash: Optional hash to remove
        """
        if content_id in self._content_shingles:
            del self._content_shingles[content_id]
        
        if content_hash and content_hash in self._seen_hashes:
            self._seen_hashes.remove(content_hash)
    
    def deduplicate_batch(
        self,
        items: List[Dict],
        title_key: str = "title",
        content_key: str = "content",
        id_key: str = "id",
    ) -> List[Dict]:
        """
        Deduplicate a batch of items.
        
        Args:
            items: List of items to deduplicate
            title_key: Key for title in item dict
            content_key: Key for content in item dict
            id_key: Key for ID in item dict
            
        Returns:
            List of unique items
        """
        unique_items = []
        
        for item in items:
            title = item.get(title_key, "")
            content = item.get(content_key, "")
            item_id = str(item.get(id_key, len(unique_items)))
            
            is_dup, content_hash, _, _ = self.is_duplicate(
                title, content, item_id, check_similarity=True
            )
            
            if not is_dup:
                # Add to index and result
                self.add_content(title, content, item_id)
                item["content_hash"] = content_hash
                unique_items.append(item)
        
        return unique_items
