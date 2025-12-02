"""
NLP Service Module

Provides language detection, translation, classification, summarization,
and keyword extraction capabilities for news processing.
"""

from typing import Optional
import re
import hashlib

from langdetect import detect, detect_langs, LangDetectException
from langdetect.lang_detect_exception import ErrorCode


# Supported languages mapping (ISO 639-1 codes)
SUPPORTED_LANGUAGES = {
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "en": "English",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "hi": "Hindi",
    "es": "Spanish",
}

# Language code normalization mapping
LANGUAGE_CODE_MAP = {
    "zh": "zh-cn",
    "zh-cn": "zh-cn",
    "zh-tw": "zh-tw",
    "en": "en",
    "ru": "ru",
    "ja": "ja",
    "ko": "ko",
    "de": "de",
    "fr": "fr",
    "hi": "hi",
    "es": "es",
}


class LanguageDetector:
    """
    Language detection service using langdetect library.
    
    Supports detection for: Chinese, English, Russian, Japanese, 
    Korean, German, French, Hindi, Spanish.
    """
    
    def __init__(self):
        self.supported_codes = set(LANGUAGE_CODE_MAP.keys())
    
    def detect(self, text: str) -> str:
        """
        Detect the language of the given text.
        
        Args:
            text: The text to analyze
            
        Returns:
            ISO 639-1 language code (normalized to supported languages)
            
        Raises:
            ValueError: If text is empty or language cannot be detected
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Clean text for better detection
        cleaned_text = self._preprocess_text(text)
        
        if not cleaned_text:
            raise ValueError("Text contains no detectable content")
        
        try:
            detected_code = detect(cleaned_text)
            return self._normalize_language_code(detected_code)
        except LangDetectException as e:
            if e.code == ErrorCode.CantDetectError:
                raise ValueError(f"Cannot detect language: {str(e)}")
            raise ValueError(f"Language detection error: {str(e)}")
    
    def detect_with_confidence(self, text: str) -> list[tuple[str, float]]:
        """
        Detect language with confidence scores.
        
        Args:
            text: The text to analyze
            
        Returns:
            List of (language_code, probability) tuples, sorted by probability
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        cleaned_text = self._preprocess_text(text)
        
        if not cleaned_text:
            raise ValueError("Text contains no detectable content")
        
        try:
            results = detect_langs(cleaned_text)
            return [
                (self._normalize_language_code(r.lang), r.prob)
                for r in results
            ]
        except LangDetectException as e:
            raise ValueError(f"Language detection error: {str(e)}")
    
    def is_supported_language(self, lang_code: str) -> bool:
        """Check if a language code is supported."""
        normalized = self._normalize_language_code(lang_code)
        return normalized in SUPPORTED_LANGUAGES
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better language detection."""
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        # Remove excessive whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    def _normalize_language_code(self, code: str) -> str:
        """Normalize language code to our supported format."""
        code_lower = code.lower()
        
        # Handle Chinese variants
        if code_lower.startswith("zh"):
            if "tw" in code_lower or "hant" in code_lower:
                return "zh-tw"
            return "zh-cn"
        
        # Map to supported codes
        if code_lower in LANGUAGE_CODE_MAP:
            return LANGUAGE_CODE_MAP[code_lower]
        
        # Return as-is if not in our mapping (will be filtered as unsupported)
        return code_lower


# Global instance for convenience
_language_detector: Optional[LanguageDetector] = None


def get_language_detector() -> LanguageDetector:
    """Get or create the global language detector instance."""
    global _language_detector
    if _language_detector is None:
        _language_detector = LanguageDetector()
    return _language_detector


def detect_language(text: str) -> str:
    """
    Convenience function to detect language of text.
    
    Args:
        text: The text to analyze
        
    Returns:
        ISO 639-1 language code
    """
    return get_language_detector().detect(text)


# ============================================================================
# Translation Service
# ============================================================================

class TranslationService:
    """
    Translation service with caching support.
    
    This is a pluggable translation service that can use different backends:
    - Mock/stub for testing
    - External API (Google Translate, DeepL, etc.)
    - Local models (MarianMT, etc.)
    
    For production, integrate with actual translation API.
    """
    
    def __init__(self, cache_enabled: bool = True):
        self.cache_enabled = cache_enabled
        self._cache: dict[str, str] = {}
        self._supported_pairs = self._build_supported_pairs()
    
    def _build_supported_pairs(self) -> set[tuple[str, str]]:
        """Build set of supported translation pairs."""
        pairs = set()
        languages = list(SUPPORTED_LANGUAGES.keys())
        for src in languages:
            for tgt in languages:
                if src != tgt:
                    pairs.add((src, tgt))
        return pairs
    
    def _get_cache_key(self, text: str, source_lang: str, target_lang: str) -> str:
        """Generate cache key for translation."""
        content = f"{source_lang}:{target_lang}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def is_pair_supported(self, source_lang: str, target_lang: str) -> bool:
        """Check if translation pair is supported."""
        src = LANGUAGE_CODE_MAP.get(source_lang.lower(), source_lang.lower())
        tgt = LANGUAGE_CODE_MAP.get(target_lang.lower(), target_lang.lower())
        return (src, tgt) in self._supported_pairs
    
    def translate(
        self, 
        text: str, 
        source_lang: str, 
        target_lang: str,
        use_cache: bool = True
    ) -> str:
        """
        Translate text from source language to target language.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            use_cache: Whether to use cached results
            
        Returns:
            Translated text
            
        Raises:
            ValueError: If languages are not supported or text is empty
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Normalize language codes
        src = LANGUAGE_CODE_MAP.get(source_lang.lower(), source_lang.lower())
        tgt = LANGUAGE_CODE_MAP.get(target_lang.lower(), target_lang.lower())
        
        if src not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Source language '{source_lang}' is not supported")
        if tgt not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Target language '{target_lang}' is not supported")
        
        if src == tgt:
            return text
        
        # Check cache
        if self.cache_enabled and use_cache:
            cache_key = self._get_cache_key(text, src, tgt)
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        # Perform translation
        translated = self._do_translate(text, src, tgt)
        
        # Cache result
        if self.cache_enabled and use_cache:
            self._cache[cache_key] = translated
        
        return translated
    
    def _do_translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Perform actual translation.
        
        This is a stub implementation that returns a marked translation.
        In production, this should call an actual translation API/model.
        
        For testing purposes, this implementation:
        1. Preserves the original text structure
        2. Adds language markers for verification
        3. Maintains semantic similarity for round-trip testing
        """
        # For production: integrate with actual translation API
        # Examples: Google Translate API, DeepL API, Azure Translator, etc.
        
        # Stub implementation for development/testing
        # Returns text with language markers to verify translation flow
        return f"[{target_lang}]{text}[/{target_lang}]"
    
    def clear_cache(self):
        """Clear the translation cache."""
        self._cache.clear()
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "enabled": self.cache_enabled,
        }


# Global instance
_translation_service: Optional[TranslationService] = None


def get_translation_service() -> TranslationService:
    """Get or create the global translation service instance."""
    global _translation_service
    if _translation_service is None:
        _translation_service = TranslationService()
    return _translation_service


def translate(text: str, source_lang: str, target_lang: str) -> str:
    """
    Convenience function to translate text.
    
    Args:
        text: Text to translate
        source_lang: Source language code
        target_lang: Target language code
        
    Returns:
        Translated text
    """
    return get_translation_service().translate(text, source_lang, target_lang)


# ============================================================================
# Text Classification Service
# ============================================================================

# Default news categories (customizable)
DEFAULT_CATEGORIES = [
    "政治",  # Politics
    "军事",  # Military
    "经济",  # Economy
    "科技",  # Technology
    "社会",  # Society
    "文化",  # Culture
    "体育",  # Sports
    "娱乐",  # Entertainment
    "国际",  # International
    "其他",  # Other
]

# Category keywords for rule-based classification
CATEGORY_KEYWORDS = {
    "政治": ["政府", "政策", "选举", "议会", "总统", "首相", "外交", "政治", "党", "国会",
             "government", "policy", "election", "parliament", "president", "prime minister"],
    "军事": ["军队", "武器", "战争", "国防", "军事", "导弹", "战斗机", "航母", "军演", "部队",
             "military", "weapon", "war", "defense", "army", "missile", "fighter", "navy"],
    "经济": ["经济", "股市", "金融", "贸易", "投资", "GDP", "通胀", "利率", "银行", "货币",
             "economy", "stock", "finance", "trade", "investment", "inflation", "bank"],
    "科技": ["科技", "技术", "人工智能", "AI", "互联网", "软件", "硬件", "创新", "研发", "数字",
             "technology", "tech", "AI", "internet", "software", "innovation", "digital"],
    "社会": ["社会", "民生", "教育", "医疗", "就业", "住房", "环境", "公共", "社区", "福利",
             "society", "education", "healthcare", "employment", "housing", "environment"],
    "文化": ["文化", "艺术", "历史", "传统", "博物馆", "文学", "音乐", "电影", "戏剧", "遗产",
             "culture", "art", "history", "tradition", "museum", "literature", "music"],
    "体育": ["体育", "足球", "篮球", "奥运", "比赛", "冠军", "运动员", "联赛", "世界杯", "赛事",
             "sports", "football", "basketball", "olympics", "match", "champion", "league"],
    "娱乐": ["娱乐", "明星", "综艺", "电视", "演员", "歌手", "网红", "直播", "游戏", "粉丝",
             "entertainment", "celebrity", "TV", "actor", "singer", "game", "fan"],
    "国际": ["国际", "全球", "联合国", "世界", "跨国", "外国", "海外", "多边", "峰会", "条约",
             "international", "global", "UN", "world", "foreign", "overseas", "summit"],
}


class TextClassifier:
    """
    Text classification service for news categorization.
    
    Uses a hybrid approach:
    1. Keyword-based classification for quick categorization
    2. Can be extended with ML models for better accuracy
    
    Supports customizable category labels.
    """
    
    def __init__(self, categories: Optional[list[str]] = None):
        self.categories = categories or DEFAULT_CATEGORIES.copy()
        self.category_keywords = CATEGORY_KEYWORDS.copy()
    
    def classify(self, text: str, top_k: int = 1) -> list[tuple[str, float]]:
        """
        Classify text into categories.
        
        Args:
            text: Text to classify
            top_k: Number of top categories to return
            
        Returns:
            List of (category, confidence) tuples, sorted by confidence
            
        Raises:
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        text_lower = text.lower()
        scores: dict[str, float] = {}
        
        # Calculate scores based on keyword matching
        for category, keywords in self.category_keywords.items():
            if category not in self.categories:
                continue
            
            score = 0.0
            for keyword in keywords:
                keyword_lower = keyword.lower()
                # Count occurrences
                count = text_lower.count(keyword_lower)
                if count > 0:
                    # Weight by keyword length (longer keywords are more specific)
                    score += count * (1 + len(keyword_lower) / 10)
            
            scores[category] = score
        
        # Normalize scores
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {k: v / total_score for k, v in scores.items()}
        else:
            # Default to "其他" if no keywords match
            scores = {"其他": 1.0}
        
        # Sort by score and return top_k
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Ensure we return at least one category
        if not sorted_scores:
            return [("其他", 1.0)]
        
        return sorted_scores[:top_k]
    
    def classify_single(self, text: str) -> str:
        """
        Classify text and return the top category.
        
        Args:
            text: Text to classify
            
        Returns:
            The most likely category
        """
        results = self.classify(text, top_k=1)
        return results[0][0] if results else "其他"
    
    def set_categories(self, categories: list[str]):
        """Update the list of valid categories."""
        self.categories = categories.copy()
    
    def add_category_keywords(self, category: str, keywords: list[str]):
        """Add keywords for a category."""
        if category not in self.category_keywords:
            self.category_keywords[category] = []
        self.category_keywords[category].extend(keywords)
    
    def get_categories(self) -> list[str]:
        """Get the list of valid categories."""
        return self.categories.copy()


# Global instance
_text_classifier: Optional[TextClassifier] = None


def get_text_classifier() -> TextClassifier:
    """Get or create the global text classifier instance."""
    global _text_classifier
    if _text_classifier is None:
        _text_classifier = TextClassifier()
    return _text_classifier


def classify_text(text: str, top_k: int = 1) -> list[tuple[str, float]]:
    """
    Convenience function to classify text.
    
    Args:
        text: Text to classify
        top_k: Number of top categories to return
        
    Returns:
        List of (category, confidence) tuples
    """
    return get_text_classifier().classify(text, top_k)


# ============================================================================
# Summarization Service
# ============================================================================


class Summarizer:
    """
    Text summarization service.
    
    Uses extractive summarization approach:
    1. Split text into sentences
    2. Score sentences by importance (position, keywords, length)
    3. Select top sentences for summary
    
    Can be extended with abstractive summarization models.
    """
    
    def __init__(self, max_sentences: int = 3, max_length: int = 300):
        self.max_sentences = max_sentences
        self.max_length = max_length
        # Sentence ending patterns for different languages
        self._sentence_endings = re.compile(r'[。！？.!?]+')
    
    def summarize(
        self, 
        text: str, 
        max_sentences: Optional[int] = None,
        max_length: Optional[int] = None
    ) -> str:
        """
        Generate a summary of the text.
        
        Args:
            text: Text to summarize
            max_sentences: Maximum number of sentences in summary
            max_length: Maximum character length of summary
            
        Returns:
            Summary text
            
        Raises:
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        max_sentences = max_sentences or self.max_sentences
        max_length = max_length or self.max_length
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if not sentences:
            return text[:max_length] if len(text) > max_length else text
        
        if len(sentences) <= max_sentences:
            summary = ' '.join(sentences)
            return summary[:max_length] if len(summary) > max_length else summary
        
        # Score sentences
        scored_sentences = self._score_sentences(sentences, text)
        
        # Select top sentences while maintaining original order
        top_indices = sorted(
            range(len(scored_sentences)),
            key=lambda i: scored_sentences[i][1],
            reverse=True
        )[:max_sentences]
        
        # Sort by original position to maintain coherence
        top_indices.sort()
        
        # Build summary
        summary_sentences = [sentences[i] for i in top_indices]
        summary = ' '.join(summary_sentences)
        
        # Truncate if needed
        if len(summary) > max_length:
            summary = summary[:max_length].rsplit(' ', 1)[0]
            if not summary.endswith(('.', '。', '!', '！', '?', '？')):
                summary += '...'
        
        return summary
    
    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Handle both Chinese and English sentence endings
        sentences = self._sentence_endings.split(text)
        # Clean and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _score_sentences(
        self, 
        sentences: list[str], 
        original_text: str
    ) -> list[tuple[str, float]]:
        """
        Score sentences by importance.
        
        Scoring factors:
        1. Position (first sentences are usually more important)
        2. Length (medium-length sentences are preferred)
        3. Keyword density
        """
        scores = []
        total_sentences = len(sentences)
        
        for i, sentence in enumerate(sentences):
            score = 0.0
            
            # Position score (first sentences get higher scores)
            position_score = 1.0 - (i / total_sentences) * 0.5
            score += position_score * 0.3
            
            # Length score (prefer medium-length sentences)
            length = len(sentence)
            if 20 <= length <= 100:
                length_score = 1.0
            elif length < 20:
                length_score = length / 20
            else:
                length_score = max(0.5, 1.0 - (length - 100) / 200)
            score += length_score * 0.3
            
            # Keyword density (sentences with more unique words)
            words = set(sentence.lower().split())
            unique_ratio = len(words) / max(len(sentence.split()), 1)
            score += unique_ratio * 0.4
            
            scores.append((sentence, score))
        
        return scores
    
    def set_max_sentences(self, max_sentences: int):
        """Set the maximum number of sentences in summary."""
        self.max_sentences = max_sentences
    
    def set_max_length(self, max_length: int):
        """Set the maximum character length of summary."""
        self.max_length = max_length


# Global instance
_summarizer: Optional[Summarizer] = None


def get_summarizer() -> Summarizer:
    """Get or create the global summarizer instance."""
    global _summarizer
    if _summarizer is None:
        _summarizer = Summarizer()
    return _summarizer


def summarize(text: str, max_sentences: int = 3, max_length: int = 300) -> str:
    """
    Convenience function to summarize text.
    
    Args:
        text: Text to summarize
        max_sentences: Maximum number of sentences
        max_length: Maximum character length
        
    Returns:
        Summary text
    """
    return get_summarizer().summarize(text, max_sentences, max_length)


# ============================================================================
# Keyword Extraction Service
# ============================================================================

import jieba
import jieba.analyse
from collections import Counter


class KeywordExtractor:
    """
    Keyword extraction service using TF-IDF and TextRank algorithms.
    
    Supports both Chinese and English text.
    Uses jieba for Chinese word segmentation.
    """
    
    def __init__(self):
        # Chinese stopwords
        self._chinese_stopwords = {
            '的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一',
            '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有',
            '看', '好', '自己', '这', '那', '他', '她', '它', '们', '这个', '那个',
            '什么', '怎么', '为什么', '哪', '哪里', '哪个', '如何', '可以', '能',
            '但是', '而且', '或者', '因为', '所以', '如果', '虽然', '然后', '之后',
        }
        # English stopwords
        self._english_stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
            'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
            'not', 'only', 'own', 'same', 'than', 'too', 'very', 'just', 'also',
            'this', 'that', 'these', 'those', 'it', 'its', 'he', 'she', 'they',
            'i', 'you', 'we', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
            'his', 'their', 'our', 'what', 'which', 'who', 'whom', 'whose',
        }
    
    def extract(self, text: str, top_k: int = 10, method: str = "tfidf") -> list[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Text to extract keywords from
            top_k: Number of keywords to extract
            method: Extraction method ('tfidf' or 'textrank')
            
        Returns:
            List of keywords
            
        Raises:
            ValueError: If text is empty or method is invalid
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        if method not in ("tfidf", "textrank"):
            raise ValueError(f"Invalid method: {method}. Use 'tfidf' or 'textrank'")
        
        # Detect if text is primarily Chinese or English
        is_chinese = self._is_chinese_text(text)
        
        if is_chinese:
            return self._extract_chinese(text, top_k, method)
        else:
            return self._extract_english(text, top_k)
    
    def _is_chinese_text(self, text: str) -> bool:
        """Check if text is primarily Chinese."""
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        return chinese_chars > len(text) * 0.3
    
    def _extract_chinese(self, text: str, top_k: int, method: str) -> list[str]:
        """Extract keywords from Chinese text using jieba."""
        if method == "tfidf":
            keywords = jieba.analyse.extract_tags(text, topK=top_k)
        else:  # textrank
            keywords = jieba.analyse.textrank(text, topK=top_k)
        
        return list(keywords)
    
    def _extract_english(self, text: str, top_k: int) -> list[str]:
        """Extract keywords from English text using TF-IDF approach."""
        # Tokenize
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        
        # Remove stopwords
        words = [w for w in words if w not in self._english_stopwords]
        
        if not words:
            return []
        
        # Calculate term frequency
        word_counts = Counter(words)
        
        # Get top-k by frequency (simple TF approach)
        top_words = word_counts.most_common(top_k)
        
        return [word for word, _ in top_words]
    
    def extract_with_scores(
        self, 
        text: str, 
        top_k: int = 10, 
        method: str = "tfidf"
    ) -> list[tuple[str, float]]:
        """
        Extract keywords with their scores.
        
        Args:
            text: Text to extract keywords from
            top_k: Number of keywords to extract
            method: Extraction method
            
        Returns:
            List of (keyword, score) tuples
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        is_chinese = self._is_chinese_text(text)
        
        if is_chinese:
            if method == "tfidf":
                return jieba.analyse.extract_tags(text, topK=top_k, withWeight=True)
            else:
                return jieba.analyse.textrank(text, topK=top_k, withWeight=True)
        else:
            # For English, return with normalized scores
            words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
            words = [w for w in words if w not in self._english_stopwords]
            
            if not words:
                return []
            
            word_counts = Counter(words)
            total = sum(word_counts.values())
            top_words = word_counts.most_common(top_k)
            
            return [(word, count / total) for word, count in top_words]


# Global instance
_keyword_extractor: Optional[KeywordExtractor] = None


def get_keyword_extractor() -> KeywordExtractor:
    """Get or create the global keyword extractor instance."""
    global _keyword_extractor
    if _keyword_extractor is None:
        _keyword_extractor = KeywordExtractor()
    return _keyword_extractor


def extract_keywords(text: str, top_k: int = 10, method: str = "tfidf") -> list[str]:
    """
    Convenience function to extract keywords.
    
    Args:
        text: Text to extract keywords from
        top_k: Number of keywords to extract
        method: Extraction method ('tfidf' or 'textrank')
        
    Returns:
        List of keywords
    """
    return get_keyword_extractor().extract(text, top_k, method)


# ============================================================================
# NLP Processing Pipeline
# ============================================================================

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class NLPProcessingResult:
    """Result of NLP processing pipeline."""
    
    # Original content
    original_text: str
    original_language: str
    
    # Translation (if applicable)
    translated_text: Optional[str] = None
    target_language: Optional[str] = None
    
    # Classification
    categories: list[tuple[str, float]] = field(default_factory=list)
    primary_category: str = ""
    
    # Summarization
    summary: str = ""
    
    # Keywords
    keywords: list[str] = field(default_factory=list)
    
    # Metadata
    processed_at: datetime = field(default_factory=datetime.utcnow)
    processing_time_ms: float = 0.0
    errors: list[str] = field(default_factory=list)


class NLPPipeline:
    """
    NLP processing pipeline that chains:
    1. Language detection
    2. Translation (if needed)
    3. Classification
    4. Summarization
    5. Keyword extraction
    
    Can be used synchronously or integrated with Celery for async processing.
    """
    
    def __init__(
        self,
        target_language: str = "zh-cn",
        translate_enabled: bool = True,
        classify_enabled: bool = True,
        summarize_enabled: bool = True,
        keywords_enabled: bool = True,
    ):
        self.target_language = target_language
        self.translate_enabled = translate_enabled
        self.classify_enabled = classify_enabled
        self.summarize_enabled = summarize_enabled
        self.keywords_enabled = keywords_enabled
        
        # Initialize components
        self._language_detector = LanguageDetector()
        self._translator = TranslationService()
        self._classifier = TextClassifier()
        self._summarizer = Summarizer()
        self._keyword_extractor = KeywordExtractor()
    
    def process(
        self,
        text: str,
        target_language: Optional[str] = None,
        skip_translation: bool = False,
    ) -> NLPProcessingResult:
        """
        Process text through the NLP pipeline.
        
        Args:
            text: Text to process
            target_language: Override default target language for translation
            skip_translation: Skip translation step
            
        Returns:
            NLPProcessingResult with all processing results
        """
        import time
        start_time = time.time()
        
        target_lang = target_language or self.target_language
        errors: list[str] = []
        
        # Initialize result
        result = NLPProcessingResult(
            original_text=text,
            original_language="",
        )
        
        # Step 1: Language Detection
        try:
            detected_lang = self._language_detector.detect(text)
            result.original_language = detected_lang
        except Exception as e:
            errors.append(f"Language detection failed: {str(e)}")
            result.original_language = "unknown"
        
        # Step 2: Translation (if enabled and needed)
        text_for_processing = text
        if (
            self.translate_enabled 
            and not skip_translation 
            and result.original_language != target_lang
            and result.original_language != "unknown"
        ):
            try:
                translated = self._translator.translate(
                    text, 
                    result.original_language, 
                    target_lang
                )
                result.translated_text = translated
                result.target_language = target_lang
                # Use translated text for further processing
                text_for_processing = translated
            except Exception as e:
                errors.append(f"Translation failed: {str(e)}")
        
        # Step 3: Classification
        if self.classify_enabled:
            try:
                categories = self._classifier.classify(text_for_processing, top_k=3)
                result.categories = categories
                result.primary_category = categories[0][0] if categories else "其他"
            except Exception as e:
                errors.append(f"Classification failed: {str(e)}")
                result.primary_category = "其他"
        
        # Step 4: Summarization
        if self.summarize_enabled:
            try:
                summary = self._summarizer.summarize(text_for_processing)
                result.summary = summary
            except Exception as e:
                errors.append(f"Summarization failed: {str(e)}")
        
        # Step 5: Keyword Extraction
        if self.keywords_enabled:
            try:
                keywords = self._keyword_extractor.extract(text_for_processing, top_k=10)
                result.keywords = keywords
            except Exception as e:
                errors.append(f"Keyword extraction failed: {str(e)}")
        
        # Finalize
        result.errors = errors
        result.processing_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def process_batch(
        self,
        texts: list[str],
        target_language: Optional[str] = None,
    ) -> list[NLPProcessingResult]:
        """
        Process multiple texts through the pipeline.
        
        Args:
            texts: List of texts to process
            target_language: Override default target language
            
        Returns:
            List of NLPProcessingResult objects
        """
        return [self.process(text, target_language) for text in texts]


# Global pipeline instance
_nlp_pipeline: Optional[NLPPipeline] = None


def get_nlp_pipeline() -> NLPPipeline:
    """Get or create the global NLP pipeline instance."""
    global _nlp_pipeline
    if _nlp_pipeline is None:
        _nlp_pipeline = NLPPipeline()
    return _nlp_pipeline


def process_news_content(text: str, target_language: str = "zh-cn") -> NLPProcessingResult:
    """
    Convenience function to process news content through the NLP pipeline.
    
    Args:
        text: News content to process
        target_language: Target language for translation
        
    Returns:
        NLPProcessingResult with all processing results
    """
    return get_nlp_pipeline().process(text, target_language)


# ============================================================================
# Celery Task Integration (for async processing)
# ============================================================================

def create_nlp_celery_task():
    """
    Create a Celery task for async NLP processing.
    
    This function returns a task that can be registered with Celery.
    Usage:
        from celery import Celery
        app = Celery('nlp_tasks')
        process_news_task = app.task(create_nlp_celery_task())
    """
    def process_news_async(
        text: str,
        target_language: str = "zh-cn",
        news_id: Optional[int] = None,
    ) -> dict:
        """
        Async NLP processing task for Celery.
        
        Args:
            text: News content to process
            target_language: Target language for translation
            news_id: Optional news ID for tracking
            
        Returns:
            Dictionary with processing results
        """
        pipeline = get_nlp_pipeline()
        result = pipeline.process(text, target_language)
        
        return {
            "news_id": news_id,
            "original_language": result.original_language,
            "translated_text": result.translated_text,
            "target_language": result.target_language,
            "primary_category": result.primary_category,
            "categories": result.categories,
            "summary": result.summary,
            "keywords": result.keywords,
            "processing_time_ms": result.processing_time_ms,
            "errors": result.errors,
        }
    
    return process_news_async
