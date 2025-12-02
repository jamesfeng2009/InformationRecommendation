"""
Unit tests for Elasticsearch indexing functionality.
Requirements: 19.2, 19.4
"""
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.elasticsearch_index import ElasticsearchIndexService, NEWS_INDEX_MAPPING
from app.services.news_indexing import NewsIndexingService
from app.models.news import News


@pytest.fixture
def mock_es_client():
    """Create a mock Elasticsearch client."""
    client = AsyncMock()
    client.indices = AsyncMock()
    return client


@pytest.fixture
def index_service(mock_es_client):
    """Create an ElasticsearchIndexService instance with mock client."""
    return ElasticsearchIndexService(mock_es_client, index_prefix="test_news")


@pytest.fixture
def indexing_service(mock_es_client):
    """Create a NewsIndexingService instance with mock client."""
    return NewsIndexingService(mock_es_client, index_name="test_news")


@pytest.fixture
def sample_news():
    """Create a sample News instance for testing."""
    return News(
        id=1,
        title="测试新闻标题",
        content="这是一篇测试新闻的内容，包含中文文本。",
        source_url="https://example.com/news/1",
        source_name="测试新闻源",
        language="zh",
        category="军事",
        author="测试作者",
        location="北京",
        keywords=["测试", "新闻"],
        summary="这是新闻摘要",
        images=["https://example.com/image1.jpg"],
        videos=["https://example.com/video1.mp4"],
        publish_time=datetime(2024, 1, 1, 12, 0, 0),
        crawl_time=datetime(2024, 1, 1, 12, 30, 0),
        hot_score=85.5,
        content_hash="abc123"
    )


class TestElasticsearchIndexService:
    """Test ElasticsearchIndexService functionality."""
    
    @pytest.mark.asyncio
    async def test_create_index_success(self, index_service, mock_es_client):
        """
        Test creating a new index with Chinese analyzer configuration.
        Requirements: 19.4 - Define news index with Chinese analyzer
        """
        # Arrange
        mock_es_client.indices.exists.return_value = False
        mock_es_client.indices.create.return_value = {"acknowledged": True}
        
        # Act
        result = await index_service.create_index("test_index")
        
        # Assert
        assert result is True
        mock_es_client.indices.exists.assert_called_once_with(index="test_index")
        mock_es_client.indices.create.assert_called_once_with(
            index="test_index",
            body=NEWS_INDEX_MAPPING
        )
    
    @pytest.mark.asyncio
    async def test_create_index_already_exists(self, index_service, mock_es_client):
        """Test creating an index that already exists."""
        # Arrange
        mock_es_client.indices.exists.return_value = True
        
        # Act
        result = await index_service.create_index("test_index")
        
        # Assert
        assert result is False
        mock_es_client.indices.create.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_delete_index_success(self, index_service, mock_es_client):
        """Test deleting an existing index."""
        # Arrange
        mock_es_client.indices.exists.return_value = True
        mock_es_client.indices.delete.return_value = {"acknowledged": True}
        
        # Act
        result = await index_service.delete_index("test_index")
        
        # Assert
        assert result is True
        mock_es_client.indices.delete.assert_called_once_with(index="test_index")
    
    @pytest.mark.asyncio
    async def test_setup_alias_new(self, index_service, mock_es_client):
        """
        Test setting up a new alias.
        Requirements: 19.4 - Set up index aliases for zero-downtime reindexing
        """
        # Arrange
        mock_es_client.indices.exists_alias.return_value = False
        mock_es_client.indices.put_alias.return_value = {"acknowledged": True}
        
        # Act
        result = await index_service.setup_alias("test_index")
        
        # Assert
        assert result is True
        mock_es_client.indices.put_alias.assert_called_once_with(
            index="test_index",
            name=index_service.alias_name
        )
    
    @pytest.mark.asyncio
    async def test_setup_alias_update(self, index_service, mock_es_client):
        """
        Test updating an existing alias (atomic switch).
        Requirements: 19.4 - Set up index aliases for zero-downtime reindexing
        """
        # Arrange
        mock_es_client.indices.exists_alias.return_value = True
        mock_es_client.indices.get_alias.return_value = {"old_index": {}}
        mock_es_client.indices.update_aliases.return_value = {"acknowledged": True}
        
        # Act
        result = await index_service.setup_alias("new_index")
        
        # Assert
        assert result is True
        mock_es_client.indices.update_aliases.assert_called_once()
        call_args = mock_es_client.indices.update_aliases.call_args
        actions = call_args[1]["body"]["actions"]
        assert len(actions) == 2
        assert actions[0]["remove"]["index"] == "old_index"
        assert actions[1]["add"]["index"] == "new_index"
    
    @pytest.mark.asyncio
    async def test_reindex(self, index_service, mock_es_client):
        """
        Test reindexing from source to destination.
        Requirements: 19.4 - Support zero-downtime reindexing
        """
        # Arrange
        mock_es_client.reindex.return_value = {"task": "task_123"}
        
        # Act
        result = await index_service.reindex("source_index", "dest_index")
        
        # Assert
        assert result["task"] == "task_123"
        mock_es_client.reindex.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize(self, index_service, mock_es_client):
        """Test initializing index and alias."""
        # Arrange
        mock_es_client.indices.exists.return_value = False
        mock_es_client.indices.create.return_value = {"acknowledged": True}
        mock_es_client.indices.exists_alias.return_value = False
        mock_es_client.indices.put_alias.return_value = {"acknowledged": True}
        
        # Act
        result = await index_service.initialize()
        
        # Assert
        assert result is True
        mock_es_client.indices.create.assert_called_once()
        mock_es_client.indices.put_alias.assert_called_once()


class TestNewsIndexingService:
    """Test NewsIndexingService functionality."""
    
    @pytest.mark.asyncio
    async def test_index_news_success(self, indexing_service, mock_es_client, sample_news):
        """
        Test indexing a single news document.
        Requirements: 19.2 - Index news documents on creation
        """
        # Arrange
        mock_es_client.index.return_value = {"result": "created"}
        
        # Act
        result = await indexing_service.index_news(sample_news)
        
        # Assert
        assert result is True
        mock_es_client.index.assert_called_once()
        call_args = mock_es_client.index.call_args
        assert call_args[1]["index"] == "test_news"
        assert call_args[1]["id"] == 1
        assert call_args[1]["document"]["title"] == "测试新闻标题"
        assert call_args[1]["document"]["content"] == "这是一篇测试新闻的内容，包含中文文本。"
    
    @pytest.mark.asyncio
    async def test_update_news_success(self, indexing_service, mock_es_client, sample_news):
        """
        Test updating an existing news document.
        Requirements: 19.2 - Update index on news modification
        """
        # Arrange
        mock_es_client.update.return_value = {"result": "updated"}
        
        # Act
        result = await indexing_service.update_news(sample_news)
        
        # Assert
        assert result is True
        mock_es_client.update.assert_called_once()
        call_args = mock_es_client.update.call_args
        assert call_args[1]["index"] == "test_news"
        assert call_args[1]["id"] == 1
    
    @pytest.mark.asyncio
    async def test_update_news_not_exists(self, indexing_service, mock_es_client, sample_news):
        """Test updating a news document that doesn't exist (should create it)."""
        # Arrange
        mock_es_client.update.side_effect = Exception("Document not found")
        mock_es_client.index.return_value = {"result": "created"}
        
        # Act
        result = await indexing_service.update_news(sample_news)
        
        # Assert
        assert result is True
        mock_es_client.index.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_news_success(self, indexing_service, mock_es_client):
        """Test deleting a news document."""
        # Arrange
        mock_es_client.delete.return_value = {"result": "deleted"}
        
        # Act
        result = await indexing_service.delete_news(1)
        
        # Assert
        assert result is True
        mock_es_client.delete.assert_called_once_with(index="test_news", id=1)
    
    def test_bulk_index_news_document_generation(self, indexing_service):
        """
        Test batch indexing document generation logic.
        Requirements: 19.2 - Batch indexing for initial data load
        
        Note: Full async_bulk testing requires integration tests with real Elasticsearch.
        This test verifies the document generation logic.
        """
        # Arrange
        news_list = [
            News(
                id=i,
                title=f"新闻{i}",
                content=f"内容{i}",
                source_url=f"https://example.com/{i}",
                source_name="测试源",
                language="zh",
                crawl_time=datetime.now()
            )
            for i in range(1, 6)
        ]
        
        # Act - Test document conversion for bulk indexing
        documents = [indexing_service._news_to_document(news) for news in news_list]
        
        # Assert
        assert len(documents) == 5
        for i, doc in enumerate(documents, start=1):
            assert doc["id"] == i
            assert doc["title"] == f"新闻{i}"
            assert doc["content"] == f"内容{i}"
            assert doc["source_name"] == "测试源"
            assert doc["language"] == "zh"
    
    @pytest.mark.asyncio
    async def test_exists(self, indexing_service, mock_es_client):
        """Test checking if a news document exists."""
        # Arrange
        mock_es_client.exists.return_value = True
        
        # Act
        result = await indexing_service.exists(1)
        
        # Assert
        assert result is True
        mock_es_client.exists.assert_called_once_with(index="test_news", id=1)
    
    @pytest.mark.asyncio
    async def test_get_document(self, indexing_service, mock_es_client):
        """Test retrieving a news document."""
        # Arrange
        mock_es_client.get.return_value = {
            "_source": {"id": 1, "title": "测试新闻"}
        }
        
        # Act
        result = await indexing_service.get_document(1)
        
        # Assert
        assert result is not None
        assert result["id"] == 1
        assert result["title"] == "测试新闻"
    
    @pytest.mark.asyncio
    async def test_count_documents(self, indexing_service, mock_es_client):
        """Test counting documents in the index."""
        # Arrange
        mock_es_client.count.return_value = {"count": 100}
        
        # Act
        result = await indexing_service.count_documents()
        
        # Assert
        assert result == 100
        mock_es_client.count.assert_called_once_with(index="test_news")
    
    def test_news_to_document_conversion(self, indexing_service, sample_news):
        """Test converting News model to Elasticsearch document."""
        # Act
        doc = indexing_service._news_to_document(sample_news)
        
        # Assert
        assert doc["id"] == 1
        assert doc["title"] == "测试新闻标题"
        assert doc["content"] == "这是一篇测试新闻的内容，包含中文文本。"
        assert doc["source_name"] == "测试新闻源"
        assert doc["language"] == "zh"
        assert doc["category"] == "军事"
        assert doc["author"] == "测试作者"
        assert doc["location"] == "北京"
        assert doc["keywords"] == ["测试", "新闻"]
        assert doc["summary"] == "这是新闻摘要"
        assert doc["hot_score"] == 85.5
        assert doc["content_hash"] == "abc123"
        assert doc["images"] == ["https://example.com/image1.jpg"]
        assert doc["videos"] == ["https://example.com/video1.mp4"]
