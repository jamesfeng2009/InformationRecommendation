from typing import Dict, Any

from elasticsearch import AsyncElasticsearch


# News index mapping with Chinese analyzer (ik_max_word)
NEWS_INDEX_MAPPING = {
    "settings": {
        "number_of_shards": 3,
        "number_of_replicas": 1,
        "analysis": {
            "analyzer": {
                "ik_max_word": {
                    "type": "custom",
                    "tokenizer": "ik_max_word"
                },
                "ik_smart": {
                    "type": "custom",
                    "tokenizer": "ik_smart"
                }
            }
        },
        "index": {
            "max_result_window": 10000
        }
    },
    "mappings": {
        "properties": {
            "id": {
                "type": "integer"
            },
            "title": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart",
                "fields": {
                    "keyword": {
                        "type": "keyword",
                        "ignore_above": 256
                    }
                }
            },
            "content": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart"
            },
            "translated_title": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart"
            },
            "translated_content": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart"
            },
            "summary": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart"
            },
            "keywords": {
                "type": "keyword"
            },
            "category": {
                "type": "keyword"
            },
            "source_name": {
                "type": "keyword"
            },
            "author": {
                "type": "keyword"
            },
            "location": {
                "type": "keyword"
            },
            "language": {
                "type": "keyword"
            },
            "source_url": {
                "type": "keyword"
            },
            "publish_time": {
                "type": "date"
            },
            "crawl_time": {
                "type": "date"
            },
            "hot_score": {
                "type": "float"
            },
            "content_hash": {
                "type": "keyword"
            },
            "images": {
                "type": "keyword"
            },
            "videos": {
                "type": "keyword"
            }
        }
    }
}


class ElasticsearchIndexService:
    """
    Service for managing Elasticsearch indices.
    """
    
    def __init__(self, es_client: AsyncElasticsearch, index_prefix: str = "news"):
        """
        Initialize Elasticsearch index service.
        
        Args:
            es_client: AsyncElasticsearch client instance
            index_prefix: Prefix for index names
        """
        self.es_client = es_client
        self.index_prefix = index_prefix
        self.current_index = f"{index_prefix}_v1"
        self.alias_name = index_prefix
    
    async def create_index(self, index_name: str = None) -> bool:
        """
        Create news index with Chinese analyzer configuration.
        
        Args:
            index_name: Optional index name, defaults to current_index
            
        Returns:
            bool: True if index created successfully
            
        Requirements: 19.4 - Define news index with Chinese analyzer
        """
        if index_name is None:
            index_name = self.current_index
        
        # Check if index already exists
        if await self.es_client.indices.exists(index=index_name):
            return False
        
        # Create index with mapping
        await self.es_client.indices.create(
            index=index_name,
            body=NEWS_INDEX_MAPPING
        )
        
        return True
    
    async def delete_index(self, index_name: str) -> bool:
        """
        Delete an index.
        
        Args:
            index_name: Name of index to delete
            
        Returns:
            bool: True if deleted successfully
        """
        if await self.es_client.indices.exists(index=index_name):
            await self.es_client.indices.delete(index=index_name)
            return True
        return False
    
    async def setup_alias(self, index_name: str = None) -> bool:
        """
        Set up index alias for zero-downtime reindexing.
        
        Args:
            index_name: Index to point alias to, defaults to current_index
            
        Returns:
            bool: True if alias set up successfully
            
        Requirements: 19.4 - Set up index aliases for zero-downtime reindexing
        """
        if index_name is None:
            index_name = self.current_index
        
        # Check if alias exists
        alias_exists = await self.es_client.indices.exists_alias(name=self.alias_name)
        
        if alias_exists:
            # Get current indices for this alias
            current_indices = await self.es_client.indices.get_alias(name=self.alias_name)
            
            # Build actions to remove old alias and add new one
            actions = []
            for old_index in current_indices.keys():
                actions.append({"remove": {"index": old_index, "alias": self.alias_name}})
            actions.append({"add": {"index": index_name, "alias": self.alias_name}})
            
            # Atomic alias switch
            await self.es_client.indices.update_aliases(body={"actions": actions})
        else:
            # Create new alias
            await self.es_client.indices.put_alias(
                index=index_name,
                name=self.alias_name
            )
        
        return True
    
    async def reindex(self, source_index: str, dest_index: str) -> Dict[str, Any]:
        """
        Reindex data from source to destination index.
        
        Args:
            source_index: Source index name
            dest_index: Destination index name
            
        Returns:
            dict: Reindex response with statistics
        """
        response = await self.es_client.reindex(
            body={
                "source": {"index": source_index},
                "dest": {"index": dest_index}
            },
            wait_for_completion=False
        )
        
        return response
    
    async def get_reindex_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get status of a reindex task.
        
        Args:
            task_id: Task ID from reindex operation
            
        Returns:
            dict: Task status information
        """
        response = await self.es_client.tasks.get(task_id=task_id)
        return response
    
    async def initialize(self) -> bool:
        """
        Initialize index and alias for first-time setup.
        
        Returns:
            bool: True if initialization successful
        """
        # Create index
        await self.create_index()
        
        # Set up alias
        await self.setup_alias()
        
        return True
    
    async def refresh_index(self, index_name: str = None) -> bool:
        """
        Refresh index to make recent changes searchable.
        
        Args:
            index_name: Index to refresh, defaults to alias_name
            
        Returns:
            bool: True if refresh successful
        """
        if index_name is None:
            index_name = self.alias_name
        
        await self.es_client.indices.refresh(index=index_name)
        return True
