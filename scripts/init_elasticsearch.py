#!/usr/bin/env python
"""
Initialize Elasticsearch index for the news recommendation system.

This script creates the news index with Chinese analyzer configuration
and sets up the index alias for zero-downtime reindexing.

Usage:
    python scripts/init_elasticsearch.py
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.elasticsearch import get_elasticsearch_client, close_elasticsearch_client
from app.services.elasticsearch_index import ElasticsearchIndexService


async def main():
    """Initialize Elasticsearch index and alias."""
    print("🚀 Initializing Elasticsearch index...")
    
    try:
        # Get Elasticsearch client
        es_client = await get_elasticsearch_client()
        print("✅ Connected to Elasticsearch")
        
        # Create index service
        service = ElasticsearchIndexService(es_client)
        
        # Check if index already exists
        index_exists = await es_client.indices.exists(index=service.current_index)
        if index_exists:
            print(f"⚠️  Index '{service.current_index}' already exists")
            response = input("Do you want to recreate it? (yes/no): ")
            if response.lower() == 'yes':
                await service.delete_index(service.current_index)
                print(f"🗑️  Deleted existing index '{service.current_index}'")
            else:
                print("❌ Aborted")
                return
        
        # Initialize index and alias
        await service.initialize()
        print(f"✅ Created index: {service.current_index}")
        print(f"✅ Created alias: {service.alias_name} -> {service.current_index}")
        
        # Verify index
        mapping = await es_client.indices.get_mapping(index=service.current_index)
        print(f"✅ Index mapping verified")
        
        # Get index info
        info = await es_client.indices.get(index=service.current_index)
        settings = info[service.current_index]['settings']['index']
        print(f"\n📊 Index Configuration:")
        print(f"   - Shards: {settings.get('number_of_shards', 'N/A')}")
        print(f"   - Replicas: {settings.get('number_of_replicas', 'N/A')}")
        
        print(f"\n🎉 Elasticsearch initialization completed successfully!")
        print(f"\n📝 Next steps:")
        print(f"   1. Start the application: uvicorn app.main:app --reload")
        print(f"   2. Create news via API: POST /api/v1/news")
        print(f"   3. Or batch index existing data: POST /api/v1/news/batch-index")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Close connection
        await close_elasticsearch_client()
        print("\n👋 Connection closed")


if __name__ == "__main__":
    asyncio.run(main())
