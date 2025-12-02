from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.elasticsearch import get_elasticsearch_client
from app.services.elasticsearch_index import ElasticsearchIndexService


router = APIRouter(prefix="/api/v1/admin/elasticsearch", tags=["elasticsearch-admin"])


class IndexCreateRequest(BaseModel):
    """Schema for index creation request."""
    index_name: str = Field(..., min_length=1, max_length=100)


class IndexCreateResponse(BaseModel):
    """Schema for index creation response."""
    success: bool
    message: str
    index_name: str


class AliasSetupRequest(BaseModel):
    """Schema for alias setup request."""
    index_name: str = Field(..., min_length=1, max_length=100)


class AliasSetupResponse(BaseModel):
    """Schema for alias setup response."""
    success: bool
    message: str
    alias_name: str
    index_name: str


class ReindexRequest(BaseModel):
    """Schema for reindex request."""
    source_index: str = Field(..., min_length=1, max_length=100)
    dest_index: str = Field(..., min_length=1, max_length=100)


class ReindexResponse(BaseModel):
    """Schema for reindex response."""
    task_id: str
    message: str


class InitializeResponse(BaseModel):
    """Schema for initialization response."""
    success: bool
    message: str
    index_name: str
    alias_name: str


async def get_index_service(
    es_client = Depends(get_elasticsearch_client)
) -> ElasticsearchIndexService:
    """Dependency to get Elasticsearch index service."""
    return ElasticsearchIndexService(es_client)


@router.post("/initialize", response_model=InitializeResponse)
async def initialize_index(
    service: ElasticsearchIndexService = Depends(get_index_service)
):
    """
    Initialize Elasticsearch index and alias for first-time setup.
    
    Requirements: 19.4 - Define news index with Chinese analyzer
    """
    try:
        await service.initialize()
        return InitializeResponse(
            success=True,
            message="Elasticsearch index and alias initialized successfully",
            index_name=service.current_index,
            alias_name=service.alias_name
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize index: {str(e)}"
        )


@router.post("/index", response_model=IndexCreateResponse)
async def create_index(
    request: IndexCreateRequest,
    service: ElasticsearchIndexService = Depends(get_index_service)
):
    """
    Create a new Elasticsearch index with Chinese analyzer configuration.
    
    Requirements: 19.4 - Define news index with Chinese analyzer
    """
    try:
        success = await service.create_index(request.index_name)
        if success:
            return IndexCreateResponse(
                success=True,
                message=f"Index '{request.index_name}' created successfully",
                index_name=request.index_name
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Index '{request.index_name}' already exists"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create index: {str(e)}"
        )


@router.delete("/index/{index_name}")
async def delete_index(
    index_name: str,
    service: ElasticsearchIndexService = Depends(get_index_service)
):
    """Delete an Elasticsearch index."""
    try:
        success = await service.delete_index(index_name)
        if success:
            return {"message": f"Index '{index_name}' deleted successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Index '{index_name}' not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete index: {str(e)}"
        )


@router.post("/alias", response_model=AliasSetupResponse)
async def setup_alias(
    request: AliasSetupRequest,
    service: ElasticsearchIndexService = Depends(get_index_service)
):
    """
    Set up or update index alias for zero-downtime reindexing.
    
    Requirements: 19.4 - Set up index aliases for zero-downtime reindexing
    """
    try:
        await service.setup_alias(request.index_name)
        return AliasSetupResponse(
            success=True,
            message=f"Alias '{service.alias_name}' now points to '{request.index_name}'",
            alias_name=service.alias_name,
            index_name=request.index_name
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to setup alias: {str(e)}"
        )


@router.post("/reindex", response_model=ReindexResponse)
async def reindex_data(
    request: ReindexRequest,
    service: ElasticsearchIndexService = Depends(get_index_service)
):
    """
    Reindex data from source to destination index.
    
    Requirements: 19.4 - Support zero-downtime reindexing
    """
    try:
        response = await service.reindex(request.source_index, request.dest_index)
        task_id = response.get("task", "")
        
        return ReindexResponse(
            task_id=task_id,
            message=f"Reindex task started from '{request.source_index}' to '{request.dest_index}'"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start reindex: {str(e)}"
        )


@router.get("/reindex/status/{task_id}")
async def get_reindex_status(
    task_id: str,
    service: ElasticsearchIndexService = Depends(get_index_service)
):
    """
    Get status of a reindex task.
    
    Requirements: 19.4 - Support zero-downtime reindexing
    """
    try:
        status_info = await service.get_reindex_status(task_id)
        return status_info
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get reindex status: {str(e)}"
        )


@router.post("/refresh")
async def refresh_index(
    service: ElasticsearchIndexService = Depends(get_index_service)
):
    """Refresh index to make recent changes searchable."""
    try:
        await service.refresh_index()
        return {"message": "Index refreshed successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to refresh index: {str(e)}"
        )
