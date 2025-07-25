"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union
from datetime import datetime

class SearchResult(BaseModel):
    """Individual search result item."""
    image_id: str = Field(..., description="Unique identifier for the image")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Similarity score (0-1)")
    image_url: str = Field(..., description="URL to access the image")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional image metadata")
    
    class Config:
        json_json_schema_extra = {
            "example": {
                "image_id": "img_12345",
                "similarity": 0.85,
                "image_url": "/images/img_12345",
                "metadata": {"category": "fashion", "color": "blue"}
            }
        }

class SearchParameters(BaseModel):
    """Search request parameters."""
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results to return")
    threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum similarity threshold")
    
    class Config:
        json_json_schema_extra = {
            "example": {
                "top_k": 10,
                "threshold": 0.5
            }
        }

class SearchResponse(BaseModel):
    """Response for image similarity search."""
    query_id: str = Field(..., description="Unique identifier for this search query")
    results: List[SearchResult] = Field(..., description="List of similar images")
    total_results: int = Field(..., ge=0, description="Total number of results returned")
    search_time_ms: float = Field(..., ge=0, description="Search time in milliseconds")
    parameters: SearchParameters = Field(..., description="Search parameters used")
    
    class Config:
        json_json_schema_extra = {
            "example": {
                "query_id": "query_1642680000",
                "results": [
                    {
                        "image_id": "img_12345",
                        "similarity": 0.95,
                        "metadata": {"category": "fashion"}
                    }
                ],
                "total_results": 1,
                "search_time_ms": 45.2,
                "parameters": {
                    "top_k": 10,
                    "threshold": 0.0
                }
            }
        }

class BatchSearchQuery(BaseModel):
    """Individual query in batch search."""
    query_index: int = Field(..., ge=0, description="Index of the query in the batch")
    query_filename: str = Field(..., description="Original filename of the query image")
    results: List[SearchResult] = Field(..., description="Search results for this query")
    total_results: int = Field(..., ge=0, description="Number of results for this query")
    error: Optional[str] = Field(None, description="Error message if query failed")

class BatchSearchResponse(BaseModel):
    """Response for batch image search."""
    batch_id: str = Field(..., description="Unique identifier for this batch")
    queries: List[BatchSearchQuery] = Field(..., description="Results for each query")
    total_queries: int = Field(..., ge=0, description="Total number of queries in batch")
    successful_queries: int = Field(..., ge=0, description="Number of successful queries")
    
    @validator('successful_queries')
    def validate_successful_count(cls, v, values):
        if 'total_queries' in values and v > values['total_queries']:
            raise ValueError('successful_queries cannot exceed total_queries')
        return v

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Overall health status")
    timestamp: float = Field(..., description="Unix timestamp of health check")
    services: Dict[str, bool] = Field(..., description="Status of individual services")
    error: Optional[str] = Field(None, description="Error message if degraded")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": 1642680000.0,
                "services": {
                    "clip_model": True,
                    "vector_db": True,
                    "clip_model_loaded": True,
                    "vector_db_connected": True
                }
            }
        }

class StatsResponse(BaseModel):
    """API usage statistics."""
    total_requests: int = Field(..., ge=0, description="Total number of requests processed")
    total_errors: int = Field(..., ge=0, description="Total number of errors")
    average_search_time: float = Field(..., ge=0, description="Average search time in seconds")
    error_rate: float = Field(..., ge=0, le=100, description="Error rate as percentage")
    total_vectors: Optional[int] = Field(None, description="Total vectors in database")
    collection_name: Optional[str] = Field(None, description="Name of the active collection")
    backend: Optional[str] = Field(None, description="Vector database backend")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_requests": 1000,
                "total_errors": 5,
                "average_search_time": 0.045,
                "error_rate": 0.5,
                "total_vectors": 50000,
                "collection_name": "fashion_images",
                "backend": "ChromaDB"
            }
        }

class CollectionInfo(BaseModel):
    """Information about the vector collection."""
    collection_name: str = Field(..., description="Name of the collection")
    total_vectors: int = Field(..., ge=0, description="Total number of vectors")
    embedding_dimension: int = Field(..., gt=0, description="Dimension of embeddings")
    backend: str = Field(..., description="Vector database backend")
    model_name: str = Field(..., description="Name of the embedding model")
    image_size: int = Field(..., gt=0, description="Input image size for model")
    
    class Config:
        json_schema_extra = {
            "example": {
                "collection_name": "fashion_images",
                "total_vectors": 25000,
                "embedding_dimension": 512,
                "backend": "ChromaDB",
                "model_name": "openai/clip-vit-base-patch32",
                "image_size": 224
            }
        }

class AddImagesRequest(BaseModel):
    """Request for adding images to collection."""
    image_ids: Optional[List[str]] = Field(None, description="Optional custom IDs for images")
    
    @validator('image_ids')
    def validate_image_ids(cls, v):
        if v is not None:
            if len(v) != len(set(v)):
                raise ValueError('image_ids must be unique')
            if len(v) > 50:
                raise ValueError('Maximum 50 images per request')
        return v

class AddImagesResponse(BaseModel):
    """Response for adding images to collection."""
    success: bool = Field(..., description="Whether the operation was successful")
    added_images: int = Field(..., ge=0, description="Number of images successfully added")
    failed_uploads: List[Dict[str, str]] = Field(..., description="Details of failed uploads")
    image_ids: List[str] = Field(..., description="IDs assigned to the added images")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "added_images": 5,
                "failed_uploads": [],
                "image_ids": ["img_1", "img_2", "img_3", "img_4", "img_5"]
            }
        }

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error type or category")
    detail: str = Field(..., description="Detailed error message")
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "detail": "Invalid file format. Only images are supported.",
                "timestamp": 1642680000.0
            }
        }

class ModelInfo(BaseModel):
    """Information about the loaded model."""
    model_name: str = Field(..., description="HuggingFace model identifier")
    embedding_dimension: int = Field(..., gt=0, description="Dimension of output embeddings")
    image_size: int = Field(..., gt=0, description="Input image size")
    device: str = Field(..., description="Device used for inference")
    is_loaded: bool = Field(..., description="Whether model is loaded in memory")
    use_half_precision: bool = Field(..., description="Whether using FP16 precision")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "openai/clip-vit-base-patch32",
                "embedding_dimension": 512,
                "image_size": 224,
                "device": "cuda",
                "is_loaded": True,
                "use_half_precision": True
            }
        }

class SystemMetrics(BaseModel):
    """System performance metrics."""
    cpu_percent: float = Field(..., ge=0, le=100, description="CPU usage percentage")
    memory_percent: float = Field(..., ge=0, le=100, description="Memory usage percentage")
    gpu_memory_used_gb: Optional[float] = Field(None, description="GPU memory used in GB")
    gpu_memory_percent: Optional[float] = Field(None, ge=0, le=100, description="GPU memory usage percentage")
    processed_images: int = Field(..., ge=0, description="Total images processed")
    failed_images: int = Field(..., ge=0, description="Total failed image processing attempts")
    
    class Config:
        json_schema_extra = {
            "example": {
                "cpu_percent": 45.2,
                "memory_percent": 62.1,
                "gpu_memory_used_gb": 2.1,
                "gpu_memory_percent": 52.5,
                "processed_images": 1000,
                "failed_images": 5
            }
        }

class SearchMetrics(BaseModel):
    """Search-specific metrics."""
    query_image_size_bytes: int = Field(..., gt=0, description="Size of query image in bytes")
    embedding_time_ms: float = Field(..., ge=0, description="Time to generate embedding")
    database_search_time_ms: float = Field(..., ge=0, description="Time for database search")
    total_time_ms: float = Field(..., ge=0, description="Total search time")
    results_count: int = Field(..., ge=0, description="Number of results returned")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query_image_size_bytes": 524288,
                "embedding_time_ms": 15.2,
                "database_search_time_ms": 3.8,
                "total_time_ms": 19.0,
                "results_count": 10
            }
        }

# Validation utilities
def validate_file_extension(filename: str) -> bool:
    """Validate if filename has supported image extension."""
    from ..config import Config
    return any(filename.lower().endswith(ext) for ext in Config.SUPPORTED_FORMATS)

def validate_image_id(image_id: str) -> bool:
    """Validate image ID format."""
    if not image_id or len(image_id) > 255:
        return False
    # Allow alphanumeric, underscore, hyphen
    import re
    return bool(re.match(r'^[a-zA-Z0-9_-]+$', image_id))

# Custom validators for common use cases
class ImageIdValidator:
    """Validator for image IDs."""
    
    @staticmethod
    def validate(value: str) -> str:
        if not validate_image_id(value):
            raise ValueError('Invalid image ID format')
        return value

class FilenameValidator:
    """Validator for filenames."""
    
    @staticmethod
    def validate(value: str) -> str:
        if not validate_file_extension(value):
            raise ValueError('Unsupported file extension')
        return value