"""
FastAPI application for visual search engine.
Provides REST API endpoints for image similarity search.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from PIL import Image
import numpy as np
import pandas as pd
import io
import os
import time
from typing import List, Dict, Optional, Any
import logging
from pathlib import Path

from .endpoints import router as search_router
from .schemas import SearchResponse, HealthResponse, StatsResponse
from ..embeddings.clip_model import get_clip_model, encode_single_image
from ..database.vector_db import get_vector_database
from ..config import Config
from ..utils.logger import get_logger
from ..utils.image_utils import validate_image, normalize_image, resize_image

# Initialize logger
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Visual Search Engine API",
    description="AI-powered visual similarity search for fashion images",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Global variables for model and database
clip_model = None
vector_db = None

# Performance metrics
request_count = 0
total_search_time = 0.0
error_count = 0

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global clip_model, vector_db
    
    try:
        logger.info("Starting up Visual Search API...")
        
        # Initialize CLIP model
        logger.info("Loading CLIP model...")
        try:
            clip_model = get_clip_model()
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            # Continue anyway - model will be loaded on demand
            clip_model = None
        
        # Initialize vector database
        logger.info("Connecting to vector database...")
        try:
            vector_db = get_vector_database()
            logger.info("Vector database connected successfully")
            
            # Try to get existing collection
            try:
                collection = vector_db.get_collection(Config.COLLECTION_NAME)
                logger.info(f"Connected to existing collection: {Config.COLLECTION_NAME}")
            except Exception as e:
                logger.warning(f"Collection {Config.COLLECTION_NAME} not found: {e}")
                logger.info("You may need to run the pipeline first to create the database")
        except Exception as e:
            logger.error(f"Failed to connect to vector database: {e}")
            vector_db = None
        
        logger.info("Visual Search API startup completed")
        
    except Exception as e:
        logger.error(f"Critical error during API startup: {str(e)}")
        # Don't raise - allow API to start in degraded mode

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Visual Search API...")
    
    # Cleanup model memory
    global clip_model
    if clip_model:
        del clip_model
    
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("Shutdown completed")

# Dependency functions
def get_clip_model_instance():
    """Dependency to get CLIP model instance."""
    global clip_model
    if clip_model is None:
        raise HTTPException(status_code=503, detail="CLIP model not initialized")
    return clip_model

def get_vector_db_instance():
    """Dependency to get vector database instance."""
    global vector_db
    if vector_db is None:
        raise HTTPException(status_code=503, detail="Vector database not initialized")
    return vector_db

# API Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AI Visual Search Engine API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global clip_model, vector_db
    
    services = {}
    status = "healthy"
    error_msg = None
    
    # Check CLIP model
    try:
        if clip_model is not None:
            services["clip_model"] = True
            # Try a quick operation to verify it's working
            # Note: This is just checking if the model exists, not doing actual inference
        else:
            services["clip_model"] = False
            status = "degraded"
    except Exception as e:
        services["clip_model"] = False
        status = "degraded"
        error_msg = f"CLIP model error: {str(e)}"
    
    # Check vector database
    try:
        if vector_db is not None:
            services["vector_db"] = True
            # Check if we can access the collection
            try:
                collection = vector_db.get_collection(Config.COLLECTION_NAME)
                services["vector_db_collection"] = True
            except Exception:
                services["vector_db_collection"] = False
                if status == "healthy":
                    status = "degraded"
        else:
            services["vector_db"] = False
            services["vector_db_collection"] = False
            status = "degraded"
    except Exception as e:
        services["vector_db"] = False
        services["vector_db_collection"] = False
        status = "degraded"
        if error_msg:
            error_msg += f" | Vector DB error: {str(e)}"
        else:
            error_msg = f"Vector DB error: {str(e)}"
    
    health_response = HealthResponse(
        status=status,
        timestamp=time.time(),
        services=services,
        error=error_msg
    )
    
    return health_response

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get API usage statistics."""
    global request_count, total_search_time, error_count, vector_db
    
    stats = {
        "total_requests": request_count,
        "total_errors": error_count,
        "average_search_time": total_search_time / max(request_count, 1),
        "error_rate": error_count / max(request_count, 1) * 100
    }
    
    # Add database stats if available
    if vector_db:
        try:
            db_stats = vector_db.get_collection_stats()
            stats.update(db_stats)
        except Exception as e:
            logger.warning(f"Could not get database stats: {e}")
    
    return stats

@app.post("/search", response_model=SearchResponse)
async def search_similar_images(
    file: UploadFile = File(...),
    top_k: int = Config.DEFAULT_TOP_K,
    threshold: float = 0.0,
    vector_db_instance = Depends(get_vector_db_instance),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Search for visually similar images.
    
    Args:
        file: Uploaded image file
        top_k: Number of similar images to return (max 50)
        threshold: Minimum similarity threshold (0.0 to 1.0)
        
    Returns:
        List of similar images with similarity scores
    """
    global request_count, total_search_time, error_count
    
    start_time = time.time()
    request_count += 1
    
    try:
        # Validate parameters
        if top_k > Config.MAX_TOP_K:
            raise HTTPException(status_code=400, detail=f"top_k cannot exceed {Config.MAX_TOP_K}")
        
        if not 0.0 <= threshold <= 1.0:
            raise HTTPException(status_code=400, detail="threshold must be between 0.0 and 1.0")
        
        # Validate file size
        if file.size and file.size > Config.MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"File size exceeds {Config.MAX_FILE_SIZE} bytes")
        
        # Read and process image (this also validates if it's a valid image)
        try:
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            
            # Additional validation: check if file extension is supported
            if file.filename:
                file_ext = os.path.splitext(file.filename.lower())[1]
                if file_ext not in Config.SUPPORTED_FORMATS:
                    raise HTTPException(status_code=400, detail=f"Unsupported file format. Supported: {Config.SUPPORTED_FORMATS}")
            
            image = normalize_image(image)
            image = resize_image(image, Config.IMAGE_SIZE)
        except HTTPException:
            raise  # Re-raise HTTP exceptions
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Generate embedding for query image
        try:
            query_embedding = encode_single_image(image, normalize=True)
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to process image")
        
        # Search for similar images
        try:
            collection = vector_db_instance.get_collection(Config.COLLECTION_NAME)
            results = vector_db_instance.search(
                collection=collection,
                query_embedding=query_embedding,
                top_k=top_k
            )
        except Exception as e:
            logger.error(f"Database search failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Search operation failed")
        
        # Filter by threshold and enrich with metadata
        filtered_results = []
        metadata_df = None
        
        # Load metadata for enriching results
        try:
            metadata_df = pd.read_csv(Config.METADATA_FILE)
        except Exception as e:
            logger.warning(f"Could not load metadata file: {e}")
        
        for result in results:
            similarity = result.get('score', 0.0)
            if similarity >= threshold:
                image_id = result.get('id', '')
                
                # Enrich with metadata if available
                enriched_result = {
                    "image_id": image_id,
                    "similarity": float(similarity),
                    "image_url": f"/images/{image_id}",  # URL to serve the image
                    "metadata": {k: v for k, v in result.items() if k not in ['score', 'id']}
                }
                
                # Add metadata from CSV if available
                if metadata_df is not None:
                    try:
                        image_idx = int(image_id)
                        if 0 <= image_idx < len(metadata_df):
                            row = metadata_df.iloc[image_idx]
                            enriched_result["metadata"].update({
                                "image_name": row.get('image_name', ''),
                                "category_name": row.get('category_name', ''),
                                "category_label": str(row.get('category_label', '')),
                                "evaluation_status": row.get('evaluation_status', ''),
                                "image_path": row.get('image_path', '')
                            })
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Could not add metadata for ID {image_id}: {e}")
                
                filtered_results.append(enriched_result)
        
        # Calculate search time
        search_time = time.time() - start_time
        total_search_time += search_time
        
        # Prepare response
        response_data = {
            "query_id": f"query_{int(time.time())}",
            "results": filtered_results,
            "total_results": len(filtered_results),
            "search_time_ms": search_time * 1000,
            "parameters": {
                "top_k": top_k,
                "threshold": threshold
            }
        }
        
        # Log search metrics in background
        background_tasks.add_task(
            log_search_metrics,
            query_image_size=len(image_data),
            results_count=len(filtered_results),
            search_time=search_time
        )
        
        return response_data
        
    except HTTPException:
        error_count += 1
        raise
    except Exception as e:
        error_count += 1
        logger.error(f"Unexpected error in search: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/search/batch")
async def search_batch_images(
    files: List[UploadFile] = File(...),
    top_k: int = Config.DEFAULT_TOP_K,
    threshold: float = 0.0,
    vector_db_instance = Depends(get_vector_db_instance)
):
    """
    Search for similar images using multiple query images.
    
    Args:
        files: List of uploaded image files
        top_k: Number of similar images to return per query
        threshold: Minimum similarity threshold
        
    Returns:
        Batch search results
    """
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    batch_results = []
    
    for i, file in enumerate(files):
        try:
            # Validate file extension if filename is available
            if file.filename:
                file_ext = os.path.splitext(file.filename.lower())[1]
                if file_ext not in Config.SUPPORTED_FORMATS:
                    batch_results.append({
                        "query_index": i,
                        "query_filename": file.filename,
                        "error": f"Unsupported file format. Supported: {Config.SUPPORTED_FORMATS}",
                        "results": []
                    })
                    continue
            
            # Process each image (similar to single search)
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            image = normalize_image(image)
            
            query_embedding = encode_single_image(image, normalize=True)
            collection = vector_db_instance.get_collection(Config.COLLECTION_NAME)
            results = vector_db_instance.search(
                collection=collection,
                query_embedding=query_embedding,
                top_k=top_k
            )
            
            # Filter results
            filtered_results = []
            metadata_df = None
            
            # Load metadata for enriching results
            try:
                metadata_df = pd.read_csv(Config.METADATA_FILE)
            except Exception as e:
                logger.warning(f"Could not load metadata file: {e}")
            
            for result in results:
                similarity = result.get('score', 0.0)
                if similarity >= threshold:
                    image_id = result.get('id', '')
                    
                    enriched_result = {
                        "image_id": image_id,
                        "similarity": float(similarity),
                        "image_url": f"/images/{image_id}"
                    }
                    
                    # Add metadata from CSV if available
                    if metadata_df is not None:
                        try:
                            image_idx = int(image_id)
                            if 0 <= image_idx < len(metadata_df):
                                row = metadata_df.iloc[image_idx]
                                enriched_result["category_name"] = row.get('category_name', '')
                        except (ValueError, IndexError):
                            pass
                    
                    filtered_results.append(enriched_result)
            
            batch_results.append({
                "query_index": i,
                "query_filename": file.filename,
                "results": filtered_results,
                "total_results": len(filtered_results)
            })
            
        except Exception as e:
            batch_results.append({
                "query_index": i,
                "query_filename": file.filename,
                "error": str(e),
                "results": [],
                "total_results": 0
            })
    
    return {
        "batch_id": f"batch_{int(time.time())}",
        "queries": batch_results,
        "total_queries": len(files),
        "successful_queries": len([r for r in batch_results if "error" not in r])
    }

@app.get("/collection/info")
async def get_collection_info(
    vector_db_instance = Depends(get_vector_db_instance)
):
    """Get information about the current collection."""
    try:
        stats = vector_db_instance.get_collection_stats()
        return {
            "collection_info": stats,
            "model_info": {
                "model_name": Config.CLIP_MODEL_NAME,
                "embedding_dimension": Config.EMBEDDING_DIM,
                "image_size": Config.IMAGE_SIZE
            }
        }
    except Exception as e:
        logger.error(f"Failed to get collection info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get collection info: {str(e)}")

@app.post("/collection/add")
async def add_images_to_collection(
    files: List[UploadFile] = File(...),
    image_ids: Optional[List[str]] = None,
    vector_db_instance = Depends(get_vector_db_instance)
):
    """
    Add new images to the collection.
    Currently disabled - use the pipeline script to add images to the collection.
    """
    raise HTTPException(status_code=501, detail="Adding images via API is not yet implemented. Use the pipeline script instead.")

@app.get("/images/{image_id}")
async def get_image_by_id(image_id: str, vector_db_instance = Depends(get_vector_db_instance)):
    """
    Serve an image by its ID from the collection.
    """
    try:
        # Load metadata to get image path
        import pandas as pd
        metadata_df = pd.read_csv(Config.METADATA_FILE)
        
        # Find the image by ID (the ID corresponds to the row index)
        try:
            image_idx = int(image_id)
            if 0 <= image_idx < len(metadata_df):
                image_path = metadata_df.iloc[image_idx]['image_path']
                
                # Convert relative path to absolute path
                if image_path.startswith('./'):
                    image_path = image_path[2:]  # Remove './'
                
                full_path = os.path.join(Config.ROOT_DIR, image_path)
                
                # Check if file exists
                if os.path.exists(full_path):
                    return FileResponse(
                        full_path,
                        media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=3600"}  # Cache for 1 hour
                    )
                else:
                    raise HTTPException(status_code=404, detail="Image file not found on disk")
            else:
                raise HTTPException(status_code=404, detail="Image ID out of range")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid image ID format")
            
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Metadata file not found")
    except Exception as e:
        logger.error(f"Error serving image {image_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error serving image")

# Background task functions
async def log_search_metrics(query_image_size: int, results_count: int, search_time: float):
    """Log search metrics for monitoring."""
    logger.info(
        f"Search metrics - Image size: {query_image_size} bytes, "
        f"Results: {results_count}, Time: {search_time:.3f}s"
    )

# Include additional routers
app.include_router(search_router, prefix="/api/v1", tags=["search"])

# Static file serving (for demo purposes)
static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "detail": "The requested endpoint does not exist"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"}
    )

# Custom middleware for request logging
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Log request
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    return response

# Run server
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        workers=1,  # Use 1 worker for development
        reload=True,  # Enable reload for development
        log_level=Config.LOG_LEVEL.lower()
    )
    