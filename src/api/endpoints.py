"""
Additional API endpoints for the Visual Search Engine.
This module contains additional routes beyond the main endpoints in main.py.
"""

from fastapi import APIRouter, HTTPException, File, UploadFile, Depends
from typing import List
import logging

from .schemas import SearchResponse
from ..embeddings.clip_model import get_clip_model
from ..database.vector_db import get_vector_database
from ..config import Config

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Note: Main search endpoints are implemented in main.py
# This router can be used for additional endpoints if needed

@router.get("/version")
async def get_api_version():
    """Get API version information."""
    return {
        "api_version": "1.0.0",
        "status": "operational",
        "endpoints": ["search", "batch_search", "health", "stats"]
    }

@router.get("/models/info")
async def get_model_info():
    """Get information about loaded models."""
    return {
        "clip_model": Config.CLIP_MODEL_NAME,
        "embedding_dimension": Config.EMBEDDING_DIM,
        "image_size": Config.IMAGE_SIZE,
        "device": Config.DEVICE
    }
