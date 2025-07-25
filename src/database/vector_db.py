import os
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from ..config import Config
from ..utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


class VectorDatabase(ABC):
    """Abstract base class for vector database operations."""

    @abstractmethod
    def get_collection(self, name: str) -> Any:
        """Get or create a collection, index, or table."""
        pass

    @abstractmethod
    def upsert(self, collection: Any, embeddings: np.ndarray, metadata: List[Dict], ids: List[str]):
        """Upsert embeddings and metadata into the collection."""
        pass

    @abstractmethod
    def search(self, collection: Any, query_embedding: np.ndarray, top_k: int) -> List[Any]:
        """Search for similar embeddings."""
        pass


class FAISSDatabase(VectorDatabase):
    """Simple FAISS-based vector database implementation."""
    
    def __init__(self, persist_directory: str):
        """Initialize FAISS database with persistent storage."""
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        self.index_path = os.path.join(persist_directory, "faiss_index.bin")
        self.metadata_path = os.path.join(persist_directory, "metadata.json")
        self.id_mapping_path = os.path.join(persist_directory, "id_mapping.json")
        
        self.index = None
        self.metadata = {}
        self.id_to_idx = {}
        self.idx_to_id = {}
        self.next_idx = 0
        
        logger.info(f"Initializing FAISS database with storage at: {persist_directory}")
        
    def get_collection(self, name: str = "default") -> "FAISSDatabase":
        """Return self as the collection."""
        return self
        
    def upsert(self, collection: Any, embeddings: np.ndarray, metadata: List[Dict], ids: List[str]):
        """Add embeddings and metadata to the FAISS index."""
        logger.info(f"Upserting {len(embeddings)} embeddings to FAISS index")
        
        # Initialize index if not exists
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            logger.info(f"Created new FAISS index with dimension {dimension}")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        start_idx = self.next_idx
        self.index.add(embeddings)
        
        # Store metadata and ID mappings
        for i, (metadata_item, id_str) in enumerate(zip(metadata, ids)):
            idx = start_idx + i
            self.metadata[idx] = metadata_item
            self.id_to_idx[id_str] = idx
            self.idx_to_id[idx] = id_str
            
        self.next_idx += len(embeddings)
        logger.info(f"Added {len(embeddings)} embeddings. Total: {self.index.ntotal}")
        
    def search(self, collection: Any, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """Search for similar embeddings."""
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty, returning empty results")
            return []
            
        # Normalize query for cosine similarity
        query_norm = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_norm)
        
        # Search
        scores, indices = self.index.search(query_norm, min(top_k, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx in self.metadata:
                result = self.metadata[idx].copy()
                result['score'] = float(score)
                result['id'] = self.idx_to_id[idx]
                results.append(result)
                
        return results
        
    def save(self):
        """Save the index and metadata to disk."""
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
            logger.info(f"Saved FAISS index to {self.index_path}")
            
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
            
        with open(self.id_mapping_path, 'w') as f:
            json.dump({
                'id_to_idx': self.id_to_idx,
                'idx_to_id': self.idx_to_id,
                'next_idx': self.next_idx
            }, f, indent=2)
            
        logger.info("Saved metadata and mappings")
        
    def load(self):
        """Load the index and metadata from disk."""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            logger.info(f"Loaded FAISS index from {self.index_path} with {self.index.ntotal} embeddings")
            
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                # Convert string keys back to integers
                metadata_str_keys = json.load(f)
                self.metadata = {int(k): v for k, v in metadata_str_keys.items()}
                
        if os.path.exists(self.id_mapping_path):
            with open(self.id_mapping_path, 'r') as f:
                mappings = json.load(f)
                self.id_to_idx = mappings['id_to_idx']
                # Convert string keys back to integers
                self.idx_to_id = {int(k): v for k, v in mappings['idx_to_id'].items()}
                self.next_idx = mappings['next_idx']
                
        logger.info(f"Loaded metadata for {len(self.metadata)} items")
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        stats = {
            "total_embeddings": self.index.ntotal if self.index else 0,
            "total_vectors": self.index.ntotal if self.index else 0,  # Add alias for frontend compatibility
            "embedding_dimension": self.index.d if self.index else 0,
            "metadata_count": len(self.metadata),
            "id_mappings_count": len(self.id_to_idx),
            "backend": "FAISS",  # Add backend information
            "collection_name": "visual_search_engine"  # Add collection name
        }
        return stats


def get_vector_database() -> VectorDatabase:
    """Factory function to get the configured vector database."""
    logger.info("Initializing FAISS vector database")
    
    # Use FAISS for simplicity and reliability
    db = FAISSDatabase(persist_directory=Config.CHROMA_PERSIST_DIRECTORY)
    db.load()  # Load existing data if available
    return db
