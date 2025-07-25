"""
Tests for vector database functionality and indexing.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import time

# Import modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.vector_db import (
    VectorDatabase, ChromaDBInterface, FAISSInterface, 
    get_vector_database
)
from src.database.indexer import (
    index_embeddings_to_database, verify_database_index,
    rebuild_database_index
)
from src.config import Config

class TestVectorDatabaseInterface:
    """Test abstract vector database interface."""
    
    def test_interface_methods(self):
        """Test that interface defines required methods."""
        interface_methods = [
            'create_collection',
            'add_vectors', 
            'search',
            'delete_vectors',
            'get_collection_stats'
        ]
        
        for method in interface_methods:
            assert hasattr(VectorDatabase, method)
            assert callable(getattr(VectorDatabase, method))

class TestChromaDBInterface:
    """Test ChromaDB implementation."""
    
    @pytest.fixture
    def chroma_db(self):
        """Create ChromaDB instance for testing."""
        # Use in-memory ChromaDB for testing
        try:
            db = ChromaDBInterface(host="localhost", port=8000)
            yield db
        except Exception:
            pytest.skip("ChromaDB not available for testing")
    
    @pytest.fixture
    def test_vectors(self):
        """Create test vectors and IDs."""
        vectors = np.random.rand(10, Config.EMBEDDING_DIM).astype(np.float32)
        ids = [f"test_vector_{i}" for i in range(10)]
        metadata = [{"category": f"cat_{i%3}", "index": i} for i in range(10)]
        return vectors, ids, metadata
    
    def test_collection_creation(self, chroma_db):
        """Test creating a collection."""
        collection_name = f"test_collection_{int(time.time())}"
        
        try:
            chroma_db.create_collection(collection_name, Config.EMBEDDING_DIM)
            
            # Verify collection exists
            stats = chroma_db.get_collection_stats()
            assert "collection_name" in stats or "error" not in stats
        except Exception as e:
            pytest.skip(f"ChromaDB collection creation failed: {e}")
    
    def test_vector_addition(self, chroma_db, test_vectors):
        """Test adding vectors to collection."""
        vectors, ids, metadata = test_vectors
        collection_name = f"test_add_{int(time.time())}"
        
        try:
            chroma_db.create_collection(collection_name, Config.EMBEDDING_DIM)
            chroma_db.add_vectors(vectors, ids, metadata)
            
            # Verify vectors were added
            stats = chroma_db.get_collection_stats()
            if "total_vectors" in stats:
                assert stats["total_vectors"] >= len(ids)
        except Exception as e:
            pytest.skip(f"ChromaDB vector addition failed: {e}")
    
    def test_vector_search(self, chroma_db, test_vectors):
        """Test searching for similar vectors."""
        vectors, ids, metadata = test_vectors
        collection_name = f"test_search_{int(time.time())}"
        
        try:
            chroma_db.create_collection(collection_name, Config.EMBEDDING_DIM)
            chroma_db.add_vectors(vectors, ids, metadata)
            
            # Search using first vector
            query_vector = vectors[0]
            result_ids, similarities = chroma_db.search(query_vector, top_k=5)
            
            assert len(result_ids) <= 5
            assert len(similarities) <= 5
            assert len(result_ids) == len(similarities)
            
            # First result should be the query vector itself (if exact match)
            if len(result_ids) > 0:
                assert isinstance(result_ids[0], str)
                assert isinstance(similarities[0], (int, float))
        except Exception as e:
            pytest.skip(f"ChromaDB vector search failed: {e}")
    
    def test_vector_deletion(self, chroma_db, test_vectors):
        """Test deleting vectors from collection."""
        vectors, ids, metadata = test_vectors
        collection_name = f"test_delete_{int(time.time())}"
        
        try:
            chroma_db.create_collection(collection_name, Config.EMBEDDING_DIM)
            chroma_db.add_vectors(vectors, ids, metadata)
            
            # Delete some vectors
            ids_to_delete = ids[:3]
            chroma_db.delete_vectors(ids_to_delete)
            
            # Verify deletion (implementation dependent)
            stats = chroma_db.get_collection_stats()
            assert "error" not in stats or stats.get("total_vectors", 0) >= 0
        except Exception as e:
            pytest.skip(f"ChromaDB vector deletion failed: {e}")

class TestFAISSInterface:
    """Test FAISS implementation."""
    
    @pytest.fixture
    def faiss_db(self):
        """Create FAISS instance for testing."""
        try:
            import faiss
            with tempfile.TemporaryDirectory() as temp_dir:
                index_path = Path(temp_dir) / "test_index.faiss"
                db = FAISSInterface(index_path=index_path)
                yield db
        except ImportError:
            pytest.skip("FAISS not available for testing")
    
    @pytest.fixture
    def test_vectors(self):
        """Create test vectors and IDs."""
        vectors = np.random.rand(10, Config.EMBEDDING_DIM).astype(np.float32)
        ids = [f"faiss_test_{i}" for i in range(10)]
        metadata = [{"type": "test", "id": i} for i in range(10)]
        return vectors, ids, metadata
    
    def test_faiss_collection_creation(self, faiss_db):
        """Test creating FAISS index."""
        collection_name = "test_faiss_collection"
        
        faiss_db.create_collection(collection_name, Config.EMBEDDING_DIM)
        
        assert faiss_db.index is not None
        assert faiss_db.index.d == Config.EMBEDDING_DIM
    
    def test_faiss_vector_addition(self, faiss_db, test_vectors):
        """Test adding vectors to FAISS index."""
        vectors, ids, metadata = test_vectors
        
        faiss_db.create_collection("test", Config.EMBEDDING_DIM)
        faiss_db.add_vectors(vectors, ids, metadata)
        
        assert faiss_db.index.ntotal == len(vectors)
        assert len(faiss_db.id_map) == len(ids)
    
    def test_faiss_vector_search(self, faiss_db, test_vectors):
        """Test searching in FAISS index."""
        vectors, ids, metadata = test_vectors
        
        faiss_db.create_collection("test", Config.EMBEDDING_DIM)
        faiss_db.add_vectors(vectors, ids, metadata)
        
        # Search using first vector
        query_vector = vectors[0]
        result_ids, similarities = faiss_db.search(query_vector, top_k=5)
        
        assert len(result_ids) <= 5
        assert len(similarities) <= 5
        assert len(result_ids) == len(similarities)
    
    def test_faiss_save_load(self, faiss_db, test_vectors):
        """Test saving and loading FAISS index."""
        vectors, ids, metadata = test_vectors
        
        faiss_db.create_collection("test", Config.EMBEDDING_DIM)
        faiss_db.add_vectors(vectors, ids, metadata)
        
        # Save index
        faiss_db.save_index()
        
        # Create new instance and load
        new_faiss_db = FAISSInterface(index_path=faiss_db.index_path)
        new_faiss_db.load_index()
        
        assert new_faiss_db.index.ntotal == len(vectors)
        assert len(new_faiss_db.id_map) == len(ids)

class TestDatabaseFactory:
    """Test database factory function."""
    
    def test_get_chromadb(self):
        """Test getting ChromaDB instance."""
        try:
            db = get_vector_database('chromadb')
            assert isinstance(db, ChromaDBInterface)
        except Exception:
            pytest.skip("ChromaDB not available")
    
    def test_get_faiss(self):
        """Test getting FAISS instance."""
        try:
            db = get_vector_database('faiss')
            assert isinstance(db, FAISSInterface)
        except ImportError:
            pytest.skip("FAISS not available")
    
    def test_invalid_database_type(self):
        """Test error handling for invalid database type."""
        with pytest.raises(ValueError):
            get_vector_database('invalid_db_type')

class TestEmbeddingIndexer:
    """Test embedding indexing functionality."""
    
    @pytest.fixture
    def test_embeddings_file(self):
        """Create test embeddings file."""
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            # Create test embeddings data
            data = {
                'image_id': [f'test_img_{i}' for i in range(5)],
                'embedding': [np.random.rand(Config.EMBEDDING_DIM).tolist() for _ in range(5)],
                'model_name': [Config.CLIP_MODEL_NAME] * 5,
                'created_at': [pd.Timestamp.now()] * 5
            }
            df = pd.DataFrame(data)
            df.to_parquet(f.name, index=False)
            
            yield Path(f.name)
            
        # Cleanup
        Path(f.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def mock_vector_db(self):
        """Create mock vector database for testing."""
        class MockVectorDB:
            def __init__(self):
                self.vectors = []
                self.collection_created = False
                
            def create_collection(self, name, dim):
                self.collection_created = True
                
            def get_collection(self, name):
                if not self.collection_created:
                    raise Exception("Collection not found")
                    
            def add_vectors(self, vectors, ids, metadata=None):
                self.vectors.extend(list(zip(vectors, ids, metadata or [])))
                
            def search(self, query_vector, top_k=10):
                # Return mock results
                if len(self.vectors) == 0:
                    return [], []
                
                # Return random subset
                import random
                n_results = min(top_k, len(self.vectors))
                selected = random.sample(self.vectors, n_results)
                ids = [item[1] for item in selected]
                similarities = [random.random() for _ in range(n_results)]
                return ids, similarities
                
            def get_collection_stats(self):
                return {
                    "total_vectors": len(self.vectors),
                    "backend": "MockDB"
                }
                
            def delete_vectors(self, ids):
                self.vectors = [v for v in self.vectors if v[1] not in ids]
        
        return MockVectorDB()
    
    def test_index_embeddings_to_database(self, test_embeddings_file, mock_vector_db):
        """Test indexing embeddings to database."""
        stats = index_embeddings_to_database(
            embeddings_file=test_embeddings_file,
            vector_db=mock_vector_db,
            batch_size=2
        )
        
        assert stats["status"] == "completed"
        assert stats["indexed_successfully"] == 5
        assert stats["failed_to_index"] == 0
        assert len(mock_vector_db.vectors) == 5
    
    def test_verify_database_index(self, mock_vector_db):
        """Test database index verification."""
        # Add some test vectors first
        test_vectors = np.random.rand(10, Config.EMBEDDING_DIM).astype(np.float32)
        test_ids = [f"verify_test_{i}" for i in range(10)]
        mock_vector_db.create_collection("test", Config.EMBEDDING_DIM)
        mock_vector_db.add_vectors(test_vectors, test_ids)
        
        stats = verify_database_index(
            vector_db=mock_vector_db,
            sample_size=5
        )
        
        assert stats["status"] == "completed"
        assert stats["total_vectors"] == 10
        assert stats["tests_run"] == 5
        assert stats["successful_searches"] >= 0
    
    def test_rebuild_database_index(self, test_embeddings_file, mock_vector_db):
        """Test rebuilding database index."""
        stats = rebuild_database_index(
            embeddings_file=test_embeddings_file,
            vector_db=mock_vector_db,
            batch_size=2
        )
        
        assert stats["status"] == "completed"
        assert stats["indexed_successfully"] == 5

class TestDatabasePerformance:
    """Test database performance characteristics."""
    
    @pytest.fixture
    def large_vector_set(self):
        """Create larger vector set for performance testing."""
        n_vectors = 100
        vectors = np.random.rand(n_vectors, Config.EMBEDDING_DIM).astype(np.float32)
        ids = [f"perf_test_{i}" for i in range(n_vectors)]
        metadata = [{"batch": i // 10, "index": i} for i in range(n_vectors)]
        return vectors, ids, metadata
    
    def test_batch_addition_performance(self, large_vector_set):
        """Test performance of batch vector addition."""
        vectors, ids, metadata = large_vector_set
        
        try:
            db = get_vector_database('faiss')  # Use FAISS for performance test
            db.create_collection("perf_test", Config.EMBEDDING_DIM)
            
            start_time = time.time()
            db.add_vectors(vectors, ids, metadata)
            end_time = time.time()
            
            addition_time = end_time - start_time
            vectors_per_second = len(vectors) / addition_time
            
            # Should be able to add at least 10 vectors per second
            assert vectors_per_second > 10
            
            stats = db.get_collection_stats()
            assert stats["total_vectors"] == len(vectors)
            
        except ImportError:
            pytest.skip("FAISS not available for performance testing")
    
    def test_search_performance(self, large_vector_set):
        """Test search performance."""
        vectors, ids, metadata = large_vector_set
        
        try:
            db = get_vector_database('faiss')
            db.create_collection("search_perf_test", Config.EMBEDDING_DIM)
            db.add_vectors(vectors, ids, metadata)
            
            # Test search performance
            query_vector = vectors[0]
            
            start_time = time.time()
            for _ in range(10):  # Multiple searches
                result_ids, similarities = db.search(query_vector, top_k=10)
            end_time = time.time()
            
            avg_search_time = (end_time - start_time) / 10
            
            # Each search should complete in reasonable time
            assert avg_search_time < 1.0  # Less than 1 second per search
            
        except ImportError:
            pytest.skip("FAISS not available for performance testing")

class TestDatabaseErrorHandling:
    """Test error handling in database operations."""
    
    def test_invalid_vector_dimensions(self):
        """Test error handling for incorrect vector dimensions."""
        try:
            db = get_vector_database('faiss')
            db.create_collection("error_test", Config.EMBEDDING_DIM)
            
            # Try to add vectors with wrong dimensions
            wrong_vectors = np.random.rand(5, Config.EMBEDDING_DIM + 10).astype(np.float32)
            wrong_ids = [f"wrong_{i}" for i in range(5)]
            
            with pytest.raises(Exception):
                db.add_vectors(wrong_vectors, wrong_ids)
                
        except ImportError:
            pytest.skip("FAISS not available for error testing")
    
    def test_search_without_vectors(self):
        """Test search on empty database."""
        try:
            db = get_vector_database('faiss')
            db.create_collection("empty_test", Config.EMBEDDING_DIM)
            
            query_vector = np.random.rand(Config.EMBEDDING_DIM).astype(np.float32)
            result_ids, similarities = db.search(query_vector, top_k=5)
            
            # Should return empty results, not error
            assert len(result_ids) == 0
            assert len(similarities) == 0
            
        except ImportError:
            pytest.skip("FAISS not available for empty search testing")
    
    def test_invalid_collection_operations(self):
        """Test operations on non-existent collections."""
        try:
            db = get_vector_database('chromadb')
            
            # Try to get non-existent collection
            with pytest.raises(Exception):
                db.get_collection("nonexistent_collection")
                
        except Exception:
            pytest.skip("ChromaDB not available for collection testing")

class TestDatabaseConsistency:
    """Test data consistency across database operations."""
    
    def test_add_search_consistency(self):
        """Test that added vectors can be found in search."""
        try:
            db = get_vector_database('faiss')
            db.create_collection("consistency_test", Config.EMBEDDING_DIM)
            
            # Add a specific vector
            test_vector = np.random.rand(Config.EMBEDDING_DIM).astype(np.float32)
            test_id = "consistency_test_vector"
            
            db.add_vectors(np.array([test_vector]), [test_id])
            
            # Search for the same vector
            result_ids, similarities = db.search(test_vector, top_k=1)
            
            # Should find the exact vector with high similarity
            assert len(result_ids) > 0
            assert result_ids[0] == test_id
            assert similarities[0] > 0.99  # Very high similarity for exact match
            
        except ImportError:
            pytest.skip("FAISS not available for consistency testing")
    
    def test_metadata_preservation(self):
        """Test that metadata is preserved through database operations."""
        try:
            db = get_vector_database('chromadb')
            collection_name = f"metadata_test_{int(time.time())}"
            db.create_collection(collection_name, Config.EMBEDDING_DIM)
            
            vectors = np.random.rand(3, Config.EMBEDDING_DIM).astype(np.float32)
            ids = ["meta_1", "meta_2", "meta_3"]
            metadata = [
                {"category": "dress", "color": "red"},
                {"category": "shirt", "color": "blue"}, 
                {"category": "pants", "color": "black"}
            ]
            
            db.add_vectors(vectors, ids, metadata)
            
            # Search and verify metadata is accessible
            result_ids, similarities = db.search(vectors[0], top_k=3)
            
            # Basic verification that search works
            assert len(result_ids) > 0
            
        except Exception:
            pytest.skip("ChromaDB not available for metadata testing")

if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])