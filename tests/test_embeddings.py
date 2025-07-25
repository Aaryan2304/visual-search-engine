"""
Tests for embedding generation and CLIP model functionality.
"""

import pytest
import numpy as np
import torch
from PIL import Image
from pathlib import Path
import tempfile
import pandas as pd

# Import modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.clip_model import CLIPModelWrapper, get_clip_model
from src.embeddings.generator import EmbeddingGenerator
from src.config import Config

class TestCLIPModelWrapper:
    """Test CLIP model wrapper functionality."""
    
    @pytest.fixture
    def clip_model(self):
        """Create a CLIP model instance for testing."""
        model = CLIPModelWrapper(use_half_precision=False)  # Disable FP16 for testing
        model.load_model()
        return model
    
    @pytest.fixture
    def test_image(self):
        """Create a test image."""
        return Image.new('RGB', (224, 224), color='red')
    
    def test_model_initialization(self, clip_model):
        """Test that the model initializes correctly."""
        assert clip_model._is_loaded
        assert clip_model.model is not None
        assert clip_model.processor is not None
    
    def test_single_image_encoding(self, clip_model, test_image):
        """Test encoding a single image."""
        embedding = clip_model.encode_single_image(test_image)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (Config.EMBEDDING_DIM,)
        assert not np.isnan(embedding).any()
        assert not np.isinf(embedding).any()
    
    def test_batch_image_encoding(self, clip_model, test_image):
        """Test encoding multiple images."""
        images = [test_image] * 3
        embeddings = clip_model.encode_images(images)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, Config.EMBEDDING_DIM)
        assert not np.isnan(embeddings).any()
        assert not np.isinf(embeddings).any()
    
    def test_embedding_normalization(self, clip_model, test_image):
        """Test that embeddings are properly normalized."""
        embedding = clip_model.encode_single_image(test_image, normalize=True)
        
        # Check that the embedding is normalized (L2 norm should be close to 1)
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-5
    
    def test_similarity_computation(self, clip_model, test_image):
        """Test similarity computation between embeddings."""
        # Create two identical images
        embedding1 = clip_model.encode_single_image(test_image)
        embedding2 = clip_model.encode_single_image(test_image)
        
        # Create database of embeddings
        database_embeddings = np.array([embedding1, embedding2])
        
        # Compute similarities
        similarities, indices = clip_model.compute_similarity(
            embedding1, database_embeddings, top_k=2
        )
        
        assert len(similarities) == 2
        assert len(indices) == 2
        assert similarities[0] > 0.99  # Should be very similar to itself
    
    def test_model_info(self, clip_model):
        """Test model information retrieval."""
        info = clip_model.get_model_info()
        
        assert isinstance(info, dict)
        assert 'model_name' in info
        assert 'is_loaded' in info
        assert 'embedding_dim' in info
        assert info['is_loaded'] is True

class TestEmbeddingGenerator:
    """Test embedding generation pipeline."""
    
    @pytest.fixture
    def temp_metadata_file(self):
        """Create temporary metadata file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Create simple test metadata
            df = pd.DataFrame({
                'image_id': ['test_001', 'test_002'],
                'filepath': ['test1.jpg', 'test2.jpg'],
                'width': [224, 224],
                'height': [224, 224]
            })
            df.to_csv(f.name, index=False)
            yield Path(f.name)
            
        # Cleanup
        Path(f.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def temp_images_dir(self):
        """Create temporary directory with test images."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test images
            for i in range(2):
                img = Image.new('RGB', (224, 224), color=(i*100, 0, 0))
                img.save(temp_path / f'test{i+1}.jpg')
            
            yield temp_path
    
    def test_generator_initialization(self, temp_metadata_file):
        """Test embedding generator initialization."""
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as output_file:
            generator = EmbeddingGenerator(
                metadata_file=temp_metadata_file,
                output_file=Path(output_file.name),
                batch_size=2
            )
            
            assert generator.metadata_file == temp_metadata_file
            assert generator.batch_size == 2
            
        Path(output_file.name).unlink(missing_ok=True)
    
    def test_metadata_loading(self, temp_metadata_file):
        """Test loading metadata from CSV."""
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as output_file:
            generator = EmbeddingGenerator(
                metadata_file=temp_metadata_file,
                output_file=Path(output_file.name)
            )
            
            metadata = generator.load_metadata()
            
            assert len(metadata) == 2
            assert 'image_id' in metadata.columns
            assert 'filepath' in metadata.columns
            
        Path(output_file.name).unlink(missing_ok=True)

class TestGlobalModelAccess:
    """Test global model access functions."""
    
    def test_get_clip_model_singleton(self):
        """Test that get_clip_model returns the same instance."""
        model1 = get_clip_model()
        model2 = get_clip_model()
        
        assert model1 is model2  # Should be the same instance
    
    def test_model_reload(self):
        """Test forcing model reload."""
        model1 = get_clip_model()
        model2 = get_clip_model(force_reload=True)
        
        # After force reload, should be different instances
        assert model1 is not model2

class TestErrorHandling:
    """Test error handling in embedding functionality."""
    
    def test_invalid_image_handling(self):
        """Test handling of invalid images."""
        model = CLIPModelWrapper(use_half_precision=False)
        model.load_model()
        
        # Test with invalid image data
        with pytest.raises((ValueError, TypeError)):
            model.encode_single_image("not_an_image")
    
    def test_empty_batch_handling(self):
        """Test handling of empty image batches."""
        model = CLIPModelWrapper(use_half_precision=False)
        model.load_model()
        
        # Test with empty list
        embeddings = model.encode_images([])
        assert len(embeddings) == 0
    
    def test_model_not_loaded_error(self):
        """Test error when model is not loaded."""
        model = CLIPModelWrapper()
        # Don't load the model
        
        test_image = Image.new('RGB', (224, 224), color='red')
        
        with pytest.raises(RuntimeError):
            model.encode_single_image(test_image)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDAFunctionality:
    """Test CUDA-specific functionality."""
    
    def test_cuda_model_loading(self):
        """Test loading model on CUDA."""
        model = CLIPModelWrapper(device='cuda')
        model.load_model()
        
        assert model.device == 'cuda'
        assert next(model.model.parameters()).is_cuda
    
    def test_memory_optimization(self):
        """Test memory optimization features."""
        model = CLIPModelWrapper(device='cuda', use_half_precision=True)
        model.load_model()
        
        # Check that model is using half precision
        assert model.use_half_precision
        
        # Test memory cleanup
        initial_memory = torch.cuda.memory_allocated()
        
        # Generate some embeddings
        test_image = Image.new('RGB', (224, 224), color='red')
        for _ in range(10):
            model.encode_single_image(test_image)
        
        # Memory should not grow indefinitely
        final_memory = torch.cuda.memory_allocated()
        memory_growth = final_memory - initial_memory
        
        # Allow some growth but not excessive
        assert memory_growth < 500 * 1024 * 1024  # Less than 500MB growth

if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])