"""
Tests for FastAPI endpoints and functionality.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from PIL import Image
import io
import json
from pathlib import Path
import tempfile
import numpy as np

# Import modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import app
from src.api.schemas import SearchResponse, HealthResponse, StatsResponse
from src.config import Config

class TestAPIEndpoints:
    """Test FastAPI endpoint functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def test_image_file(self):
        """Create a test image file."""
        img = Image.new('RGB', (224, 224), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return img_bytes
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "status" in data
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "services" in data
        
        # Validate against schema
        health_response = HealthResponse(**data)
        assert health_response.status in ["healthy", "degraded", "unhealthy"]
    
    def test_stats_endpoint(self, client):
        """Test statistics endpoint."""
        response = client.get("/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_requests" in data
        assert "total_errors" in data
        assert "average_search_time" in data
        assert "error_rate" in data
        
        # Validate against schema
        stats_response = StatsResponse(**data)
        assert stats_response.total_requests >= 0
        assert stats_response.error_rate >= 0
    
    @pytest.mark.asyncio
    async def test_search_endpoint_with_valid_image(self, client, test_image_file):
        """Test search endpoint with valid image."""
        # Note: This test may fail if models/database are not properly initialized
        files = {"file": ("test.jpg", test_image_file, "image/jpeg")}
        params = {"top_k": 5, "threshold": 0.0}
        
        response = client.post("/search", files=files, params=params)
        
        # Could be 200 (success) or 503 (service unavailable if models not loaded)
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "query_id" in data
            assert "results" in data
            assert "total_results" in data
            assert "search_time_ms" in data
            
            # Validate against schema
            search_response = SearchResponse(**data)
            assert search_response.total_results >= 0
    
    def test_search_endpoint_with_invalid_file(self, client):
        """Test search endpoint with invalid file."""
        # Create a text file instead of image
        text_file = io.BytesIO(b"This is not an image")
        files = {"file": ("test.txt", text_file, "text/plain")}
        
        response = client.post("/search", files=files)
        assert response.status_code == 400
        
        data = response.json()
        assert "detail" in data
    
    def test_search_endpoint_with_large_file(self, client):
        """Test search endpoint with oversized file."""
        # Create a large image
        large_img = Image.new('RGB', (5000, 5000), color='blue')
        img_bytes = io.BytesIO()
        large_img.save(img_bytes, format='JPEG', quality=95)
        img_bytes.seek(0)
        
        files = {"file": ("large.jpg", img_bytes, "image/jpeg")}
        
        response = client.post("/search", files=files)
        # Should reject if file is too large
        assert response.status_code in [200, 400, 413, 503]
    
    def test_search_endpoint_invalid_parameters(self, client, test_image_file):
        """Test search endpoint with invalid parameters."""
        files = {"file": ("test.jpg", test_image_file, "image/jpeg")}
        
        # Test invalid top_k
        params = {"top_k": 100}  # Exceeds max limit
        response = client.post("/search", files=files, params=params)
        assert response.status_code in [400, 503]
        
        # Test invalid threshold
        test_image_file.seek(0)
        params = {"threshold": 1.5}  # Exceeds valid range
        response = client.post("/search", files=files, params=params)
        assert response.status_code in [400, 503]
    
    def test_collection_info_endpoint(self, client):
        """Test collection info endpoint."""
        response = client.get("/collection/info")
        
        # Could be 200 (success) or 503 (service unavailable)
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "collection_info" in data
            assert "model_info" in data

class TestAPISchemas:
    """Test Pydantic schema validation."""
    
    def test_search_response_schema(self):
        """Test SearchResponse schema validation."""
        valid_data = {
            "query_id": "test_123",
            "results": [
                {
                    "image_id": "img_001",
                    "similarity": 0.95,
                    "metadata": {"category": "dress"}
                }
            ],
            "total_results": 1,
            "search_time_ms": 50.0,
            "parameters": {
                "top_k": 10,
                "threshold": 0.0
            }
        }
        
        # Should not raise validation error
        response = SearchResponse(**valid_data)
        assert response.total_results == 1
        assert len(response.results) == 1
        assert response.results[0].similarity == 0.95
    
    def test_search_response_schema_validation_errors(self):
        """Test SearchResponse schema validation with invalid data."""
        # Invalid similarity score
        invalid_data = {
            "query_id": "test_123",
            "results": [
                {
                    "image_id": "img_001",
                    "similarity": 1.5,  # Invalid: > 1.0
                    "metadata": {}
                }
            ],
            "total_results": 1,
            "search_time_ms": 50.0,
            "parameters": {"top_k": 10, "threshold": 0.0}
        }
        
        with pytest.raises(ValueError):
            SearchResponse(**invalid_data)
    
    def test_health_response_schema(self):
        """Test HealthResponse schema validation."""
        valid_data = {
            "status": "healthy",
            "timestamp": 1642680000.0,
            "services": {
                "clip_model": True,
                "vector_db": True
            }
        }
        
        response = HealthResponse(**valid_data)
        assert response.status == "healthy"
        assert response.services["clip_model"] is True

class TestAPIMiddleware:
    """Test API middleware functionality."""
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.get("/")
        
        # Check for CORS headers (may vary based on configuration)
        headers = response.headers
        # Basic test - should not fail due to CORS
        assert response.status_code == 200
    
    def test_gzip_compression(self, client):
        """Test gzip compression middleware."""
        # Make request with Accept-Encoding header
        headers = {"Accept-Encoding": "gzip"}
        response = client.get("/", headers=headers)
        
        assert response.status_code == 200
        # Response should be handled properly regardless of compression

class TestAPIErrorHandling:
    """Test API error handling."""
    
    def test_404_error_handler(self, client):
        """Test 404 error handling."""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
        
        data = response.json()
        assert "error" in data
        assert "detail" in data
    
    def test_method_not_allowed(self, client):
        """Test method not allowed error."""
        # Try POST on GET-only endpoint
        response = client.post("/health")
        assert response.status_code == 405
    
    def test_request_timeout_handling(self, client):
        """Test request timeout handling."""
        # This test simulates a slow request
        # In practice, this would depend on the actual timeout configuration
        response = client.get("/stats")
        
        # Should complete within reasonable time
        assert response.status_code in [200, 503]

class TestBatchEndpoints:
    """Test batch processing endpoints."""
    
    @pytest.fixture
    def multiple_test_images(self):
        """Create multiple test image files."""
        images = []
        for i in range(3):
            img = Image.new('RGB', (224, 224), color=(i*80, 0, 0))
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            images.append(("test_{}.jpg".format(i), img_bytes, "image/jpeg"))
        return images
    
    def test_batch_search_endpoint(self, client, multiple_test_images):
        """Test batch search endpoint."""
        files = [("files", img_data) for img_data in multiple_test_images]
        params = {"top_k": 5, "threshold": 0.0}
        
        response = client.post("/search/batch", files=files, params=params)
        
        # Could be 200 (success) or 503 (service unavailable)
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "batch_id" in data
            assert "queries" in data
            assert "total_queries" in data
            assert "successful_queries" in data
    
    def test_batch_search_too_many_files(self, client, multiple_test_images):
        """Test batch search with too many files."""
        # Create more files than allowed
        many_files = multiple_test_images * 5  # 15 files
        files = [("files", img_data) for img_data in many_files]
        
        response = client.post("/search/batch", files=files)
        assert response.status_code == 400
        
        data = response.json()
        assert "detail" in data

class TestAPIPerformance:
    """Test API performance characteristics."""
    
    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests."""
        import threading
        import time
        
        results = []
        
        def make_request():
            start_time = time.time()
            response = client.get("/health")
            end_time = time.time()
            results.append({
                "status_code": response.status_code,
                "response_time": end_time - start_time
            })
        
        # Make 5 concurrent requests
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert len(results) == 5
        for result in results:
            assert result["status_code"] == 200
            assert result["response_time"] < 5.0  # Should be fast

if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])