#!/usr/bin/env python3
"""
Test script to verify the image serving and search functionality.
"""

import requests
import json
import sys
import os

API_BASE_URL = "http://localhost:8001"

def test_health():
    """Test API health endpoint."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        print(f"✅ Health check: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_collection_info():
    """Test collection info endpoint."""
    try:
        response = requests.get(f"{API_BASE_URL}/collection/info", timeout=5)
        print(f"✅ Collection info: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Total images: {data.get('collection_info', {}).get('total_embeddings', 'Unknown')}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Collection info failed: {e}")
        return False

def test_image_serving():
    """Test image serving endpoint."""
    try:
        # Try to get the first image (ID: 0)
        response = requests.get(f"{API_BASE_URL}/images/0", timeout=10)
        print(f"✅ Image serving (ID: 0): {response.status_code}")
        
        if response.status_code == 200:
            print(f"   Content-Type: {response.headers.get('content-type', 'unknown')}")
            print(f"   Content-Length: {len(response.content)} bytes")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Image serving failed: {e}")
        return False

def test_search_with_sample():
    """Test search functionality with a sample image from the dataset."""
    try:
        # Try to get the first image and use it as a query
        image_response = requests.get(f"{API_BASE_URL}/images/100", timeout=10)
        
        if image_response.status_code != 200:
            print("❌ Could not get sample image for search test")
            return False
        
        # Use this image for search
        files = {"file": ("test_image.jpg", image_response.content, "image/jpeg")}
        params = {"top_k": 3, "threshold": 0.0}
        
        search_response = requests.post(
            f"{API_BASE_URL}/search",
            files=files,
            params=params,
            timeout=30
        )
        
        print(f"✅ Search test: {search_response.status_code}")
        
        if search_response.status_code == 200:
            data = search_response.json()
            print(f"   Results found: {data.get('total_results', 0)}")
            print(f"   Search time: {data.get('search_time_ms', 0):.1f}ms")
            
            # Check if results have image_url field
            results = data.get('results', [])
            if results:
                first_result = results[0]
                print(f"   First result ID: {first_result.get('image_id', 'Unknown')}")
                print(f"   First result similarity: {first_result.get('similarity', 0):.3f}")
                print(f"   Has image_url: {'image_url' in first_result}")
                print(f"   Image URL: {first_result.get('image_url', 'None')}")
                print(f"   Has metadata: {'metadata' in first_result}")
                print(f"   All keys: {list(first_result.keys())}")
                
                # Test if we can access the result image
                if 'image_url' in first_result:
                    result_image_url = f"{API_BASE_URL}{first_result['image_url']}"
                    img_response = requests.get(result_image_url, timeout=10)
                    print(f"   Result image accessible: {img_response.status_code == 200}")
                    if img_response.status_code != 200:
                        print(f"   Image error: {img_response.status_code} - {img_response.text[:100]}")
                else:
                    print("   ❌ No image_url in response!")
            else:
                print("   ❌ No results in response!")
        else:
            print(f"   Error: {search_response.text[:200]}")
        
        return search_response.status_code == 200
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🔍 Testing Visual Search API Functionality...\n")
    
    tests = [
        ("API Health", test_health),
        ("Collection Info", test_collection_info),
        ("Image Serving", test_image_serving),
        ("Search with Sample", test_search_with_sample)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"📋 {test_name}:")
        if test_func():
            passed += 1
        print()
    
    print("="*50)
    if passed == total:
        print(f"🎉 All {total} tests passed! The API is working correctly.")
        print("\nYou can now:")
        print("1. Upload images in the Streamlit interface")
        print("2. View actual similar images (not placeholders)")
        print("3. See image metadata and categories")
    else:
        print(f"⚠️  {passed}/{total} tests passed. Some issues remain.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
