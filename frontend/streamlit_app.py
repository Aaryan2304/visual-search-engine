"""
Streamlit frontend for the AI Visual Search Engine.
Provides an interactive web interface for image similarity search.
"""

import streamlit as st
import requests
import pandas as pd
from PIL import Image
import io
import time
from typing import List, Dict, Any, Optional
import json
import base64
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="AI Visual Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_BASE_URL = "http://localhost:8001"  # Updated to match API port
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
SUPPORTED_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-bottom: 2rem;
        border-radius: 10px;
    }
    
    .result-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    
    .metric-box {
        text-align: center;
        padding: 1rem;
        background: #2b2b2b;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        color: white;
        border: 1px solid #3d3d3d;
    }
    
    .metric-box h4 {
        color: #ffffff;
        margin: 0 0 0.5rem 0;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    .metric-box p {
        color: #cccccc;
        margin: 0;
        font-size: 0.9rem;
    }
    
    .similarity-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        font-size: 0.8rem;
    }
    
    .similarity-high { background-color: #28a745; }
    .similarity-medium { background-color: #ffc107; color: black; }
    .similarity-low { background-color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# Utility functions
@st.cache_data
def check_api_health() -> Dict[str, Any]:
    """Check if the API is healthy and return status."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.json() if response.status_code == 200 else {"status": "error"}
    except:
        return {"status": "offline"}

def get_api_stats() -> Dict[str, Any]:
    """Get API usage statistics."""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        return response.json() if response.status_code == 200 else {}
    except:
        return {}

def get_collection_info() -> Dict[str, Any]:
    """Get information about the vector collection."""
    try:
        response = requests.get(f"{API_BASE_URL}/collection/info", timeout=5)
        return response.json() if response.status_code == 200 else {}
    except:
        return {}

def search_similar_images(
    image_file: Any, 
    top_k: int = 10, 
    threshold: float = 0.0
) -> Dict[str, Any]:
    """Search for similar images using the API."""
    files = {"file": image_file}
    params = {"top_k": top_k, "threshold": threshold}
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/search",
            files=files,
            params=params,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code} - {response.text}"}
    except requests.exceptions.Timeout:
        return {"error": "Request timeout - the search took too long"}
    except requests.exceptions.ConnectionError:
        return {"error": "Connection error - API server might be down"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def get_similarity_badge_class(similarity: float) -> str:
    """Get CSS class for similarity badge based on score."""
    if similarity >= 0.8:
        return "similarity-high"
    elif similarity >= 0.5:
        return "similarity-medium"
    else:
        return "similarity-low"

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string for display."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

# Main application
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 AI Visual Search Engine</h1>
        <p>Find visually similar images using state-of-the-art CLIP embeddings</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🛠️ Controls")
        
        # API Health Check
        health_status = check_api_health()
        if health_status.get("status") == "healthy":
            st.success("✅ API Status: Healthy")
        elif health_status.get("status") == "degraded":
            st.warning("⚠️ API Status: Degraded")
        else:
            st.error("❌ API Status: Offline")
        
        st.markdown("---")
        
        # Search Parameters
        st.subheader("Search Parameters")
        top_k = st.slider(
            "Number of Results",
            min_value=1,
            max_value=50,
            value=10,
            help="Maximum number of similar images to return"
        )
        
        threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            help="Minimum similarity score (0.0 = no filter)"
        )
        
        st.markdown("---")
        
        # Collection Info
        st.subheader("📊 Collection Info")
        collection_info = get_collection_info()
        if collection_info:
            col_info = collection_info.get("collection_info", {})
            model_info = collection_info.get("model_info", {})
            
            # Get total images - try both field names
            total_images = col_info.get("total_vectors") or col_info.get("total_embeddings") or "N/A"
            if isinstance(total_images, int):
                total_images = f"{total_images:,}"  # Format with commas
            
            st.metric("Total Images", total_images)
            st.metric("Model", model_info.get("model_name", "N/A").split("/")[-1])
            st.metric("Backend", col_info.get("backend", "N/A"))
            
            # Show dataset type if available
            collection_name = col_info.get("collection_name", "")
            if "visual_search_engine" in str(collection_name).lower() or "deepfashion" in str(collection_name).lower():
                st.info("🎯 **DeepFashion Dataset**\n- 50 clothing categories\n- 1000+ attributes\n- Fashion-focused embeddings")
        else:
            st.error("❌ Could not load collection info")
        
        st.markdown("---")
        
        # API Statistics
        st.subheader("📈 API Stats")
        stats = get_api_stats()
        if stats:
            st.metric("Total Requests", stats.get("total_requests", 0))
            st.metric("Error Rate", f"{stats.get('error_rate', 0):.1f}%")
            st.metric("Avg Search Time", f"{stats.get('average_search_time', 0) * 1000:.1f}ms")
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🖼️ Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image to search for similar images",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help=f"Supported formats: {', '.join(SUPPORTED_FORMATS)}\nMax size: {format_file_size(MAX_FILE_SIZE)}"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Query Image", use_column_width=True)
            
            # File info
            file_size = len(uploaded_file.getvalue())
            st.info(f"📁 File: {uploaded_file.name}\n📏 Size: {format_file_size(file_size)}\n🖼️ Dimensions: {image.size[0]}x{image.size[1]}")
            
            # Search button
            if st.button("🔍 Search Similar Images", type="primary"):
                if file_size > MAX_FILE_SIZE:
                    st.error(f"File too large! Maximum size is {format_file_size(MAX_FILE_SIZE)}")
                else:
                    # Perform search
                    with st.spinner("Searching for similar images..."):
                        # Reset file pointer
                        uploaded_file.seek(0)
                        search_results = search_similar_images(uploaded_file, top_k, threshold)
                    
                    # Store results in session state
                    st.session_state['search_results'] = search_results
                    st.session_state['query_image'] = image
    
    with col2:
        st.subheader("🎯 Search Results")
        
        # Display search results
        if 'search_results' in st.session_state:
            results = st.session_state['search_results']
            
            if 'error' in results:
                st.error(f"Search Error: {results['error']}")
            
            elif 'results' in results:
                search_info = results
                
                # Search metrics
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-box">
                        <h4>{search_info.get('total_results', 0)}</h4>
                        <p>Results Found</p>
                    </div>
                    <div class="metric-box">
                        <h4>{search_info.get('search_time_ms', 0):.1f}ms</h4>
                        <p>Search Time</p>
                    </div>
                    <div class="metric-box">
                        <h4>{search_info.get('parameters', {}).get('top_k', 0)}</h4>
                        <p>Max Results</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if search_info['total_results'] > 0:
                    st.markdown("### Similar Images")
                    
                    # Display results in a grid
                    results_data = search_info['results']
                    
                    # Create columns for grid layout - only create columns for actual results
                    cols_per_row = 3
                    
                    for i in range(0, len(results_data), cols_per_row):
                        # Only create columns for the remaining results
                        remaining_results = len(results_data) - i
                        num_cols = min(cols_per_row, remaining_results)
                        cols = st.columns(num_cols)
                        
                        for j in range(num_cols):
                            result = results_data[i + j]
                            
                            with cols[j]:
                                # Get image URL from API
                                image_url = f"{API_BASE_URL}{result.get('image_url', '')}"
                                
                                try:
                                    # Try to load and display the actual image
                                    response = requests.get(image_url, timeout=10)
                                    if response.status_code == 200:
                                        image_data = Image.open(io.BytesIO(response.content))
                                        st.image(
                                            image_data,
                                            caption=f"ID: {result['image_id']}",
                                            use_column_width=True
                                        )
                                    else:
                                        # Fallback to placeholder if image can't be loaded
                                        st.image(
                                            "https://via.placeholder.com/200x200?text=Image+Error",
                                            caption=f"ID: {result['image_id']} (Error loading)"
                                        )
                                except Exception as e:
                                    # Fallback to placeholder on any error
                                    st.image(
                                        "https://via.placeholder.com/200x200?text=Load+Error",
                                        caption=f"ID: {result['image_id']} (Load failed)"
                                    )
                                
                                # Display metadata if available
                                metadata = result.get('metadata', {})
                                if metadata.get('category_name'):
                                    st.caption(f"📂 {metadata['category_name']}")
                                
                                # Similarity badge
                                similarity = result['similarity']
                                badge_class = get_similarity_badge_class(similarity)
                                st.markdown(f"""
                                <div class="similarity-badge {badge_class}">
                                    {similarity:.2%}
                                </div>
                                """, unsafe_allow_html=True)
                else:
                    st.info("No similar images found. Try adjusting the threshold or using a different image.")
            
            else:
                if 'error' in search_info:
                    st.error(f"❌ Search failed: {search_info['error']}")
                else:
                    st.error("❌ Unexpected error occurred during search")

if __name__ == "__main__":
    main()
                                