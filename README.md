# AI-Powered Visual Search Engine

## 📋 Project Overview
A scalable visual search system that finds visually similar images using CLIP embeddings and FAISS vector database. This system is optimized for fashion images using the DeepFashion dataset and works efficiently on both CPU and GPU environments (RTX 3050+ recommended for optimal performance).

## 🏗️ Architecture
- **Frontend**: Streamlit web interface with intuitive search functionality
- **Backend**: FastAPI with async endpoints and comprehensive error handling
- **ML Pipeline**: CLIP embeddings with PyTorch and optimized batch processing
- **Database**: FAISS vector database with persistent local storage
- **Deployment**: Local development environment

## ✨ Features
- **Real-time Visual Search**: Upload an image and find similar fashion items instantly
- **REST API**: Complete API with interactive documentation
- **Scalable Architecture**: Handles 100K+ images efficiently
- **GPU Optimized**: Automatic GPU detection and optimization
- **Comprehensive Logging**: Error handling and performance monitoring
- **Easy Setup**: Simple installation and configuration process

## 📊 Performance & Specifications
- **Hardware**: Optimized for RTX 3050+ (4GB VRAM), works on CPU
- **Memory**: 8GB+ RAM recommended, 16GB for large datasets
- **Batch Processing**: Dynamic batch sizing (8-32 images based on hardware)
- **Search Latency**: <100ms for similarity search
- **Dataset Support**: 100K+ images with room for scaling
- **Storage**: Efficient FAISS indexing with 512-dimensional embeddings

## 🔧 Technical Implementation
- **Vector Database**: FAISS (Facebook AI Similarity Search) for high-performance similarity search
- **Storage**: Local persistent storage with automatic save/load functionality
- **Similarity Metric**: Cosine similarity using normalized embeddings
- **Index Type**: Flat index for exact search results
- **No External Dependencies**: Self-contained database that doesn't require separate server processes

## 🚀 Quick Start

### Prerequisites
- **Python**: 3.9+ (3.11 recommended for best performance)
- **Memory**: 8GB+ RAM (16GB recommended for large datasets)
- **GPU**: NVIDIA GPU optional but recommended (RTX 3050+ for optimal performance)
- **Storage**: 2GB+ free space (more for larger datasets)
- **OS**: Windows 10+, macOS 10.14+, or modern Linux distribution

### 1. Clone and Setup
```bash
# Clone the repository
git clone <your-repo>
cd visual-search-engine

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify setup
python scripts/verify_setup.py
```

### 2. Dataset Setup
Download the **DeepFashion Category and Attribute Prediction Benchmark**:
- Download from: [Google Drive](https://drive.google.com/drive/folders/0B7EVK8r0v71pWGplNFhjc01NbzQ?resourcekey=0-BU3lAk-Nc7HscJu-CyC1yA&usp=sharing)
- Extract to `data/deepfashion/`

**Expected folder structure:**
```
data/deepfashion/
├── Anno_coarse/              # Main annotation files
│   ├── list_category_cloth.txt      # 50 clothing categories
│   ├── list_category_img.txt        # Image-category mappings
│   ├── list_attr_cloth.txt          # 1000+ clothing attributes
│   ├── list_attr_img.txt            # Image-attribute mappings
│   ├── list_bbox.txt                # Bounding box annotations
│   └── list_landmarks.txt           # Fashion landmark annotations (8 landmarks)
├── Anno_fine/                # Fine-grained train/val/test splits
│   ├── train.txt, val.txt, test.txt  # Data partition files
│   ├── train_*.txt, val_*.txt, test_*.txt  # Split-specific annotations
│   └── list_*.txt              # Category and attribute definitions
├── Eval/                     # Evaluation protocols
│   └── list_eval_partition.txt      # Overall train/val/test splits
├── Img/                      # ~289K fashion images (~5,620 categories)
│   └── [clothing_category_folders]/  # Images organized by item names
└── README.txt                # Official dataset documentation
```

### 3. Run the Pipeline

#### Quick Start (Recommended)
```bash
# Navigate to project directory
cd visual-search-engine

# Option 1: Quick test with 1000 images
python scripts/run_pipeline.py --data-dir ./data/deepfashion --max-images 1000 --complete

# Option 2: Full pipeline with all images (slow, 2+ hours)
python scripts/run_pipeline.py --data-dir ./data/deepfashion --complete
```

#### Step-by-Step Execution
If you prefer to run individual steps:
```bash
# Step 1: Process dataset metadata
python scripts/run_pipeline.py --data-dir ./data/deepfashion --step data

# Step 2: Generate embeddings
python scripts/run_pipeline.py --data-dir ./data/deepfashion --step embeddings

# Step 3: Setup vector database
python scripts/run_pipeline.py --data-dir ./data/deepfashion --step database

# Step 4: Validate system
python scripts/run_pipeline.py --data-dir ./data/deepfashion --step validate
```

#### Launch Services
After completing the pipeline, start both services:

**Option 1: Quick Start (Recommended)**
```bash
# Terminal 1: Start API server
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8001

# Terminal 2: Start web interface
streamlit run frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

**Option 2: Background Services**
```bash
# Start API in background
start /b python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8001

# Start Streamlit in background
start /b streamlit run frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

🌐 **Access Your Application:**
- **Web Interface**: http://localhost:8501
- **API Documentation**: http://localhost:8001/docs
- **API Health Check**: http://localhost:8001/health

### 4. Access Services & Usage

#### Web Interface (Recommended for Users)
- **URL**: http://localhost:8501
- **Features**: Upload images, browse results, adjust similarity threshold
- **Best For**: End users, demonstrations, testing

#### API Interface (For Developers)
- **Base URL**: http://localhost:8001
- **Documentation**: http://localhost:8001/docs (Interactive Swagger UI)
- **Health Check**: http://localhost:8001/health
- **Search Endpoint**: POST /search with image upload
- **Best For**: Integration, automation, custom applications

#### Local Database
- **Location**: `data/processed/`
- **Type**: FAISS vector index with metadata
- **Size**: ~50MB for 10K images, scales linearly

### 5. Common Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--max-images` | Limit number of images (for testing) | `1000` |
| `--batch-size` | Images per batch (adjust for your GPU) | `16` |
| `--step` | Run specific step only | `data`, `embeddings`, `database`, `validate` |
| `--complete` | Run all pipeline steps | (flag only) |

### 6. Troubleshooting

**Import Error Fix:**
```bash
# Ensure you're in the project root directory
cd visual-search-engine

# If still failing, set Python path explicitly
set PYTHONPATH=%CD% && python scripts/run_pipeline.py --data-dir ./data/deepfashion
```

**Memory Issues:**
```bash
# Reduce batch size for lower VRAM
python scripts/run_pipeline.py --data-dir ./data/deepfashion --batch-size 8 --max-images 500 --complete
```

**Database Issues:**
```bash
# Clear database and start fresh (if needed)
rmdir /s data\processed\chroma_db\
```

**Quick Setup Verification:**
```bash
# Run verification script to check setup
python scripts/verify_setup.py
```

## 🗃️ Database Information

The system uses **FAISS (Facebook AI Similarity Search)** for vector storage:
- **No external server required**: Runs entirely locally
- **High performance**: Optimized for similarity search  
- **Persistent storage**: Automatically saves and loads index data
- **Simple setup**: No configuration needed

### Database Files:
```
data/processed/chroma_db/
├── faiss_index.bin        # FAISS index with embeddings
├── metadata.json          # Image metadata
└── id_mapping.json        # ID mappings
```

### Migration from ChromaDB to FAISS:
If you previously used ChromaDB, the system has been updated to use FAISS for better reliability and easier setup. No manual migration is needed - simply run the pipeline steps again and the system will automatically create the new FAISS database.

## 🛠️ Development

### Manual Service Startup
```bash
# Start API server (Terminal 1)
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8001

# Start web interface (Terminal 2)  
streamlit run frontend/streamlit_app.py --server.port 8501
```

### Project Structure
```
visual-search-engine/
├── 📁 data/                     # Dataset and processed files
│   ├── deepfashion/             # DeepFashion dataset (download required)
│   └── processed/               # Generated embeddings and database
│       ├── embeddings.npy       # CLIP embeddings
│       ├── metadata.csv         # Image metadata
│       ├── image_mappings.json  # Image path mappings
│       └── chroma_db/           # FAISS vector database
├── 📁 src/                      # Core application code
│   ├── api/                     # FastAPI backend
│   │   ├── main.py              # API application entry point
│   │   ├── endpoints.py         # API route definitions
│   │   └── schemas.py           # Pydantic models
│   ├── embeddings/              # CLIP model and embedding generation
│   │   ├── clip_model.py        # CLIP model wrapper
│   │   └── generator.py         # Embedding generation pipeline
│   ├── database/                # Vector database operations
│   │   ├── vector_db.py         # FAISS database wrapper
│   │   └── indexer.py           # Indexing operations
│   ├── data/                    # Data processing utilities
│   │   └── preprocessor.py      # Dataset preprocessing
│   └── utils/                   # Shared utilities
│       ├── image_utils.py       # Image processing helpers
│       └── logger.py            # Logging configuration
├── 📁 frontend/                 # Streamlit web interface
│   └── streamlit_app.py         # Main web application
├── 📁 scripts/                  # Automation and setup scripts
│   ├── run_pipeline.py          # Main pipeline orchestrator
│   └── verify_setup.py          # Setup verification
├── 📁 tests/                    # Test suite
│   ├── test_api.py              # API tests
│   ├── test_embeddings.py       # Embedding tests
│   ├── test_database.py         # Database tests
│   └── test_api_integration.py  # Integration tests
├── 📁 docs/                     # Documentation
│   └── DATASET_ANALYSIS.md      # Dataset structure analysis
├── 📁 extras/                   # Optional features
│   └── monitoring/              # Prometheus/Grafana configs
├── 📁 docker/                   # Docker configurations (optional)
│   ├── api.Dockerfile           # API container
│   └── frontend.Dockerfile      # Frontend container
├── 📁 logs/                     # Application logs
├── 📄 requirements.txt          # Python dependencies
├── 📄 docker_compose.yml        # Docker Compose configuration
└── 📄 README.md                 # This file
```

### Running Tests
```bash
pytest tests/
```

## 📚 References

- [DeepFashion Dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
- [DeepFashion Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.pdf)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [FAISS Documentation](https://faiss.ai/)

## 📝 Citation

If you use this system or the DeepFashion dataset, please cite:

```bibtex
@inproceedings{liu2016deepfashion,
  author = {Ziwei Liu, Ping Luo, Shi Qiu, Xiaogang Wang, and Xiaoou Tang},
  title = {DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations},
  booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2016} 
}
```

## 📝 License

MIT License - see LICENSE file for details.
