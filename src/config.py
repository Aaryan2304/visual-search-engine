import os
import torch

# =================================================================================
# Helper Functions and Classes
# =================================================================================
def get_device_info():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        return f"CUDA ({gpu_count}x {gpu_name})"
    return "CPU"

class ModelOptimization:
    GRADIENT_CHECKPOINTING = True
    TORCH_COMPILE = False

# =================================================================================
# Main Configuration Class
# =================================================================================
class Config:
    # --- Base Directories
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SRC_DIR = os.path.join(ROOT_DIR, "src")
    LOG_DIR = os.path.join(ROOT_DIR, "logs")

    # --- Data Directories
    DATA_DIR = os.path.join(ROOT_DIR, "data/deepfashion")
    PROCESSED_DIR = os.path.join(ROOT_DIR, "data/processed") # <-- ADD THIS LINE

    # --- Dynamically set file paths
    METADATA_FILE = os.path.join(PROCESSED_DIR, "metadata.csv")
    EMBEDDINGS_PATH = os.path.join(PROCESSED_DIR, "embeddings.npy")
    IMAGE_MAPPINGS_PATH = os.path.join(PROCESSED_DIR, "image_mappings.json")
    FAISS_INDEX_PATH = os.path.join(PROCESSED_DIR, "faiss_index.bin")

    # Logging Configuration
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True) # <-- Also good to add this

    # Machine Learning Model Configuration
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

    # Data Processing Configuration
    EMBEDDING_BATCH_SIZE = 64
    NUM_WORKERS = 4 if torch.cuda.is_available() else 0
    
    # Image Processing Configuration
    IMAGE_SIZE = 224  # Standard size for CLIP model
    SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    # Model Configuration
    EMBEDDING_DIM = 512  # CLIP embedding dimension
    MAX_MEMORY_GB = 4  # Available memory for processing

    # Vector Database Configuration
    VECTOR_DB_TYPE = "chroma"
    COLLECTION_NAME = "visual_search_engine"
    CHROMA_HOST = "localhost"  # localhost for local development
    CHROMA_PORT = 8000
    CHROMA_PERSIST_DIRECTORY = os.path.join(PROCESSED_DIR, "chroma_db")  # Local persistent storage
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "YOUR_API_KEY")
    PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", "gcp-starter")

    # API and Service Configuration
    API_HOST = "0.0.0.0"
    API_PORT = 8001  # Changed to 8001 to match README
    DEFAULT_TOP_K = 10
    MAX_TOP_K = 50
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_TOP_K = 50
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB