import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast

from ..config import Config, ModelOptimization, get_device_info
from ..utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# --- Global model cache to avoid reloading ---
_model_cache = {}

def get_clip_model():
    """
    Initializes and returns the CLIP model and processor.
    Uses a global cache to ensure the model is loaded only once.
    """
    if "model" in _model_cache:
        logger.info("Returning cached CLIP model.")
        return _model_cache["model"], _model_cache["processor"]

    logger.info(f"Loading CLIP model: {Config.CLIP_MODEL_NAME}")
    logger.info(f"Using device: {get_device_info()} with dtype: {Config.DTYPE}")

    # Load model components from Hugging Face
    model = CLIPModel.from_pretrained(Config.CLIP_MODEL_NAME).to(Config.DEVICE)
    processor = CLIPProcessor.from_pretrained(Config.CLIP_MODEL_NAME)
    
    # --- Apply Optimizations ---
    model.eval() # Set model to evaluation mode

    if ModelOptimization.GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing for memory savings.")

    if ModelOptimization.TORCH_COMPILE and hasattr(torch, "compile"):
        logger.info("Applying torch.compile() for performance boost.")
        model = torch.compile(model)

    # Cache the model and processor
    _model_cache["model"] = model
    _model_cache["processor"] = processor

    return model, processor

def encode_image(image_path: str):
    """
    Encodes a single image into a CLIP embedding.

    Args:
        image_path (str): The file path to the image.

    Returns:
        torch.Tensor: The image embedding vector.
    """
    model, processor = get_clip_model()
    image = Image.open(image_path).convert("RGB")

    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt", padding=True).to(
            Config.DEVICE
        )
        embedding = model.get_image_features(**inputs)

    return embedding.cpu().numpy()

def encode_single_image(image: Image.Image, normalize: bool = True):
    """
    Encodes a single PIL Image into a CLIP embedding.

    Args:
        image (Image.Image): The PIL Image object.
        normalize (bool): Whether to normalize the embedding vector.

    Returns:
        np.ndarray: The image embedding vector.
    """
    model, processor = get_clip_model()
    
    # Ensure image is in RGB format
    if image.mode != "RGB":
        image = image.convert("RGB")

    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt", padding=True).to(
            Config.DEVICE
        )
        embedding = model.get_image_features(**inputs)
        
        # Convert to numpy
        embedding_np = embedding.cpu().numpy()
        
        # Normalize if requested
        if normalize:
            # L2 normalization for cosine similarity
            norm = np.linalg.norm(embedding_np, axis=1, keepdims=True)
            embedding_np = embedding_np / (norm + 1e-8)  # Add small epsilon to avoid division by zero

    return embedding_np.squeeze()  # Remove batch dimension