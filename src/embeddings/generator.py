import json
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image

from ..config import Config
from .clip_model import get_clip_model
from ..utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

class ImageDataset(Dataset):
    """Custom PyTorch Dataset for loading images from metadata."""
    def __init__(self, metadata, processor):
        self.metadata = metadata
        self.processor = processor

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image_path = self.metadata.iloc[idx]["image_path"]
        try:
            image = Image.open(image_path).convert("RGB")
            # The processor handles transformations and tokenization
            processed = self.processor(images=image, return_tensors="pt", padding=True)
            # Squeeze to remove the batch dimension added by the processor
            return processed["pixel_values"].squeeze(0), idx
        except Exception as e:
            logger.warning(f"Could not load image {image_path}: {e}. Skipping.")
            return None, None

def collate_fn(batch):
    """Custom collate function to filter out None values from failed image loads."""
    batch = list(filter(lambda x: x[0] is not None, batch))
    if not batch:
        return None, None
    pixel_values, indices = zip(*batch)
    return torch.stack(pixel_values), list(indices)

class EmbeddingGenerator:
    """Generates and saves image embeddings using the CLIP model."""

    def __init__(self):
        self.model, self.processor = get_clip_model()
        self.device = Config.DEVICE
        self.metadata_path = Config.METADATA_FILE
        self.embeddings_path = Config.EMBEDDINGS_PATH
        self.mappings_path = Config.IMAGE_MAPPINGS_PATH

    def run(self):
        """Executes the embedding generation pipeline."""
        logger.info("Starting embedding generation.")
        
        # Load metadata
        if not os.path.exists(self.metadata_path):
            logger.error(f"Metadata file not found at {self.metadata_path}")
            raise FileNotFoundError("Run the 'data' step first to generate metadata.")
        
        metadata_df = pd.read_csv(self.metadata_path)
        logger.info(f"Loaded metadata with {len(metadata_df)} records.")

        # Set up dataset and dataloader
        dataset = ImageDataset(metadata_df, self.processor)
        dataloader = DataLoader(
            dataset,
            batch_size=Config.EMBEDDING_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn
        )

        all_embeddings = []
        image_mappings = {}

        # Generate embeddings
        with torch.no_grad():
            for batch_pixels, batch_indices in tqdm(dataloader, desc="Generating Embeddings"):
                if batch_pixels is None:
                    continue

                # Move data to the configured device
                batch_pixels = batch_pixels.to(self.device, dtype=Config.DTYPE)
                
                # Get embeddings from the model
                batch_embeddings = self.model.get_image_features(pixel_values=batch_pixels)
                all_embeddings.extend(batch_embeddings.cpu().numpy())

                # Create mappings from original index to image path
                for i, original_idx in enumerate(batch_indices):
                    image_mappings[len(all_embeddings) - len(batch_indices) + i] = metadata_df.iloc[original_idx]['image_path']

        # Save results
        if all_embeddings:
            embeddings_array = np.array(all_embeddings)
            logger.info(f"Saving {len(embeddings_array)} embeddings to {self.embeddings_path}")
            np.save(self.embeddings_path, embeddings_array)

            logger.info(f"Saving image mappings to {self.mappings_path}")
            with open(self.mappings_path, 'w') as f:
                json.dump(image_mappings, f)
            
            logger.info("Embedding generation completed successfully.")
        else:
            logger.warning("No embeddings were generated. Please check image paths and data.")