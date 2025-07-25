import json
import os
import pandas as pd
from tqdm import tqdm

from ..config import Config
from ..utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


class DataPreprocessor:
    """Preprocesses the DeepFashion dataset to create a unified metadata file."""

    def __init__(self, base_dir):
        """
        Initializes the DataPreprocessor.
        Args:
            base_dir (str): The root directory of the DeepFashion dataset.
        """
        self.base_dir = base_dir
        self.image_dir = os.path.join(self.base_dir, "Img")
        self.anno_coarse_dir = os.path.join(self.base_dir, "Anno_coarse")
        self.eval_file = os.path.join(self.base_dir, "Eval", "list_eval_partition.txt")
        self.category_file = os.path.join(self.anno_coarse_dir, "list_category_img.txt")
        self.category_cloth_file = os.path.join(self.anno_coarse_dir, "list_category_cloth.txt")
        self.output_file = Config.METADATA_FILE

    def _load_partition_file(self):
        """Loads the evaluation partition file to map images to train/val/test sets."""
        logger.info(f"Loading partition file from: {self.eval_file}")
        return pd.read_csv(
            self.eval_file,
            sep=r"\s+",
            skiprows=2,
            names=["image_name", "evaluation_status"],
        )

    def _load_category_mappings(self):
        """Loads category mappings for images."""
        logger.info(f"Loading category mappings from: {self.category_file}")
        category_df = pd.read_csv(
            self.category_file,
            sep=r"\s+",
            skiprows=2,
            names=["image_name", "category_label"],
        )
        
        # Load category names
        logger.info(f"Loading category names from: {self.category_cloth_file}")
        category_names_df = pd.read_csv(
            self.category_cloth_file,
            sep=r"\s+",
            skiprows=2,
            names=["category_name", "category_type"],
        )
        category_names_df['category_label'] = range(1, len(category_names_df) + 1)
        
        # Merge to get category names
        category_df = category_df.merge(category_names_df[['category_label', 'category_name']], 
                                       on='category_label', how='left')
        
        return category_df

    def _process_images(self, partition_df, category_df):
        """Process images and create metadata records."""
        records = []
        
        # Merge partition and category data
        merged_df = partition_df.merge(category_df, on='image_name', how='inner')
        logger.info(f"Processing {len(merged_df)} images with annotations.")

        for _, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Processing Images"):
            image_name = row['image_name']
            image_path = os.path.join(self.base_dir, image_name)  # Full path to image
            
            # Check if image exists
            if os.path.exists(image_path):
                records.append({
                    "image_name": image_name,
                    "image_path": image_path,
                    "evaluation_status": row['evaluation_status'],
                    "category_label": row['category_label'],
                    "category_name": row.get('category_name', 'unknown'),
                })
            else:
                logger.warning(f"Image not found: {image_path}")
        
        return pd.DataFrame(records)

    def run(self):
        """
        Executes the data preprocessing pipeline:
        1. Loads partition data.
        2. Loads annotations.
        3. Merges them into a single metadata file.
        """
        logger.info(f"Starting data preprocessing for directory: {self.base_dir}")

        # Step 1: Load partition file
        logger.info("Loading partition file...")
        partition_df = self._load_partition_file()
        logger.info(f"Loaded {len(partition_df)} records from partition file.")

        # Step 2: Load category mappings
        logger.info("Loading category mappings...")
        category_df = self._load_category_mappings()
        logger.info(f"Loaded {len(category_df)} category mappings.")

        # Step 3: Process images and create metadata
        logger.info("Processing images and creating metadata...")
        metadata_df = self._process_images(partition_df, category_df)
        logger.info(f"Created final metadata with {len(metadata_df)} records.")

        if not metadata_df.empty:
            # Save to CSV
            logger.info(f"Saving metadata to {self.output_file}")
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            metadata_df.to_csv(self.output_file, index=False)
            logger.info("Metadata file saved successfully.")
            
            # Log some statistics
            logger.info(f"Train images: {len(metadata_df[metadata_df['evaluation_status'] == 'train'])}")
            logger.info(f"Val images: {len(metadata_df[metadata_df['evaluation_status'] == 'val'])}")
            logger.info(f"Test images: {len(metadata_df[metadata_df['evaluation_status'] == 'test'])}")
            logger.info(f"Unique categories: {metadata_df['category_name'].nunique()}")
        else:
            logger.error("No images were processed. Cannot create metadata file.")
            raise ValueError("No valid images found in the dataset.")