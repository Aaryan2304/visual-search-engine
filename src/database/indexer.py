import json
import os
import numpy as np
from tqdm import tqdm
import math

from ..config import Config
from .vector_db import get_vector_database
from ..utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

class Indexer:
    """Handles indexing of embeddings into the vector database."""

    def __init__(self):
        """Initializes the Indexer."""
        self.db = get_vector_database()
        self.collection = self.db.get_collection(Config.COLLECTION_NAME)
        self.embeddings_path = Config.EMBEDDINGS_PATH
        self.image_mappings_path = Config.IMAGE_MAPPINGS_PATH
        self.batch_size = 128  # Define a reasonable batch size

    def _load_data(self):
        """Loads embeddings and image mappings from files."""
        if not os.path.exists(self.embeddings_path) or not os.path.exists(self.image_mappings_path):
            logger.error("Embeddings file or image mappings file not found.")
            raise FileNotFoundError("Required data files are missing.")
        
        logger.info(f"Loading embeddings from {self.embeddings_path}")
        embeddings = np.load(self.embeddings_path)
        
        logger.info(f"Loading image mappings from {self.image_mappings_path}")
        with open(self.image_mappings_path, 'r') as f:
            image_mappings = json.load(f)
            
        return embeddings, image_mappings

    def run(self):
        """
        Executes the indexing process:
        1. Loads embeddings and mappings.
        2. Upserts them into the vector database in batches.
        """
        try:
            embeddings, image_mappings = self._load_data()
            num_embeddings = len(embeddings)
            num_batches = math.ceil(num_embeddings / self.batch_size)

            logger.info(f"Starting indexing for {num_embeddings} embeddings in {num_batches} batches.")

            for i in tqdm(range(0, num_embeddings, self.batch_size), desc="Indexing Batches"):
                batch_end = min(i + self.batch_size, num_embeddings)
                
                # Prepare batch data
                batch_embeddings = embeddings[i:batch_end]
                batch_ids = [str(idx) for idx in range(i, batch_end)]
                batch_metadata = [
                    {"image_path": image_mappings[str(idx)]} for idx in range(i, batch_end)
                ]

                # Upsert batch to the database
                self.db.upsert(
                    collection=self.collection,
                    embeddings=np.array(batch_embeddings),
                    metadata=batch_metadata,
                    ids=batch_ids
                )
            
            # Save the database to disk
            if hasattr(self.db, 'save'):
                self.db.save()
                logger.info("Vector database saved to disk")
            
            logger.info("Indexing completed successfully.")

        except FileNotFoundError as e:
            logger.error(f"Indexing failed: {e}")
            # Re-raise the exception to be caught by the pipeline script
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during indexing: {e}", exc_info=True)
            # Re-raise for visibility
            raise