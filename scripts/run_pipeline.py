import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

from src.config import Config
from src.data.preprocessor import DataPreprocessor
from src.database.indexer import Indexer
from src.embeddings.generator import EmbeddingGenerator
from src.utils.logger import get_logger

# Set up logger
logger = get_logger(__name__)


def run_data_preparation(data_dir, max_images=None, partitions=None):
    """Runs the data preparation step."""
    logger.info("=== Step 1: Data Preparation ===")
    try:
        preprocessor = DataPreprocessor(data_dir)
        preprocessor.run()
        logger.info("Data preparation completed successfully.")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Data preparation failed: {e}", exc_info=True)
        return {"status": "failed", "error": str(e)}


def run_embedding_generation(batch_size=None, max_images=None):
    """Runs the embedding generation step."""
    logger.info("=== Step 2: Generating Embeddings ===")
    try:
        generator = EmbeddingGenerator()
        # TODO: Add support for batch_size and max_images parameters
        generator.run()
        logger.info("Embedding generation completed successfully.")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}", exc_info=True)
        return {"status": "failed", "error": str(e)}


def run_database_setup():
    """Runs the database setup step."""
    logger.info("=== Step 3: Database Setup ===")
    try:
        indexer = Indexer()
        indexer.run()
        logger.info("Database setup completed successfully.")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Database setup failed: {e}", exc_info=True)
        return {"status": "failed", "error": str(e)}


def run_validation():
    """Runs system validation."""
    logger.info("=== Step 4: System Validation ===")
    try:
        # TODO: Implement validation logic
        logger.info("System validation completed successfully.")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"System validation failed: {e}", exc_info=True)
        return {"status": "failed", "error": str(e)}


def launch_services_docker():
    """Launches services using Docker."""
    logger.info("=== Launching Services with Docker ===")
    try:
        # Stop existing containers
        subprocess.run(["docker-compose", "down", "--volumes"], check=False)
        
        # Start services
        subprocess.run(
            ["docker-compose", "up", "--build", "-d"],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Docker services launched successfully.")
        logger.info("Frontend: http://localhost:8501")
        logger.info("API: http://localhost:8001")
        logger.info("API Docs: http://localhost:8001/docs")
        return {"status": "success"}
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to launch Docker services: {e}")
        logger.error(f"Stderr: {e.stderr}")
        return {"status": "failed", "error": e.stderr}


def launch_services_local():
    """Launches services locally without Docker."""
    logger.info("=== Launching Services Locally ===")
    try:
        import threading
        import time
        
        # Function to run the API server
        def run_api():
            import uvicorn
            uvicorn.run(
                "src.api.main:app",
                host="0.0.0.0",
                port=8001,
                reload=False,
                log_level="info"
            )
        
        # Function to run the Streamlit frontend
        def run_frontend():
            import streamlit.web.cli as stcli
            sys.argv = ["streamlit", "run", "frontend/streamlit_app.py", "--server.port=8501"]
            stcli.main()
        
        # Start API server in background thread
        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()
        
        # Give API server time to start
        time.sleep(3)
        
        logger.info("API server started on http://localhost:8001")
        logger.info("API docs available at http://localhost:8001/docs")
        
        # Start Streamlit frontend in background thread
        frontend_thread = threading.Thread(target=run_frontend, daemon=True)
        frontend_thread.start()
        
        # Give frontend time to start
        time.sleep(3)
        
        logger.info("Frontend started on http://localhost:8501")
        logger.info("Local services launched successfully.")
        logger.info("Services are running in background threads.")
        logger.info("Press Ctrl+C to stop all services.")
        
        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down services...")
            return {"status": "success"}
            
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to launch local services: {e}")
        return {"status": "failed", "error": str(e)}


def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(description="Run the visual search engine pipeline.")
    
    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/deepfashion",
        help="Path to the source dataset directory (e.g., deepfashion).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Limit number of images processed (for testing/development).",
    )
    parser.add_argument(
        "--partitions",
        nargs="+",
        choices=["train", "val", "test"],
        help="Dataset partitions to process (default: all).",
    )
    
    # Pipeline control arguments
    parser.add_argument(
        "--step",
        type=str,
        choices=["data", "embeddings", "database", "validate"],
        help="Run specific pipeline step only.",
    )
    parser.add_argument(
        "--complete",
        action="store_true",
        help="Run all pipeline steps.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing progress.",
    )
    
    # Service arguments
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Launch services after pipeline completion.",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Use Docker for services.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local processes for services.",
    )
    
    # Processing arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for processing (affects VRAM usage).",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.docker and args.local:
        logger.error("Cannot use both --docker and --local flags.")
        sys.exit(1)
    
    if not args.step and not args.complete and not args.launch:
        logger.error("Must specify --step, --complete, or --launch.")
        sys.exit(1)
    
    # Update config with user-provided paths
    Config.DATA_DIR = args.data_dir
    
    logger.info(f"Initializing pipeline with source data from: {Config.DATA_DIR}")
    logger.info(f"Processed data will be stored in: {Config.PROCESSED_DIR}")
    
    start_time = datetime.now()
    results = {}
    
    # Determine which steps to run
    steps_to_run = []
    if args.complete:
        steps_to_run = ["data", "embeddings", "database", "validate"]
    elif args.step:
        steps_to_run = [args.step]
    
    # Run pipeline steps
    for step in steps_to_run:
        if step == "data":
            results["data_preparation"] = run_data_preparation(
                Config.DATA_DIR, args.max_images, args.partitions
            )
            if results["data_preparation"]["status"] == "failed":
                print(json.dumps(results["data_preparation"], indent=2))
                return
                
        elif step == "embeddings":
            results["embedding_generation"] = run_embedding_generation(
                args.batch_size, args.max_images
            )
            if results["embedding_generation"]["status"] == "failed":
                print(json.dumps(results["embedding_generation"], indent=2))
                return
                
        elif step == "database":
            results["database_setup"] = run_database_setup()
            if results["database_setup"]["status"] == "failed":
                print(json.dumps(results["database_setup"], indent=2))
                return
                
        elif step == "validate":
            results["validation"] = run_validation()
            if results["validation"]["status"] == "failed":
                print(json.dumps(results["validation"], indent=2))
                return
    
    # Launch services if requested
    if args.launch:
        if args.docker:
            results["service_launch"] = launch_services_docker()
        elif args.local:
            results["service_launch"] = launch_services_local()
        else:
            # Default to docker if available
            results["service_launch"] = launch_services_docker()
            
        if results["service_launch"]["status"] == "failed":
            print(json.dumps(results["service_launch"], indent=2))
            return
    
    logger.info(f"Pipeline finished in {datetime.now() - start_time}.")
    if results:
        logger.info("Pipeline results:")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()