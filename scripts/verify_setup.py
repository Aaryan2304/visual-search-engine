#!/usr/bin/env python3
"""
Quick verification script to check if the visual search engine is properly set up.
Run this before starting the API to catch common configuration issues.
"""

import os
import sys
import importlib
from pathlib import Path

def check_python_version():
    """Check Python version compatibility."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Python 3.9+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'torch', 'transformers', 'fastapi', 'uvicorn', 
        'streamlit', 'faiss', 'PIL', 'numpy', 'pandas'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'PIL':
                importlib.import_module('PIL')
            elif package == 'faiss':
                importlib.import_module('faiss')
            else:
                importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing.append(package)
    
    return len(missing) == 0, missing

def check_project_structure():
    """Check if project files are in place."""
    required_files = [
        'src/config.py',
        'src/api/main.py', 
        'src/embeddings/clip_model.py',
        'src/database/vector_db.py',
        'frontend/streamlit_app.py',
        'scripts/run_pipeline.py'
    ]
    
    missing = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (missing)")
            missing.append(file_path)
    
    return len(missing) == 0, missing

def check_config():
    """Test configuration loading."""
    try:
        # Add current directory to Python path for imports
        import sys
        import os
        current_dir = os.getcwd()
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        from src.config import Config
        print(f"✅ Config loaded successfully")
        print(f"   - Device: {Config.DEVICE}")
        print(f"   - Model: {Config.CLIP_MODEL_NAME}")
        print(f"   - Image size: {Config.IMAGE_SIZE}")
        return True
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False

def check_api_imports():
    """Test API module imports."""
    try:
        # Add current directory to Python path for imports
        import sys
        import os
        current_dir = os.getcwd()
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
            
        from src.api.main import app
        print("✅ API imports successful")
        return True
    except Exception as e:
        print(f"❌ API import error: {e}")
        return False

def check_data_directory():
    """Check if data directory exists."""
    data_dir = Path("data/deepfashion")
    if data_dir.exists():
        print(f"✅ Dataset directory exists: {data_dir}")
        
        # Check for key files
        key_files = [
            "Anno_coarse/list_eval_partition.txt",
            "Anno_coarse/list_category_img.txt",
            "Img/"
        ]
        
        for file_path in key_files:
            full_path = data_dir / file_path
            if full_path.exists():
                print(f"   ✅ {file_path}")
            else:
                print(f"   ❌ {file_path} (missing)")
        
        return True
    else:
        print(f"❌ Dataset directory not found: {data_dir}")
        print("   Download DeepFashion dataset first")
        return False

def main():
    """Run all verification checks."""
    print("🔍 Visual Search Engine Setup Verification\n")
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", lambda: check_dependencies()[0]),
        ("Project Structure", lambda: check_project_structure()[0]),
        ("Configuration", check_config),
        ("API Imports", check_api_imports),
        ("Data Directory", check_data_directory)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n📋 {check_name}:")
        try:
            passed = check_func()
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"❌ {check_name} failed with error: {e}")
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 All checks passed! You're ready to run the visual search engine.")
        print("\nNext steps:")
        print("1. Run pipeline: python scripts/run_pipeline.py --data-dir ./data/deepfashion --max-images 1000 --complete")
        print("2. Start API: python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8001")
        print("3. Start frontend: streamlit run frontend/streamlit_app.py --server.port 8501")
    else:
        print("❌ Some checks failed. Please fix the issues above before proceeding.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
