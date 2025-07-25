#!/usr/bin/env python3
"""
Deployment preparation script for Visual Search Engine.
This script helps prepare the application for cloud deployment.
"""

import os
import sys
import json
import shutil
from pathlib import Path

def create_render_yaml():
    """Create render.yaml for Render.com deployment."""
    render_config = {
        "services": [{
            "type": "web",
            "name": "visual-search-engine",
            "env": "python",
            "plan": "free",
            "buildCommand": "python -m pip install --upgrade pip && pip install -r requirements.txt",
            "startCommand": "streamlit run frontend/streamlit_app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --server.enableCORS false",
            "envVars": [
                {"key": "PYTHON_VERSION", "value": "3.11.0"},
                {"key": "ENVIRONMENT", "value": "production"},
                {"key": "DEBUG", "value": "false"},
                {"key": "DEVICE", "value": "cpu"},
                {"key": "BATCH_SIZE", "value": "8"}
            ]
        }]
    }
    
    with open("render.yaml", "w") as f:
        import yaml
        yaml.dump(render_config, f, default_flow_style=False)
    
    print("✅ Created render.yaml for Render.com deployment")

def create_procfile():
    """Create Procfile for Heroku deployment."""
    procfile_content = "web: streamlit run frontend/streamlit_app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true\n"
    
    with open("Procfile", "w") as f:
        f.write(procfile_content)
    
    print("✅ Created Procfile for Heroku deployment")

def create_runtime_txt():
    """Create runtime.txt for Heroku deployment."""
    with open("runtime.txt", "w") as f:
        f.write("python-3.11.0\n")
    
    print("✅ Created runtime.txt for Heroku deployment")

def create_production_env():
    """Create .env.production template."""
    env_content = """# Production Environment Variables
ENVIRONMENT=production
DEBUG=false
API_HOST=0.0.0.0
API_PORT=8001
STREAMLIT_PORT=8501
DEVICE=cpu
BATCH_SIZE=8
MAX_SEARCH_RESULTS=10
LOG_LEVEL=INFO
MAX_FILE_SIZE=10485760
ALLOWED_EXTENSIONS=.jpg,.jpeg,.png,.webp
"""
    
    with open(".env.production", "w") as f:
        f.write(env_content)
    
    print("✅ Created .env.production template")

def update_gitignore():
    """Update .gitignore for deployment."""
    gitignore_additions = """
# Deployment files
.env.production
.env.local

# Large data files (don't deploy to git)
data/deepfashion/Img/
data/processed/embeddings.npy
*.npy
*.parquet

# Logs
logs/
*.log

# Cache
__pycache__/
.pytest_cache/
"""
    
    with open(".gitignore", "a") as f:
        f.write(gitignore_additions)
    
    print("✅ Updated .gitignore for deployment")

def optimize_requirements():
    """Create optimized requirements.txt for deployment."""
    # Read current requirements
    with open("requirements.txt", "r") as f:
        requirements = f.read()
    
    # Create deployment-optimized version
    optimized_requirements = """# Core ML/DL Libraries (CPU optimized for deployment)
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0

# Computer Vision & Image Processing
Pillow>=9.5.0

# Numerical Computing
numpy>=1.24.0
pandas>=2.0.0

# Vector Database
faiss-cpu>=1.7.4

# API Framework
fastapi>=0.100.0
uvicorn[standard]>=0.22.0
python-multipart>=0.0.6

# Web Framework
streamlit>=1.24.0

# HTTP Requests
requests>=2.31.0

# Utilities
python-dotenv>=1.0.0
pydantic>=2.0.0
tqdm>=4.65.0

# Logging
loguru>=0.7.0
"""
    
    with open("requirements.deployment.txt", "w") as f:
        f.write(optimized_requirements)
    
    print("✅ Created requirements.deployment.txt (optimized for cloud)")

def create_dockerfile():
    """Create Dockerfile for container deployment."""
    dockerfile_content = """FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \\
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true"]
"""
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    print("✅ Created Dockerfile for container deployment")

def verify_deployment_readiness():
    """Verify the application is ready for deployment."""
    print("\n🔍 Verifying deployment readiness...")
    
    issues = []
    
    # Check if data is processed
    if not os.path.exists("data/processed/metadata.csv"):
        issues.append("❌ No processed data found. Run pipeline first.")
    
    # Check file sizes
    if os.path.exists("data/processed/embeddings.npy"):
        size_mb = os.path.getsize("data/processed/embeddings.npy") / (1024 * 1024)
        if size_mb > 100:
            issues.append(f"⚠️  Large embeddings file ({size_mb:.1f}MB). Consider smaller dataset for free tier.")
    
    # Check requirements
    if not os.path.exists("requirements.txt"):
        issues.append("❌ requirements.txt not found")
    
    # Check main files
    required_files = [
        "frontend/streamlit_app.py",
        "src/api/main.py",
        "README.md"
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            issues.append(f"❌ Required file missing: {file}")
    
    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("✅ Application is ready for deployment!")
        return True

def main():
    """Main deployment preparation function."""
    print("🚀 Preparing Visual Search Engine for Cloud Deployment")
    print("=" * 55)
    
    # Check if we're in the right directory
    if not os.path.exists("frontend/streamlit_app.py"):
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Create deployment files
    try:
        print("\n📝 Creating deployment configuration files...")
        create_render_yaml()
        create_procfile()
        create_runtime_txt()
        create_production_env()
        create_dockerfile()
        optimize_requirements()
        update_gitignore()
        
        print("\n🔍 Verifying deployment readiness...")
        ready = verify_deployment_readiness()
        
        print("\n📋 Deployment Files Created:")
        print("  ✅ render.yaml (for Render.com)")
        print("  ✅ Procfile (for Heroku)")
        print("  ✅ runtime.txt (Python version)")
        print("  ✅ Dockerfile (for containers)")
        print("  ✅ .env.production (environment template)")
        print("  ✅ requirements.deployment.txt (optimized)")
        
        print("\n🌐 Next Steps:")
        print("  1. Test locally: streamlit run frontend/streamlit_app.py")
        print("  2. Commit to Git: git add . && git commit -m 'Prepare for deployment'")
        print("  3. Push to GitHub: git push origin main")
        print("  4. Deploy on your chosen platform:")
        print("     • Render.com: Connect GitHub repo")
        print("     • Streamlit Cloud: Use share.streamlit.io")
        print("     • Heroku: heroku create && git push heroku main")
        
        if ready:
            print("\n🎉 Your application is ready for cloud deployment!")
        else:
            print("\n⚠️  Please fix the issues above before deploying.")
            
    except Exception as e:
        print(f"❌ Error during preparation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
