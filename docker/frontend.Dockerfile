# Frontend Dockerfile for Visual Search Engine
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (only what's needed for Streamlit)
RUN pip install --no-cache-dir streamlit requests pandas pillow plotly

# Copy frontend code
COPY frontend/ ./frontend/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]