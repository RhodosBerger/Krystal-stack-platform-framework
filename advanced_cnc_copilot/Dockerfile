FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cmake \
    make \
    libpq-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies (with CMAKE_ARGS for llama-cpp-python if needed)
# optimizing for CPU build in docker by default
RUN CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_OPENBLAS=ON" pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Default command (overridden in docker-compose)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
