FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
# Create a dummy requirements if not exists, or just install key libs directly for this milestone
RUN pip install --no-cache-dir fastapi uvicorn redis numpy scipy celery[redis] requests python-multipart jinja2 aiofiles

# Copy Codebase
COPY . .

# Expose Port
EXPOSE 8000

# Command
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
