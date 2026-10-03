# ==============================================================================
# KRYSTAL-STACK: MODULAR MULTI-PROTOCOL CONTAINER PLATFORM
# ==============================================================================
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 1. System Dependencies & Networking Utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    openssl \
    ca-certificates \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Python Dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# 3. Copy Application Code & Assets
COPY . /app/

# 4. Generate Default TLS Certificates if missing
RUN mkdir -p /app/krystal_web_hub/certs && \
    if [ ! -f /app/krystal_web_hub/certs/cert.pem ]; then \
        openssl req -x509 -newkey rsa:2048 \
        -keyout /app/krystal_web_hub/certs/key.pem \
        -out /app/krystal_web_hub/certs/cert.pem \
        -days 365 -nodes -subj "/CN=krystal.mesh/O=KrystalStack/C=SK"; \
    fi

# 5. Expose Ports:
#   - 8080: Web Hub & 3D WebGL Studio (HTTP & WebSocket)
#   - 8089: Engine Core CMS & AST Backend (HTTP/REST)
#   - 8443: Secure Commerce, Inventory & Underdog Gateway (HTTPS/TLS)
EXPOSE 8080 8089 8443

# 6. Default Command (Can be overridden by docker-compose)
CMD ["python3", "-u", "-m", "krystal_web_hub.server", "8080"]
