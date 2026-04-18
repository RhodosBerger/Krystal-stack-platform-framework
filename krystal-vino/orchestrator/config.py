"""Configuration management for Krystal Vino Orchestrator"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    environment: str = "development"  # development, staging, production

    # CORS
    cors_origins: List[str] = ["*"]

    # Rust Core Connection
    core_service_host: str = "localhost"
    core_service_port: int = 50051
    core_service_timeout: int = 5000  # milliseconds

    # OpenVINO Configuration
    openvino_device: str = "GPU"  # CPU, GPU, HETERO
    openvino_models_path: str = "./models"

    # Metrics
    metrics_max_samples: int = 10000
    metrics_flush_interval: int = 60  # seconds

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
