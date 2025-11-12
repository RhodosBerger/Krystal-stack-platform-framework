"""
Logging configuration for Dev-conditional Server Engine
"""

import logging
import logging.handlers
import os


def setup_logging():
    """Setup logging configuration"""

    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            # Console handler
            logging.StreamHandler(),

            # File handler with rotation
            logging.handlers.RotatingFileHandler(
                "logs/devcontitional.log",
                maxBytes=10485760,  # 10MB
                backupCount=5
            ),

            # Error file handler
            logging.handlers.RotatingFileHandler(
                "logs/errors.log",
                maxBytes=10485760,  # 10MB
                backupCount=5
            )
        ]
    )

    # Set specific logger for errors
    error_logger = logging.getLogger("errors")
    error_handler = logging.FileHandler("logs/errors.log")
    error_handler.setLevel(logging.ERROR)
    error_logger.addHandler(error_handler)

    # Set logging levels for specific modules
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("websockets").setLevel(logging.INFO)

    logging.info("Logging configuration initialized")