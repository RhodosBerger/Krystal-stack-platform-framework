"""
Main application entry point for FANUC RISE v2.1 Advanced CNC Copilot
"""
from fastapi import FastAPI
import logging
from datetime import datetime

from backend.cms.app_factory import create_app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the application using the factory pattern
app = create_app()

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False,  # Disable reload in production
        log_level="info"
    )
```