"""
Dev-conditional Main Application
Autonomous Server Application Engine with LLM Integration
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os
import logging
from contextlib import asynccontextmanager

from .api import workflow, codegen, llm
from .websocket import ws_handler
from .config import settings
from .storage.database import init_db
from .logging_config import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    setup_logging()
    logging.info("Starting Dev-conditional Server Engine...")

    # Initialize database
    await init_db()
    logging.info("Database initialized")

    # TODO: Initialize Kafka, Redis, and other services
    logging.info("Application startup complete")

    yield

    # Shutdown
    logging.info("Shutting down Dev-conditional Server Engine...")


# Create FastAPI application
app = FastAPI(
    title="Dev-conditional API",
    description="Autonomous Server Application Engine with LLM Integration",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(workflow.router, prefix="/api/workflow", tags=["workflow"])
app.include_router(codegen.router, prefix="/api/codegen", tags=["codegen"])
app.include_router(llm.router, prefix="/api/llm", tags=["llm"])

# Include WebSocket
app.include_router(ws_handler.router, prefix="/ws")

# Mount static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/generated", StaticFiles(directory="generated_projects"), name="generated")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Dev-conditional Autonomous Server Engine",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "database": "connected",  # TODO: Implement actual health checks
        "services": {
            "redis": "connected",  # TODO: Implement actual health checks
            "kafka": "connected"   # TODO: Implement actual health checks
        }
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions"""
    logging.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )