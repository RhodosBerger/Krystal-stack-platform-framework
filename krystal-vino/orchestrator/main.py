"""
Krystal Vino Orchestrator

High-level coordination layer that interfaces with:
- Rust real-time control core (via gRPC)
- LLM inference engines
- Hardware abstraction layer
- External systems (monitoring, logging, etc.)
"""

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import structlog
import os

from config import Settings
from handlers import control_handler, inference_handler, metrics_handler

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Load configuration
settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    logger.info("krystal_vino_startup", version="0.1.0", environment=settings.environment)

    # Initialize Rust core connection
    await control_handler.initialize()

    yield

    # Cleanup
    logger.info("krystal_vino_shutdown")
    await control_handler.shutdown()


# Create FastAPI application
app = FastAPI(
    title="Krystal Vino Orchestrator",
    description="Real-time hardware optimization via statistical governors",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health & Status Endpoints
# ============================================================================

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "krystal-vino-orchestrator",
        "version": "0.1.0",
    }


@app.get("/status")
async def status():
    """Get system status"""
    control_status = await control_handler.get_status()
    return {
        "orchestrator": "healthy",
        "control_core": control_status,
        "governors": await control_handler.list_governors(),
    }


# ============================================================================
# Control Endpoints
# ============================================================================

@app.post("/api/control/decide")
async def control_decide(request: dict):
    """
    Make a safety decision using the control core.

    Request body:
    {
        "input": {"speed": 100, "vibration": 0.5, ...},
        "context": {"material": "aluminum", "tool": "endmill", ...}
    }
    """
    try:
        decision = await control_handler.make_decision(
            request.get("input", {}),
            request.get("context", {})
        )
        return {"success": True, "decision": decision}
    except Exception as e:
        logger.error("decision_error", error=str(e))
        return {"success": False, "error": str(e)}, 500


@app.post("/api/control/observe")
async def control_observe(metrics: dict):
    """
    Record metrics for governors to learn from.

    Request body:
    {
        "vibration": 0.5,
        "temperature": 65.2,
        "speed": 100,
        ...
    }
    """
    try:
        await control_handler.record_metrics(metrics)
        return {"success": True, "recorded": True}
    except Exception as e:
        logger.error("metrics_error", error=str(e))
        return {"success": False, "error": str(e)}, 500


# ============================================================================
# Governor Management Endpoints
# ============================================================================

@app.get("/api/governors")
async def list_governors():
    """List all registered governors"""
    governors = await control_handler.list_governors()
    return {"governors": governors}


@app.get("/api/governors/{governor_name}")
async def get_governor_info(governor_name: str):
    """Get information about a specific governor"""
    info = await control_handler.get_governor_info(governor_name)
    if info:
        return info
    return {"error": "Governor not found"}, 404


@app.post("/api/governors/{governor_name}/reset")
async def reset_governor(governor_name: str):
    """Reset a governor to initial state"""
    try:
        await control_handler.reset_governor(governor_name)
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}, 500


# ============================================================================
# Metrics Endpoints
# ============================================================================

@app.get("/api/metrics/{metric_name}")
async def get_metric_stats(metric_name: str):
    """Get statistics for a metric"""
    stats = await metrics_handler.get_stats(metric_name)
    if stats:
        return stats
    return {"error": "Metric not found"}, 404


@app.get("/api/metrics")
async def list_metrics():
    """List all available metrics"""
    return await metrics_handler.list_metrics()


# ============================================================================
# WebSocket for Real-Time Streaming
# ============================================================================

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """
    WebSocket endpoint for real-time telemetry streaming.

    Clients can stream metrics and receive governor decisions.
    """
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()

            # Process decision request
            decision = await control_handler.make_decision(
                data.get("input", {}),
                data.get("context", {})
            )

            # Record metrics for learning
            if "metrics" in data:
                await control_handler.record_metrics(data["metrics"])

            # Send decision back
            await websocket.send_json({
                "decision_id": decision.get("id"),
                "allowed": decision.get("allowed"),
                "confidence": decision.get("confidence"),
                "safe_limit": decision.get("safe_limit"),
            })

    except Exception as e:
        logger.error("websocket_error", error=str(e))
    finally:
        await websocket.close()


# ============================================================================
# Inference Endpoints (OpenVINO integration)
# ============================================================================

@app.post("/api/inference/predict")
async def inference_predict(request: dict):
    """
    Run inference on a model.

    Request body:
    {
        "model": "tool_wear_predictor",
        "input": [[...]]  # Input tensor
    }
    """
    try:
        result = await inference_handler.predict(
            request["model"],
            request["input"]
        )
        return {"success": True, "output": result}
    except Exception as e:
        logger.error("inference_error", error=str(e))
        return {"success": False, "error": str(e)}, 500


# ============================================================================
# Error Handling
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error("unhandled_exception", error=str(exc), traceback=exc)
    return {
        "error": "Internal server error",
        "detail": str(exc) if settings.environment == "development" else None,
    }, 500


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.environment == "development",
        log_level="info",
    )
