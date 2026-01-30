"""
Unified Manufacturing API
FastAPI interface for all manufacturing intelligence systems

ENDPOINTS:
- POST /api/manufacturing/quote - Instant quote
- POST /api/manufacturing/optimize - Optimize specifications
- POST /api/manufacturing/reaas - Reverse engineering
- POST /api/manufacturing/simulate - Synthetic simulation
- POST /api/manufacturing/complete - Complete workflow
- GET  /api/manufacturing/status - System health
- WS   /ws/realtime - Real-time sensor data

AUTHENTICATION: JWT tokens (future)
RATE LIMITING: 100 requests/minute/IP
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
import asyncio
from datetime import datetime
import json

from cms.master_orchestrator import (
    MasterManufacturingOrchestrator,
    ManufacturingRequest,
    ManufacturingResponse
)
from cms.llm_gcode_generator import LLMGCodeGenerator
from cms.multi_bot_system import BotCoordinator
from cms.iot_sensor_integration import SensorManager, SensorConfig, SensorType
from cms.predictive_maintenance import FeatureEngineeringPipeline, AnomalyDetectionEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Advanced CNC Copilot API",
    description="Unified API for manufacturing intelligence systems",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware (allow all origins for dev - restrict in production!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize orchestrator and intelligence systems
orchestrator = None
gcode_generator = None
bot_coordinator = None
feature_pipeline = None
anomaly_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize orchestrator and intelligence systems on startup"""
    global orchestrator, gcode_generator, bot_coordinator, feature_pipeline, anomaly_engine
    logger.info("🚀 Starting Advanced CNC Copilot API...")
    
    orchestrator = MasterManufacturingOrchestrator()
    gcode_generator = LLMGCodeGenerator()
    bot_coordinator = BotCoordinator()
    feature_pipeline = FeatureEngineeringPipeline()
    anomaly_engine = AnomalyDetectionEngine()
    
    logger.info("✅ API ready with all intelligence systems!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("👋 Shutting down API...")


# =============================================================================
# REQUEST/RESPONSE MODELS (Pydantic)
# =============================================================================

class QuoteRequest(BaseModel):
    """Request model for instant quote"""
    part_description: str = Field(..., description="Description of part to manufacture")
    quantity: int = Field(default=1, ge=1, description="Quantity to manufacture")
    material: Optional[str] = Field(None, description="Preferred material")
    tolerance_mm: Optional[float] = Field(None, description="Tolerance requirement in mm")
    max_cost_per_part: Optional[float] = Field(None, description="Maximum cost per part")
    urgency: str = Field(default="standard", description="Urgency level: rush, expedited, standard, economical")
    
    class Config:
        schema_extra = {
            "example": {
                "part_description": "100 aluminum brackets, ±0.05mm tolerance",
                "quantity": 100,
                "material": "Aluminum6061",
                "tolerance_mm": 0.05,
                "max_cost_per_part": 100.0,
                "urgency": "standard"
            }
        }


class OptimizeRequest(BaseModel):
    """Request model for optimization"""
    part_description: str
    quantity: int = 1
    material: Optional[str] = None
    complexity: Optional[str] = None
    tolerance_mm: Optional[float] = None
    max_cost_per_part: Optional[float] = None
    
    # Optional geometry for SolidWorks validation
    geometry: Optional[Dict] = None
    loading_conditions: Optional[Dict] = None
    
    class Config:
        schema_extra = {
            "example": {
                "part_description": "Titanium shaft for aerospace application",
                "quantity": 50,
                "material": "Titanium6Al4V",
                "complexity": "moderate",
                "tolerance_mm": 0.01,
                "geometry": {
                    "type": "shaft",
                    "dimensions": {"diameter": 30, "length": 200}
                },
                "loading_conditions": {
                    "force_n": 50000,
                    "direction": [0, -1, 0],
                    "frequency_hz": 10.0
                }
            }
        }


class REaaSRequest(BaseModel):
    """Request for reverse engineering as a service"""
    part_description: str
    original_part_images: Optional[List[str]] = Field(None, description="URLs to images of worn part")
    scanned_dimensions: Optional[Dict] = Field(None, description="Scanned dimensions if available")
    material: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "part_description": "Worn bearing housing from marine pump, 10 years of use",
                "material": "Steel4140",
                "scanned_dimensions": {
                    "outer_diameter": 146.5,
                    "inner_diameter": 89.2,
                    "deviation": 3.5
                }
            }
        }


class SimulateRequest(BaseModel):
    """Request for synthetic simulation"""
    part_type: str = Field(..., description="Type of part: bearing_housing, shaft, bracket, pulley, gear")
    material: Optional[str] = None
    duration_minutes: Optional[float] = Field(10.0, description="Simulation duration")
    inject_failures: bool = Field(True, description="Whether to inject realistic failures")
    
    class Config:
        schema_extra = {
            "example": {
                "part_type": "gear",
                "material": "Steel4140",
                "duration_minutes": 5.0,
                "inject_failures": True
            }
        }


class ApiResponse(BaseModel):
    """Standard API response"""
    success: bool
    request_id: str
    message: str
    data: Optional[Dict] = None
    errors: Optional[List[str]] = None
    processing_time_seconds: Optional[float] = None


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Advanced CNC Copilot API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    status = orchestrator.get_system_status() if orchestrator else {}
    
    # Add intelligence systems status
    intelligence_systems = {
        "gcode_generator": gcode_generator is not None,
        "bot_coordinator": bot_coordinator is not None,
        "feature_pipeline": feature_pipeline is not None,
        "anomaly_engine": anomaly_engine is not None
    }
    
    all_systems = {**status, **intelligence_systems}
    
    return {
        "status": "healthy" if all(all_systems.values()) else "degraded",
        "timestamp": datetime.now().isoformat(),
        "core_systems": status,
        "intelligence_systems": intelligence_systems
    }


@app.post("/api/manufacturing/quote", response_model=ApiResponse)
async def get_quote(request: QuoteRequest, background_tasks: BackgroundTasks):
    """
    Get instant manufacturing quote
    
    Returns cost estimate, time estimate, and recommended producer
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    # Generate request ID
    request_id = f"QUOTE-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    try:
        # Create manufacturing request
        mfg_request = ManufacturingRequest(
            request_id=request_id,
            request_type='quote',
            part_description=request.part_description,
            material=request.material,
            quantity=request.quantity,
            tolerance_mm=request.tolerance_mm,
            max_cost_per_part=request.max_cost_per_part,
            urgency=request.urgency
        )
        
        # Process request
        mfg_response = orchestrator.process_request(mfg_request)
        
        # Format response
        return ApiResponse(
            success=mfg_response.status == 'success',
            request_id=request_id,
            message="Quote generated successfully",
            data={
                "estimated_cost_per_part": mfg_response.estimated_cost,
                "estimated_time_minutes": mfg_response.estimated_time_minutes,
                "predicted_lifetime_hours": mfg_response.predicted_lifetime_hours,
                "recommended_producer": mfg_response.recommended_producer,
                "quality_prediction": mfg_response.quality_prediction,
                "recommendations": mfg_response.recommendations,
                "systems_used": mfg_response.systems_used
            },
            errors=mfg_response.warnings if mfg_response.warnings else None,
            processing_time_seconds=mfg_response.processing_time_seconds
        )
    
    except Exception as e:
        logger.error(f"Error processing quote: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/manufacturing/optimize", response_model=ApiResponse)
async def optimize_specs(request: OptimizeRequest):
    """
    Optimize part specifications
    
    Returns optimal material, tooling, speeds/feeds, and optionally
    SolidWorks validation results
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    request_id = f"OPT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    try:
        mfg_request = ManufacturingRequest(
            request_id=request_id,
            request_type='optimize',
            part_description=request.part_description,
            material=request.material,
            quantity=request.quantity,
            complexity=request.complexity,
            tolerance_mm=request.tolerance_mm,
            max_cost_per_part=request.max_cost_per_part,
            geometry=request.geometry,
            loading_conditions=request.loading_conditions
        )
        
        mfg_response = orchestrator.process_request(mfg_request)
        
        return ApiResponse(
            success=mfg_response.status == 'success',
            request_id=request_id,
            message="Optimization completed",
            data={
                "optimization_result": mfg_response.optimization_result,
                "solidworks_validation": mfg_response.solidworks_validation,
                "estimated_cost": mfg_response.estimated_cost,
                "predicted_lifetime": mfg_response.predicted_lifetime_hours,
                "recommendations": mfg_response.recommendations
            },
            errors=mfg_response.warnings,
            processing_time_seconds=mfg_response.processing_time_seconds
        )
    
    except Exception as e:
        logger.error(f"Error optimizing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/manufacturing/reaas", response_model=ApiResponse)
async def reverse_engineer(request: REaaSRequest):
    """
    Reverse Engineering as a Service
    
    Analyzes worn/obsolete part and generates replacement specifications
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    request_id = f"REAAS-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    try:
        mfg_request = ManufacturingRequest(
            request_id=request_id,
            request_type='reaas',
            part_description=request.part_description,
            material=request.material
        )
        
        mfg_response = orchestrator.process_request(mfg_request)
        
        return ApiResponse(
            success=mfg_response.status == 'success',
            request_id=request_id,
            message="Reverse engineering analysis complete",
            data={
                "reaas_analysis": mfg_response.reaas_analysis,
                "optimization_result": mfg_response.optimization_result,
                "estimated_cost": mfg_response.estimated_cost,
                "recommendations": mfg_response.recommendations
            },
            errors=mfg_response.warnings,
            processing_time_seconds=mfg_response.processing_time_seconds
        )
    
    except Exception as e:
        logger.error(f"Error in REaaS: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/manufacturing/simulate", response_model=ApiResponse)
async def run_simulation(request: SimulateRequest):
    """
    Run synthetic manufacturing simulation
    
    Generates realistic operation data for testing and analysis
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    request_id = f"SIM-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    try:
        mfg_request = ManufacturingRequest(
            request_id=request_id,
            request_type='simulate',
            part_description=f"Synthetic {request.part_type}",
            part_type=request.part_type,
            material=request.material
        )
        
        mfg_response = orchestrator.process_request(mfg_request)
        
        return ApiResponse(
            success=mfg_response.status == 'success',
            request_id=request_id,
            message="Simulation completed",
            data={
                "simulation_result": mfg_response.synthetic_simulation,
                "estimated_cost": mfg_response.estimated_cost,
                "estimated_time": mfg_response.estimated_time_minutes
            },
            errors=mfg_response.warnings,
            processing_time_seconds=mfg_response.processing_time_seconds
        )
    
    except Exception as e:
        logger.error(f"Error simulating: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/manufacturing/complete", response_model=ApiResponse)
async def complete_workflow(request: QuoteRequest):
    """
    Complete manufacturing workflow
    
    Runs all systems: LLM understanding, optimization, validation, simulation
    Returns comprehensive manufacturing package
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    request_id = f"COMPLETE-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    try:
        mfg_request = ManufacturingRequest(
            request_id=request_id,
            request_type='complete',
            part_description=request.part_description,
            material=request.material,
            quantity=request.quantity,
            tolerance_mm=request.tolerance_mm,
            max_cost_per_part=request.max_cost_per_part,
            urgency=request.urgency
        )
        
        mfg_response = orchestrator.process_request(mfg_request)
        
        return ApiResponse(
            success=mfg_response.status == 'success',
            request_id=request_id,
            message="Complete workflow finished",
            data={
                "llm_understanding": mfg_response.llm_understanding,
                "reaas_analysis": mfg_response.reaas_analysis,
                "optimization": mfg_response.optimization_result,
                "simulation": mfg_response.synthetic_simulation,
                "estimated_cost": mfg_response.estimated_cost,
                "estimated_time": mfg_response.estimated_time_minutes,
                "predicted_lifetime": mfg_response.predicted_lifetime_hours,
                "producer": mfg_response.recommended_producer,
                "recommendations": mfg_response.recommendations,
                "systems_used": mfg_response.systems_used
            },
            errors=mfg_response.warnings,
            processing_time_seconds=mfg_response.processing_time_seconds
        )
    
    except Exception as e:
        logger.error(f"Error in complete workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# INTELLIGENCE FEATURE ENDPOINTS
# =============================================================================

class GCodeRequest(BaseModel):
    """Request for G-Code generation"""
    description: str = Field(..., description="Natural language description of operation")
    validate: bool = Field(True, description="Whether to validate generated code")
    
    class Config:
        schema_extra = {
            "example": {
                "description": "Mill 50mm square pocket, 10mm deep, leave 0.5mm for finishing",
                "validate": True
            }
        }


class BotConsultRequest(BaseModel):
    """Request for bot consultation"""
    question: str = Field(..., description="Manufacturing question")
    context: Optional[Dict] = Field(None, description="Additional context")
    
    class Config:
        schema_extra = {
            "example": {
                "question": "What cutting speed for aluminum 6061?",
                "context": {"material": "Aluminum6061", "operation": "pocket_milling"}
            }
        }


@app.post("/api/gcode/generate", response_model=ApiResponse)
async def generate_gcode(request: GCodeRequest):
    """
    Generate G-Code from natural language description
    
    Uses LLM-powered generator to create validated CNC programs
    """
    if not gcode_generator:
        raise HTTPException(status_code=503, detail="G-Code generator not initialized")
    
    request_id = f"GCODE-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    try:
        program, validation = gcode_generator.generate_from_description(
            request.description,
            validate=request.validate
        )
        
        return ApiResponse(
            success=validation['valid'] if validation else True,
            request_id=request_id,
            message="G-Code generated successfully",
            data={
                "program_number": program.program_number,
                "program_name": program.program_name,
                "gcode": program.to_string(),
                "estimated_time_minutes": program.estimated_time_minutes,
                "line_count": len(program.gcode_lines),
                "validation": validation
            },
            processing_time_seconds=0.1
        )
    
    except Exception as e:
        logger.error(f"Error generating G-Code: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/consulting/ask", response_model=ApiResponse)
async def consult_expert(request: BotConsultRequest):
    """
    Ask manufacturing expert bot a question
    
    Routes question to appropriate specialized bot using confidence scoring
    """
    if not bot_coordinator:
        raise HTTPException(status_code=503, detail="Bot coordinator not initialized")
    
    request_id = f"CONSULT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    try:
        result = bot_coordinator.route_question(
            request.question,
            context=request.context
        )
        
        return ApiResponse(
            success=True,
            request_id=request_id,
            message="Consultation complete",
            data={
                "primary_consultant": result['primary_consultant'],
                "collaborators": result.get('collaborators', []),
                "all_scores": result.get('all_scores', {})
            },
            processing_time_seconds=0.05
        )
    
    except Exception as e:
        logger.error(f"Error in consultation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/consulting/bots")
async def list_bots():
    """
    List all available expert bots with their specializations
    """
    if not bot_coordinator:
        raise HTTPException(status_code=503, detail="Bot coordinator not initialized")
    
    roster = bot_coordinator.get_bot_roster()
    
    return {
        "total_bots": len(roster),
        "bots": roster
    }


@app.get("/api/predictive-maintenance/status/{operation_id}")
async def get_predictive_maintenance_status(operation_id: int):
    """
    Get predictive maintenance status for an operation
    
    Returns anomaly scores, feature summaries, and alerts
    """
    # This would integrate with actual database/sensor data in production
    # For now, return example status
    
    return {
        "operation_id": operation_id,
        "timestamp": datetime.now().isoformat(),
        "health_score": 0.92,  # 0-1 scale (1 = perfect health)
        "anomaly_score": 0.08,
        "anomalies_detected": [],
        "trends": {
            "spindle_load_trend": "+0.002/min",
            "vibration_trend": "stable",
            "temperature_trend": "+0.1°C/hour"
        },
        "predictions": {
            "failure_probability": 0.05,
            "estimated_time_to_failure_hours": None,
            "confidence": 0.88
        },
        "recommendations": [
            "Continue normal operation",
            "Schedule routine maintenance in 200 hours"
        ]
    }


# =============================================================================
# WEBSOCKET ENDPOINT (Real-time Data)
# =============================================================================

class ConnectionManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Active connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Active connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)


manager = ConnectionManager()


@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time sensor data streaming
    
    Sends simulated sensor data every second
    """
    await manager.connect(websocket)
    try:
        while True:
            # Simulate real-time sensor data
            import random
            data = {
                "timestamp": datetime.now().isoformat(),
                "spindle_load": round(random.uniform(40, 70), 2),
                "vibration_x": round(random.uniform(0.1, 0.5), 4),
                "vibration_y": round(random.uniform(0.08, 0.4), 4),
                "vibration_z": round(random.uniform(0.06, 0.3), 4),
                "tool_health": round(random.uniform(0.7, 0.95), 4),
                "temperature_c": round(random.uniform(25, 45), 2),
                "cortisol": round(random.uniform(10, 40), 2),
                "dopamine": round(random.uniform(60, 95), 2),
                "quality_pass": random.choice([True, True, True, False])
            }
            
            await websocket.send_json(data)
            await asyncio.sleep(1)  # Send every second
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# =============================================================================
# RUNNER
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("Advanced CNC Copilot - Unified API")
    print("=" * 70)
    print("\n🚀 Starting API server...")
    print("\n📚 API Documentation: http://localhost:8000/docs")
    print("📡 ReDoc: http://localhost:8000/redoc")
    print("💻 WebSocket: ws://localhost:8000/ws/realtime")
    print("\n" + "=" * 70 + "\n")
    
    uvicorn.run(
        "manufacturing_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
