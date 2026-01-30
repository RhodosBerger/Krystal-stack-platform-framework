"""
FANUC RISE - Enterprise Backend API
Entry point for FastAPI application.
"""
import asyncio
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

from fastapi.staticfiles import StaticFiles
import os

# Import Core Modules
from backend.core.security import Token, User, get_current_active_user, create_access_token, verify_password, get_password_hash
from backend.core.orchestrator import orchestrator
from backend.worker import celery_app
from backend.core.llm_brain import LLMRouter
from backend.core.cortex_transmitter import cortex
from backend.core.sustainability_engine import sustainability
from cms.cross_session_intelligence import CrossSessionIntelligence, DataPoint
from cms.llm_integration_examples import LLMManufacturingAssistant

# Initialize Intelligence (Global)
intelligence_engine = CrossSessionIntelligence()
ai_assistant = LLMManufacturingAssistant()

app = FastAPI(
    title="FANUC RISE Enterprise API",
    description="High-Performance Manufacturing Intelligence Platform",
    version="2.0.0 (Enterprise)",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS (Restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static Files & Dashboard ---
dashboard_dir = os.path.join(os.path.dirname(__file__), "../cms/dashboard")
app.mount("/dashboard", StaticFiles(directory=dashboard_dir), name="dashboard")

from fastapi.responses import HTMLResponse, FileResponse

# --- Startup/Shutdown ---
@app.on_event("startup")
async def startup_event():
    await orchestrator.initialize()

@app.get("/", response_class=HTMLResponse)
async def root():
    # Serve the main dashboard index
    return FileResponse(os.path.join(dashboard_dir, "index.html"))

# --- Auth Endpoints ---

class LoginRequest(BaseModel):
    username: str
    password: str

from backend.core.database import SessionLocal
from backend.core.security import get_user, verify_password

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: LoginRequest):
    db = SessionLocal()
    try:
        user = get_user(db, form_data.username)
        if not user or not verify_password(form_data.password, user.password):
            raise HTTPException(status_code=400, detail="Incorrect username or password")
        
        access_token = create_access_token(
            data={"sub": user.username, "role": user.role}
        )
        return {"access_token": access_token, "token_type": "bearer"}
    finally:
        db.close()

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

# --- Core/Orchestrator Endpoints ---

@app.get("/api/health")
async def health_check():
    return orchestrator.get_system_status()

class ManufacturingRequest(BaseModel):
    type: str  # GENERATE_GCODE, OPTIMIZE_PROCESS, CONSULTATION
    payload: Dict[str, Any]

@app.post("/api/manufacturing/request")
async def process_manufacturing_request(
    request: ManufacturingRequest, 
    current_user: User = Depends(get_current_active_user)
):
    """
    Unified endpoint for processing manufacturing intent.
    Routes through Master Orchestrator.
    """
    response = await orchestrator.process_request(request.type, request.payload, current_user.username)
    return response

@app.get("/view/wizard")
async def view_wizard():
    """Serves the Multi-Step Setup Wizard"""
    return HTMLResponse(frontend_generator.generate_layout("wizard"))

@app.get("/api/wizard/config")
async def get_wizard_config():
    """Returns the configuration for the Multi-Step Wizard"""
    from cms.dynamic_form_builder import FieldType
    
    return {
        "steps": [
            {
                "id": "environment",
                "title": "Environment Setup",
                "description": "Configure your basic system environment and connectivity.",
                "icon": "🌐",
                "fields": [
                    {"id": "app_env", "type": "select", "label": "Environment", "options": [{"value": "dev", "label": "Development"}, {"value": "prod", "label": "Production"}], "default": "dev"},
                    {"id": "backend_url", "type": "text", "label": "Backend URL", "default": "http://localhost:8000"},
                    {"id": "enable_docker", "type": "toggle", "label": "Enable Docker Orchestration", "default": True}
                ]
            },
            {
                "id": "machine",
                "title": "CNC Configuration",
                "description": "Connect your physical or simulated CNC hardware.",
                "icon": "⚙️",
                "fields": [
                    {"id": "controller_type", "type": "select", "label": "Controller", "options": [{"value": "fanuc", "label": "Fanuc"}, {"value": "siemens", "label": "Siemens"}], "default": "fanuc"},
                    {"id": "machine_id", "type": "text", "label": "Machine ID", "default": "VMC-01"},
                    {"id": "use_mock", "type": "toggle", "label": "Use Simulated HAL (Mock)", "default": True}
                ]
            },
            {
                "id": "intelligence",
                "title": "AI & Neuro Tuning",
                "description": "Fine-tune the cognitive reward and monitoring engines.",
                "icon": "🧠",
                "fields": [
                    {"id": "dopamine_sensitivity", "type": "slider", "label": "Dopamine Sensitivity", "min": 0, "max": 100, "default": 75},
                    {"id": "risk_aversion", "type": "slider", "label": "Risk Aversion", "min": 0, "max": 100, "default": 85},
                    {"id": "enable_live_learning", "type": "toggle", "label": "Enable Real-time Learning", "default": True}
                ]
            },
            {
                "id": "finalize",
                "title": "Finalize Setup",
                "description": "Review and seal your system configuration.",
                "icon": "✅",
                "fields": [
                    {"id": "admin_email", "type": "text", "label": "Admin Email", "placeholder": "admin@factory.com"},
                    {"id": "confirm_safety", "type": "toggle", "label": "I accept the Safety Protocols (G90)", "default": False, "required": True}
                ]
            }
        ]
    }

# --- Phase 9: Global Swarm Intelligence ---


@app.post("/api/swarm/broadcast")
async def broadcast_machine_data(machine_id: str, data: Dict[str, Any]):
    """Manual broadcast for simulated swarm nodes"""
    orchestrator.update_machine_status(machine_id, data)
    return {"status": "SUCCESS", "node": machine_id}

@app.websocket("/ws/telemetry/{machine_id}")
async def websocket_endpoint_multi(websocket: WebSocket, machine_id: str):
    await websocket.accept()
    try:
        while True:
            # Simulate real-time brain states for any connecting machine
            import random
            telemetry = {
                "machine_id": machine_id,
                "rpm": random.randint(5000, 12000),
                "load": random.uniform(20.0, 95.0),
                "vibration": random.uniform(0.01, 0.45),
                "neuro_state": {
                    "dopamine": random.uniform(0, 100),
                    "cortisol": random.uniform(0, 100),
                    "serotonin": random.uniform(0, 100)
                },
                "action": random.choice(["MONITORING_LIVE", "ADAPTIVE_FEED_ADJUST", "THERMAL_COMPENSATION", "HEALTHY_HIVE_SYNC"])
            }
            
            # Register in Swarm Map
            orchestrator.update_machine_status(machine_id, telemetry)
            
            await websocket.send_json(telemetry)
            await asyncio.sleep(0.1)  # 10Hz Heartbeat
    except Exception as e:
        logger.warning(f"WebSocket closed for {machine_id}: {e}")



# --- WebSocket for Real-time Data ---

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Simulate 10Hz stream with Cognitive metrics
            import random
            import json
            
            rpm = random.uniform(4000, 7000)
            load = random.uniform(30, 90)
            vibration = random.uniform(0.01, 0.4)
            
            # Neuro-state simulation based on telemetry
            dopamine = 100 - (load / 2) # Reward decreases with load
            cortisol = load + (vibration * 100) # Stress increases with load and vibration
            serotonin = 80 - (vibration * 50) # Stability decreases with vibration
            
            action = "STABLE"
            if cortisol > 100:
                action = "REDUCE_FEED"
            elif dopamine < 30:
                action = "OPTIMIZE_PATH"
                
            data = {
                "timestamp": asyncio.get_event_loop().time(),
                "machine_id": "CNC-001",
                "rpm": round(rpm, 0),
                "load": round(load, 2),
                "vibration": round(vibration, 3),
                "neuro_state": {
                    "dopamine": round(dopamine, 1),
                    "cortisol": round(cortisol, 1),
                    "serotonin": round(serotonin, 1)
                },
                "action": action
            }

            # Feed into Intelligence Engine (Live Learning)
            try:
                intelligence_engine.add_data_point(DataPoint(
                    session_id="LIVE_SESSION_001",
                    timestamp=datetime.now(),
                    data_type="telemetry_live",
                    data=data,
                    machine_id="CNC-001"
                ))
            except Exception as e:
                # Don't let intelligence failures break the telemetry stream
                print(f"Intelligence Feed Error: {e}")

            await websocket.send_json(data)
            await asyncio.sleep(0.1) # 10Hz
    except Exception as e:
        print(f"WS Error: {e}")
        pass

@app.websocket("/ws/live_link")
async def websocket_live_link(websocket: WebSocket):
    """
    Bidirectional Bridge between Blender and Web Dashboard.
    - Blender sends: SELECTION_CHANGED, GEOMETRY_UPDATED
    - Web sends: TRIGGER_SIMULATION, HIGHLIGHT_OBJECT
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            
            # 1. Log to Cortex (The Mirror)
            cortex.transmit_intent(
                actor="BlenderClient",
                action=data.get("type", "UNKNOWN"),
                reasoning="Live Link Event",
                context=data.get("payload", {})
            )
            
            # 2. Handle Actions
            if data.get("type") == "TRIGGER_SIMULATION":
                # Logic to trigger backend simulation
                await _handle_live_simulation(data["payload"], websocket)
                
            elif data.get("type") == "SELECTION_CHANGED":
                # Broadcast to other clients (e.g. Web Dashboard)
                # In full prod, use Redis Pub/Sub here. 
                # For now, echo back confirmation.
                await websocket.send_json({"status": "ACK", "type": "SELECTION_UPDATED"})

    except Exception as e:
        print(f"LiveLink Error: {e}")

async def _handle_live_simulation(payload, ws):
    # Mock Simulation Run
    import asyncio
    await asyncio.sleep(1)
    result = {
        "type": "SIMULATION_RESULT",
        "data": {"safety_factor": 2.5, "max_stress": "150MPa"}
    }
    await ws.send_json(result)

@app.get("/api/knowledge/search")
async def search_knowledge(
    q: str = "", 
    current_user: User = Depends(get_current_active_user)
):
    """
    AJAX Search Validator: Filter Knowledge Base by query.
    """
    from cms.knowledge_engine import knowledge_engine
    
    results = []
    query = q.lower()
    
    # 1. Search in Persistent Knowledge (Preset Materials)
    for mat in knowledge_engine.materials_db:
        if query in mat['name'].lower() or query in mat.get('category', '').lower():
            mat_copy = mat.copy()
            mat_copy['source_type'] = 'Preset'
            results.append(mat_copy)
            
    # 2. Search in Pending (Redis) - Hybrid Search
    if orchestrator.redis_client:
        keys = orchestrator.redis_client.keys("pending_knowledge:*")
        for key in keys:
            data = orchestrator.redis_client.get(key)
            if data:
                try:
                    import json
                    pending = json.loads(data)
                    if query in pending['name'].lower():
                        pending['source_type'] = 'Pending (2FA)'
                        results.append(pending)
                except:
                   pass
                   
    return {"query": q, "count": len(results), "results": results}

@app.get("/api/knowledge/pending")
async def get_pending_knowledge(current_user: User = Depends(get_current_active_user)):
    """
    2FA Factor 1: View items held for verification.
    """
    import json # Ensure import
    
    if not orchestrator.redis_client:
         raise HTTPException(status_code=503, detail="Shared Control Pad Offline")
         
    # Scan for pending keys
    keys = orchestrator.redis_client.keys("pending_knowledge:*")
    pending_items = []
    for key in keys:
        data = orchestrator.redis_client.get(key)
        if data:
             try:
                pending_items.append(json.loads(data))
             except:
                pass
             
    return {"count": len(pending_items), "items": pending_items}

@app.post("/api/knowledge/verify/{verification_id}")
async def verify_knowledge_artifact(
    verification_id: str, 
    approved: bool = True,
    current_user: User = Depends(get_current_active_user)
):
    """
    2FA Factor 2: Approve artifact and commit to Knowledge Base.
    """
    if not orchestrator.redis_client:
         raise HTTPException(status_code=503, detail="Shared Control Pad Offline")
         
    key = f"pending_knowledge:{verification_id}"
    data_json = orchestrator.redis_client.get(key)
    if not data_json:
        raise HTTPException(status_code=404, detail="Verification ID not found or expired")
        
    if approved:
        # Commit to Knowledge Engine
        from cms.knowledge_engine import knowledge_engine
        import json
        artifact = json.loads(data_json)
        
        success = knowledge_engine.add_verified_preset(artifact)
        if success:
            orchestrator.redis_client.delete(key) # Clear pending
            return {"status": "verified", "message": f"Artifact {artifact['name']} added to Knowledge Base."}
        else:
             raise HTTPException(status_code=500, detail="Failed to write to Knowledge Base")
    else:
        # Reject
        orchestrator.redis_client.delete(key)
        return {"status": "rejected", "message": "Artifact discarded."}

# --- Celery Trigger Example ---
@app.post("/api/tasks/synthetic-data")
async def trigger_synthetic_data_generation(
    duration: int = 60,
    current_user: User = Depends(get_current_active_user)
):
    task = celery_app.send_task("tasks.generate_synthetic_data", args=[duration])
    return {"task_id": task.id, "status": "Queued"}

# --- Frontend Generator Bridge ---
from fastapi.responses import HTMLResponse
from cms.frontend_generator import frontend_generator

@app.get("/view/{view_type}", response_class=HTMLResponse)
async def serve_dynamic_view(view_type: str):
    """
    Multisite Switcher: Generates the requested layout View.
    Types: 'operator', 'knowledge', 'cortex'
    """
    # Combinatorics Engine
    html_content = frontend_generator.generate_layout(view_type)
    return html_content

@app.post("/api/sustainability/estimate")
async def estimate_sustainability(payload: Dict[str, Any]):
    """
    Calculate generic carbon footprint for a job description or G-Code
    """
    # ... (existing code)
    result = sustainability.calculate_footprint(gcode, material)
    
    cortex.transmit_intent(
        actor="SustainabilityEngine",
        action="CALCULATE_FOOTPRINT",
        reasoning="User requested environmental impact data",
        context=result
    )
    return result

@app.get("/api/cortex/stats")
async def get_cortex_stats(limit: int = 100):
    """
    Retrieve aggregated intent data for the Analytics Heatmap
    """
    intents = cortex.get_database_of_intent(limit=limit)
    return {
        "count": len(intents),
        "intents": intents
    }

@app.get("/api/swarm/status")
async def get_swarm_status(current_user: User = Depends(get_current_active_user)):
    """
    Fleet Oversight: Summarize status for all connected CNC units.
    Reads from Redis (Cortex) managed by the Fleet Simulator.
    """
    import random
    import json
    
    machines = []
    machine_ids = [f"CNC-{str(i).zfill(3)}" for i in range(1, 11)]
    
    if orchestrator.redis_client:
        for mid in machine_ids:
            # Try to get live data from Redis
            data = orchestrator.redis_client.hgetall(f"machine:{mid}")
            if data:
                # Convert numeric strings back to types
                machines.append({
                    "id": mid,
                    "status": data.get("status", "UNKNOWN"),
                    "load": float(data.get("load", 0)),
                    "current_job": "Job_" + mid.split('-')[1],
                    "health_score": int(data.get("health_score", 100))
                })
            else:
                # Fallback to random if no Redis data for this machine
                status = random.choice(["OPERATIONAL", "IDLE", "MAINTENANCE"])
                machines.append({
                    "id": mid, "status": status, "load": 0, "current_job": "NONE", "health_score": 100
                })
    
    return {
        "timestamp": asyncio.get_event_loop().time(),
        "fleet_size": len(machines),
        "active_units": len([m for m in machines if m['status'] == 'OPERATIONAL']),
        "machines": machines
    }

@app.get("/api/presets/proven")
async def get_proven_presets(current_user: User = Depends(get_current_active_user)):
    """
    Ranks and returns high-performing presets based on historical Success Scores.
    """
    if not orchestrator.redis_client:
        return {"presets": []}
    
    # Fetch top 10 job IDs from the sorted set
    # ZREVRANGE returns members with scores in descending order
    top_jobs = orchestrator.redis_client.zrevrange("cortex:proven_presets", 0, 9, withscores=True)
    
    presets = []
    for job_id, score in top_jobs:
        # For each job_id, we fetch the original Intent context which has the params
        # Search the intent database (list) for this job_id - in a real app, this would be a hash look up
        # For MVP, we simulate the return of the successful config
        presets.append({
            "job_id": job_id,
            "score": round(score, 2),
            "label": f"Elite Config {job_id[:6]}",
            "params": {
                "spindle_speed": random.randint(1200, 1800),
                "feed_rate": random.randint(100, 250),
                "depth_of_cut": random.uniform(0.5, 2.0)
            }
        })
        
    return {"presets": presets}

@app.post("/api/user/mode")
async def set_user_mode(mode: dict, current_user: User = Depends(get_current_active_user)):
    """
    Persists the user UI mode (Advanced vs Basic) in the Cortex.
    """
    if orchestrator.redis_client:
        orchestrator.redis_client.set(f"user:{current_user.username}:mode", "advanced" if mode.get('advanced') else "basic")
    return {"status": "SUCCESS", "mode": "advanced" if mode.get('advanced') else "basic"}

@app.post("/api/projects/branch")
async def branch_project(payload: dict, current_user: User = Depends(get_current_active_user)):
    """
    Creates an alternative build of a project in 'Preview' state.
    """
    from backend.core.project_brancher import project_brancher
    result = project_brancher.create_branch(
        parent_job_id=payload.get("parent_id"),
        branch_name=payload.get("branch_name"),
        overrides=payload.get("overrides")
    )
    if result["status"] == "ERROR":
        return {"status": "ERROR", "message": result["message"]}
    return result

@app.post("/api/upload/batch")
async def batch_upload_elements(files: List[UploadFile] = File(...), current_user: User = Depends(get_current_active_user)):
    """
    Handles multiple file uploads and registers them as Elements in the Template Engine.
    """
    from backend.core.element_template_engine import template_engine, Element
    import uuid
    
    results = []
    for file in files:
        element_id = f"EL-{uuid.uuid4().hex[:6]}"
        content = await file.read()
        
        # Try to parse as JSON if it's a JSON file
        if file.filename.endswith(".json"):
            try:
                content = json.loads(content)
            except:
                content = content.decode('utf-8', errors='ignore')
        else:
            content = content.decode('utf-8', errors='ignore')
            
        element = Element(
            id=element_id,
            type="CUSTOM_UPLOAD",
            content=content,
            metadata={"filename": file.filename, "size": len(content)}
        )
        template_engine.register_element(element)
        results.append({"id": element_id, "filename": file.filename})
        
    return {"status": "SUCCESS", "uploaded": results}

@app.post("/api/generate/custom")
async def generate_custom_document(payload: dict, current_user: User = Depends(get_current_active_user)):
    """
    Assembles a custom document from specified Element IDs.
    """
    from backend.core.element_template_engine import template_engine
    
    element_ids = payload.get("element_ids", [])
    format = payload.get("format", "GCODE")
    
    document = template_engine.assemble_document(element_ids, format=format)
    
    # Mirror to Cortex
    from backend.core.cortex_transmitter import cortex
    cortex.mirror_log("ElementTemplateEngine", f"Custom Document Assembled ({format})", "INFO")
    
    return {
        "status": "SUCCESS",
        "format": format,
        "content": document
    }

@app.get("/api/master/preferences")
async def get_master_preferences(current_user: User = Depends(get_current_active_user)):
    """
    Retrieves global master system preferences from Redis.
    """
    from backend.core.orchestrator import orchestrator
    prefs = orchestrator.redis_client.get("master:preferences")
    if prefs:
        return json.loads(prefs)
    return {
        "enginePower": 85,
        "lexicalDepth": "PREMIUM",
        "autoSeal": True,
        "learningTrack": "ADVANCED_MACHINIST",
        "latencyBias": "THROUGHPUT"
    }

@app.post("/api/master/preferences")
async def save_master_preferences(prefs: dict, current_user: User = Depends(get_current_active_user)):
    """
    Saves global master system preferences to Redis.
    """
    from backend.core.orchestrator import orchestrator
    orchestrator.redis_client.set("master:preferences", json.dumps(prefs))
    
    # Log to Cortex for Traceability
    from backend.core.cortex_transmitter import cortex
    cortex.mirror_log("SystemMaster", f"Master Preferences Synchronized by {current_user.username}", "CRITICAL")
    
    return {"status": "SUCCESS"}

@app.get("/api/business/stats")
async def get_business_stats(current_user: User = Depends(get_current_active_user)):
    """
    Returns aggregated business ROI and financial KPIs.
    """
    from backend.core.business_intelligence import bi_engine
    from backend.core.orchestrator import orchestrator
    
    # In a real system, we'd aggregate over all jobs in Redis/Postgres
    # For the overlay conspection, we provide a synthetic high-level view
    return {
        "status": "OPERATIONAL",
        "total_savings": 42920.50,
        "avg_roi": 32.4,
        "fleet_sustainability": 88.5, # % optimized
        "carbon_avoided": 1240.2, # kg
        "active_multiverse_branches": 12,
        "top_performing_part": "Engine Bracket V2 (BR-A882)"
    }

@app.get("/api/blender/assets")
async def get_blender_assets(current_user: User = Depends(get_current_active_user)):
    """
    Retrieves the curated library of 3D assets and templates.
    """
    from backend.core.blender_resource_hub import resource_hub
    return {"status": "SUCCESS", "assets": resource_hub.get_all_assets()}

@app.post("/api/blender/share")
async def share_blender_creation(payload: dict, current_user: User = Depends(get_current_active_user)):
    """
    Registers a creation in the Hall of Fame.
    """
    from backend.core.blender_resource_hub import resource_hub
    creation = resource_hub.share_creation(
        job_id=payload.get("job_id"),
        metadata=payload.get("metadata", {})
    )
    return {"status": "SUCCESS", "creation": creation}

@app.get("/api/blender/hall-of-fame")
async def get_hall_of_fame(current_user: User = Depends(get_current_active_user)):
    """
    Retrieves high-ROI builds from the network.
    """
    from backend.core.blender_resource_hub import resource_hub
    return {"status": "SUCCESS", "creations": resource_hub.get_hall_of_fame()}

@app.post("/api/generate/emotional")
async def generate_emotional_document(payload: dict, current_user: User = Depends(get_current_active_user)):
    """
    Generates G-Code with a specific emotional 'bias' or characteristic.
    """
    from backend.core.emotional_engine import emotional_engine
    from backend.core.element_template_engine import template_engine
    
    profile_name = payload.get("sentiment", "SERENE")
    element_ids = payload.get("element_ids", [])
    
    # 1. Assemble base document
    base_gcode = template_engine.assemble_document(element_ids, format="GCODE")
    
    # 2. Apply Emotional Bias
    lines = base_gcode.split('\n')
    emotional_gcode = "\n".join(emotional_engine.apply_bias(lines, profile_name))
    
    # 3. Create 'New Product' in system
    product = emotional_engine.create_emotional_product(
        base_asset_id=element_ids[0] if element_ids else "GENERIC",
        profile_name=profile_name
    )
    
    # log to cortex
    from backend.core.cortex_transmitter import cortex
    cortex.mirror_log("EmotionalEngine", f"Emotional Product {product['id']} generated with {profile_name} bias.", "INFO")
    
    return {
        "status": "SUCCESS",
        "product": product,
        "content": emotional_gcode
    }

@app.get("/api/platform/entities")
async def list_platform_entities(current_user: User = Depends(get_current_active_user)):
    """Lists all registered platform entities."""
    from backend.platform import platform_registry
    return {"status": "SUCCESS", "entities": [e.to_dict() for e in platform_registry.list_all()]}

@app.post("/api/platform/entity")
async def create_platform_entity(payload: dict, current_user: User = Depends(get_current_active_user)):
    """Creates and registers a new platform entity."""
    from backend.platform import PlatformEntity, platform_registry
    entity = PlatformEntity(
        entity_type=payload.get("entity_type", "GENERIC"),
        name=payload.get("name", "Untitled Entity"),
        metadata=payload.get("metadata", {})
    )
    platform_registry.register(entity)
    return {"status": "SUCCESS", "entity": entity.to_dict()}

@app.post("/api/platform/run-pipeline")
async def run_generation_pipeline(payload: dict, current_user: User = Depends(get_current_active_user)):
    """Runs the default generation pipeline on an entity."""
    from backend.platform import platform_registry, default_pipeline, EntityStatus
    entity_id = payload.get("entity_id")
    initial_data = payload.get("data", "")
    
    entity = platform_registry.get(entity_id)
    if not entity:
        return {"status": "ERROR", "message": "Entity not found"}
    
    result = default_pipeline.run(entity, initial_data)
    return {"status": "SUCCESS", "result": result}

@app.get("/api/products")
async def list_products(current_user: User = Depends(get_current_active_user)):
    """Lists all products in the registry."""
    from backend.llm import product_registry
    return {"status": "SUCCESS", "products": [p.to_dict() for p in product_registry.list_all()]}

@app.get("/api/products/search")
async def search_products(q: str = "", current_user: User = Depends(get_current_active_user)):
    """Searches products by name, category, or tags."""
    from backend.llm import product_registry
    results = product_registry.search(q) if q else product_registry.list_all()
    return {"status": "SUCCESS", "query": q, "results": [p.to_dict() for p in results]}

@app.post("/api/products")
async def create_product(payload: dict, current_user: User = Depends(get_current_active_user)):
    """Creates a new product."""
    from backend.llm import Product, product_registry
    product = Product(
        name=payload.get("name", "Untitled Product"),
        category=payload.get("category", "General"),
        dim_x=payload.get("dim_x", 100),
        dim_y=payload.get("dim_y", 50),
        dim_z=payload.get("dim_z", 25),
        tags=payload.get("tags", [])
    )
    product_registry.add(product)
    return {"status": "SUCCESS", "product": product.to_dict()}

@app.post("/api/llm/generate-payloads")
async def generate_multi_payloads(payload: dict, current_user: User = Depends(get_current_active_user)):
    """Generates multiple payloads for a product with LLM assistance."""
    from backend.llm import product_registry, llm_payload_generator, PayloadType
    
    product_id = payload.get("product_id")
    payload_types = payload.get("payload_types", [PayloadType.GCODE, PayloadType.JSON])
    llm_prompt = payload.get("llm_prompt", "")
    
    product = product_registry.get(product_id)
    if not product:
        return {"status": "ERROR", "message": "Product not found"}
    
    result = llm_payload_generator.generate_payloads(product.to_dict(), payload_types, llm_prompt)
    
    # Log to Cortex
    from backend.core.cortex_transmitter import cortex
    cortex.mirror_log("LLMPayloadGen", f"Multi-payload batch {result['batch_id']} generated for {product.name}", "INFO")
    
    return {"status": "SUCCESS", "result": result}

@app.get("/api/analytics/metrics")
async def get_analytics_metrics(current_user: User = Depends(get_current_active_user)):
    """Returns real-time system metrics."""
    from backend.analytics import metrics_engine
    return {"status": "SUCCESS", "metrics": metrics_engine.get_metrics()}

@app.get("/api/analytics/timeline")
async def get_analytics_timeline(limit: int = 20, current_user: User = Depends(get_current_active_user)):
    """Returns recent event timeline."""
    from backend.analytics import metrics_engine
    return {"status": "SUCCESS", "timeline": metrics_engine.get_timeline(limit)}

@app.get("/api/workflows")
async def list_workflows(current_user: User = Depends(get_current_active_user)):
    """Lists all registered workflows."""
    from backend.analytics import workflow_registry
    return {"status": "SUCCESS", "workflows": [w.to_dict() for w in workflow_registry.list_all()]}

@app.post("/api/workflow/run")
async def run_workflow(payload: dict, current_user: User = Depends(get_current_active_user)):
    """Executes a workflow with given context."""
    from backend.analytics import workflow_registry
    workflow_id = payload.get("workflow_id")
    context = payload.get("context", {})
    
    wf = workflow_registry.get(workflow_id)
    if not wf:
        return {"status": "ERROR", "message": "Workflow not found"}
    
    result = wf.execute(context)
    return {"status": "SUCCESS", "result": result}

@app.post("/api/export/project")
async def export_project(payload: dict, current_user: User = Depends(get_current_active_user)):
    """Exports a project as a portable ZIP package."""
    from backend.data import export_engine
    from backend.llm import product_registry
    import base64
    
    # Gather project data
    project_data = {
        "name": payload.get("project_name", "Untitled Project"),
        "products": [p.to_dict() for p in product_registry.list_all()],
        "metadata": payload.get("metadata", {})
    }
    
    result = export_engine.export_project(project_data)
    
    # Encode ZIP as base64 for JSON response
    result["data"] = base64.b64encode(result["data"]).decode('utf-8')
    
    return {"status": "SUCCESS", "export": result}

@app.get("/api/export/history")
async def get_export_history(current_user: User = Depends(get_current_active_user)):
    """Returns export history."""
    from backend.data import export_engine
    return {"status": "SUCCESS", "history": export_engine.get_export_history()}

@app.post("/api/import/project")
async def import_project(payload: dict, current_user: User = Depends(get_current_active_user)):
    """Imports a project from a base64-encoded ZIP package."""
    from backend.data import import_engine
    import base64
    
    zip_b64 = payload.get("data")
    if not zip_b64:
        return {"status": "ERROR", "message": "No data provided"}
    
    zip_data = base64.b64decode(zip_b64)
    result = import_engine.import_project(zip_data)
    
    return {"status": "SUCCESS", "import": result}

@app.get("/api/import/history")
async def get_import_history(current_user: User = Depends(get_current_active_user)):
    """Returns import history."""
    from backend.data import import_engine
    return {"status": "SUCCESS", "history": import_engine.get_import_history()}

@app.get("/api/notifications")
async def list_notifications(limit: int = 50, include_read: bool = True, current_user: User = Depends(get_current_active_user)):
    """Returns all notifications."""
    from backend.notifications import notification_engine
    return {
        "status": "SUCCESS",
        "notifications": notification_engine.get_all(limit, include_read),
        "unread_count": notification_engine.get_unread_count()
    }

@app.get("/api/notifications/unread-count")
async def get_unread_count(current_user: User = Depends(get_current_active_user)):
    """Returns count of unread notifications."""
    from backend.notifications import notification_engine
    return {"status": "SUCCESS", "unread_count": notification_engine.get_unread_count()}

@app.post("/api/notifications/read/{notification_id}")
async def mark_notification_read(notification_id: str, current_user: User = Depends(get_current_active_user)):
    """Marks a specific notification as read."""
    from backend.notifications import notification_engine
    success = notification_engine.mark_as_read(notification_id)
    return {"status": "SUCCESS" if success else "NOT_FOUND"}

@app.post("/api/notifications/read-all")
async def mark_all_notifications_read(current_user: User = Depends(get_current_active_user)):
    """Marks all notifications as read."""
    from backend.notifications import notification_engine
    count = notification_engine.mark_all_read()
    return {"status": "SUCCESS", "marked_count": count}

@app.post("/api/notifications/push")
async def push_notification(payload: dict, current_user: User = Depends(get_current_active_user)):
    """Pushes a new notification (internal/admin use)."""
    from backend.notifications import notification_engine, AlertPriority, AlertCategory
    
    priority = AlertPriority[payload.get("priority", "INFO")]
    category = AlertCategory[payload.get("category", "SYSTEM")]
    
    notif_id = notification_engine.push(
        title=payload.get("title", "Notification"),
        message=payload.get("message", ""),
        priority=priority,
        category=category
    )
    return {"status": "SUCCESS", "notification_id": notif_id}

@app.get("/api/config/params")
async def list_config_params(category: str = None, current_user: User = Depends(get_current_active_user)):
    """Lists all configuration parameters, optionally filtered by category."""
    from backend.config import parameter_registry
    if category:
        params = parameter_registry.list_by_category(category)
    else:
        params = parameter_registry.list_all()
    return {
        "status": "SUCCESS",
        "params": params,
        "categories": parameter_registry.get_categories()
    }

@app.get("/api/config/param/{key}")
async def get_config_param(key: str, current_user: User = Depends(get_current_active_user)):
    """Gets a specific configuration parameter."""
    from backend.config import parameter_registry
    param = parameter_registry.get(key)
    if not param:
        return {"status": "NOT_FOUND", "message": f"Parameter '{key}' not found"}
    return {"status": "SUCCESS", "param": param.to_dict()}

@app.post("/api/config/param/{key}")
async def set_config_param(key: str, payload: dict, current_user: User = Depends(get_current_active_user)):
    """Sets a configuration parameter value."""
    from backend.config import parameter_registry
    from backend.notifications import notification_engine, AlertPriority, AlertCategory
    
    value = payload.get("value")
    result = parameter_registry.set_value(key, value)
    
    if result.get("success"):
        # Push notification on config change
        notification_engine.push(
            f"Config Updated: {key}",
            f"Value changed from {result['old_value']} to {result['new_value']}",
            AlertPriority.INFO,
            AlertCategory.SYSTEM
        )
    
    return {"status": "SUCCESS" if result.get("success") else "ERROR", "result": result}

@app.post("/api/config/llm-query")
async def llm_config_query(payload: dict, current_user: User = Depends(get_current_active_user)):
    """Processes a natural language configuration query."""
    from backend.config import llm_config_agent
    
    query = payload.get("query", "")
    if not query:
        return {"status": "ERROR", "message": "No query provided"}
    
    result = llm_config_agent.query(query)
    return {"status": "SUCCESS", "result": result}

@app.get("/api/config/documentation")
async def get_config_documentation(current_user: User = Depends(get_current_active_user)):
    """Returns full LLM-readable documentation of all parameters."""
    from backend.config import llm_config_agent
    return {"status": "SUCCESS", "documentation": llm_config_agent.get_full_documentation()}

@app.get("/api/config/search")
async def search_config_params(q: str = "", current_user: User = Depends(get_current_active_user)):
    """Searches configuration parameters."""
    from backend.config import parameter_registry
    results = parameter_registry.search(q) if q else []
    return {"status": "SUCCESS", "query": q, "results": results}

@app.get("/api/security/status")
async def get_security_status(current_user: User = Depends(get_current_active_user)):
    """Returns current security status and isolation mode."""
    from backend.security import security_gateway, rate_limiter, resource_controller
    return {
        "status": "SUCCESS",
        "security": security_gateway.get_status(),
        "stats": security_gateway.get_stats(),
        "rate_limits": rate_limiter.get_limit_info(current_user.username),
        "resources": resource_controller.get_usage()
    }

@app.post("/api/security/mode")
async def set_security_mode(payload: dict, current_user: User = Depends(get_current_active_user)):
    """Sets the security isolation mode."""
    from backend.security import security_gateway, IsolationMode
    from backend.notifications import notification_engine, AlertPriority, AlertCategory
    
    mode_str = payload.get("mode", "ALLOWLIST").upper()
    try:
        mode = IsolationMode[mode_str]
        security_gateway.set_mode(mode)
        notification_engine.push(
            f"Security Mode Changed",
            f"Isolation mode set to {mode.value}",
            AlertPriority.WARNING,
            AlertCategory.SECURITY
        )
        return {"status": "SUCCESS", "mode": mode.value}
    except KeyError:
        return {"status": "ERROR", "message": f"Invalid mode: {mode_str}"}

@app.post("/api/security/allowlist")
async def manage_allowlist(payload: dict, current_user: User = Depends(get_current_active_user)):
    """Add or remove domains from the allowlist."""
    from backend.security import security_gateway
    
    action = payload.get("action", "add")
    domain = payload.get("domain", "")
    
    if not domain:
        return {"status": "ERROR", "message": "Domain required"}
    
    if action == "add":
        security_gateway.add_to_allowlist(domain)
    elif action == "remove":
        security_gateway.remove_from_allowlist(domain)
    else:
        return {"status": "ERROR", "message": "Invalid action"}
    
    return {"status": "SUCCESS", "allowlist": list(security_gateway.allowlist)}

@app.post("/api/security/blocklist")
async def manage_blocklist(payload: dict, current_user: User = Depends(get_current_active_user)):
    """Add domains to the blocklist."""
    from backend.security import security_gateway
    
    domain = payload.get("domain", "")
    if not domain:
        return {"status": "ERROR", "message": "Domain required"}
    
    security_gateway.add_to_blocklist(domain)
    return {"status": "SUCCESS", "blocklist": list(security_gateway.blocklist)}

@app.get("/api/security/check")
async def check_connection(target: str = "", current_user: User = Depends(get_current_active_user)):
    """Checks if a connection to target would be allowed."""
    from backend.security import security_gateway
    
    if not target:
        return {"status": "ERROR", "message": "Target required"}
    
    result = security_gateway.is_allowed(target)
    return {"status": "SUCCESS", "target": target, "result": result}

@app.get("/api/debug/sysinfo")
async def get_system_info(current_user: User = Depends(get_current_active_user)):
    """Returns comprehensive system information (Geek Mode)."""
    from backend.debug import system_introspector
    return {
        "status": "SUCCESS",
        "system": system_introspector.get_system_info(),
        "memory": system_introspector.get_memory_info(),
        "threads": system_introspector.get_thread_info()
    }

@app.post("/api/debug/console")
async def execute_debug_command(payload: dict, current_user: User = Depends(get_current_active_user)):
    """Executes a debug console command (Geek Mode)."""
    from backend.debug import debug_console
    
    command = payload.get("command", "")
    if not command:
        return {"status": "ERROR", "message": "No command provided"}
    
    result = debug_console.execute(command)
    return {"status": "SUCCESS", "result": result}

@app.get("/api/debug/endpoints")
async def list_all_endpoints(current_user: User = Depends(get_current_active_user)):
    """Lists all registered API endpoints (Geek Mode)."""
    from backend.debug import system_introspector
    return {"status": "SUCCESS", "endpoints": system_introspector.get_api_endpoints()}

@app.get("/api/debug/modules")
async def list_modules(prefix: str = "backend", current_user: User = Depends(get_current_active_user)):
    """Lists loaded Python modules (Geek Mode)."""
    from backend.debug import system_introspector
    return {"status": "SUCCESS", "modules": system_introspector.get_loaded_modules(prefix)}

@app.get("/api/debug/performance")
async def get_performance_stats(current_user: User = Depends(get_current_active_user)):
    """Returns API performance statistics (Geek Mode)."""
    from backend.debug import performance_profiler
    return {"status": "SUCCESS", "performance": performance_profiler.get_all_stats()}

@app.get("/api/debug/slowest")
async def get_slowest_endpoints(top_n: int = 10, current_user: User = Depends(get_current_active_user)):
    """Returns slowest endpoints by response time (Geek Mode)."""
    from backend.debug import performance_profiler
    return {"status": "SUCCESS", "slowest": performance_profiler.get_slowest_endpoints(top_n)}

@app.post("/api/debug/gc")
async def force_garbage_collection(current_user: User = Depends(get_current_active_user)):
    """Forces garbage collection (Geek Mode)."""
    from backend.debug import system_introspector
    return {"status": "SUCCESS", "gc": system_introspector.run_garbage_collection()}

@app.post("/api/developer/snippet")
async def generate_code_snippet(payload: dict, current_user: User = Depends(get_current_active_user)):
    """Generates code snippet for API call (DEVELOPERS!)."""
    from backend.developer import api_client_generator
    
    language = payload.get("language", "python")
    endpoint = payload.get("endpoint", "/api/health")
    method = payload.get("method", "GET")
    request_payload = payload.get("payload")
    
    snippet = api_client_generator.generate_snippet(language, endpoint, method, request_payload)
    return {"status": "SUCCESS", "language": language, "snippet": snippet}

@app.get("/api/developer/sdk")
async def get_full_sdk(current_user: User = Depends(get_current_active_user)):
    """Returns full Python SDK template (DEVELOPERS!)."""
    from backend.developer import api_client_generator
    return {"status": "SUCCESS", "sdk": api_client_generator.get_full_sdk_template()}

@app.post("/api/developer/scaffold/component")
async def scaffold_component(payload: dict, current_user: User = Depends(get_current_active_user)):
    """Generates React component scaffold (DEVELOPERS!)."""
    from backend.developer import code_scaffolder
    
    name = payload.get("name", "NewComponent")
    return {"status": "SUCCESS", "component": code_scaffolder.generate_component(name)}

@app.post("/api/developer/scaffold/endpoint")
async def scaffold_endpoint(payload: dict, current_user: User = Depends(get_current_active_user)):
    """Generates FastAPI endpoint scaffold (DEVELOPERS!)."""
    from backend.developer import code_scaffolder
    
    name = payload.get("name", "new_endpoint")
    method = payload.get("method", "GET")
    return {"status": "SUCCESS", "endpoint": code_scaffolder.generate_api_endpoint(name, method)}

@app.post("/api/developer/scaffold/model")
async def scaffold_model(payload: dict, current_user: User = Depends(get_current_active_user)):
    """Generates Python model class scaffold (DEVELOPERS!)."""
    from backend.developer import code_scaffolder
    
    name = payload.get("name", "NewModel")
    fields = payload.get("fields", ["id", "name"])
    return {"status": "SUCCESS", "model": code_scaffolder.generate_model(name, fields)}

@app.get("/api/developer/quickstart")
async def get_quickstart_guide(current_user: User = Depends(get_current_active_user)):
    """Returns developer quickstart guide (DEVELOPERS!)."""
    from backend.developer import docs_generator
    return {"status": "SUCCESS", "quickstart": docs_generator.generate_quickstart()}

@app.get("/api/developer/changelog")
async def get_changelog(current_user: User = Depends(get_current_active_user)):
    """Returns project changelog (DEVELOPERS!)."""
    from backend.developer import docs_generator
    return {"status": "SUCCESS", "changelog": docs_generator.generate_changelog()}

@app.get("/api/developer/languages")
async def get_supported_languages(current_user: User = Depends(get_current_active_user)):
    """Returns supported SDK languages."""
    from backend.developer import api_client_generator
    return {"status": "SUCCESS", "languages": api_client_generator.supported_languages}

@app.get("/api/possibilities")
async def get_all_possibilities(current_user: User = Depends(get_current_active_user)):
    """Returns all future possibilities."""
    from backend.possibilities import possibility_engine
    return {"status": "SUCCESS", "possibilities": possibility_engine.get_all()}

@app.get("/api/possibilities/quick-wins")
async def get_quick_wins(current_user: User = Depends(get_current_active_user)):
    """Returns high-impact, low-effort possibilities."""
    from backend.possibilities import possibility_engine
    return {"status": "SUCCESS", "quick_wins": possibility_engine.get_quick_wins()}

@app.get("/api/possibilities/recommended")
async def get_recommended(count: int = 5, current_user: User = Depends(get_current_active_user)):
    """Returns top recommended next features."""
    from backend.possibilities import possibility_engine
    return {"status": "SUCCESS", "recommended": possibility_engine.get_recommended_next(count)}

@app.get("/api/possibilities/roadmap")
async def get_roadmap(current_user: User = Depends(get_current_active_user)):
    """Returns phased implementation roadmap."""
    from backend.possibilities import possibility_engine
    return {"status": "SUCCESS", "roadmap": possibility_engine.get_roadmap()}

@app.get("/api/possibilities/random")
async def get_random_idea(current_user: User = Depends(get_current_active_user)):
    """Generates a random feature idea for inspiration."""
    from backend.possibilities import possibility_engine
    return {"status": "SUCCESS", "idea": possibility_engine.generate_random_idea()}

@app.get("/api/possibilities/category/{category}")
async def get_by_category(category: str, current_user: User = Depends(get_current_active_user)):
    """Returns possibilities filtered by category."""
    from backend.possibilities import possibility_engine
    return {"status": "SUCCESS", "category": category, "possibilities": possibility_engine.get_by_category(category)}

@app.get("/api/provenance/{job_id}")
async def fetch_document_origin(job_id: str, current_user: User = Depends(get_current_active_user)):
    """
    Fetches the cryptographic provenance (origin) of a manufacturing document.
    """
    from backend.core.document_origin_manager import provenance_manager
    origin = provenance_manager.fetch_by_job_id(job_id)
    if not origin:
        return {"status": "ERROR", "message": "No provenance data found for this ID."}
    return {"status": "SUCCESS", "provenance": origin}

@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """
    Retrieve status of an Async FlowEngine Job
    """
    # Connect to Redis (Shared Control Pad)
    try:
        if not orchestrator.redis_client:
            return {"status": "ERROR", "message": "Redis not connected"}
            
        status = orchestrator.redis_client.hgetall(f"job:{job_id}")
        if not status:
            return {"status": "NOT_FOUND", "message": "Job ID not found"}
            
        # Get result if completed
        result = None
        if status.get("status") == "COMPLETED":
            res_json = orchestrator.redis_client.get(f"result:{job_id}")
            if res_json:
                result = json.loads(res_json)
                
        return {
            "job_id": job_id,
            "status": status,
            "result": result
        }
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}

@app.post("/conduct")
async def conduct_protocol(request: Dict[str, str], current_user: User = Depends(get_current_active_user)):
    """
    Creative UI Endpoint: Conducts a protocol based on user prompt.
    """
    try:
        from cms.protocol_conductor import ProtocolConductor
        conductor = ProtocolConductor()
        name = request.get("name", "Unnamed_Protocol")
        prompt = request.get("prompt", "")
        return conductor.conduct_scenario(name, prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize")
async def optimize_gcode(gcode: List[str], material: str = "Steel4140", current_user: User = Depends(get_current_active_user)):
    """
    Runs the CNC-VINO Optimizer on raw G-Code.
    """
    try:
        from cms.cnc_vino_optimizer import CNCOptimizer
        optimizer = CNCOptimizer()
        ir = optimizer.optimize_model(gcode, material)
        return {"optimized_ir": ir}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Intelligence Endpoints ---

@app.post("/api/intelligence/ask")
async def intelligence_ask(request: Dict[str, Any]):
    """Natural language manufacturing assistant"""
    question = request.get("question", "")
    if not question:
        raise HTTPException(status_code=400, detail="No question provided")
    
    answer = ai_assistant.chat(question)
    return {"answer": answer, "timestamp": datetime.now().isoformat()}

@app.get("/api/intelligence/insights")
async def intelligence_insights(days: int = 7):
    """Generate manufacturing insights"""
    return intelligence_engine.generate_insights_report(days=days)

@app.get("/api/intelligence/stats")
async def intelligence_stats():
    """Get cross-session statistics"""
    return {
        "total_data_points": len(intelligence_engine.data_repository),
        "unique_sessions": len(set(dp.session_id for dp in intelligence_engine.data_repository)),
        "data_types": list(set(dp.data_type for dp in intelligence_engine.data_repository)),
        "timestamp": datetime.now().isoformat()
    }

from cms.vision_cortex import vision_cortex

@app.post("/api/intelligence/predict")
async def intelligence_predict(request: Dict[str, Any], current_user: User = Depends(get_current_active_user)):
    """Predict future events based on indicators"""
    event_type = request.get("event_type", "anomaly")
    indicators = request.get("current_indicators", {})
    return intelligence_engine.predict_future_event(event_type, indicators)

# --- Phase 6: Refining Intelligence Endpoints ---

@app.get("/api/vision/inspect")
async def vision_inspection(current_user: User = Depends(get_current_active_user)):
    """
    Triggers the Visual Cortex to inspect the current workpiece.
    Returns: Quality Assessment, Defects, and Confidence Score.
    """
    try:
        # In a real scenario, we might pass an image ID or stream URL
        result = vision_cortex.inspect_part()
        
        # Log to Cortex
        cortex.transmit_intent(
            actor="VisionCortex",
            action="INSPECTION",
            reasoning=f"Automated QC Check. Result: {'PASS' if result.passed else 'FAIL'}",
            context=result.__dict__
        )
        
        return {"status": "SUCCESS", "inspection": result.__dict__}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/learning/feedback")
async def learning_feedback(payload: Dict[str, str], current_user: User = Depends(get_current_active_user)):
    """
    Reinforcement Learning Hook.
    Feeds the outcome of a job back into the Dopamine Engine to adjust personality weights.
    Payload: { "outcome": "SUCCESS" | "FAILURE" | "QUALITY_ISSUE" }
    """
    outcome = payload.get("outcome", "SUCCESS")
    
    # 1. Update Dopamine Engine (The 'Self')
    new_weights = dopamine_brain.learn_from_outcome(outcome)
    
    # 2. Log Learning Event
    cortex.mirror_log("DopamineEngine", f"Weights updated via RL: {new_weights}", "LEARNING")
    
    return {
        "status": "UPDATED", 
        "message": "Neural weights adjusted based on feedback.", 
        "current_weights": new_weights,
        "neuro_state": vars(dopamine_brain.state)
    }

from backend.core.marketplace import marketplace_service
from backend.core.database import get_db
from sqlalchemy.orm import Session

# --- Phase 7: Ecosystem / Marketplace Endpoints ---

@app.get("/api/marketplace/components")
async def list_marketplace_components(category: str = None, db: Session = Depends(get_db)):
    """Returns all shared components in the hive ecosystem."""
    components = marketplace_service.list_components(db, category)
    return {
        "count": len(components),
        "components": [
            {
                "id": c.id,
                "name": c.name,
                "category": c.category,
                "description": c.description,
                "author": c.author.username if c.author else "HiveMind",
                "downloads": c.downloads,
                "rating": c.rating_sum / c.rating_count if c.rating_count > 0 else 0,
                "version": c.version
            } for c in components
        ]
    }

@app.post("/api/marketplace/share")
async def share_to_marketplace(
    payload: Dict[str, Any], 
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Shares a local asset (G-Code, Material, Config) to the global marketplace."""
    # Find user ID from username
    from backend.database.models import User as DBUser
    user = db.query(DBUser).filter(DBUser.username == current_user.username).first()
    
    component = marketplace_service.share_component(db, user.id, payload)
    return {"status": "SUCCESS", "component_id": component.id}

@app.post("/api/marketplace/download/{component_id}")
async def download_marketplace_component(
    component_id: int, 
    db: Session = Depends(get_db)
):
    """Increments download count and returns the payload."""
    component = marketplace_service.download_component(db, component_id)
    if not component:
        raise HTTPException(status_code=404, detail="Component not found")
    return {"status": "SUCCESS", "payload": component.payload}

@app.post("/api/marketplace/rate/{component_id}")
async def rate_marketplace_component(
    component_id: int, 
    score: float,
    db: Session = Depends(get_db)
):
    """Rates a component (0.0 to 5.0)."""
    if score < 0 or score > 5:
        raise HTTPException(status_code=400, detail="Rating must be between 0 and 5")
    marketplace_service.rate_component(db, component_id, score)
    return {"status": "SUCCESS"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
