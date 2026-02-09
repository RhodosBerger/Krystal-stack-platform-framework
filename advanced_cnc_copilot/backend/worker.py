"""
Celery Worker Configuration
"""
from celery import Celery
import sys
import os
import json
import redis
import time
from datetime import datetime
# Ensure we can import from root 'cms'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Configuration
BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

# Redis Client for Shared Control Pad
try:
    redis_client = redis.Redis.from_url(
        RESULT_BACKEND, 
        decode_responses=True, 
        socket_connect_timeout=1
    )
    redis_client.ping()
except Exception:
    print("Warning: Redis unavailable. Worker running in limited mode.")
    redis_client = None

celery_app = Celery(
    "rise_worker",
    broker=BROKER_URL,
    backend=RESULT_BACKEND
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

# --- Tasks ---

@celery_app.task(name="tasks.generate_synthetic_data")
def generate_synthetic_data_task(duration_seconds: int = 60):
    """
    Background task to generate synthetic sensor data.
    Simulates heavy computation.
    """
    print(f"Starting synthetic data generation for {duration_seconds}s...")
    time.sleep(5) # Simulating work
    return {"status": "completed", "samples_generated": duration_seconds * 10}

@celery_app.task(name="tasks.validate_solidworks_design")
def validate_solidworks_design_task(file_path: str):
    """
    Simulates checking a design against SolidWorks API (via COM/Interop)
    """
    print(f"Validating design: {file_path}")
    time.sleep(3)
    return {"valid": True, "fea_result": "Safety Factor 2.5"}

@celery_app.task(name="tasks.generate_gcode")
def generate_gcode_task(description: str, material: str, job_id: str):
    """
    Async FlowEngine Task: G-Code Generation
    """
    # Lazy Import inside task context
    from backend.core.cortex_transmitter import cortex
    
    print(f"[{job_id}] Processing G-Code Request: {description}")
    cortex.mirror_log("Worker", f"Start Processing Job {job_id}", "INFO")
    
    try:
        from backend.cms.llm_gcode_generator import LLMGCodeGenerator
        engine = LLMGCodeGenerator()
        
        full_desc = f"{description} in {material}"
        program, validation = engine.generate_from_description(full_desc, validate=True)
        
        result = {
            "program_name": program.program_name,
            "gcode": program.to_string(),
            "estimated_time": program.estimated_time_minutes,
            "validation": validation,
            "engine_used": "CMS LLMGCodeGenerator (Async Worker)"
        }
        
        # Update Shared Control Pad (Redis)
        if redis_client:
            redis_client.hset(f"job:{job_id}", mapping={
                "status": "COMPLETED",
                "result_summary": "G-Code Generated (Async)",
                "completed_at": datetime.now().isoformat()
            })
            redis_client.setex(f"result:{job_id}", 3600, json.dumps(result))
        
        cortex.mirror_log("Worker", f"Completed Job {job_id} successfully", "INFO")
        return result
        
    except Exception as e:
        print(f"[{job_id}] Error: {e}")
        cortex.mirror_log("Worker", f"Job {job_id} Failed: {e}", "ERROR")
        # Update Failure State
        if redis_client:
            redis_client.hset(f"job:{job_id}", mapping={
                "status": "FAILED",
                "error_msg": str(e)
            })
        raise

@celery_app.task(name="tasks.optimize_process")
def optimize_process_task(description: str, constraints: dict, job_id: str):
    """
    Async FlowEngine Task: Process Optimization
    """
    from backend.core.cortex_transmitter import cortex
    cortex.mirror_log("Worker", f"Start Optimization Job {job_id}", "INFO")
    
    try:
        from backend.cms.producer_effectiveness_engine import PartOptimizationBot
        optimizer = PartOptimizationBot()
        
        result = optimizer.optimize_complete_project(description, constraints)
        
        # Phase 17: Immutable Audit Signing
        from backend.core.provenance_agent import provenance
        signature = provenance.sign_optimization(
            job_id=job_id,
            original_hash="PRE_AUTH_HASH_001", # Placeholder for original G-Code hash
            modified_hash=provenance.generate_gcode_hash([str(result)]),
            reasoning=f"Autonomous Optimization via {optimizer.__class__.__name__}",
            actor="CORTEX_WORKER"
        )
        result["provenance_signature"] = signature

        # Update Redis
        if redis_client:
            redis_client.hset(f"job:{job_id}", mapping={
                "status": "COMPLETED",
                "result_summary": "Optimization Complete & Signed",
                "provenance_sig": signature,
                "completed_at": datetime.now().isoformat()
            })
            redis_client.setex(f"result:{job_id}", 3600, json.dumps(result))
        
        return result
        
    except Exception as e:
        cortex.mirror_log("Worker", f"Optimization Job {job_id} Failed: {e}", "ERROR")
        if redis_client:
            redis_client.hset(f"job:{job_id}", mapping={"status": "FAILED", "error_msg": str(e)})
        raise

@celery_app.task(name="tasks.check_industry_blogs")
def check_industry_blogs_task():
    """
    Crawler Task: Ping relevant CNC blogs for new presets/knowledge.
    (Factor 1: Find & Hold for Verification)
    """
    # Lazy Import
    from backend.core.cortex_transmitter import cortex
    
    print("Checking industry blogs (Machinery's Handbook, CNC Cookbook)...")
    
    # Mock finding a new article -> Transform to Preset Candidate
    verification_id = f"VERIFY-{int(time.time())}"
    new_preset_candidate = {
        "id": verification_id,
        "name": "SuperAlloy-Inconel718", # Extracted from title
        "category": "SuperAlloy",
        "machinability_rating": 0.2,
        "source": "CNC Cookbook",
        "timestamp": datetime.now().isoformat(),
        "operations": {
             "milling": { "cutting_speed_sfm": 50, "notes": "From Blog: High Speed Machining" }
        }
    }
    
    # Store in Pending State (Redis)
    # Expiration 24h
    if redis_client:
        redis_client.setex(f"pending_knowledge:{verification_id}", 86400, json.dumps(new_preset_candidate))
    
    # Notify Cortex
    cortex.mirror_log("Crawler", f"Found Candidate matching 'Inconel'. Held for 2FA: {verification_id}", "WARNING")
    
    print(f"Found new article. Held for Verification: {verification_id}")
    return {"status": "pending_verification", "verification_id": verification_id, "data": new_preset_candidate}

@celery_app.task(name="tasks.generate_propedeutics")
def generate_propedeutics_task(topic: str):
    """
    Knowledge Task: Generate educational content using KnowledgeEngine
    """
    try:
        from backend.cms.knowledge_engine import knowledge_engine
        content = knowledge_engine.generate_propedeutics(topic)
        return {"status": "completed", "topic": topic, "content_snippet": content[:100] + "..."}
    except Exception as e:
        return {"status": "error", "message": str(e)}