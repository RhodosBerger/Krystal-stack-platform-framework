"""
Master Orchestrator for FANUC RISE (Enterprise Edition)
Coordinates Intelligence, Economic Engine, and CNC Control.
Integrates specialized CMS engines for high-capability performance.
"""
import asyncio
import logging
import json
import sys
import os
from typing import Dict, Any, List, Optional
from datetime import datetime

# Ensure we can import from root 'cms'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Configuring Logging
logger = logging.getLogger("MasterOrchestrator")
logger.setLevel(logging.INFO)

# Import Enterprise Intelligence Modules
CMS_AVAILABLE = False
try:
    logger.info(f"System Path: {sys.path}")
    from cms.llm_gcode_generator import LLMGCodeGenerator
    from cms.multi_bot_system import BotCoordinator
    from cms.producer_effectiveness_engine import PartOptimizationBot
    
    # Shadow Council Imports
    from cms.message_bus import global_bus
    from cms.interaction_supervisor import InteractionSupervisor
    
    CMS_AVAILABLE = True
except Exception as e:
    import traceback
    logger.error(f"Failed to import CMS modules: {e}")
    logger.error(traceback.format_exc())
    CMS_AVAILABLE = False

# Import Core Systems
from backend.core.llm_brain import LLMRouter
from backend.core.cortex_transmitter import cortex
from backend.core.manufacturing_lexicon import Lexicon, GCodeCompiler
from backend.core.document_origin_manager import provenance_manager
from backend.core.project_brancher import project_brancher

class MasterOrchestrator:
    """
    The Central Nervous System of the FANUC RISE implementation.
    Coordinates between:
    - User API Requests
    - Hybrid LLM Brain (OpenAI/Ollama)
    - Specialized CMS Engines (G-Code, Optimization, Consulting)
    - Celery Workers
    """
    
    def __init__(self):
        self.status = "INITIALIZING"
        self.systems = {
            "llm": "ONLINE",
            "database": "ONLINE",
            "iot_simulator": "STANDBY",
            "cnc_controller": "READY",
            "cms_engines": "LOADING",
            "shared_memory": "OFFLINE",
            "semantic_lexicon": "READY",
            "shadow_council": "OFFLINE"
        }
        
        self.llm_router = LLMRouter()
        self.lexicon = Lexicon(version="PREMIUM")
        self.active_jobs = {}
        self.cms_available = False
        
        # Phase 9: Global Swarm Registry
        self.machine_registry = {}  # { machine_id: { status: str, load: float, activity: str, last_seen: datetime } }
        
        # Initialize Redis for Shared Control Pad
        try:
            import redis
            # Connect to the Shared Memory System
            self.redis_client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)
            self.redis_client.ping()
            self.systems["shared_memory"] = "ONLINE"
            logger.info("✅ Shared Control Pad (Redis) Connected")
            cortex.mirror_log("MasterOrchestrator", "Shared Control Pad Connected", "INFO")
        except Exception as e:
            logger.error(f"❌ Shared Control Pad Failed: {e}")
            self.systems["shared_memory"] = "OFFLINE"
            self.redis_client = None

        # Initialize Enterprise Engines
        if CMS_AVAILABLE:
            try:
                self.gcode_engine = LLMGCodeGenerator()
                self.bot_coordinator = BotCoordinator()
                self.optimizer_engine = PartOptimizationBot()
                
                # Initialize Shadow Council
                self.supervisor = InteractionSupervisor()
                
                # Import New Evolutionary Modules (Phases 9-16)
                from backend.core.geometry.voxelizer import voxelizer
                from backend.core.simulation_agent import physicist
                from backend.core.generative_logic import runtime as gen_runtime
                from backend.core.evolution_runtime import EvolutionRuntime
                
                # Initialize The Lifecycle Manager
                self.evolution_runtime = EvolutionRuntime(backend_refs={
                    "voxelizer": voxelizer,
                    "physicist": physicist,
                    "builder": gen_runtime
                })
                
                self.systems["cms_engines"] = "ONLINE"
                self.systems["evolution_runtime"] = "ONLINE"
                self.systems["shadow_council"] = "ONLINE"
                self.cms_available = True
                
                logger.info("✅ CMS Enterprise Engines & Evolution Runtime Loaded")
                cortex.mirror_log("MasterOrchestrator", "Evolution Runtime Activated", "INFO")
            except Exception as e:
                logger.error(f"❌ Error initializing CMS Engines: {e}")
                cortex.mirror_log("MasterOrchestrator", f"CMS Engines Failed: {e}", "ERROR")
                self.systems["cms_engines"] = "FAILED"
                self.cms_available = False
        else:
            self.systems["cms_engines"] = "MISSING"
            logger.warning("⚠️ CMS Modules not found. Running in degraded mode.")

    async def initialize(self):
        """Startup sequence"""
        logger.info("🚀 FlowEngine (Master Orchestrator) Starting...")
        
        # Start Shadow Council Supervisor
        if CMS_AVAILABLE and hasattr(self, 'supervisor'):
            asyncio.create_task(self.supervisor.start())
            logger.info("👁️ Shadow Council (Supervisor) Activated")
        
        # Verify DB Connection
        try:
            from backend.core.database import engine
            with engine.connect() as conn:
                logger.info("✅ Database (PostgreSQL) Connected")
                self.systems["database"] = "ONLINE"
        except Exception as e:
            logger.error(f"❌ Database Connection Failed: {e}")
            self.systems["database"] = "OFFLINE"
            
        self.status = "OPERATIONAL"
        logger.info("✅ System Operational")

    async def _wait_for_audit(self, job_id: str, timeout: int = 5) -> Dict[str, Any]:
        """
        Listens for a VALIDATION_RESULT matching the job_id.
        """
        future = asyncio.get_event_loop().create_future()

        async def _audit_listener(msg):
            if msg.payload.get("original_plan_id") == job_id:
                if not future.done():
                    future.set_result(msg.payload)

        # Subscribe
        global_bus.subscribe("VALIDATION_RESULT", _audit_listener)
        
        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Audit timed out for {job_id}")
            return {"status": "TIMEOUT", "errors": ["Audit service unavailable"]}
        finally:
            # Cleanup listener if possible, but basic bus might not support unsubscribe easily
            # For this prototype, we rely on the check inside the listener
            pass

    async def process_request(self, request_type: str, payload: Dict[str, Any], user: str) -> Dict[str, Any]:
        """
        FlowEngine Pipeline:
        1. DataCreator Input -> 2. PrecisionAnchor Logic -> 3. Shared Memory Update -> 4. SpecificOutput
        """
        logger.info(f"FlowEngine: Processing {request_type} from {user}")
        
        # Transmit Intent
        cortex.transmit_intent(
            actor=f"DataCreator:{user}",
            action=request_type,
            reasoning="User Request via API",
            context=payload
        )
        
        try:
            if request_type == "GENERATE_GCODE":
                return await self._handle_gcode_generation(payload, user)
            elif request_type == "GENERATE_SEMANTIC_PROGRAM":
                return await self._handle_semantic_program(payload)
            elif request_type == "BRANCH_PROJECT":
                return await self._handle_branching(payload)
            elif request_type == "OPTIMIZE_PROCESS":
                return await self._handle_optimization(payload)
            elif request_type == "ANALYZE_GEOMETRY":
                return await self._handle_geometry_analysis(payload)
            elif request_type == "CONSULTATION" or request_type == "CONSULT_BOT":
                return await self._handle_consultation(payload)
            elif request_type == "ANALYZE_MARKET":
                return await self._handle_market_analysis(payload)
            else:
                return {"status": "ERROR", "message": f"Unknown request type: {request_type}"}
        except Exception as e:
            logger.error(f"FlowEngine Error: {e}", exc_info=True)
            cortex.mirror_log("MasterOrchestrator", f"FlowEngine Error: {e}", "ERROR")
            return {"status": "ERROR", "message": str(e)}

    async def _handle_gcode_generation(self, payload: Dict, user_id: str) -> Dict:
        """
        FlowEngine Route: G-Code Generation (Async)
        Includes SHADOW COUNCIL AUDIT.
        """
        description = payload.get("description", "")
        material = payload.get("material", "Aluminum")
        job_id = f"JOB-{int(datetime.now().timestamp())}"
        
        # --- SHADOW COUNCIL: AUDIT PHASE ---
        if CMS_AVAILABLE and hasattr(self, 'supervisor'):
            logger.info(f"[{job_id}] Requesting Audit...")
            
            # Construct a Draft Plan for the Auditor (Proposing standard parameters)
            # In a real system, the Creator (LLM) would generate this first.
            # Here we simulate the proposal based on defaults to check material safety.
            draft_plan = {
                "job_id": job_id,
                "action": "MILLING",
                "material": material,
                "rpm": payload.get("rpm", 8000), # Default if not specified
                "feed": payload.get("feed", 1000)
            }
            
            await global_bus.publish("DRAFT_PLAN", draft_plan, sender_id="ORCHESTRATOR")
            
            audit_result = await self._wait_for_audit(job_id)
            
            if audit_result.get("status") == "FAIL":
                error_msg = f"Audit Rejected Plan: {audit_result.get('errors')}"
                logger.warning(error_msg)
                cortex.mirror_log("Auditor", error_msg, "SECURITY_BLOCK")
                return {
                    "status": "BLOCKED",
                    "job_id": job_id,
                    "message": "Safety Audit Failed",
                    "errors": audit_result.get("errors")
                }
            elif audit_result.get("status") == "WARNING":
                logger.warning(f"Audit Warning: {audit_result.get('warnings')}")
        
        # 1. Update Shared Control Pad (Redis) - Initial State
        if self.redis_client:
            self.redis_client.hset(f"job:{job_id}", mapping={
                "type": "GENERATE_GCODE",
                "status": "QUEUED",
                "user": user_id,
                "description": description,
                "material": material,
                "created_at": datetime.now().isoformat()
            })

        # 2. Dispatch to Async Worker (FlowEngine)
        try:
            # Lazy import to avoid circular dependency
            from backend.worker import celery_app
            
            task = celery_app.send_task(
                "tasks.generate_gcode",
                args=[description, material, job_id]
            )
            
            logger.info(f"🚀 Dispatched Job {job_id} to Worker (Task ID: {task.id})")
            cortex.mirror_log("MasterOrchestrator", f"Dispatched Job {job_id} to Worker", "INFO")
            
            return {
                "status": "QUEUED", 
                "job_id": job_id, 
                "task_id": task.id,
                "message": "Job queued for processing. Poll /api/jobs/{job_id} for results."
            }
            
        except Exception as e:
            logger.error(f"Failed to dispatch worker task: {e}")
            cortex.mirror_log("MasterOrchestrator", f"Dispatch Failed: {e}", "ERROR")
            
            # Fallback to Synchronous/Local Execution if Worker fails
            if self.cms_available:
                logger.info("⚠️ Worker dispatch failed. Falling back to local CMS Engine...")
                loop = asyncio.get_event_loop()
                full_desc = f"{description} in {material}"
                
                program, validation = await loop.run_in_executor(
                    None, 
                    lambda: self.gcode_engine.generate_from_description(full_desc, validate=True)
                )
                
                result = {
                    "program_name": program.program_name,
                    "gcode": program.to_string(),
                    "estimated_time": program.estimated_time_minutes,
                    "validation": validation,
                    "engine_used": "CMS LLMGCodeGenerator (Synchronous Fallback)"
                }
                
                # Update Redis since we finished locally
                if self.redis_client:
                    self.redis_client.hset(f"job:{job_id}", mapping={
                        "status": "COMPLETED",
                        "result_summary": "G-Code Generated (Sync Fallback)"
                    })
                    self.redis_client.setex(f"result:{job_id}", 3600, json.dumps(result))
                    
                return {"status": "COMPLETED", "job_id": job_id, "data": result}
            else:
                 return {"status": "ERROR", "message": "Async dispatch failed and local CMS unavailable."}

    async def _handle_optimization(self, payload: Dict) -> Dict:
        """
        Route to ProducerEffectivenessEngine (CMS) via FlowEngine (Async)
        """
        description = payload.get("description", "Standard part")
        constraints = payload.get("constraints", {})
        job_id = f"JOB-OPT-{int(datetime.now().timestamp())}"
        
        # 1. Update Shared Control Pad (Redis) - Initial State
        if self.redis_client:
            self.redis_client.hset(f"job:{job_id}", mapping={
                "type": "OPTIMIZE_PROCESS",
                "status": "QUEUED",
                "description": description,
                "created_at": datetime.now().isoformat()
            })

        # 2. Dispatch to Async Worker
        try:
            from backend.worker import celery_app
            
            task = celery_app.send_task(
                "tasks.optimize_process",
                args=[description, constraints, job_id]
            )
            
            logger.info(f"🚀 Dispatched Optimization {job_id} to Worker")
            cortex.mirror_log("MasterOrchestrator", f"Dispatched Optimization {job_id}", "INFO")
            
            return {
                "status": "QUEUED",
                "job_id": job_id,
                "task_id": task.id,
                "message": "Optimization queued. Poll /api/jobs/{job_id} for results."
            }
            
        except Exception as e:
            logger.error(f"Failed to dispatch optimization: {e}")
            cortex.mirror_log("MasterOrchestrator", f"Opt Dispatch Failed: {e}", "ERROR")
            return {"status": "ERROR", "message": str(e)}
        else:
            return {"status": "error", "message": "Optimization Engine Unavailable"}

    async def _handle_consultation(self, payload: Dict) -> Dict:
        """
        Route to MultiBotSystem (CMS)
        """
        question = payload.get("question", "")
        
        if CMS_AVAILABLE and self.systems["cms_engines"] == "ONLINE":
            logger.info("🧠 Using CMS BotCoordinator...")
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.bot_coordinator.route_question(question)
            )
            return {"status": "success", "data": result}
        else:
            # Brain fallback
            response = self.llm_router.query(
                 system_prompt="You are a Manufacturing Consultant.",
                 user_prompt=question
            )
            return {"status": "success", "data": {"response": response, "source": "LLM Fallback"}}

    async def _handle_market_analysis(self, payload: Dict) -> Dict:
        return {"analysis": "Market data module not yet connected."}

    async def _handle_geometry_analysis(self, payload: Dict) -> Dict:
        """
        Route to Neural Voxelizer (Phase 9)
        """
        try:
            from backend.core.geometry.voxelizer import voxelizer
            result = voxelizer.process_geometry_intent(payload)
            return {"status": "COMPLETED", "data": result}
        except Exception as e:
            logger.error(f"Geometry Analysis Failed: {e}")
            return {"status": "ERROR", "message": str(e)}

    async def _handle_semantic_program(self, payload: Dict) -> Dict:
        """
        Translates a list of Lexicon Intent Primitives into a G-Code program.
        """
        intents = payload.get("intents", [])
        job_id = payload.get("job_id", f"JOB-SEM-{int(datetime.now().timestamp())}")
        
        # --- SEQUENCE CHI: DOCUMENT PROVENANCE ---
        origin_uid = provenance_manager.create_origin_tag(
            intent_payload={**payload, "job_id": job_id},
            evolution_id=payload.get("evolution_id", "GEN_1")
        )
        
        lexicon_calls = []
        for item in intents:
            name = item.get("primitive")
            params = item.get("params", {})
            
            if hasattr(self.lexicon, name):
                method = getattr(self.lexicon, name)
                lexicon_calls.append(method(**params))
            else:
                logger.warning(f"Lexicon Primitive Not Found: {name}")
                
        # Pass the Origin UID to the Compiler
        compiled_gcode = GCodeCompiler.compile(lexicon_calls, origin_uid=origin_uid)
        
        return {
            "status": "COMPLETED",
            "job_id": job_id,
            "origin_uid": origin_uid,
            "gcode": compiled_gcode,
            "lexicon_version": self.lexicon.version,
            "primitives_used": len(lexicon_calls)
        }

    async def _handle_branching(self, payload: Dict) -> Dict:
        """
        FlowEngine Route: Create an alternative build (branch).
        """
        parent_id = payload.get("parent_id")
        branch_name = payload.get("branch_name", "Alternative Build")
        overrides = payload.get("overrides", {})
        
        result = project_brancher.create_branch(parent_id, branch_name, overrides)
        
        if result["status"] == "SUCCESS":
            cortex.mirror_log("MasterOrchestrator", f"Project Branched: {branch_name} from {parent_id}", "INFO")
            
        return result

    def get_system_status(self):
        return {
            "global_status": self.status,
            "systems": self.systems,
            "timestamp": datetime.now().isoformat()
        }

    # --- Swarm Intelligence Methods ---

    def update_machine_status(self, machine_id: str, data: Dict[str, Any]):
        """Updates the status of a specific machine in the swarm registry"""
        self.machine_registry[machine_id] = {
            "status": data.get("status", "ONLINE"),
            "load": data.get("load", 0.0),
            "rpm": data.get("rpm", 0),
            "vibration": data.get("vibration", 0.0),
            "activity": data.get("action", "MONITORING"),
            "neuro": data.get("neuro_state", {}),
            "last_seen": datetime.now().isoformat()
        }
        
    def get_swarm_status(self):
        """Returns the full factory-wide swarm status"""
        return {
            "node_count": len(self.machine_registry),
            "machines": self.machine_registry,
            "timestamp": datetime.now().isoformat()
        }

# Global Instance
orchestrator = MasterOrchestrator()
