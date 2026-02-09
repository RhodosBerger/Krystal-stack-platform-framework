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
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from enum import Enum

class CoherenceState(Enum):
    INVALID = 0
    SHARED = 1
    EXCLUSIVE = 2
    MODIFIED = 3

# Ensure we can import from root 'cms'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Configuring Logging
logger = logging.getLogger("MasterOrchestrator")
logger.setLevel(logging.INFO)

# Import Enterprise Intelligence Modules
CMS_AVAILABLE = False
try:
    logger.info(f"System Path: {sys.path}")
    from backend.cms.llm_gcode_generator import LLMGCodeGenerator
    from backend.cms.multi_bot_system import BotCoordinator
    from backend.cms.producer_effectiveness_engine import PartOptimizationBot
    
    # Shadow Council Imports
    from backend.cms.message_bus import global_bus
    from backend.core.council import council
    from backend.cms.interaction_supervisor import InteractionSupervisor
    from backend.cms.translation_adapter import TranslationAdapter, SaaSMetric
    from backend.cms.cross_session_intelligence import CrossSessionIntelligence
    from backend.cms.intervention_agent import InterventionAgent
    from backend.cms.economic_engine import EconomicEngine
    from backend.cms.swarm_brain import SwarmBrain
    from backend.cms.twin_engine import TwinEngine
    
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
from backend.core.marketplace import marketplace_service
from backend.core.database import SessionLocal

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
            "shadow_council": "OFFLINE",
            "scalpel_active": "OFFLINE",
            "swarm_brain": "OFFLINE",
            "digital_twin": "OFFLINE"
        }
        
        self.llm_router = LLMRouter()
        self.lexicon = Lexicon(version="PREMIUM")
        self.active_jobs = {}
        self.machine_registry = {} # Phase 9: Global Registry
        self.cms_available = False
        
        # Phase 9: Global Swarm Registry
        self.swarm_state = {} # Stores the latest telemetry for each machine
        self.active_marketplace_map = {} # { job_id: component_id } for Anti-Fragile rewards
        
        try:
            from backend.agent.manufacturing import ManufacturingAgent
            # Use the engines already initialized in LLMRouter
            self.agent = ManufacturingAgent(self.llm_router.engine, self.llm_router.processor)
            logger.info("🤖 Manufacturing Agent (Cognitive) Initialized.")
        except ImportError as e:
            logger.error(f"Failed to load ManufacturingAgent: {e}")
            self.agent = None

        # Initialize Redis for Shared Control Pad
        try:
            logger.info("🔌 Connecting to Redis (Shared Memory)...")
            import redis
            # Connect to the Shared Memory System
            # use a short timeout to avoid hanging if redis is down
            self.redis_client = redis.Redis(
                host='127.0.0.1', 
                port=6379, 
                db=0, 
                decode_responses=True,
                socket_connect_timeout=0.5
            )
            # self.redis_client.ping() # DISABLE ACTIVE PING TO PREVENT HANGS ON WINDOWS
            self.systems["shared_memory"] = "ONLINE"
            logger.info("✅ Shared Control Pad (Redis) Client Initialized (Lazy)")
            cortex.mirror_log("MasterOrchestrator", "Shared Control Pad Connected", "INFO")
        except Exception as e:
            logger.error(f"❌ Shared Control Pad Failed (Running in Degraded Mode): {e}")
            self.systems["shared_memory"] = "OFFLINE"
            self.redis_client = None

        # Phase 12: MESI Coherence Registry
        self.coherence_map = {} # machine_id -> (CoherenceState, owner_agent)

        # Initialize Enterprise Engines
        if CMS_AVAILABLE:
            try:
                self.gcode_engine = LLMGCodeGenerator()
                self.bot_coordinator = BotCoordinator()
                self.optimizer_engine = PartOptimizationBot()
                
                # Initialize Shadow Council
                self.supervisor = InteractionSupervisor()
                self.scalpel = InterventionAgent()
                self.swarm_logic = SwarmBrain(self)
                self.twin = TwinEngine()
                
                from backend.core.robotics_agent import robotics
                self.robotics = robotics
                
                # Phase 15: Cognitive Repair
                from backend.core.cognitive_repair_agent import repair_agent
                self.repair_agent = repair_agent
                
                # Phase 17: Provenance & Auditing
                from backend.core.provenance_agent import provenance
                self.provenance = provenance
                
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
                self.systems["scalpel_active"] = "ONLINE"
                self.systems["swarm_brain"] = "ONLINE"
                self.systems["digital_twin"] = "ONLINE"
                self.systems["robotics_agent"] = "ONLINE"
                self.systems["cognitive_repair"] = "ONLINE"
                self.systems["provenance_audit"] = "ONLINE"
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
            
            # Start The Scalpel (Theory 4: Adaptive Interventions)
            asyncio.create_task(self.scalpel.start())
            logger.info("🔪 The Scalpel (Adaptive Intervention) Activated")
            
            # Start Stability Listener (Theory 5: Anti-Fragile Rewards)
            global_bus.subscribe("STABILITY_SCORE_UPDATE", self._stability_listener)
            logger.info("🏆 Anti-Fragile Reward Listener (Marketplace) Activated")
            
            # Start Swarm Brain (Phase 9: Recruitment)
            asyncio.create_task(self.swarm_logic.start())
            logger.info("🐝 Swarm Brain (Coordination) Activated")
            
            # Start Vision Cortex (Phase 8: Optical Grounding)
            from backend.cms.vision_cortex import vision_cortex
            asyncio.create_task(vision_cortex.start())
            logger.info("👁️ Vision Cortex (Optical Grounding) Activated")
            
            # Start The Truth Engine (Theory 2: Time Travel)
            asyncio.create_task(self._truth_engine_loop())
        
        # Start Swarm Simulation
        asyncio.create_task(self._swarm_simulation_loop())
        
        # Verify DB Connection
        # Verify DB Connection (SKIPPED TO PREVENT HANGS)
        try:
            # from backend.core.database import engine
            # with engine.connect() as conn:
            #     logger.info("✅ Database (PostgreSQL) Connected")
            self.systems["database"] = "SKIPPED"
        except Exception as e:
            logger.error(f"❌ Database Connection Failed: {e}")
            self.systems["database"] = "OFFLINE"
            
        self.status = "OPERATIONAL"
        logger.info("✅ System Operational")

    async def _swarm_simulation_loop(self):
        """Simulates a living factory floor with autonomous agents."""
        import random
        
        # Initialize some mock machines if empty
        if not self.machine_registry:
            for i in range(1, 13):
                mid = f"CNC-VMC-{i:02d}"
                self.update_machine_status(mid, {
                    "status": "ONLINE", 
                    "load": random.uniform(20, 80),
                    "action": "IDLE"
                })
        
        while True:
            try:
                for mid, data in self.machine_registry.items():
                    # 1. Fluctuate Telemetry
                    current_load = data.get("load", 0)
                    new_load = max(0, min(100, current_load + random.uniform(-5, 5)))
                    
                    current_rpm = data.get("rpm", 0)
                    target_rpm = 8000 if data["status"] == "ONLINE" else 0
                    new_rpm = current_rpm + (target_rpm - current_rpm) * 0.1 + random.uniform(-50, 50)
                    
                    # 2. Random State Transitions
                    if random.random() < 0.05: # 5% chance to change state
                        states = ["ONLINE", "IDLE", "MAINTENANCE", "ERROR"]
                        weights = [0.7, 0.2, 0.08, 0.02]
                        new_status = random.choices(states, weights=weights)[0]
                        data["status"] = new_status
                        data["action"] = f"TRANSITION_TO_{new_status}"
                    
                    # 3. Update Registry (MESI Coherent)
                    self.update_machine_status(mid, {
                        "status": data["status"],
                        "load": new_load,
                        "rpm": int(new_rpm),
                        "vibration": random.uniform(0.01, 0.15) * (new_load / 50),
                        "action": data.get("action", "MONITORING"),
                        "neuro_state": {
                            "dopamine": random.uniform(40, 90),
                            "cortisol": random.uniform(10, 60)
                        }
                    }, agent_id="SWARM_SIM")
                
                await asyncio.sleep(2) # Update every 2 seconds
            except Exception as e:
                logger.error(f"Swarm Sim Error: {e}")
                await asyncio.sleep(5)

    async def _truth_engine_loop(self):
        """
        Theory 2: The Truth Engine (LLM Time Travel).
        Periodically correlactes cross-session data to find 'Black Swan' insights.
        """
        from backend.main import intelligence_engine
        logger.info("🕵️ Truth Engine (Time Travel) Active.")
        
        while True:
            await asyncio.sleep(30) # Correlate every 30 seconds
            try:
                logger.info("[TRUTH_ENGINE] Correlating historical telemetry...")
                connections = intelligence_engine.connect_unrelated_events(time_window_hours=24)
                
                if connections:
                    for conn in connections:
                        msg = f"[TIME_TRAVEL] Discovered correlation: {conn['connections'][0]['reasoning']}"
                        logger.info(msg)
                        cortex.transmit_intent(
                            actor="TruthEngine",
                            action="DISCOVER_CORRELATION",
                            reasoning=msg,
                            context=conn
                        )
            except Exception as e:
                logger.error(f"Truth Engine Error: {e}")

    async def _stability_listener(self, msg):
        """
        Theory 5: Anti-Fragile Feedback Loop.
        Rewards marketplace components that survive high-vibration/stress.
        """
        payload = msg.payload
        score = payload.get("score", 0)
        stress = payload.get("stress_factor", 0)
        
        # If we are stable (score > 0.7) and under stress (stress > 0.3)
        if score > 0.7 and stress > 0.3:
            # Check if any active job is using a marketplace component
            db = SessionLocal()
            try:
                for job_id, component_id in self.active_marketplace_map.items():
                    # Reward the survivor
                    marketplace_service.record_success(db, component_id, stress)
            finally:
                db.close()

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
            logger.info(f"[{job_id}] Requesting Audit from Shadow Council...")
            
            # Construct a Draft Plan for the Auditor
            draft_plan = {
                "job_id": job_id,
                "action": "MILLING",
                "material": material,
                "rpm": payload.get("rpm", 8000), 
                "feed": payload.get("feed", 1000)
            }
            
            # Publish Intent
            await global_bus.publish("DRAFT_PLAN", draft_plan, sender_id="ORCHESTRATOR")
            
            # Wait for Verdict
            audit_result = await self._wait_for_audit(job_id)
            
            if audit_result.get("status") == "FAIL":
                error_msg = f"Audit Rejected Plan: {audit_result.get('errors')}"
                logger.warning(error_msg)
                cortex.mirror_log("Auditor", error_msg, "SECURITY_BLOCK")
                return {
                    "status": "BLOCKED",
                    "job_id": job_id,
                    "message": "Safety Audit Failed - Shadow Council Veto",
                    "errors": audit_result.get("errors")
                }
            elif audit_result.get("status") == "WARNING":
                logger.warning(f"Audit Warning: {audit_result.get('warnings')}")
            else:
                logger.info(f"[{job_id}] Audit Passed. Proceeding.")
        
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
        
        # 1. Theory 6: The Great Translation (Pre-Optimization)
        if CMS_AVAILABLE:
            strategy = constraints.get("strategy", "Balanced")
            adapted_strategy = TranslationAdapter.laptop_mode_to_machining_strategy(strategy)
            
            # --- PHASE 6: EMERALD LOGIC (PROFIT OPTIMIZATION) ---
            # Suggest mode based on Profit Rate curves
            suggested_mode = EconomicEngine.optimize_mode(payload.get("economic_req", {}), {})
            logger.info(f"[EMERALD] Economic Logic suggests: {suggested_mode}")
            
            logger.info(f"[THEORY_6] Mapping UI Strategy '{strategy}' -> Internal Strategy '{adapted_strategy}'")
            constraints["internal_strategy"] = adapted_strategy
            constraints["economic_mode"] = suggested_mode
            
            # Simulated SaaS Churn Translation
            mock_churn = 0.05 # 5% Churn
            translation = TranslationAdapter.software_to_machining(SaaSMetric.CHURN_RATE, mock_churn)
            logger.info(f"[THEORY_6] SaaS Churn ({mock_churn*100}%) -> {translation['physical_mapping']} Adjustment: {translation['adjustment']}x")
            constraints["wear_bias"] = translation["adjustment"]

        # 2. Update Shared Control Pad (Redis) - Initial State
        if self.redis_client:
            self.redis_client.hset(f"job:{job_id}", mapping={
                "type": "OPTIMIZE_PROCESS",
                "status": "QUEUED",
                "description": description,
                "created_at": datetime.now().isoformat(),
                "theory_6_translation": "ACTIVE"
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
            # Brain fallback -> Cognitive Agent
            if self.agent:
                response = self.agent.run_task(f"Consultation: {question}")
                return {"status": "success", "data": {"response": response, "source": "ManufacturingAgent"}}
            else:
                # Basic Router Fallback
                response = self.llm_router.query(
                    system_prompt="You are a Manufacturing Consultant.",
                    user_prompt=question
                )
                return {"status": "success", "data": {"response": response, "source": "LLM Fallback"}}

    async def _handle_market_analysis(self, payload: Dict) -> Dict:
        return {"analysis": "Market data module not yet connected."}

    async def _handle_geometry_analysis(self, payload: Dict) -> Dict:
        """
        Route to Neural Voxelizer (Phase 9) and Twin Engine (Phase 10)
        """
        try:
            from backend.core.geometry.voxelizer import voxelizer
            
            # 1. Neural Voxelization
            voxel_result = voxelizer.process_geometry_intent(payload)
            
            # 2. Twin Engine Feasibility (TFSM)
            features = payload.get("features", [])
            twin_result = self.twin.check_tfsm_feasibility({}, features)
            
            # 3. Apply Theory 4: Quadratic Mantinel (Smoothing)
            curvature_mean = voxel_result["graphs"]["curvature"]["mean"]
            adaptive_fro = voxelizer.graph_engine.calculate_adaptive_feed_override(curvature_mean)
            
            return {
                "status": "COMPLETED", 
                "voxel_data": voxel_result,
                "twin_simulation": twin_result,
                "theory_4_smoothing": {
                    "suggested_fro": adaptive_fro,
                    "reason": f"Geometric curvature ({curvature_mean:.3f}) requires FRO limitation."
                }
            }
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

    def get_emerald_metrics(self) -> Dict[str, Any]:
        """
        Returns Phase 6 economic metrics for the Manager Persona.
        """
        # Simulated metrics based on current swarm state
        m1_load = sum(m["load"] for m in self.machine_registry.values()) / (len(self.machine_registry) or 1)
        m2_load = m1_load * 0.85 # Slight unbalance mock
        
        balance = EconomicEngine.calculate_spindle_balance(m1_load, m2_load)
        profit_rate = EconomicEngine.calculate_profit_rate(500.0, 200.0 + (m1_load * 0.5), 60.0)
        
        return {
            "profit_rate_min": profit_rate,
            "spindle_balance": balance,
            "fleet_gravity": sum(m.get("load", 0) for m in self.machine_registry.values()) / 100.0,
            "high_churn_alerts": [
                {"id": "SCR-99", "name": "Turbo_Finisher_v2.gcode", "churn_rate": 0.85, "reason": "Excessive Tool Attrition (Theory 6)"}
            ],
            "economic_tier": "PREMIUM"
        }

    def get_system_status(self):
        return {
            "global_status": self.status,
            "systems": self.systems,
            "timestamp": datetime.now().isoformat()
        }

    # --- Swarm Intelligence Methods ---

    def update_machine_status(self, machine_id: str, data: Dict[str, Any], agent_id: str = "SYSTEM"):
        """
        Updates the status of a specific machine.
        Enforces MESI Coherence (Theory 1: ActionGate).
        """
        state, owner = self.coherence_map.get(machine_id, (CoherenceState.INVALID, None))

        # Check if agent has right to write (Exclusive or Modified)
        if owner and owner != agent_id and state in [CoherenceState.EXCLUSIVE, CoherenceState.MODIFIED]:
            logger.warning(f"❌ MESI Collision: Agent '{agent_id}' tried to update machine '{machine_id}' owned by '{owner}'")
            return False

        self.machine_registry[machine_id] = {
            "status": data.get("status", "ONLINE"),
            "load": data.get("load", 0.0),
            "rpm": data.get("rpm", 0),
            "vibration": data.get("vibration", 0.0),
            "activity": data.get("action", "MONITORING"),
            "neuro": data.get("neuro_state", {}),
            "last_seen": datetime.now().isoformat(),
            "last_agent": agent_id
        }
        
        # Transition to MODIFIED state
        self.coherence_map[machine_id] = (CoherenceState.MODIFIED, agent_id)

        # Phase 14: Check for Robotic Handshake opportunity
        if data.get("status") == "CYCLE_STOP":
            self.robotics.calculate_cobot_handshake(machine_id, "CYCLE_STOP")

        return True

    def acquire_machine_ownership(self, machine_id: str, agent_id: str) -> bool:
        """KrystalStack Protocol: Request Exclusive access to a machine."""
        state, owner = self.coherence_map.get(machine_id, (CoherenceState.INVALID, None))
        
        if owner and owner != agent_id and state != CoherenceState.INVALID:
            return False # Already owned or shared
            
        self.coherence_map[machine_id] = (CoherenceState.EXCLUSIVE, agent_id)
        cortex.mirror_log("Orchestrator", f"MESI: Agent '{agent_id}' acquired EXCLUSIVE access to '{machine_id}'", "INFO")
        return True
        
    def get_swarm_status(self):
        """Returns the full factory-wide swarm status"""
        return {
            "node_count": len(self.machine_registry),
            "machines": self.machine_registry,
            "timestamp": datetime.now().isoformat()
        }

    async def execute_repair_cycle(self, strategy: Dict[str, Any], machine_id: str):
        """
        Executes autonomous re-work actions.
        Tactic: Conclusion Leveling (Reflex -> Deep Thought).
        """
        logger.info(f"🛠️ [COGNITIVE_REPAIR] Executing Level {strategy['conclusion_level']} strategy for {machine_id}")
        
        for action in strategy.get("actions", []):
            if action["action"] == "HARM_SHIFT":
                # Apply micro-FRO adjustment via the Scalpel
                logger.info(f"Applying Harmonic Shift: {action['adjustment']} FRO")
                # (Scalpel integration call here)
                
            elif action["action"] == "TRIGGER_ROBOTIC_DEBURR":
                # Trigger sub-task to RoboticsAgent
                logger.info("Transferring part to Robotics for Deburring.")
                params = action.get("params", {})
                self.robotics.generate_deburring_path(0.5, {"length": 150, "height": 20})
                
        cortex.mirror_log("Orchestrator", f"Repair cycle complete for {machine_id}.", "SUCCESS")

# Global Instance
orchestrator = MasterOrchestrator()
