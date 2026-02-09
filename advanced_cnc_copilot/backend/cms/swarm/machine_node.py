"""
Swarm Intelligence - Machine Node
Individual machine node that wraps the Shadow Council to interact with the Hive Mind
"""

import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import hashlib
import threading
import logging

from ..services.shadow_council import ShadowCouncil
from ..services.dopamine_engine import DopamineEngine
from ..services.economics_engine import EconomicsEngine
from .hive_mind import HiveMind, TraumaSignature, TraumaSeverity, SurvivorBadge


logger = logging.getLogger(__name__)


@dataclass
class JobParameters:
    """Parameters for a specific job to be processed by the machine node"""
    material: str
    operation_type: str
    parameters: Dict[str, float]
    gcode_signature: str
    job_id: str
    estimated_duration: float  # in hours
    priority: int = 1  # 1-5, with 5 being highest priority


class MachineNode:
    """
    Machine Node - Individual machine in the swarm that interacts with the Hive Mind.
    
    Responsibilities:
    - Check Hive for existing traumas before executing operations
    - Run jobs through local Shadow Council
    - Report failures to Hive as trauma signatures
    - Query Hive for high-scoring strategies (Survivor Badges)
    - Maintain local state synchronized with Hive
    """
    
    def __init__(self, machine_id: str, hive: HiveMind, shadow_council: ShadowCouncil):
        self.machine_id = machine_id
        self.hive = hive
        self.shadow_council = shadow_council
        self.local_trauma_cache = {}  # Cache of recently checked traumas
        self.local_badge_cache = {}  # Cache of available survivor badges
        self.active_jobs = {}  # Currently running jobs
        self.completed_jobs = []  # History of completed jobs
        self.status = "idle"  # idle, running, error, maintenance
        self.heartbeat_interval = 30  # seconds
        self._stop_heartbeat = threading.Event()
        
        # Start heartbeat thread
        self._heartbeat_thread = threading.Thread(target=self._send_heartbeat, daemon=True)
        self._heartbeat_thread.start()
        
        print(f"[NODE-{self.machine_id}] Initialized and connected to Hive")
    
    def _send_heartbeat(self):
        """Send periodic heartbeat to Hive to indicate machine is alive"""
        while not self._stop_heartbeat.wait(self.heartbeat_interval):
            status_info = {
                'status': self.status,
                'active_jobs': len(self.active_jobs),
                'last_heartbeat': datetime.utcnow().isoformat(),
                'machine_id': self.machine_id
            }
            self.hive.update_machine_status(self.machine_id, status_info)
    
    def check_hive_for_traumas(self, job_params: JobParameters) -> Optional[TraumaSignature]:
        """
        Check the Hive for any previously recorded traumas for this operation.
        
        Args:
            job_params: Parameters for the job to check
            
        Returns:
            TraumaSignature if dangerous operation found, None otherwise
        """
        # First check local cache
        cache_key = self._generate_cache_key(job_params)
        if cache_key in self.local_trauma_cache:
            cached_time, cached_trauma = self.local_trauma_cache[cache_key]
            # Cache for 5 minutes
            if (datetime.utcnow() - cached_time).seconds < 300:
                return cached_trauma
        
        # Check Hive for existing trauma
        trauma = self.hive.check_for_existing_trauma(
            material=job_params.material,
            operation_type=job_params.operation_type,
            parameters=job_params.parameters,
            gcode_signature=job_params.gcode_signature
        )
        
        # Update local cache
        self.local_trauma_cache[cache_key] = (datetime.utcnow(), trauma)
        
        if trauma:
            print(f"[NODE-{self.machine_id}] WARNING: Operation matches known trauma: "
                  f"{trauma.failure_reason} (Severity: {trauma.severity.value})")
        
        return trauma
    
    def get_survivor_strategies(self, material: str, operation_type: str) -> List[SurvivorBadge]:
        """
        Get high-scoring survivor strategies from the Hive for this material/operation.
        
        Args:
            material: Material type
            operation_type: Type of operation
            
        Returns:
            List of high-scoring survivor badges
        """
        # Check local cache first
        cache_key = f"{material}:{operation_type}"
        if cache_key in self.local_badge_cache:
            cached_time, cached_badges = self.local_badge_cache[cache_key]
            # Cache for 2 minutes
            if (datetime.utcnow() - cached_time).seconds < 120:
                return cached_badges
        
        # Get from Hive
        badges = self.hive.get_high_anti_fragile_strategies(material, operation_type)
        
        # Update local cache
        self.local_badge_cache[cache_key] = (datetime.utcnow(), badges)
        
        return badges
    
    def run_job(self, job_params: JobParameters) -> Dict[str, Any]:
        """
        Run a job through the machine node, checking Hive and local Shadow Council.
        
        Args:
            job_params: Parameters for the job to run
            
        Returns:
            Dictionary with job results and status
        """
        print(f"[NODE-{self.machine_id}] Starting job {job_params.job_id}")
        self.status = "running"
        
        try:
            # 1. Check Hive for existing traumas
            trauma = self.check_hive_for_traumas(job_params)
            if trauma:
                # Operation is known to be dangerous, abort immediately
                result = {
                    'job_id': job_params.job_id,
                    'status': 'aborted_known_trauma',
                    'reason': f'Known trauma: {trauma.failure_reason}',
                    'severity': trauma.severity.value,
                    'timestamp': datetime.utcnow().isoformat(),
                    'machine_id': self.machine_id
                }
                
                print(f"[NODE-{self.machine_id}] ABORTED job {job_params.job_id} due to known trauma")
                return result
            
            # 2. Check for high-scoring strategies in Hive
            survivor_strategies = self.get_survivor_strategies(
                job_params.material, 
                job_params.operation_type
            )
            
            if survivor_strategies:
                print(f"[NODE-{self.machine_id}] Found {len(survivor_strategies)} survivor strategies for "
                      f"{job_params.material} {job_params.operation_type}")
                
                # Use the highest scoring strategy if parameters are compatible
                best_strategy = survivor_strategies[0]
                print(f"[NODE-{self.machine_id}] Using survivor strategy: {best_strategy.strategy_id} "
                      f"(Score: {best_strategy.anti_fragile_score:.3f})")
            
            # 3. Run job through local Shadow Council
            print(f"[NODE-{self.machine_id}] Submitting job to local Shadow Council")
            
            # Create a state representation for the Shadow Council
            current_state = {
                'material': job_params.material,
                'operation_type': job_params.operation_type,
                'estimated_duration': job_params.estimated_duration,
                'priority': job_params.priority,
                'machine_id': self.machine_id,
                **job_params.parameters  # Unpack all parameters
            }
            
            # Run the Shadow Council evaluation
            council_decision = self.shadow_council.evaluate_strategy(current_state, int(self.machine_id.replace('M', '')))
            
            if not council_decision['council_approval']:
                # Local Shadow Council rejected the job
                result = {
                    'job_id': job_params.job_id,
                    'status': 'rejected_by_local_council',
                    'reason': 'Local Shadow Council vetoed operation',
                    'council_feedback': council_decision.get('reasoning_trace', []),
                    'timestamp': datetime.utcnow().isoformat(),
                    'machine_id': self.machine_id
                }
                
                print(f"[NODE-{self.machine_id}] LOCAL COUNCIL rejected job {job_params.job_id}")
                return result
            
            # 4. Simulate job execution (in real system, this would be actual CNC operation)
            print(f"[NODE-{self.machine_id}] Executing job {job_params.job_id} - Simulating operation")
            
            # Simulate the operation - sometimes we'll simulate a failure for testing
            operation_success = self._simulate_operation(job_params)
            
            if operation_success:
                # Job completed successfully
                result = {
                    'job_id': job_params.job_id,
                    'status': 'completed_successfully',
                    'council_decision': council_decision,
                    'timestamp': datetime.utcnow().isoformat(),
                    'machine_id': self.machine_id,
                    'duration_actual': job_params.estimated_duration * 0.95  # Simulated actual duration
                }
                
                # Award a survivor badge for successful completion
                self._award_survivor_badge(job_params, council_decision)
                
                print(f"[NODE-{self.machine_id}] COMPLETED job {job_params.job_id} successfully")
            else:
                # Simulated failure - report to Hive as trauma
                result = {
                    'job_id': job_params.job_id,
                    'status': 'failed_simulation',
                    'reason': 'Simulated failure during operation',
                    'council_decision': council_decision,
                    'timestamp': datetime.utcnow().isoformat(),
                    'machine_id': self.machine_id
                }
                
                # Create trauma signature for the failure
                trauma_sig = TraumaSignature(
                    material=job_params.material,
                    operation_type=job_params.operation_type,
                    parameters=job_params.parameters,
                    failure_reason='simulated_operation_failure',
                    severity=TraumaSeverity.HIGH,  # Simulated severity
                    timestamp=datetime.utcnow(),
                    machine_id=self.machine_id,
                    gcode_signature=job_params.gcode_signature
                )
                
                # Register trauma with Hive
                self.hive.register_trauma(trauma_sig)
                
                print(f"[NODE-{self.machine_id}] SIMULATED FAILURE reported for job {job_params.job_id}")
        
        except Exception as e:
            # Handle any exceptions during job execution
            result = {
                'job_id': job_params.job_id,
                'status': 'error',
                'reason': str(e),
                'timestamp': datetime.utcnow().isoformat(),
                'machine_id': self.machine_id
            }
            
            print(f"[NODE-{self.machine_id}] ERROR in job {job_params.job_id}: {str(e)}")
        
        finally:
            self.status = "idle"
        
        return result
    
    def _simulate_operation(self, job_params: JobParameters) -> bool:
        """
        Simulate the actual operation to determine success/failure.
        In a real system, this would be the actual CNC operation.
        
        Args:
            job_params: Parameters for the job to simulate
            
        Returns:
            True if operation succeeds, False if it fails
        """
        # Simulate based on parameters - higher feed rates and aggressive parameters
        # have higher chance of failure
        feed_rate = job_params.parameters.get('feed_rate', 1000)
        rpm = job_params.parameters.get('rpm', 2000)
        depth = job_params.parameters.get('depth', 1.0)
        
        # Calculate risk factor based on parameters
        risk_factor = 0.0
        if feed_rate > 2500:
            risk_factor += 0.3
        if rpm > 8000:
            risk_factor += 0.2
        if depth > 3.0:
            risk_factor += 0.4
        if job_params.material.lower() == 'titanium':
            risk_factor += 0.3
        elif job_params.material.lower() == 'inconel':
            risk_factor += 0.5
        
        # Simulate success/failure based on risk factor
        import random
        success_probability = max(0.1, 1.0 - risk_factor)  # At least 10% chance of success
        
        return random.random() < success_probability
    
    def _award_survivor_badge(self, job_params: JobParameters, council_decision: Dict[str, Any]):
        """
        Award a survivor badge for successfully completed job.
        
        Args:
            job_params: The parameters of the successful job
            council_decision: The council decision for the job
        """
        # Calculate anti-fragile score based on various factors
        base_score = council_decision.get('final_fitness', 0.5)
        
        # Adjust for environmental stresses if any
        stress_resistance = 1.0  # Would be calculated based on real environmental factors
        complexity_factor = len(job_params.parameters) / 10.0  # More parameters = more complex
        
        anti_fragile_score = min(1.0, base_score * stress_resistance * (1 + complexity_factor))
        
        # Create strategy ID based on material, operation, and parameters
        strategy_id = self._generate_strategy_id(job_params)
        
        badge = SurvivorBadge(
            strategy_id=strategy_id,
            material=job_params.material,
            operation_type=job_params.operation_type,
            success_count=1,
            failure_count=0,
            total_runs=1,
            anti_fragile_score=anti_fragile_score,
            metrics={
                'base_fitness': base_score,
                'stress_resistance': stress_resistance,
                'complexity': complexity_factor,
                'parameters': job_params.parameters
            },
            last_updated=datetime.utcnow()
        )
        
        # Award the badge to the Hive
        self.hive.award_survivor_badge(badge)
    
    def _generate_strategy_id(self, job_params: JobParameters) -> str:
        """Generate a unique strategy ID based on job parameters."""
        params_str = str(sorted(job_params.parameters.items()))
        strategy_data = f"{job_params.material}:{job_params.operation_type}:{params_str}"
        return hashlib.sha256(strategy_data.encode()).hexdigest()[:16]  # Short hash
    
    def _generate_cache_key(self, job_params: JobParameters) -> str:
        """Generate a cache key for the job parameters."""
        params_str = str(sorted(job_params.parameters.items()))
        return f"{job_params.material}:{job_params.operation_type}:{params_str}:{job_params.gcode_signature}"
    
    def get_node_status(self) -> Dict[str, Any]:
        """Get the current status of this machine node."""
        return {
            'machine_id': self.machine_id,
            'status': self.status,
            'active_jobs': len(self.active_jobs),
            'completed_jobs_count': len(self.completed_jobs),
            'local_trauma_cache_size': len(self.local_trauma_cache),
            'local_badge_cache_size': len(self.local_badge_cache),
            'last_updated': datetime.utcnow().isoformat()
        }
    
    def shutdown(self):
        """Gracefully shut down the machine node."""
        print(f"[NODE-{self.machine_id}] Shutting down...")
        self._stop_heartbeat.set()
        self._heartbeat_thread.join(timeout=5)
        print(f"[NODE-{self.machine_id}] Shutdown complete")


# Example usage and testing
if __name__ == "__main__":
    print("Machine Node initialized successfully.")
    print("Ready to connect to Hive Mind and process jobs.")
    
    # Example would require actual ShadowCouncil instance:
    # from ..app_factory import create_app_with_components
    # 
    # # Initialize components
    # app, components = create_app_with_components()
    # shadow_council = components['shadow_council']
    # 
    # # Create Hive and Machine Node
    # hive = HiveMind()
    # node = MachineNode("M001", hive, shadow_council)
    # 
    # # Create a test job
    # test_job = JobParameters(
    #     material="Aluminum-6061",
    #     operation_type="face_mill",
    #     parameters={"feed_rate": 2000, "rpm": 4000, "depth": 1.5},
    #     gcode_signature="gcode_hash_abc123",
    #     job_id="JOB_TEST_001",
    #     estimated_duration=0.5
    # )
    # 
    # # Run the job
    # result = node.run_job(test_job)
    # print(f"Job result: {result}")