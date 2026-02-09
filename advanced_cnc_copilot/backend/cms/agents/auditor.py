from typing import Dict, Any, List
from dataclasses import dataclass
import logging
from datetime import datetime


@dataclass
class MachineLimits:
    """Machine operational limits"""
    max_power_kw: float
    max_torque_nm: float
    max_rpm: float
    max_feed_rate: float
    max_temperature_c: float
    safe_resonance_zones: List[float]  # RPM ranges to avoid due to resonance


@dataclass
class ParallelPlan:
    """Represents a draft parallel machining plan"""
    id: str
    operations: List[Dict[str, Any]]  # List of operations for S1 and S2
    timestamp: datetime
    creator_id: str


class AuditorAgent:
    """
    Implements Shadow Council's "Death Penalty Function" for MTM parallel plans.
    Applies deterministic validation to probabilistic AI proposals.
    """
    
    def __init__(self, machine_limits: MachineLimits):
        self.machine_limits = machine_limits
        self.logger = logging.getLogger(__name__)
    
    def audit_parallel_plan(self, draft_plan: ParallelPlan) -> Dict[str, Any]:
        """
        Audits a parallel machining plan against physics and safety constraints.
        Implements the "Death Penalty Function" for constraint violations.
        """
        audit_result = {
            "plan_id": draft_plan.id,
            "approved": True,
            "fitness": 1.0,
            "veto_reason": "",
            "warnings": [],
            "audit_trace": [],
            "timestamp": datetime.utcnow()
        }
        
        # 1. Total Power/Torque Check
        power_check = self._check_total_torque(draft_plan.operations)
        if not power_check["passed"]:
            audit_result["approved"] = False
            audit_result["fitness"] = 0.0
            audit_result["veto_reason"] = power_check["reason"]
            audit_result["audit_trace"].append(power_check["trace"])
            return audit_result  # Immediate rejection if power exceeded
        
        # 2. Vibration Resonance Check
        resonance_check = self._check_vibration_resonance(draft_plan.operations)
        if not resonance_check["passed"]:
            audit_result["approved"] = False
            audit_result["fitness"] = 0.0
            audit_result["veto_reason"] = resonance_check["reason"]
            audit_result["audit_trace"].append(resonance_check["trace"])
            return audit_result  # Immediate rejection if resonance detected
        
        # 3. Additional safety checks
        safety_checks = self._perform_additional_safety_checks(draft_plan.operations)
        for check in safety_checks:
            if not check["passed"]:
                audit_result["approved"] = False
                audit_result["fitness"] = 0.0
                audit_result["veto_reason"] = check["reason"]
                audit_result["audit_trace"].append(check["trace"])
                return audit_result  # Immediate rejection for any safety violation
        
        # If all checks pass, approve the plan
        audit_result["audit_trace"].append("All safety and physics constraints satisfied")
        return audit_result
    
    def _check_total_torque(self, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check if total torque for S1 and S2 at any time step exceeds machine limits.
        If Torque(S1) + Torque(S2) > Machine.MaxPower, return fitness = 0.
        """
        result = {
            "passed": True,
            "reason": "",
            "trace": ""
        }
        
        # Simulate operations over time to check torque loads
        time_steps = self._simulate_time_steps(operations)
        
        for time_step in time_steps:
            s1_torque = time_step.get('s1_torque', 0.0)
            s2_torque = time_step.get('s2_torque', 0.0)
            total_torque = s1_torque + s2_torque
            
            if total_torque > self.machine_limits.max_power_kw * 1000:  # Convert kW to W if needed
                result["passed"] = False
                result["reason"] = f"Death Penalty: Combined torque {total_torque}W exceeds machine limit {self.machine_limits.max_power_kw * 1000}W"
                result["trace"] = f"Time Step {time_step['time']}: S1 Torque={s1_torque}W, S2 Torque={s2_torque}W, Total={total_torque}W"
                self.logger.warning(result["reason"])
                return result
        
        result["trace"] = "Total torque remained within machine limits throughout all operations"
        return result
    
    def _check_vibration_resonance(self, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check for vibration resonance conditions.
        If RPM(S1) is harmonic multiple of RPM(S2), suggest 10% RPM shift (Destructive Interference).
        """
        result = {
            "passed": True,
            "reason": "",
            "trace": ""
        }
        
        # Check for harmonic relationships between spindles
        for op_pair in self._get_simultaneous_operations(operations):
            s1_rpm = op_pair['s1'].get('rpm', 0.0)
            s2_rpm = op_pair['s2'].get('rpm', 0.0)
            
            if s1_rpm > 0 and s2_rpm > 0:
                # Check for harmonic multiples (within 5% tolerance)
                ratio_s1_s2 = s1_rpm / s2_rpm
                ratio_s2_s1 = s2_rpm / s1_rpm
                
                # Common harmonic ratios
                harmonics = [0.5, 1.0, 2.0, 0.33, 0.25, 1.5, 3.0]
                
                for harmonic in harmonics:
                    if abs(ratio_s1_s2 - harmonic) < 0.05 or abs(ratio_s2_s1 - harmonic) < 0.05:
                        # Resonance detected - recommend destructive interference
                        new_s1_rpm = s1_rpm * 1.1  # 10% shift
                        new_s2_rpm = s2_rpm * 0.9  # Opposite shift
                        
                        result["passed"] = False
                        result["reason"] = f"Resonance detected: RPM {s1_rpm} and {s2_rpm} are harmonic multiples. Recommend destructive interference via RPM adjustment: S1={new_s1_rpm}, S2={new_s2_rpm}"
                        result["trace"] = f"Harmonic ratio {harmonic} detected between {s1_rpm} and {s2_rpm} RPM"
                        
                        self.logger.warning(f"Resonance condition detected: {result['reason']}")
                        return result
        
        result["trace"] = "No vibration resonance conditions detected between spindle operations"
        return result
    
    def _perform_additional_safety_checks(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Performs additional safety checks beyond torque and resonance.
        """
        checks = []
        
        # Temperature check
        temp_check = self._check_temperature_limits(operations)
        checks.append(temp_check)
        
        # Feed rate check
        feed_check = self._check_feed_rates(operations)
        checks.append(feed_check)
        
        # RPM check
        rpm_check = self._check_rpm_limits(operations)
        checks.append(rpm_check)
        
        return [c for c in checks if not c["passed"]]  # Return only failed checks
    
    def _check_temperature_limits(self, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check if operations exceed temperature limits"""
        result = {
            "passed": True,
            "reason": "",
            "trace": ""
        }
        
        for op in operations:
            temp = op.get('temperature', 0.0)
            if temp > self.machine_limits.max_temperature_c:
                result["passed"] = False
                result["reason"] = f"Temperature {temp}°C exceeds limit {self.machine_limits.max_temperature_c}°C"
                result["trace"] = f"Operation {op.get('id', 'unknown')} exceeds temperature limits"
                self.logger.warning(result["reason"])
                return result
        
        result["trace"] = "All operations remain within temperature limits"
        return result
    
    def _check_feed_rates(self, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check if operations exceed feed rate limits"""
        result = {
            "passed": True,
            "reason": "",
            "trace": ""
        }
        
        for op in operations:
            feed_rate = op.get('feed_rate', 0.0)
            if feed_rate > self.machine_limits.max_feed_rate:
                result["passed"] = False
                result["reason"] = f"Feed rate {feed_rate} exceeds limit {self.machine_limits.max_feed_rate}"
                result["trace"] = f"Operation {op.get('id', 'unknown')} exceeds feed rate limits"
                self.logger.warning(result["reason"])
                return result
        
        result["trace"] = "All operations remain within feed rate limits"
        return result
    
    def _check_rpm_limits(self, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check if operations exceed RPM limits"""
        result = {
            "passed": True,
            "reason": "",
            "trace": ""
        }
        
        for op in operations:
            rpm = op.get('rpm', 0.0)
            if rpm > self.machine_limits.max_rpm:
                result["passed"] = False
                result["reason"] = f"RPM {rpm} exceeds limit {self.machine_limits.max_rpm}"
                result["trace"] = f"Operation {op.get('id', 'unknown')} exceeds RPM limits"
                self.logger.warning(result["reason"])
                return result
        
        result["trace"] = "All operations remain within RPM limits"
        return result
    
    def _simulate_time_steps(self, operations: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """
        Simulate operations over time to check torque loads at each step.
        """
        # This is a simplified simulation - in reality, this would be more complex
        time_steps = []
        
        # Create time steps based on operation timing
        for i, op in enumerate(operations):
            time_step = {
                'time': i,
                's1_torque': op.get('s1_torque', 0.0) if 's1' in op.get('spindle', '') else 0.0,
                's2_torque': op.get('s2_torque', 0.0) if 's2' in op.get('spindle', '') else 0.0
            }
            time_steps.append(time_step)
        
        return time_steps
    
    def _get_simultaneous_operations(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify pairs of operations that occur simultaneously on different spindles.
        """
        simultaneous_pairs = []
        
        # Group operations by time slot
        time_groups = {}
        for op in operations:
            time_slot = op.get('time_slot', 0)
            if time_slot not in time_groups:
                time_groups[time_slot] = []
            time_groups[time_slot].append(op)
        
        # Find pairs where one operation is on S1 and another on S2 in same time slot
        for time_slot, ops in time_groups.items():
            s1_ops = [op for op in ops if 's1' in op.get('spindle', '').lower()]
            s2_ops = [op for op in ops if 's2' in op.get('spindle', '').lower()]
            
            if s1_ops and s2_ops:
                # Pair first S1 with first S2 (simplified)
                simultaneous_pairs.append({
                    's1': s1_ops[0],
                    's2': s2_ops[0],
                    'time_slot': time_slot
                })
        
        return simultaneous_pairs