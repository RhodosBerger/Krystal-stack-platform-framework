"""
Fleet Intelligence & Collective Learning - Hive Mind Protocol
Enables real-time knowledge sharing between CNC machines, creating emergent collective intelligence behaviors
through distributed consensus algorithms and cross-machine pattern recognition.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import uuid
import logging
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor


class TraumaType(Enum):
    """Types of trauma events that can be shared across the fleet"""
    TOOL_BREAKAGE = "tool_breakage"
    THERMAL_RUNAWAY = "thermal_runaway"
    VIBRATION_ANOMALY = "vibration_anomaly"
    SERVO_JERK = "servo_jerk"
    COLLISION_EVENT = "collision_event"
    COOLANT_FAILURE = "coolant_failure"
    SPINDLE_OVERLOAD = "spindle_overload"


@dataclass
class TraumaEvent:
    """Represents a traumatic event that occurred on a machine and needs to be shared"""
    event_id: str
    machine_id: str
    trauma_type: TraumaType
    parameters_at_failure: Dict[str, Any]
    timestamp: datetime
    cost_impact: float
    material: str
    operation_type: str
    gcode_signature: str
    recovery_actions: List[str]
    severity_level: float  # 0.0 to 1.0 scale


@dataclass
class FleetConsensusVote:
    """Vote in the distributed consensus process"""
    vote_id: str
    machine_id: str
    proposal: Dict[str, Any]
    vote_decision: bool
    confidence: float
    reasoning: str
    timestamp: datetime


@dataclass
class CrossMachinePattern:
    """Detected pattern across multiple machines"""
    pattern_id: str
    pattern_type: str
    affected_machines: List[str]
    common_factors: Dict[str, Any]
    severity_score: float
    prediction_accuracy: float
    timestamp: datetime
    suggested_prevention: str


class HiveMind:
    """
    The Hive Mind - Central nervous system for fleet-wide intelligence sharing.
    Enables real-time knowledge sharing between CNC machines, creating emergent collective intelligence.
    """
    
    def __init__(self):
        # Initialize with empty collections
        self.trauma_registry: List[TraumaEvent] = []
        self.survivor_badges: List[Dict[str, Any]] = []
        self.fleet_consensus_records: List[Dict[str, Any]] = []
        self.cross_machine_patterns: List[CrossMachinePattern] = []
        self.fleet_knowledge_base: Dict[str, Any] = {}
        self.machine_statuses: Dict[str, Any] = {}
        self.fleet_metrics: Dict[str, Any] = {}
        
        self.sync_lock = threading.RLock()  # Thread safety for shared resources
        self.logger = logging.getLogger(__name__)
    
    def register_trauma_event(self, machine_id: str, trauma_type: TraumaType,
                            parameters: Dict[str, Any], cost_impact: float,
                            material: str, operation_type: str,
                            gcode_signature: str, recovery_actions: Optional[List[str]] = None) -> TraumaEvent:
        """
        Register a trauma event in the global registry to share with all fleet members.
        
        Args:
            machine_id: ID of the machine that experienced the trauma
            trauma_type: Type of trauma event
            parameters: Parameters at the time of failure
            cost_impact: Financial cost of the trauma
            material: Material being machined
            operation_type: Type of operation
            gcode_signature: Hash of the G-Code that caused the issue
            recovery_actions: Actions taken to recover from the trauma
            
        Returns:
            TraumaEvent record of the registered trauma
        """
        with self.sync_lock:
            event_id = f"TRAUMA_{uuid.uuid4().hex[:8]}"
            
            trauma_event = TraumaEvent(
                event_id=event_id,
                machine_id=machine_id,
                trauma_type=trauma_type,
                parameters_at_failure=parameters,
                timestamp=datetime.utcnow(),
                cost_impact=cost_impact,
                material=material,
                operation_type=operation_type,
                gcode_signature=gcode_signature,
                recovery_actions=recovery_actions or [],
                severity_level=self._calculate_severity_level(trauma_type, cost_impact, parameters)
            )
            
            # Add to global trauma registry
            self.trauma_registry.append(trauma_event)
            
            # Broadcast to all machines in fleet
            self._broadcast_trauma_to_fleet(trauma_event)
            
            self.logger.info(f"Hive Mind registered trauma: {trauma_type.value} on {machine_id}, "
                           f"impact: ${cost_impact:.2f}, severity: {trauma_event.severity_level:.2f}")
            
            return trauma_event
    
    def _calculate_severity_level(self, trauma_type: TraumaType, cost_impact: float, 
                                parameters: Dict[str, Any]) -> float:
        """
        Calculate severity level based on trauma type, cost impact, and parameters.
        
        Args:
            trauma_type: Type of trauma
            cost_impact: Financial cost of the trauma
            parameters: Parameters at time of failure
            
        Returns:
            Severity level from 0.0 to 1.0
        """
        base_severity = {
            TraumaType.TOOL_BREAKAGE: 0.6,
            TraumaType.THERMAL_RUNAWAY: 0.8,
            TraumaType.VIBRATION_ANOMALY: 0.4,
            TraumaType.SERVO_JERK: 0.7,
            TraumaType.COLLISION_EVENT: 0.9,
            TraumaType.COOLANT_FAILURE: 0.5,
            TraumaType.SPINDLE_OVERLOAD: 0.7
        }.get(trauma_type, 0.5)
        
        # Adjust based on cost impact (higher costs = higher severity)
        cost_factor = min(1.0, cost_impact / 500.0)  # Normalize against $500 baseline
        
        # Adjust based on parameter extremes
        parameter_extremes = 0.0
        for param_name, param_value in parameters.items():
            if isinstance(param_value, (int, float)):
                # Check for extreme values that might have caused the trauma
                if param_name == 'spindle_load' and param_value > 90:
                    parameter_extremes += 0.2
                elif param_name == 'temperature' and param_value > 65:
                    parameter_extremes += 0.2
                elif param_name == 'vibration_x' and param_value > 1.5:
                    parameter_extremes += 0.15
                elif param_name == 'vibration_y' and param_value > 1.5:
                    parameter_extremes += 0.15
                elif param_name == 'feed_rate' and param_value > 4000:
                    parameter_extremes += 0.1
                elif param_name == 'rpm' and param_value > 10000:
                    parameter_extremes += 0.1
        
        # Combine factors
        severity = min(1.0, base_severity + (cost_factor * 0.3) + (parameter_extremes * 0.2))
        return severity
    
    def _broadcast_trauma_to_fleet(self, trauma_event: TraumaEvent):
        """
        Broadcast trauma event to all machines in the fleet for immediate protection.
        This implements "Industrial Telepathy" - machines learn from failures they've never experienced.
        """
        # In a real system, this would send to all connected machines
        # For simulation, we'll just log the broadcast
        self.logger.info(f"Broadcasting trauma {trauma_event.event_id} to fleet: "
                        f"Preventing {trauma_event.trauma_type.value} on {trauma_event.material} {trauma_event.operation_type}")
        
        # Update fleet knowledge base with new trauma
        trauma_key = f"{trauma_event.material}:{trauma_event.operation_type}:{trauma_event.gcode_signature}"
        if trauma_key not in self.fleet_knowledge_base:
            self.fleet_knowledge_base[trauma_key] = []
        self.fleet_knowledge_base[trauma_key].append(trauma_event)
    
    def award_survivor_badge(self, strategy_id: str, machine_id: str, 
                           material: str, operation_type: str,
                           fitness_score: float, improvement_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Award a survivor badge to a strategy that successfully operated under stress conditions.
        
        Args:
            strategy_id: ID of the successful strategy
            machine_id: Machine that executed the strategy
            material: Material the strategy was for
            operation_type: Type of operation
            fitness_score: Fitness score achieved
            improvement_metrics: Metrics showing improvement over baseline
            
        Returns:
            Badge record with details
        """
        with self.sync_lock:
            badge_id = f"BADGE_{uuid.uuid4().hex[:8]}"
            
            # Determine badge level based on fitness score
            if fitness_score >= 0.95:
                badge_level = "DIAMOND"
                badge_value = 100.0
            elif fitness_score >= 0.85:
                badge_level = "PLATINUM"
                badge_value = 75.0
            elif fitness_score >= 0.70:
                badge_level = "GOLD"
                badge_value = 50.0
            elif fitness_score >= 0.50:
                badge_level = "SILVER"
                badge_value = 25.0
            else:
                badge_level = "BRONZE"
                badge_value = 10.0
            
            badge = {
                'badge_id': badge_id,
                'strategy_id': strategy_id,
                'machine_id': machine_id,
                'material': material,
                'operation_type': operation_type,
                'fitness_score': fitness_score,
                'badge_level': badge_level,
                'improvement_metrics': improvement_metrics,
                'timestamp': datetime.utcnow().isoformat(),
                'badge_value': badge_value,  # Economic value of the badge
                'anti_fragile_score': self._calculate_anti_fragile_score(improvement_metrics, fitness_score)
            }
            
            # Add to global survivor badge registry
            self.survivor_badges.append(badge)
            
            # Broadcast successful strategy to fleet
            self._broadcast_successful_strategy(badge)
            
            self.logger.info(f"Awarded {badge_level} survivor badge to {strategy_id} on {machine_id}, "
                           f"fitness: {fitness_score:.3f}, value: ${badge_value:.2f}")
            
            return badge
    
    def _calculate_anti_fragile_score(self, improvement_metrics: Dict[str, float], 
                                    fitness_score: float) -> float:
        """
        Calculate the anti-fragile score based on how well the strategy performs under stress.
        """
        # Anti-fragile score combines fitness with improvement under stress
        stress_resilience = improvement_metrics.get('stress_resilience', 0.5)
        performance_under_stress = improvement_metrics.get('performance_under_stress', 0.5)
        
        anti_fragile_score = (fitness_score * 0.4 + stress_resilience * 0.3 + performance_under_stress * 0.3)
        return anti_fragile_score
    
    def _broadcast_successful_strategy(self, badge: Dict[str, Any]):
        """
        Broadcast successful strategy to all fleet members for adoption.
        """
        strategy_key = f"{badge['material']}:{badge['operation_type']}:{badge['strategy_id']}"
        
        if strategy_key not in self.fleet_knowledge_base:
            self.fleet_knowledge_base[strategy_key] = []
        
        self.fleet_knowledge_base[strategy_key].append({
            'type': 'success',
            'badge': badge,
            'timestamp': badge['timestamp']
        })
        
        self.logger.info(f"Broadcasting successful strategy {badge['strategy_id']} to fleet: "
                        f"For {badge['material']} {badge['operation_type']}")
    
    def initiate_fleet_consensus(self, proposal: Dict[str, Any], 
                               participating_machines: List[str]) -> Dict[str, Any]:
        """
        Initiate a distributed consensus process across fleet machines.
        
        Args:
            proposal: The proposal to get consensus on
            participating_machines: List of machines to participate in consensus
            
        Returns:
            Consensus result with aggregated votes and final decision
        """
        with self.sync_lock:
            consensus_id = f"CONS_{uuid.uuid4().hex[:8]}"
            
            self.logger.info(f"Initiating fleet consensus {consensus_id} for proposal affecting {len(participating_machines)} machines")
            
            # Collect votes from participating machines
            votes = []
            for machine_id in participating_machines:
                vote = self._request_machine_vote(machine_id, proposal, consensus_id)
                votes.append(vote)
            
            # Apply consensus algorithm
            consensus_result = self._apply_consensus_algorithm(votes, proposal)
            
            # Store consensus record
            consensus_record = {
                'consensus_id': consensus_id,
                'proposal': proposal,
                'votes': votes,
                'result': consensus_result,
                'timestamp': datetime.utcnow().isoformat(),
                'participating_machines': participating_machines
            }
            
            self.fleet_consensus_records.append(consensus_record)
            
            self.logger.info(f"Fleet consensus {consensus_id} completed: {consensus_result['decision']}, "
                           f"approval_rate: {consensus_result['approval_rate']:.2f}")
            
            return consensus_result
    
    def _request_machine_vote(self, machine_id: str, proposal: Dict[str, Any], 
                            consensus_id: str) -> FleetConsensusVote:
        """
        Request a vote from a specific machine on a proposal.
        In real implementation, this would be a network call to the machine.
        """
        # Simulate machine voting based on local Shadow Council decision
        # In a real system, this would call the machine's Shadow Council to evaluate the proposal
        vote_id = f"VOTE_{uuid.uuid4().hex[:8]}"
        
        # Simulate machine evaluation
        # Calculate vote based on local constraints and experience
        machine_experience = self._get_machine_experience_with_material_op(machine_id, 
                                                                         proposal.get('material', 'unknown'),
                                                                         proposal.get('operation_type', 'unknown'))
        
        # Determine vote based on proposal and machine experience
        if proposal.get('requires_high_rpm', False) and machine_experience.get('thermal_issues', 0) > 0.5:
            vote_decision = False
            confidence = 0.8
            reasoning = "Machine has history of thermal issues with high RPM operations"
        elif proposal.get('aggressive_feed_rate', False) and machine_experience.get('vibration_issues', 0) > 0.4:
            vote_decision = False
            confidence = 0.75
            reasoning = "Machine has history of vibration issues with aggressive feed rates"
        else:
            vote_decision = True
            confidence = 0.9
            reasoning = "Proposal aligns with machine's operational strengths"
        
        vote = FleetConsensusVote(
            vote_id=vote_id,
            machine_id=machine_id,
            proposal=proposal,
            vote_decision=vote_decision,
            confidence=confidence,
            reasoning=reasoning,
            timestamp=datetime.utcnow()
        )
        
        return vote
    
    def _apply_consensus_algorithm(self, votes: List[FleetConsensusVote], 
                                 proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply consensus algorithm to aggregate votes.
        Uses weighted majority voting based on machine experience and confidence.
        """
        # Count positive and negative votes with weights
        weighted_approve = 0.0
        weighted_reject = 0.0
        total_weight = 0.0
        
        for vote in votes:
            weight = vote.confidence
            total_weight += weight
            
            if vote.vote_decision:
                weighted_approve += weight
            else:
                weighted_reject += weight
        
        # Calculate approval rate
        approval_rate = weighted_approve / total_weight if total_weight > 0 else 0.0
        
        # Determine consensus decision (simple majority with threshold)
        consensus_threshold = 0.6  # Require 60% approval for consensus
        decision = approval_rate >= consensus_threshold
        
        # Calculate overall confidence
        overall_confidence = sum(vote.confidence for vote in votes) / len(votes) if votes else 0.0
        
        return {
            'decision': decision,
            'approval_rate': approval_rate,
            'overall_confidence': overall_confidence,
            'total_votes': len(votes),
            'weighted_approve': weighted_approve,
            'weighted_reject': weighted_reject,
            'consensus_threshold': consensus_threshold,
            'reasoning_trace': [
                f"Weighted approval rate: {approval_rate:.3f}",
                f"Threshold: {consensus_threshold}",
                f"Decision: {'APPROVE' if decision else 'REJECT'}"
            ]
        }
    
    def _get_machine_experience_with_material_op(self, machine_id: str, 
                                                material: str, operation_type: str) -> Dict[str, float]:
        """
        Get a machine's experience with specific material and operation type.
        """
        # Simulate retrieving machine experience
        # In real implementation, this would query the machine's local database
        experience = {
            'thermal_issues': 0.2,  # 0.0 to 1.0 scale
            'vibration_issues': 0.1,
            'tool_wear_rate': 0.005,  # mm/min wear rate
            'success_rate': 0.85,
            'efficiency_score': 0.78
        }
        
        # Adjust based on material and operation type
        if material.lower() == 'titanium' or material.lower() == 'inconel':
            experience['thermal_issues'] = 0.4  # Harder materials cause more thermal issues
            experience['tool_wear_rate'] = 0.008
        
        if operation_type.lower() == 'drill' or operation_type.lower() == 'tap':
            experience['vibration_issues'] = 0.3  # More prone to vibration
        
        return experience
    
    def detect_cross_machine_patterns(self, lookback_hours: int = 24) -> List[CrossMachinePattern]:
        """
        Detect patterns across multiple machines that indicate systemic issues or opportunities.
        
        Args:
            lookback_hours: How many hours of data to analyze for patterns
            
        Returns:
            List of detected patterns across the fleet
        """
        with self.sync_lock:
            self.logger.info(f"Detecting cross-machine patterns in last {lookback_hours} hours")
            
            # Analyze trauma registry for patterns
            pattern_start_time = datetime.utcnow() - timedelta(hours=lookback_hours)
            recent_traumas = [t for t in self.trauma_registry if t.timestamp >= pattern_start_time]
            
            detected_patterns = []
            
            # Group traumas by common factors
            material_patterns = self._group_by_material_and_operation(recent_traumas)
            parameter_patterns = self._analyze_parameter_correlations(recent_traumas)
            temporal_patterns = self._analyze_temporal_correlations(recent_traumas)
            
            # Process material patterns
            for pattern_group in material_patterns:
                if len(pattern_group['events']) > 1:  # Multiple machines experienced similar trauma
                    pattern = CrossMachinePattern(
                        pattern_id=f"PAT_{uuid.uuid4().hex[:8]}",
                        pattern_type="material_operation_correlation",
                        affected_machines=[e.machine_id for e in pattern_group['events']],
                        common_factors={
                            'material': pattern_group['material'],
                            'operation': pattern_group['operation'],
                            'gcode_signature': pattern_group['gcode_signature']
                        },
                        severity_score=self._calculate_pattern_severity(pattern_group['events']),
                        prediction_accuracy=0.85,  # Based on historical accuracy
                        timestamp=datetime.utcnow(),
                        suggested_prevention=f"Avoid {pattern_group['operation']} on {pattern_group['material']} "
                                           f"with parameters similar to {pattern_group['gcode_signature']}"
                    )
                    detected_patterns.append(pattern)
            
            # Process parameter patterns
            for param_pattern in parameter_patterns:
                if param_pattern['frequency'] > 1:  # Multiple machines with similar parameter issues
                    pattern = CrossMachinePattern(
                        pattern_id=f"PAT_{uuid.uuid4().hex[:8]}",
                        pattern_type="parameter_correlation",
                        affected_machines=param_pattern['affected_machines'],
                        common_factors={
                            'parameters': param_pattern['common_parameters'],
                            'frequency': param_pattern['frequency']
                        },
                        severity_score=param_pattern['severity_score'],
                        prediction_accuracy=0.78,
                        timestamp=datetime.utcnow(),
                        suggested_prevention=f"Reduce aggression in {list(param_pattern['common_parameters'].keys())} parameters"
                    )
                    detected_patterns.append(pattern)
            
            # Process temporal patterns
            for temp_pattern in temporal_patterns:
                pattern = CrossMachinePattern(
                    pattern_id=f"PAT_{uuid.uuid4().hex[:8]}",
                    pattern_type="temporal_correlation",
                    affected_machines=temp_pattern['affected_machines'],
                    common_factors={
                        'time_offset': temp_pattern['time_offset'],
                        'common_trigger': temp_pattern['common_trigger']
                    },
                    severity_score=temp_pattern['severity_score'],
                    prediction_accuracy=0.72,
                    timestamp=datetime.utcnow(),
                    suggested_prevention=f"Monitor for {temp_pattern['common_trigger']} {temp_pattern['time_offset']} after similar operations"
                )
                detected_patterns.append(pattern)
            
            # Store detected patterns
            self.cross_machine_patterns.extend(detected_patterns)
            
            self.logger.info(f"Detected {len(detected_patterns)} cross-machine patterns in last {lookback_hours} hours")
            
            return detected_patterns
    
    def _group_by_material_and_operation(self, traumas: List[TraumaEvent]) -> List[Dict[str, Any]]:
        """
        Group trauma events by material and operation type to identify common patterns.
        """
        groups = {}
        
        for trauma in traumas:
            key = f"{trauma.material}:{trauma.operation_type}:{trauma.gcode_signature}"
            
            if key not in groups:
                groups[key] = {
                    'material': trauma.material,
                    'operation': trauma.operation_type,
                    'gcode_signature': trauma.gcode_signature,
                    'events': []
                }
            
            groups[key]['events'].append(trauma)
        
        return list(groups.values())
    
    def _analyze_parameter_correlations(self, traumas: List[TraumaEvent]) -> List[Dict[str, Any]]:
        """
        Analyze parameter correlations across trauma events to identify risky parameter combinations.
        """
        patterns = []
        
        # Group by similar parameter combinations
        param_groups = {}
        for trauma in traumas:
            # Create a signature based on extreme parameters
            extreme_params = {}
            for param, value in trauma.parameters_at_failure.items():
                if isinstance(value, (int, float)):
                    # Consider parameters that were at extreme values
                    if (param == 'spindle_load' and value > 85) or \
                       (param == 'temperature' and value > 60) or \
                       (param == 'vibration_x' and value > 1.2) or \
                       (param == 'feed_rate' and value > 3500) or \
                       (param == 'rpm' and value > 9000):
                        extreme_params[param] = value
            
            if extreme_params:  # Only consider if there were extreme parameters
                param_signature = tuple(sorted(extreme_params.items()))
                param_sig_str = str(param_signature)
                
                if param_sig_str not in param_groups:
                    param_groups[param_sig_str] = {
                        'common_parameters': extreme_params,
                        'affected_machines': [],
                        'events': [],
                        'frequency': 0,
                        'severity_score': 0.0
                    }
                
                param_groups[param_sig_str]['affected_machines'].append(trauma.machine_id)
                param_groups[param_sig_str]['events'].append(trauma)
                param_groups[param_sig_str]['frequency'] += 1
                param_groups[param_sig_str]['severity_score'] += trauma.severity_level
        
        # Calculate average severity for each group
        for param_group in param_groups.values():
            param_group['severity_score'] /= param_group['frequency']  # Average severity
            patterns.append(param_group)
        
        return patterns
    
    def _analyze_temporal_correlations(self, traumas: List[TraumaEvent]) -> List[Dict[str, Any]]:
        """
        Analyze temporal correlations to identify patterns where one machine's failure 
        leads to similar failures in other machines.
        """
        patterns = []
        
        # Look for temporal clustering of similar failures
        for i, trauma1 in enumerate(traumas):
            for trauma2 in traumas[i+1:]:
                # Check if similar failure happened on different machine within time window
                time_diff = abs((trauma1.timestamp - trauma2.timestamp).total_seconds())
                
                if (trauma1.machine_id != trauma2.machine_id and
                    trauma1.trauma_type == trauma2.trauma_type and
                    trauma1.material == trauma2.material and
                    time_diff <= 3600):  # Within 1 hour
                    
                    # Calculate severity based on cost impact
                    severity = (trauma1.severity_level + trauma2.severity_level) / 2
                    
                    pattern = {
                        'affected_machines': [trauma1.machine_id, trauma2.machine_id],
                        'time_offset': time_diff,
                        'common_trigger': f"{trauma1.trauma_type.value} on {trauma1.material}",
                        'severity_score': severity
                    }
                    patterns.append(pattern)
        
        return patterns
    
    def _calculate_pattern_severity(self, events: List[TraumaEvent]) -> float:
        """
        Calculate severity of a pattern based on the traumas in the pattern.
        """
        if not events:
            return 0.0
        
        total_severity = sum(e.severity_level for e in events)
        avg_severity = total_severity / len(events)
        
        # Boost severity if multiple machines experienced the same issue
        machine_diversity_factor = len(set(e.machine_id for e in events)) / len(events)
        return avg_severity * (1 + machine_diversity_factor)
    
    def get_fleet_intelligence_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report on fleet intelligence and collective learning.
        
        Returns:
            Dictionary with fleet intelligence metrics
        """
        with self.sync_lock:
            # Calculate fleet-wide metrics
            total_traumas = len(self.trauma_registry)
            total_survivor_badges = len(self.survivor_badges)
            total_consensus_events = len(self.fleet_consensus_records)
            total_patterns = len(self.cross_machine_patterns)
            
            # Calculate economic impact
            total_trauma_cost = sum(t.cost_impact for t in self.trauma_registry)
            total_badge_value = sum(b['badge_value'] for b in self.survivor_badges)
            
            # Calculate learning metrics
            prevented_events_estimate = self._estimate_prevented_events()
            fleet_efficiency_improvement = self._calculate_fleet_efficiency_improvement()
            collective_resilience_score = self._calculate_collective_resilience_score()
            
            report = {
                'fleet_intelligence_summary': {
                    'total_traumas_shared': total_traumas,
                    'total_survivor_badges_awarded': total_survivor_badges,
                    'total_consensus_events': total_consensus_events,
                    'total_cross_machine_patterns_detected': total_patterns,
                    'total_economic_impact_of_traumas': total_trauma_cost,
                    'total_value_of_shared_success': total_badge_value,
                    'estimated_events_prevented_by_sharing': prevented_events_estimate,
                    'fleet_efficiency_improvement_percentage': fleet_efficiency_improvement,
                    'collective_resilience_score': collective_resilience_score,
                    'report_generated': datetime.utcnow().isoformat()
                },
                'top_material_operation_patterns': self._get_top_material_operation_patterns(),
                'most_shared_traumas': self._get_most_shared_traumas(),
                'highest_value_strategies': self._get_highest_value_strategies(),
                'fleet_health_metrics': self._calculate_fleet_health_metrics()
            }
            
            return report
    
    def _estimate_prevented_events(self) -> int:
        """
        Estimate how many events have been prevented by sharing trauma across the fleet.
        """
        # Based on trauma registry and sharing, estimate prevented events
        # This is a simplified estimation - in real system would be more complex
        if not self.trauma_registry:
            return 0
        
        # Assume each trauma shared prevents 3-5 similar events across fleet
        avg_prevention_multiplier = 4.0
        return int(len(self.trauma_registry) * avg_prevention_multiplier)
    
    def _calculate_fleet_efficiency_improvement(self) -> float:
        """
        Calculate estimated efficiency improvement from fleet intelligence.
        """
        # Based on successful strategies shared across fleet
        if not self.survivor_badges:
            return 0.0
        
        # Average improvement from survivor badges
        avg_improvement = sum(
            badge.get('improvement_metrics', {}).get('efficiency_improvement', 0.0) 
            for badge in self.survivor_badges
        ) / len(self.survivor_badges) if self.survivor_badges else 0.0
        
        return avg_improvement * 100  # Convert to percentage
    
    def _calculate_collective_resilience_score(self) -> float:
        """
        Calculate the collective resilience score of the fleet.
        """
        if not self.survivor_badges:
            return 0.3  # Baseline resilience if no badges yet
        
        # Average the anti-fragile scores
        avg_anti_fragile = sum(
            badge.get('anti_fragile_score', 0.5) 
            for badge in self.survivor_badges
        ) / len(self.survivor_badges)
        
        return avg_anti_fragile
    
    def _get_top_material_operation_patterns(self) -> List[Dict[str, Any]]:
        """
        Get the most common material-operation combinations that have caused issues.
        """
        material_op_counts = {}
        
        for trauma in self.trauma_registry:
            key = f"{trauma.material}:{trauma.operation_type}"
            if key not in material_op_counts:
                material_op_counts[key] = {
                    'material': trauma.material,
                    'operation': trauma.operation_type,
                    'count': 0,
                    'total_cost': 0.0,
                    'avg_severity': 0.0
                }
            
            record = material_op_counts[key]
            record['count'] += 1
            record['total_cost'] += trauma.cost_impact
            record['avg_severity'] += trauma.severity_level
        
        for record in material_op_counts.values():
            record['avg_severity'] /= record['count']
        
        # Sort by count and return top 10
        sorted_records = sorted(
            material_op_counts.values(),
            key=lambda x: x['count'],
            reverse=True
        )[:10]
        
        return sorted_records
    
    def _get_most_shared_traumas(self) -> List[Dict[str, Any]]:
        """
        Get the traumas that were most widely shared across the fleet.
        """
        # Group traumas by similarity (same material, operation, parameters)
        trauma_groups = {}
        for trauma in self.trauma_registry:
            key = f"{trauma.material}:{trauma.operation_type}:{trauma.trauma_type.value}"
            if key not in trauma_groups:
                trauma_groups[key] = []
            trauma_groups[key].append(trauma)
        
        # Find groups with most traumas
        most_shared = sorted(
            trauma_groups.values(),
            key=len,
            reverse=True
        )[:5]
        
        return [
            {
                'group_key': f"{tr[0].material}:{tr[0].operation_type}:{tr[0].trauma_type.value}",
                'trauma_count': len(tr),
                'total_cost': sum(t.cost_impact for t in tr),
                'affected_machines': list(set(t.machine_id for t in tr)),
                'earliest_occurrence': min(t.timestamp for t in tr).isoformat(),
                'latest_occurrence': max(t.timestamp for t in tr).isoformat()
            } for tr in most_shared if tr
        ]
    
    def _get_highest_value_strategies(self) -> List[Dict[str, Any]]:
        """
        Get the strategies with the highest economic value (from survivor badges).
        """
        sorted_badges = sorted(
            self.survivor_badges,
            key=lambda b: b.get('badge_value', 0.0),
            reverse=True
        )[:10]
        
        return [
            {
                'strategy_id': badge['strategy_id'],
                'material': badge['material'],
                'operation_type': badge['operation_type'],
                'fitness_score': badge['fitness_score'],
                'badge_level': badge['badge_level'],
                'badge_value': badge['badge_value'],
                'anti_fragile_score': badge['anti_fragile_score']
            } for badge in sorted_badges
        ]
    
    def _calculate_fleet_health_metrics(self) -> Dict[str, float]:
        """
        Calculate overall health metrics for the fleet.
        """
        if not self.trauma_registry:
            return {
                'trauma_frequency_per_hour': 0.0,
                'average_trauma_severity': 0.0,
                'trauma_recovery_time_average': 0.0
            }
        
        # Calculate trauma frequency
        earliest = min(t.timestamp for t in self.trauma_registry)
        latest = max(t.timestamp for t in self.trauma_registry)
        time_span_hours = (latest - earliest).total_seconds() / 3600 if latest != earliest else 1.0
        trauma_frequency = len(self.trauma_registry) / time_span_hours if time_span_hours > 0 else 0.0
        
        # Calculate average severity
        avg_severity = sum(t.severity_level for t in self.trauma_registry) / len(self.trauma_registry)
        
        return {
            'trauma_frequency_per_hour': trauma_frequency,
            'average_trauma_severity': avg_severity,
            'total_traumas_recorded': len(self.trauma_registry)
        }


# Example usage and testing
if __name__ == "__main__":
    print("Hive Mind Protocol initialized successfully.")
    print("Ready to coordinate fleet intelligence and collective learning.")
    
    # Example usage would be:
    # hive_mind = HiveMind()
    # 
    # # Register a trauma event from one machine
    # trauma_event = hive_mind.register_trauma_event(
    #     machine_id="M001",
    #     trauma_type=TraumaType.TOOL_BREAKAGE,
    #     parameters={'feed_rate': 4500, 'rpm': 11000, 'depth': 2.5},
    #     cost_impact=350.00,
    #     material="Inconel-718",
    #     operation_type="face_mill",
    #     gcode_signature="GCODE_HASH_ABC123",
    #     recovery_actions=["reduce_feed_rate", "inspect_tool", "update_parameters"]
    # )
    # 
    # # Award a survivor badge for a successful strategy
    # survivor_badge = hive_mind.award_survivor_badge(
    #     strategy_id="STRAT_ALUMINUM_FACE_MILL_001",
    #     machine_id="M002",
    #     material="Aluminum-6061",
    #     operation_type="face_mill",
    #     fitness_score=0.87,
    #     improvement_metrics={
    #         'stress_resilience': 0.92,
    #         'performance_under_stress': 0.85,
    #         'efficiency_improvement': 0.15
    #     }
    # )
    # 
    # # Initiate fleet consensus for a parameter change
    # consensus_result = hive_mind.initiate_fleet_consensus(
    #     proposal={
    #         'change_type': 'feed_rate_increase',
    #         'new_value': 2500,
    #         'material': 'Aluminum-6061',
    #         'operation': 'face_mill',
    #         'justification': 'Improved efficiency based on successful runs'
    #     },
    #     participating_machines=['M001', 'M002', 'M003', 'M004']
    # )
    # 
    # # Detect cross-machine patterns
    # patterns = hive_mind.detect_cross_machine_patterns(lookback_hours=24)
    # print(f"Detected {len(patterns)} cross-machine patterns")
    # 
    # # Generate fleet intelligence report
    # report = hive_mind.get_fleet_intelligence_report()
    # print(f"Fleet intelligence report generated with {report['fleet_intelligence_summary']['total_traumas_shared']} traumas shared")