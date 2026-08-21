#!/usr/bin/env python3
"""
Windows Optimization Agent - Practical Implementation

A sophisticated system optimization agent that uses the economic resource trading framework
to intelligently manage Windows system resources. This agent demonstrates practical
applications of the framework with real-world scenarios.
"""

import time
import threading
import uuid
from typing import Dict, List, Optional
from dataclasses import dataclass

from windows_system_utility import (
    WindowsResourceManager, ResourceType, Priority, AllocationRequest
)


@dataclass
class OptimizationPolicy:
    """Policy for system optimization."""
    name: str
    resource_type: ResourceType
    threshold: float  # When to trigger optimization
    action: str  # What action to take
    bid_credits: float  # How much to bid for resources
    cooldown_seconds: float = 5.0  # Minimum time between optimizations


class WindowsOptimizationAgent:
    """
    Intelligent Windows optimization agent that trades system resources
    based on real-time system state and optimization policies.
    """
    
    def __init__(self, agent_id: str = "OptimizerAgent"):
        self.agent_id = agent_id
        self.resource_manager = WindowsResourceManager()
        self.policies: List[OptimizationPolicy] = []
        self.active_allocations: List[str] = []  # allocation IDs
        self._running = False
        self._thread = None
        
        # Initialize optimization policies
        self._init_policies()
    
    def _init_policies(self):
        """Initialize default optimization policies."""
        self.policies = [
            OptimizationPolicy(
                name="Thermal Management",
                resource_type=ResourceType.CPU_CORE,
                threshold=80.0,  # CPU % above which to act
                action="reduce_cpu_cores",
                bid_credits=25.0
            ),
            OptimizationPolicy(
                name="Memory Pressure Relief",
                resource_type=ResourceType.MEMORY,
                threshold=85.0,  # Memory % above which to act
                action="memory_cleanup",
                bid_credits=50.0
            ),
            OptimizationPolicy(
                name="Performance Boost",
                resource_type=ResourceType.CPU_CORE,
                threshold=20.0,  # CPU % below which to act (idle detection)
                action="optimize_boost",
                bid_credits=75.0
            ),
            OptimizationPolicy(
                name="Disk Optimization",
                resource_type=ResourceType.DISK_IO,
                threshold=70.0,  # Disk % above which to act
                action="defrag_scheduled",
                bid_credits=10.0
            )
        ]
    
    def _evaluate_policies(self, telemetry) -> List[OptimizationPolicy]:
        """Evaluate which policies should be triggered based on telemetry."""
        triggered_policies = []
        
        # Map telemetry values to policy thresholds
        metric_map = {
            ResourceType.CPU_CORE: telemetry.cpu_percent,
            ResourceType.MEMORY: telemetry.memory_percent,
            ResourceType.DISK_IO: telemetry.disk_percent,
        }
        
        for policy in self.policies:
            current_value = metric_map.get(policy.resource_type, 0)
            if policy.threshold > 50:  # High threshold (e.g., 80%) - trigger if above
                if current_value >= policy.threshold:
                    triggered_policies.append(policy)
            else:  # Low threshold (e.g., 20%) - trigger if below
                if current_value <= policy.threshold:
                    triggered_policies.append(policy)
        
        return triggered_policies
    
    def _execute_policy(self, policy: OptimizationPolicy, amount: float = 1.0) -> bool:
        """Execute a triggered optimization policy."""
        print(f"[{self.agent_id}] Executing policy: {policy.name}")
        
        request = AllocationRequest(
            request_id=str(uuid.uuid4()),
            agent_id=self.agent_id,
            resource_type=policy.resource_type,
            amount=amount,
            priority=Priority.NORMAL,
            bid_credits=policy.bid_credits,
            duration_ms=5000  # Hold for 5 seconds
        )
        
        allocation = self.resource_manager.allocate_resource(request)
        if allocation:
            self.active_allocations.append(allocation.allocation_id)
            print(f"  [SUCCESS] Allocated {allocation.amount} {allocation.resource_type.value} "
                  f"(ID: {allocation.allocation_id[:8]})")
            
            # Execute the specific action
            self._perform_action(policy.action, allocation)
            return True
        else:
            print(f"  [FAILED] Could not allocate {policy.resource_type.value}")
            return False
    
    def _perform_action(self, action: str, allocation):
        """Perform the specific optimization action."""
        actions = {
            "reduce_cpu_cores": self._reduce_cpu_cores,
            "memory_cleanup": self._memory_cleanup,
            "optimize_boost": self._optimize_boost,
            "defrag_scheduled": self._defrag_scheduled,
        }
        
        action_func = actions.get(action, self._default_action)
        action_func(allocation)
    
    def _reduce_cpu_cores(self, allocation):
        """Reduce CPU core usage to manage thermal issues."""
        print(f"    Action: Reducing CPU core usage to manage thermal load")
        # In a real implementation, this would adjust process affinity
        # or throttle CPU usage for specific applications
    
    def _memory_cleanup(self, allocation):
        """Perform memory cleanup operations."""
        print(f"    Action: Initiating memory cleanup and optimization")
        # In a real implementation, this would trigger system cleanup
        # or memory management operations
    
    def _optimize_boost(self, allocation):
        """Boost performance when system is idle."""
        print(f"    Action: Optimizing for performance boost")
        # In a real implementation, this would enable performance modes
        # or adjust process priorities
    
    def _defrag_scheduled(self, allocation):
        """Schedule disk defragmentation."""
        print(f"    Action: Scheduling disk optimization")
        # In a real implementation, this would schedule defrag operations
    
    def _default_action(self, allocation):
        """Default action when specific action is not found."""
        print(f"    Action: Default optimization for {allocation.resource_type.value}")
    
    def optimize_once(self) -> int:
        """Run one optimization cycle."""
        # Collect telemetry
        telemetry = self.resource_manager.collect_telemetry()
        
        # Evaluate policies
        triggered = self._evaluate_policies(telemetry)
        
        executed_count = 0
        for policy in triggered:
            success = self._execute_policy(policy)
            if success:
                executed_count += 1
        
        # Print system status
        status = self.resource_manager.get_system_status()
        print(f"  System Status: CPU={telemetry.cpu_percent:.1f}%, "
              f"Memory={telemetry.memory_percent:.1f}%, "
              f"Processes={telemetry.process_count}")
        
        return executed_count
    
    def start_continuous_optimization(self, interval_seconds: float = 10.0):
        """Start continuous optimization in a background thread."""
        if self._running:
            print(f"[{self.agent_id}] Optimization already running")
            return
        
        self._running = True
        
        def optimization_loop():
            print(f"[{self.agent_id}] Starting continuous optimization (every {interval_seconds}s)")
            while self._running:
                try:
                    executed = self.optimize_once()
                    if executed > 0:
                        print(f"  Executed {executed} optimizations")
                    time.sleep(interval_seconds)
                except Exception as e:
                    print(f"[{self.agent_id}] Optimization error: {e}")
                    time.sleep(interval_seconds)
        
        self._thread = threading.Thread(target=optimization_loop, daemon=True)
        self._thread.start()
    
    def stop_optimization(self):
        """Stop continuous optimization."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        print(f"[{self.agent_id}] Optimization stopped")
    
    def get_status(self) -> Dict:
        """Get optimization agent status."""
        return {
            "agent_id": self.agent_id,
            "running": self._running,
            "active_allocations": len(self.active_allocations),
            "resource_status": self.resource_manager.get_agent_status(self.agent_id),
            "policies_count": len(self.policies)
        }


def demo_optimization_agent():
    """Demonstrate the Windows Optimization Agent."""
    print("=" * 80)
    print("WINDOWS OPTIMIZATION AGENT - INTELLIGENT RESOURCE MANAGEMENT")
    print("=" * 80)
    
    # Create optimization agent
    agent = WindowsOptimizationAgent("GamePerformanceAgent")
    print("[OK] Optimization agent created")

    # Show initial status
    status = agent.get_status()
    print("[OK] Initial status:")
    print(f"  - Policies: {status['policies_count']}")
    print(f"  - Agent credits: {status['resource_status']['credits']}")
    print(f"  - Active allocations: {status['active_allocations']}")
    
    # Run a few optimization cycles manually
    print("\n--- Manual Optimization Cycles ---")
    for i in range(3):
        print(f"\nCycle {i+1}:")
        executed = agent.optimize_once()
        print(f"  Completed cycle {i+1}, executed {executed} optimizations")
        time.sleep(2)
    
    # Show status after manual cycles
    status = agent.get_status()
    print(f"\n[OK] Status after manual optimization:")
    print(f"  - Agent credits: {status['resource_status']['credits']:.1f}")
    print(f"  - Resources used: {status['resource_status']['total_resources_used']:.1f}")
    
    # Demonstrate policy configuration
    print("\n--- Policy Configuration ---")
    print("Current policies:")
    for i, policy in enumerate(agent.policies):
        print(f"  {i+1}. {policy.name}: {policy.resource_type.value} > {policy.threshold}% "
              f"-> {policy.action} (bid: {policy.bid_credits} credits)")
    
    # Add a custom policy
    custom_policy = OptimizationPolicy(
        name="Custom Performance",
        resource_type=ResourceType.NETWORK,
        threshold=50.0,
        action="optimize_network",
        bid_credits=15.0
    )
    agent.policies.append(custom_policy)
    print(f"  + Added custom policy: {custom_policy.name}")
    
    # Show system resource availability
    print("\n--- System Resource Status ---")
    sys_status = agent.resource_manager.get_system_status()
    for resource, stats in sys_status["resource_pools"].items():
        print(f"  {resource}: {stats['allocated']:.1f}/{stats['total']:.1f} "
              f"({stats['allocation_rate']:.1%} allocated)")
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION AGENT DEMONSTRATION COMPLETE")
    print("This agent can run continuously to optimize Windows systems")
    print("using economic resource trading principles")
    print("=" * 80)


def demo_continuous_optimization():
    """Demonstrate continuous optimization (short run)."""
    print("\n" + "=" * 80)
    print("CONTINUOUS OPTIMIZATION DEMO")
    print("=" * 80)
    
    agent = WindowsOptimizationAgent("ContinuousOptimizer")
    
    print("Starting 30 seconds of continuous optimization...")
    agent.start_continuous_optimization(interval_seconds=5.0)
    
    # Let it run for a bit
    time.sleep(15)  # Only run for 15 seconds to keep demo short
    
    agent.stop_optimization()
    status = agent.get_status()
    print(f"Stopped. Final status: {status['active_allocations']} allocations")
    
    print("\nThis framework enables sophisticated Windows system optimization")
    print("with economic resource trading, UUID process tracking, and intelligent scheduling")
    print("=" * 80)


if __name__ == "__main__":
    demo_optimization_agent()
    demo_continuous_optimization()