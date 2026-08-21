#!/usr/bin/env python3
"""
GAMESA Breaking Script - Complete System Integration (Wave 2)

Real-time optimization for ALL applications across ALL platforms.
Integrates unified_brain, app_optimizer, universal_platform, kernel_tuning,
derived_features, recurrent_logic, generators, and advanced_allocation.

Wave 2 Features:
- Visual Dashboard: Real-time web monitoring
- Crystal Core: Hexadecimal memory pool (0x7FFF0000)
- Neural Optimizer: On-device learning

Usage:
    python breakingscript.py                    # Run with defaults
    python breakingscript.py --status           # Show platform info
    python breakingscript.py --benchmark        # Run benchmarks
    python breakingscript.py --dashboard        # Enable web dashboard
    python breakingscript.py --neural           # Enable neural optimizer
    python breakingscript.py --crystal          # Enable Crystal Core
    python breakingscript.py --wave2            # Enable all Wave 2 features
"""

import sys
import os
import time
import signal
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Import all GAMESA modules
from universal_platform import create_universal_platform
from app_optimizer import create_app_optimizer
from kernel_tuning import KernelTuner, safe_smt_gate
from derived_features import create_derived_system
from recurrent_logic import create_recurrent_system
from generators import create_generator_system
from advanced_allocation import create_allocation_system
from unified_brain import UnifiedBrain
from metrics_logger import FeatureFlags
from platform_hal import get_safety_profile

# Wave 2 modules (optional)
try:
    from visual_dashboard import create_dashboard, DashboardMetrics
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    print("Note: Visual Dashboard not available (missing Flask/SocketIO)")

try:
    from crystal_core import get_crystal_core, get_cache_manager
    CRYSTAL_AVAILABLE = True
except ImportError:
    CRYSTAL_AVAILABLE = False
    print("Note: Crystal Core not available")

try:
    from neural_optimizer import create_neural_optimizer
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False
    print("Note: Neural Optimizer not available (missing numpy)")


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("GAMESA")


class GAMESAOptimizer:
    """Complete GAMESA optimization system with Wave 2 features."""

    def __init__(self, enable_dashboard=False, enable_neural=False, enable_crystal=False):
        logger.info("=" * 70)
        logger.info(" GAMESA - Generalized Adaptive Management & Execution System")
        logger.info(" Wave 2: Advanced Intelligence & Distributed Systems")
        logger.info("=" * 70)

        # Detect platform
        logger.info("[1/10] Detecting platform...")
        self.platform = create_universal_platform()
        info = self.platform.get_platform_info()
        logger.info(f"  Platform: {info['architecture']} / {info['vendor']}")
        logger.info(f"  CPU: {info['model']}")
        logger.info(f"  Cores: {info['cores']} physical, {info['threads']} logical")
        logger.info(f"  SIMD: {info['simd_width']}-bit")
        logger.info(f"  Accelerators: {', '.join(info['accelerators'])}")

        # Get safety profile
        self.safety = get_safety_profile(info['model'])
        logger.info(f"  Safety: {self.safety.name}")
        logger.info(f"    Thermal throttle: {self.safety.thermal_throttle}°C")
        logger.info(f"    TDP max: {self.safety.tdp_max}W")

        # Initialize brain
        logger.info("[2/10] Initializing Unified Brain...")
        flags = FeatureFlags.production()  # Safe defaults
        self.brain = UnifiedBrain(flags=flags)

        # Application optimizer
        logger.info("[3/10] Initializing Application Optimizer...")
        self.app_opt = create_app_optimizer()

        # Kernel tuning (non-invasive recommendations only)
        logger.info("[4/10] Initializing Kernel Tuner...")
        self.kernel = KernelTuner()
        logger.info(f"  Topology: {self.kernel.topology.physical_cores} cores, SMT={self.kernel.topology.smt_enabled}")

        # Derived features
        logger.info("[5/10] Initializing Derived Features...")
        self.derived = create_derived_system()

        # Recurrent logic
        logger.info("[6/10] Initializing Recurrent Logic...")
        self.recurrent = create_recurrent_system()

        # Allocation system
        logger.info("[7/10] Initializing Allocation System...")
        self.allocation = create_allocation_system()

        # Wave 2: Crystal Core
        self.crystal = None
        self.cache_manager = None
        if enable_crystal and CRYSTAL_AVAILABLE:
            logger.info("[8/10] Initializing Crystal Core Memory Pool...")
            self.crystal = get_crystal_core()
            self.cache_manager = get_cache_manager()
            logger.info(f"  Crystal Core ready at 0x{self.crystal.BASE_ADDRESS:08X}")
        else:
            logger.info("[8/10] Crystal Core disabled")

        # Wave 2: Neural Optimizer
        self.neural = None
        if enable_neural and NEURAL_AVAILABLE:
            logger.info("[9/10] Initializing Neural Optimizer...")
            self.neural = create_neural_optimizer()
            logger.info("  Neural models: Thermal predictor, Policy network, Anomaly detector")
        else:
            logger.info("[9/10] Neural Optimizer disabled")

        # Wave 2: Visual Dashboard
        self.dashboard = None
        self.dashboard_thread = None
        if enable_dashboard and DASHBOARD_AVAILABLE:
            logger.info("[10/10] Initializing Visual Dashboard...")
            self.dashboard = create_dashboard(port=8080)
            logger.info("  Dashboard will be available at http://localhost:8080")
        else:
            logger.info("[10/10] Visual Dashboard disabled")

        logger.info("Initialization complete!\n")

        self.running = False
        self.cycle_count = 0
        self.prev_state = None  # For neural reward calculation

    def collect_telemetry(self) -> Dict[str, float]:
        """Collect system telemetry."""
        telemetry = self.platform.telemetry.collect()

        # Add defaults for missing values
        defaults = {
            "temperature": 65.0,
            "thermal_headroom": 20.0,
            "power_draw": self.safety.tdp_sustained,
            "cpu_util": 0.5,
            "gpu_util": 0.3,
            "memory_util": 0.6,
            "fps": 60.0,
            "latency": 10.0,
        }

        for key, val in defaults.items():
            if key not in telemetry or telemetry[key] is None:
                telemetry[key] = val

        # Calculate thermal headroom
        if "temperature" in telemetry:
            temp = telemetry["temperature"]
            telemetry["thermal_headroom"] = max(0, self.safety.thermal_throttle - temp)

        return telemetry

    def process_cycle(self):
        """Run one optimization cycle with Wave 2 features."""
        self.cycle_count += 1

        # Collect telemetry
        telemetry = self.collect_telemetry()

        # Process through brain
        brain_result = self.brain.process(telemetry)
        decision = brain_result["decision"]

        # Process derived features
        derived_result = self.derived.process(telemetry)

        # Recurrent predictions
        recurrent_result = self.recurrent.process(telemetry)

        # Application analysis
        app_status = self.app_opt.get_system_status()

        # Kernel recommendations
        smt_rec = safe_smt_gate(telemetry.get("thermal_headroom", 20))

        # Update allocation
        self.allocation.update_telemetry(telemetry)

        # Wave 2: Neural Optimizer
        neural_result = None
        if self.neural:
            neural_result = self.neural.process(telemetry)

            # Provide reward for policy learning
            if self.prev_state is not None:
                reward = self._compute_neural_reward(telemetry, derived_result)
                # Convert state to normalized array for neural network
                current_state = self._telemetry_to_state(telemetry)
                prev_state_array = self._telemetry_to_state(self.prev_state)
                action = neural_result.get("neural_action", "noop")
                self.neural.provide_reward(reward, prev_state_array, action, current_state)

            self.prev_state = telemetry.copy()

        # Wave 2: Crystal Core - Store telemetry in shared memory
        if self.crystal and self.cycle_count % 5 == 0:
            # Every 5 cycles, persist critical telemetry to Crystal Core
            telemetry_bytes = str(telemetry).encode('utf-8')
            block_id = self.crystal.allocate(len(telemetry_bytes), owner="gamesa_telemetry")
            if block_id:
                self.crystal.write(block_id, telemetry_bytes)

        # Wave 2: Visual Dashboard
        if self.dashboard:
            self._update_dashboard(telemetry, decision, derived_result, app_status)

        # Log every 10 cycles
        if self.cycle_count % 10 == 0:
            self._log_status(telemetry, decision, derived_result, smt_rec, app_status, neural_result)

        return {
            "telemetry": telemetry,
            "brain": decision,
            "derived": derived_result,
            "recurrent": recurrent_result,
            "apps": app_status,
            "kernel": smt_rec,
            "neural": neural_result,
        }

    def _log_status(self, telemetry, decision, derived, smt, apps, neural=None):
        """Log current status."""
        logger.info(f"=== Cycle {self.cycle_count} ===")
        logger.info(f"  Temp: {telemetry.get('temperature', 0):.1f}°C "
                   f"(headroom: {telemetry.get('thermal_headroom', 0):.1f}°C)")
        logger.info(f"  CPU: {telemetry.get('cpu_util', 0)*100:.0f}%  "
                   f"GPU: {telemetry.get('gpu_util', 0)*100:.0f}%  "
                   f"Power: {telemetry.get('power_draw', 0):.1f}W")
        logger.info(f"  Brain: {decision['action']} (source: {decision['source']})")
        logger.info(f"  Game State: {derived['game']['state']}, "
                   f"Power: {derived['power']['state']}")
        logger.info(f"  Thermal: {derived['thermal']['action']}, "
                   f"Anomalies: {derived['anomaly']['anomaly_count']}")
        logger.info(f"  SMT: {smt['action']} - {smt['reason']}")
        logger.info(f"  Apps: {len(apps['by_category'])} categories active")

        # Wave 2 features
        if neural:
            logger.info(f"  Neural: {neural.get('neural_action', 'N/A')}, "
                       f"Anomaly: {neural.get('anomaly', False)}")
        if self.crystal:
            stats = self.crystal.get_stats()
            logger.info(f"  Crystal: {stats['utilization']*100:.1f}% used, "
                       f"{stats['blocks']} blocks")
        logger.info("")

    def _compute_neural_reward(self, telemetry: Dict, derived: Dict) -> float:
        """Compute reward for neural policy learning."""
        reward = 0.0

        # Thermal reward (stay cool)
        thermal_headroom = telemetry.get("thermal_headroom", 20)
        if thermal_headroom > 15:
            reward += 0.3
        elif thermal_headroom < 5:
            reward -= 0.2

        # Performance reward (high FPS, low latency)
        fps = telemetry.get("fps", 60)
        if fps > 60:
            reward += 0.2

        # Power efficiency
        power = telemetry.get("power_draw", 20)
        if power < self.safety.tdp_sustained:
            reward += 0.1

        # Anomaly penalty
        if derived["anomaly"]["anomaly_count"] > 0:
            reward -= 0.3

        return max(-1.0, min(1.0, reward))

    def _telemetry_to_state(self, telemetry: Dict):
        """Convert telemetry dict to numpy array for neural network."""
        try:
            import numpy as np
            return np.array([
                telemetry.get("temperature", 70) / 100.0,
                telemetry.get("thermal_headroom", 20) / 30.0,
                telemetry.get("cpu_util", 0.5),
                telemetry.get("gpu_util", 0.5),
                telemetry.get("power_draw", 20) / 30.0,
                telemetry.get("fps", 60) / 120.0,
                telemetry.get("latency", 10) / 50.0,
                telemetry.get("memory_util", 0.6),
            ], dtype='float32')
        except ImportError:
            return None

    def _update_dashboard(self, telemetry: Dict, decision: Dict, derived: Dict, apps: Dict):
        """Update visual dashboard with latest metrics."""
        if not DASHBOARD_AVAILABLE:
            return

        metrics = DashboardMetrics(
            timestamp=time.time(),
            temperature=telemetry.get("temperature", 65),
            thermal_headroom=telemetry.get("thermal_headroom", 20),
            cpu_util=telemetry.get("cpu_util", 0.5),
            gpu_util=telemetry.get("gpu_util", 0.3),
            memory_util=telemetry.get("memory_util", 0.6),
            power_draw=telemetry.get("power_draw", 20),
            fps=telemetry.get("fps", 60),
            latency=telemetry.get("latency", 10),
            brain_decision=decision.get("action", "unknown"),
            brain_source=decision.get("source", "unknown"),
            game_state=derived["game"]["state"],
            power_state=derived["power"]["state"],
            thermal_action=derived["thermal"]["action"],
            anomaly_count=derived["anomaly"]["anomaly_count"],
            active_apps=len(apps.get("by_category", {})),
        )

        self.dashboard.update_metrics(metrics)

    def run(self, duration_sec: Optional[int] = None):
        """Run optimization loop."""
        logger.info("Starting GAMESA optimization loop...")
        if self.dashboard:
            logger.info("Dashboard available at http://localhost:8080")
        logger.info("Press Ctrl+C to stop\n")

        self.running = True
        start_time = time.time()

        # Start dashboard in background if enabled
        if self.dashboard:
            self.dashboard_thread = threading.Thread(
                target=lambda: self.dashboard.run(host='0.0.0.0', debug=False),
                daemon=True
            )
            self.dashboard_thread.start()
            time.sleep(2)  # Give dashboard time to start

        try:
            while self.running:
                self.process_cycle()
                time.sleep(1.0)

                if duration_sec and (time.time() - start_time) >= duration_sec:
                    logger.info(f"Duration limit reached ({duration_sec}s)")
                    break

        except KeyboardInterrupt:
            logger.info("\nKeyboard interrupt - stopping...")
        finally:
            self.stop()

    def stop(self):
        """Stop optimization."""
        self.running = False

        # Cleanup Wave 2 resources
        if self.dashboard:
            logger.info("Stopping dashboard...")
            self.dashboard.stop()

        if self.crystal:
            logger.info("Cleaning up Crystal Core...")
            stats = self.crystal.get_stats()
            logger.info(f"  Final stats: {stats['allocations']} allocations, "
                       f"{stats['utilization']*100:.1f}% peak usage")
            self.crystal.cleanup()

        logger.info(f"GAMESA stopped after {self.cycle_count} cycles")

    def benchmark(self):
        """Run benchmarks."""
        from benchmark_harness import BenchmarkHarness

        logger.info("Running GAMESA benchmarks...\n")
        harness = BenchmarkHarness()
        results = harness.run_all()

        print("\n" + harness.generate_report())


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="GAMESA Breaking Script - Wave 2 Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python breakingscript.py --status                # Show platform info
  python breakingscript.py --benchmark             # Run performance tests
  python breakingscript.py --wave2                 # Run with all Wave 2 features
  python breakingscript.py --dashboard --duration 60  # Dashboard for 60 seconds
  python breakingscript.py --neural --crystal      # Neural + Crystal Core
        """
    )

    # Core options
    parser.add_argument("--status", action="store_true",
                       help="Show platform status and exit")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run benchmarks")
    parser.add_argument("--duration", type=int,
                       help="Run for N seconds")

    # Wave 2 features
    parser.add_argument("--dashboard", action="store_true",
                       help="Enable visual dashboard (http://localhost:8080)")
    parser.add_argument("--neural", action="store_true",
                       help="Enable neural optimizer (requires numpy)")
    parser.add_argument("--crystal", action="store_true",
                       help="Enable Crystal Core memory pool")
    parser.add_argument("--wave2", action="store_true",
                       help="Enable all Wave 2 features")

    args = parser.parse_args()

    # Wave 2 mode enables all features
    if args.wave2:
        args.dashboard = True
        args.neural = True
        args.crystal = True

    # Create optimizer with Wave 2 features
    optimizer = GAMESAOptimizer(
        enable_dashboard=args.dashboard,
        enable_neural=args.neural,
        enable_crystal=args.crystal
    )

    if args.status:
        logger.info("Status check complete - see above")
        return

    if args.benchmark:
        optimizer.benchmark()
        return

    optimizer.run(duration_sec=args.duration)


if __name__ == "__main__":
    main()
