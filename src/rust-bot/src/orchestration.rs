//! Orchestration - Full Cycle Driver
//!
//! Drives the complete GAMESA cycle:
//! 1. Check Python/C prerequisites
//! 2. Collect idle telemetry
//! 3. Run C demos when idle
//! 4. Inspect Vulkan scenes
//! 5. Apply combat latency policy
//! 6. Evaluate micro-inference rules
//! 7. Fetch grid replicas
//! 8. Call performance advisor
//! 9. Log all events

use crate::{
    TelemetrySnapshot, Action, Domain, CpuState, GpuState,
    CpuGovernor, GpuTelemetry, GridEngine, PolicyEngine, MicroInference,
    FeatureRegistry, TimeSeriesLogger, Config,
};
use crate::trust::{TrustEngine, TrustedDecision};
use crate::combat_latency_policy::{CombatLatencyPolicy, PinningDirective};
use crate::vulkan_inspector::{VulkanInspector, SceneAnalysis};
use crate::stack_injector::StackInjector;
use crate::grid_injector::GridInjector;
use crate::fallback::{FallbackManager, TrustLevel};
use crate::events::{Event, EventKind, EventBus};

use std::time::{Duration, Instant};
use chrono::Utc;

/// Orchestrator state
pub struct Orchestrator {
    // Core components
    pub config: Config,
    pub cpu: CpuGovernor,
    pub gpu: GpuTelemetry,
    pub grid: GridEngine,
    pub policy: PolicyEngine,
    pub inference: MicroInference,
    pub features: FeatureRegistry,
    pub trust: TrustEngine,
    pub logger: TimeSeriesLogger,
    pub events: EventBus,

    // Specialized modules
    pub combat_policy: CombatLatencyPolicy,
    pub vulkan: VulkanInspector,
    pub stack_injector: StackInjector,
    pub grid_injector: GridInjector,
    pub fallback: FallbackManager,

    // State
    last_cycle: Instant,
    cycle_count: u64,
    idle_window_ms: u64,
}

/// Cycle result
pub struct CycleResult {
    pub decision: TrustedDecision,
    pub pinning: Option<PinningDirective>,
    pub scene: Option<SceneAnalysis>,
    pub trust_level: TrustLevel,
    pub cycle_time_us: u64,
}

impl Orchestrator {
    pub fn new(config: Config) -> Self {
        Self {
            trust: TrustEngine::new(&config),
            cpu: CpuGovernor::new(),
            gpu: GpuTelemetry::new(),
            grid: GridEngine::new(8, 8, 8),
            policy: PolicyEngine::new(),
            inference: MicroInference::new(),
            features: FeatureRegistry::new(),
            logger: TimeSeriesLogger::new(&config.log_path),
            events: EventBus::new(1000),
            combat_policy: CombatLatencyPolicy::new(),
            vulkan: VulkanInspector::new(),
            stack_injector: StackInjector::new(),
            grid_injector: GridInjector::new(),
            fallback: FallbackManager::new(),
            last_cycle: Instant::now(),
            cycle_count: 0,
            idle_window_ms: 0,
            config,
        }
    }

    /// Run a full orchestration cycle
    pub fn run_cycle(&mut self, telemetry: TelemetrySnapshot) -> CycleResult {
        let cycle_start = Instant::now();
        self.cycle_count += 1;

        // 1. Check prerequisites
        let python_ok = self.check_python_bridge();
        let c_runtime_ok = self.check_c_runtime();

        // 2. Collect and validate telemetry
        let (validated_telemetry, trust_level) = self.validate_telemetry(&telemetry);
        self.cpu.update(&validated_telemetry);
        self.gpu.update(&validated_telemetry);

        // 3. Check idle window for C demo execution
        self.update_idle_window(&validated_telemetry);
        if self.is_safe_for_demo() && self.features.is_enabled("c_demos") {
            self.run_c_demo_if_idle();
        }

        // 4. Inspect Vulkan scene
        let scene = if self.features.is_enabled("vulkan_inspector") {
            Some(self.vulkan.inspect_scene(&validated_telemetry))
        } else {
            None
        };

        // 5. Apply combat latency policy
        let pinning = if self.features.is_enabled("combat_latency") {
            self.combat_policy.evaluate(&validated_telemetry, scene.as_ref())
        } else {
            None
        };

        // Log pinning directive
        if let Some(ref pin) = pinning {
            self.log_event(EventKind::Directive, &format!("Pinning: {:?}", pin));
        }

        // 6. Evaluate micro-inference rules
        let inference_action = if self.features.is_enabled("micro_inference") {
            Some(self.inference.predict(&validated_telemetry))
        } else {
            None
        };

        // 7. Fetch grid replicas from C
        if self.features.is_enabled("grid_sync") {
            self.sync_grid_state();
        }

        // 8. Get policy decision
        let policy_result = self.policy.evaluate(&validated_telemetry, &self.features);

        // 9. Create trusted decision (merging all inputs)
        let decision = self.trust.create_decision(
            policy_result,
            inference_action,
            &validated_telemetry,
        );

        // Apply trust level override if needed
        let final_decision = if trust_level == TrustLevel::Low {
            self.fallback.force_cooldown(&decision)
        } else {
            decision
        };

        // Log the cycle
        self.logger.log(&validated_telemetry);
        self.log_event(EventKind::Decision, &format!("Decision: {:?}", final_decision.action));

        let cycle_time = cycle_start.elapsed();

        CycleResult {
            decision: final_decision,
            pinning,
            scene,
            trust_level,
            cycle_time_us: cycle_time.as_micros() as u64,
        }
    }

    // Prerequisites

    fn check_python_bridge(&self) -> bool {
        if !self.features.is_enabled("python_bridge") {
            return false;
        }
        // Would check IPC socket/shared memory
        true
    }

    fn check_c_runtime(&self) -> bool {
        // Would check C runtime availability
        true
    }

    // Telemetry validation

    fn validate_telemetry(&mut self, telemetry: &TelemetrySnapshot) -> (TelemetrySnapshot, TrustLevel) {
        // Merge Python and Rust telemetry with fallback validation
        let trust_level = self.fallback.validate_telemetry(telemetry);

        // If divergence detected, use Rust-only telemetry
        let validated = if trust_level == TrustLevel::Low {
            self.fallback.get_fallback_telemetry(telemetry)
        } else {
            telemetry.clone()
        };

        (validated, trust_level)
    }

    // Idle window management

    fn update_idle_window(&mut self, telemetry: &TelemetrySnapshot) {
        if telemetry.cpu_util < 0.1 && telemetry.gpu_util < 0.1 {
            self.idle_window_ms += 16; // Assume 16ms tick
        } else {
            self.idle_window_ms = 0;
        }
    }

    fn is_safe_for_demo(&self) -> bool {
        self.idle_window_ms > 1000 // 1 second of idle
    }

    fn run_c_demo_if_idle(&mut self) {
        // Call C memory_tier_demo via stack injector
        self.stack_injector.run_memory_demo();
        self.log_event(EventKind::StateChange, "Ran memory_tier_demo during idle");
    }

    // Grid sync

    fn sync_grid_state(&mut self) {
        // Fetch grid replicas from C runtime
        if let Some(replica) = self.grid_injector.fetch_replica() {
            // Update local grid state
            for cell in replica.cells {
                self.grid.set_signal(cell.x, cell.y, cell.z, cell.signal);
            }
        }
    }

    // Event logging

    fn log_event(&mut self, kind: EventKind, message: &str) {
        self.events.emit(Event {
            timestamp: Utc::now(),
            kind,
            source: "orchestrator".into(),
            data: serde_json::json!({ "message": message }),
        });
    }

    // Statistics

    pub fn get_stats(&self) -> OrchestratorStats {
        OrchestratorStats {
            cycle_count: self.cycle_count,
            idle_window_ms: self.idle_window_ms,
            features_enabled: self.features.get_enabled(),
            cpu_state: self.cpu.get_state(),
            gpu_state: self.gpu.get_state(),
            grid_summary: self.grid.get_summary(),
        }
    }
}

#[derive(Debug)]
pub struct OrchestratorStats {
    pub cycle_count: u64,
    pub idle_window_ms: u64,
    pub features_enabled: Vec<String>,
    pub cpu_state: CpuState,
    pub gpu_state: GpuState,
    pub grid_summary: crate::GridSummary,
}
