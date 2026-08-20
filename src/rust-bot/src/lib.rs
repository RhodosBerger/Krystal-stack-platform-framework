//! GAMESA Rust Bot - Trusted Decision/Telemetry Engine
//!
//! Middle layer that Python guardian or C runtime can query for:
//! - Trusted decisions (validated, signed responses)
//! - Telemetry aggregation and validation
//! - Policy evaluation with formal guarantees
//! - Feature registry and adaptation

pub mod types;
pub mod config;
pub mod events;
pub mod trust;
pub mod cpu_governor;
pub mod gpu_telemetry;
pub mod grid_engine;
pub mod policy;
pub mod micro_inference;
pub mod feature_registry;
pub mod time_series;
pub mod orchestration;
pub mod combat_latency_policy;
pub mod vulkan_inspector;
pub mod stack_injector;
pub mod grid_injector;
pub mod fallback;
pub mod unified_core;

pub use types::*;
pub use config::Config;
pub use events::{Event, EventBus};
pub use trust::{TrustEngine, TrustedDecision};
pub use cpu_governor::CpuGovernor;
pub use gpu_telemetry::GpuTelemetry;
pub use grid_engine::{GridEngine, HexCell};
pub use policy::{PolicyEngine, PolicyResult};
pub use micro_inference::MicroInference;
pub use feature_registry::FeatureRegistry;
pub use time_series::TimeSeriesLogger;
pub use orchestration::{Orchestrator, CycleResult, OrchestratorStats};
pub use combat_latency_policy::{CombatLatencyPolicy, PinningDirective};
pub use vulkan_inspector::{VulkanInspector, SceneAnalysis, GovernorSuggestion};
pub use stack_injector::{StackInjector, BoostInjection};
pub use grid_injector::{GridInjector, GridReplica, GridCell as InjectorCell};
pub use fallback::{FallbackManager, TrustLevel};
pub use unified_core::{UnifiedCore, HardwareTelemetry, HardwarePreset, Phase, SystemMode};

/// Main orchestrator combining all components
pub struct GamesaBot {
    pub config: Config,
    pub trust: TrustEngine,
    pub events: EventBus,
    pub cpu: CpuGovernor,
    pub gpu: GpuTelemetry,
    pub grid: GridEngine,
    pub policy: PolicyEngine,
    pub inference: MicroInference,
    pub features: FeatureRegistry,
    pub logger: TimeSeriesLogger,
}

impl GamesaBot {
    pub fn new(config: Config) -> Self {
        Self {
            trust: TrustEngine::new(&config),
            events: EventBus::new(1000),
            cpu: CpuGovernor::new(),
            gpu: GpuTelemetry::new(),
            grid: GridEngine::new(8, 8, 8),
            policy: PolicyEngine::new(),
            inference: MicroInference::new(),
            features: FeatureRegistry::new(),
            logger: TimeSeriesLogger::new(&config.log_path),
            config,
        }
    }

    /// Process telemetry and return trusted decision
    pub fn process(&mut self, telemetry: TelemetrySnapshot) -> TrustedDecision {
        // Log telemetry
        self.logger.log(&telemetry);

        // Update subsystems
        self.cpu.update(&telemetry);
        self.gpu.update(&telemetry);

        // Evaluate policy
        let policy_result = self.policy.evaluate(&telemetry, &self.features);

        // Run micro inference if needed
        let inference_hint = if policy_result.needs_inference {
            Some(self.inference.predict(&telemetry))
        } else {
            None
        };

        // Generate trusted decision
        self.trust.create_decision(
            policy_result,
            inference_hint,
            &telemetry,
        )
    }

    /// Get current system state
    pub fn get_state(&self) -> SystemState {
        SystemState {
            cpu_state: self.cpu.get_state(),
            gpu_state: self.gpu.get_state(),
            grid_state: self.grid.get_summary(),
            feature_flags: self.features.get_enabled(),
        }
    }
}
