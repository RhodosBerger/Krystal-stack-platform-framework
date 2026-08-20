//! Combat Latency Policy - Thread Pinning for Gaming
//!
//! Detects combat scenarios and emits pinning directives:
//! - Main thread → P-cores
//! - Background threads → E-cores

use crate::TelemetrySnapshot;
use crate::vulkan_inspector::SceneAnalysis;
use serde::{Deserialize, Serialize};

/// Pinning directive for thread affinity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PinningDirective {
    /// Pin main thread to performance cores
    PinMainToP { core_mask: u32 },
    /// Pin background to efficiency cores
    PinBackgroundToE { core_mask: u32 },
    /// Pin to specific cores
    PinToCore { thread_id: u64, core: u32 },
    /// Release all pinning
    Release,
    /// No action needed
    NoAction,
}

/// Combat detection thresholds
#[derive(Debug, Clone)]
pub struct CombatThresholds {
    pub l3_miss_rate_high: f64,
    pub gpu_util_combat: f64,
    pub frametime_spike_ms: f64,
}

impl Default for CombatThresholds {
    fn default() -> Self {
        Self {
            l3_miss_rate_high: 0.15,     // 15% L3 miss = memory pressure
            gpu_util_combat: 0.8,         // 80% GPU = heavy rendering
            frametime_spike_ms: 20.0,     // >20ms = performance issue
        }
    }
}

pub struct CombatLatencyPolicy {
    thresholds: CombatThresholds,
    in_combat: bool,
    combat_start_time: Option<std::time::Instant>,
    p_core_mask: u32,
    e_core_mask: u32,
}

impl CombatLatencyPolicy {
    pub fn new() -> Self {
        Self {
            thresholds: CombatThresholds::default(),
            in_combat: false,
            combat_start_time: None,
            p_core_mask: 0x0F,  // Cores 0-3 (P-cores)
            e_core_mask: 0xF0,  // Cores 4-7 (E-cores)
        }
    }

    pub fn with_core_masks(p_cores: u32, e_cores: u32) -> Self {
        Self {
            p_core_mask: p_cores,
            e_core_mask: e_cores,
            ..Self::new()
        }
    }

    /// Evaluate telemetry and scene for combat scenario
    pub fn evaluate(
        &mut self,
        telemetry: &TelemetrySnapshot,
        scene: Option<&SceneAnalysis>,
    ) -> Option<PinningDirective> {
        let is_combat = self.detect_combat(telemetry, scene);

        if is_combat && !self.in_combat {
            // Entering combat
            self.in_combat = true;
            self.combat_start_time = Some(std::time::Instant::now());
            return Some(PinningDirective::PinMainToP {
                core_mask: self.p_core_mask,
            });
        } else if !is_combat && self.in_combat {
            // Exiting combat
            self.in_combat = false;
            self.combat_start_time = None;
            return Some(PinningDirective::Release);
        }

        None
    }

    fn detect_combat(&self, telemetry: &TelemetrySnapshot, scene: Option<&SceneAnalysis>) -> bool {
        // High GPU utilization indicates heavy rendering
        let gpu_intense = telemetry.gpu_util > self.thresholds.gpu_util_combat;

        // Frametime spike indicates performance pressure
        let frametime_spike = telemetry.frametime_ms > self.thresholds.frametime_spike_ms;

        // Scene complexity from Vulkan inspector
        let scene_complex = scene.map_or(false, |s| s.is_combat_likely());

        // Combat if GPU intense AND (frametime spike OR scene suggests combat)
        gpu_intense && (frametime_spike || scene_complex)
    }

    pub fn is_in_combat(&self) -> bool {
        self.in_combat
    }

    pub fn combat_duration_ms(&self) -> Option<u64> {
        self.combat_start_time.map(|t| t.elapsed().as_millis() as u64)
    }
}

impl Default for CombatLatencyPolicy {
    fn default() -> Self {
        Self::new()
    }
}
