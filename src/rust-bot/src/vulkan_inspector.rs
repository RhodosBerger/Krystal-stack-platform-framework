//! Vulkan Inspector - Scene Analysis for GPU Workloads

use crate::TelemetrySnapshot;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneAnalysis {
    pub draw_calls: u32,
    pub triangles: u64,
    pub textures_bound: u32,
    pub compute_dispatches: u32,
    pub render_passes: u32,
    pub complexity_score: f64,
    pub suggested_governor: GovernorSuggestion,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum GovernorSuggestion {
    Boost,
    Balanced,
    Cooldown,
}

impl SceneAnalysis {
    pub fn is_combat_likely(&self) -> bool {
        // High draw calls + particles usually means combat
        self.draw_calls > 5000 || self.complexity_score > 0.7
    }
}

pub struct VulkanInspector {
    last_analysis: Option<SceneAnalysis>,
    frame_history: Vec<f64>,
}

impl VulkanInspector {
    pub fn new() -> Self {
        Self {
            last_analysis: None,
            frame_history: Vec::with_capacity(60),
        }
    }

    pub fn inspect_scene(&mut self, telemetry: &TelemetrySnapshot) -> SceneAnalysis {
        // Estimate scene complexity from telemetry
        let complexity = (telemetry.gpu_util + (1.0 - telemetry.thermal_headroom())) / 2.0;

        self.frame_history.push(telemetry.frametime_ms);
        if self.frame_history.len() > 60 {
            self.frame_history.remove(0);
        }

        let suggested = if telemetry.temp_gpu > 80.0 {
            GovernorSuggestion::Cooldown
        } else if complexity > 0.7 {
            GovernorSuggestion::Boost
        } else {
            GovernorSuggestion::Balanced
        };

        let analysis = SceneAnalysis {
            draw_calls: (telemetry.gpu_util * 10000.0) as u32,
            triangles: (telemetry.gpu_util * 5_000_000.0) as u64,
            textures_bound: (telemetry.memory_util * 100.0) as u32,
            compute_dispatches: if telemetry.gpu_util > 0.5 { 50 } else { 10 },
            render_passes: 3,
            complexity_score: complexity,
            suggested_governor: suggested,
        };

        self.last_analysis = Some(analysis.clone());
        analysis
    }

    pub fn get_last_analysis(&self) -> Option<&SceneAnalysis> {
        self.last_analysis.as_ref()
    }
}

impl Default for VulkanInspector {
    fn default() -> Self { Self::new() }
}
