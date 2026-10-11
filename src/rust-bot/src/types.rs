//! Core types shared across the Rust bot

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Telemetry snapshot from C runtime or Python guardian
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetrySnapshot {
    pub timestamp: DateTime<Utc>,
    pub cpu_util: f64,
    pub gpu_util: f64,
    pub memory_util: f64,
    pub temp_cpu: f64,
    pub temp_gpu: f64,
    pub frametime_ms: f64,
    pub power_draw: f64,
    pub zone_id: u32,
    pub pe_core_mask: u32,
}

/// CPU state from governor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuState {
    pub frequency_mhz: u32,
    pub governor: String,
    pub p_cores_active: u32,
    pub e_cores_active: u32,
    pub load_average: [f64; 3],
}

/// GPU state from telemetry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuState {
    pub clock_mhz: u32,
    pub vram_used_mb: u64,
    pub vram_total_mb: u64,
    pub temperature: f64,
    pub power_draw_w: f64,
    pub utilization: f64,
}

/// Grid summary for hex engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridSummary {
    pub dimensions: (usize, usize, usize),
    pub active_cells: usize,
    pub total_cells: usize,
    pub hottest_cell: Option<(usize, usize, usize)>,
}

/// Complete system state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemState {
    pub cpu_state: CpuState,
    pub gpu_state: GpuState,
    pub grid_state: GridSummary,
    pub feature_flags: Vec<String>,
}

/// Preset configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresetConfig {
    pub id: Uuid,
    pub name: String,
    pub cpu_multiplier: f64,
    pub gpu_offset_mhz: i32,
    pub power_limit_percent: f64,
    pub thermal_limit: f64,
    pub priority: u8,
}

/// Action to take
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Action {
    Boost { zone_id: u32, strength: f64 },
    Throttle { zone_id: u32, target: f64 },
    Migrate { zone_id: u32, target_cores: u32 },
    ApplyPreset { preset: PresetConfig },
    Idle { zone_id: u32 },
    NoOp,
}

/// Domain for signal classification
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum Domain {
    Safety = 0,
    Thermal = 1,
    User = 2,
    Performance = 3,
    Power = 4,
}

/// Signal kind
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignalKind {
    ThermalWarning,
    FrametimeSpike,
    CpuBottleneck,
    GpuBottleneck,
    IdleDetected,
    UserBoostRequest,
    PowerLimit,
}

impl TelemetrySnapshot {
    pub fn fps(&self) -> f64 {
        1000.0 / self.frametime_ms.max(0.001)
    }

    pub fn thermal_headroom(&self) -> f64 {
        let max_temp = self.temp_cpu.max(self.temp_gpu);
        ((95.0 - max_temp) / 25.0).clamp(0.0, 1.0)
    }

    pub fn is_bottlenecked(&self) -> Option<&'static str> {
        if self.cpu_util > 0.95 && self.gpu_util < 0.7 {
            Some("cpu")
        } else if self.gpu_util > 0.95 && self.cpu_util < 0.7 {
            Some("gpu")
        } else {
            None
        }
    }
}
