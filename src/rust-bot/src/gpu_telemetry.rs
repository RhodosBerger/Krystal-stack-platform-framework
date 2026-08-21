//! GPU Telemetry - Graphics card monitoring

use crate::{TelemetrySnapshot, GpuState};

pub struct GpuTelemetry {
    state: GpuState,
    vram_history: Vec<u64>,
}

impl GpuTelemetry {
    pub fn new() -> Self {
        Self {
            state: GpuState {
                clock_mhz: 1800,
                vram_used_mb: 0,
                vram_total_mb: 8192,
                temperature: 50.0,
                power_draw_w: 0.0,
                utilization: 0.0,
            },
            vram_history: Vec::with_capacity(60),
        }
    }

    pub fn update(&mut self, telemetry: &TelemetrySnapshot) {
        self.state.utilization = telemetry.gpu_util;
        self.state.temperature = telemetry.temp_gpu;
        self.state.power_draw_w = telemetry.power_draw * 0.7; // GPU portion
        self.state.vram_used_mb = (telemetry.memory_util * self.state.vram_total_mb as f64) as u64;

        self.vram_history.push(self.state.vram_used_mb);
        if self.vram_history.len() > 60 {
            self.vram_history.remove(0);
        }
    }

    pub fn get_state(&self) -> GpuState {
        self.state.clone()
    }

    pub fn vram_pressure(&self) -> f64 {
        self.state.vram_used_mb as f64 / self.state.vram_total_mb as f64
    }

    pub fn is_thermal_throttling(&self) -> bool {
        self.state.temperature > 83.0
    }
}

impl Default for GpuTelemetry {
    fn default() -> Self {
        Self::new()
    }
}
