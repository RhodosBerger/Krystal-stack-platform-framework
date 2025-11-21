//! CPU Governor - Frequency and core management

use crate::{TelemetrySnapshot, CpuState};

pub struct CpuGovernor {
    state: CpuState,
    history: Vec<f64>,
}

impl CpuGovernor {
    pub fn new() -> Self {
        Self {
            state: CpuState {
                frequency_mhz: 3000,
                governor: "performance".into(),
                p_cores_active: 8,
                e_cores_active: 8,
                load_average: [0.0, 0.0, 0.0],
            },
            history: Vec::with_capacity(60),
        }
    }

    pub fn update(&mut self, telemetry: &TelemetrySnapshot) {
        self.history.push(telemetry.cpu_util);
        if self.history.len() > 60 {
            self.history.remove(0);
        }

        // Update load average simulation
        let avg = self.history.iter().sum::<f64>() / self.history.len() as f64;
        self.state.load_average = [avg, avg * 0.9, avg * 0.8];
    }

    pub fn get_state(&self) -> CpuState {
        self.state.clone()
    }

    pub fn set_governor(&mut self, gov: &str) {
        self.state.governor = gov.to_string();
    }

    pub fn set_pe_ratio(&mut self, p_ratio: f64) {
        let total = self.state.p_cores_active + self.state.e_cores_active;
        self.state.p_cores_active = (total as f64 * p_ratio) as u32;
        self.state.e_cores_active = total - self.state.p_cores_active;
    }
}

impl Default for CpuGovernor {
    fn default() -> Self {
        Self::new()
    }
}
