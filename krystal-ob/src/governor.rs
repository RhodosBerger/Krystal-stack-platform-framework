use crate::telemetry::TelemetryData;

pub struct EconomicGovernor;

impl EconomicGovernor {
    pub fn new() -> Self {
        Self
    }

    /// Governor Check pre udelenie pretaktovania
    pub fn is_overclock_allowed(&self, data: &TelemetryData) -> bool {
        let temp_ok = data.gpu_temp < 68.0;
        let power_ok = data.power_percent < 80.0;
        let idle_ok = data.idle_percent > 85.0;
        // Zabezpečuje stiahnutie výkonu pre edge úlohy s prioritou (napr. vizion AI model nad 5 priority)
        let priority_ok = data.edge_task_priority <= 5; 

        temp_ok && power_ok && idle_ok && priority_ok
    }
}
