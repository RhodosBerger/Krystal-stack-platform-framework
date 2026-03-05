use serde::{Serialize, Deserialize};
use nvml_wrapper::Nvml;
use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor};
use log::{warn, error};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryData {
    pub gpu_temp: f64,
    pub vram_temp: f64,
    pub power_percent: f64,
    pub idle_percent: f64,
    pub core_clock: i32,
    pub mem_clock: i32,
    pub fan_rpm: i32,
    pub cpu_temp: f64,
    pub voltage: f64,
    pub ambient_temp: f64,
    pub mining_hashrate: f64,
    pub rejected_shares_percent: f64,
    pub edge_task_priority: i32,
}

use std::sync::Arc;
use crate::mqtt::EdgeStatusState;
use std::sync::atomic::Ordering;

pub struct TelemetryCollector {
    nvml: Option<Nvml>,
    edge_state: Arc<EdgeStatusState>,
}

impl TelemetryCollector {
    pub fn new(edge_state: Arc<EdgeStatusState>) -> Self {
        let nvml = match Nvml::init() {
            Ok(n) => Some(n),
            Err(e) => {
                warn!("Nepodarilo sa inicializovat NVML: {}. Pouzijem fallbacks.", e);
                None
            }
        };
        Self { nvml, edge_state }
    }

    pub async fn collect_telemetry(&self) -> TelemetryData {
        let mut data = TelemetryData {
            gpu_temp: 64.0,
            vram_temp: 84.0,
            power_percent: 72.0,
            idle_percent: 88.0,
            core_clock: 1450,
            mem_clock: 7100,
            fan_rpm: 1800,
            cpu_temp: 42.0,
            voltage: 0.850,
            ambient_temp: 23.0,
            mining_hashrate: 15.2,
            rejected_shares_percent: 0.1,
            edge_task_priority: self.edge_state.priority.load(Ordering::Relaxed),
        };

        if let Some(ref nvml) = self.nvml {
            if let Ok(device) = nvml.device_by_index(0) {
                if let Ok(temp) = device.temperature(TemperatureSensor::Gpu) {
                    data.gpu_temp = temp as f64;
                }
                
                if let Ok(util) = device.utilization_rates() {
                    data.idle_percent = 100.0 - (util.gpu as f64);
                }
                
                if let Ok(power) = device.power_usage() {
                    if let Ok(limit) = device.enforced_power_limit() {
                        data.power_percent = (power as f64 / limit as f64) * 100.0;
                    }
                }
                
                if let Ok(core_clock) = device.clock_info(Clock::Graphics) {
                    data.core_clock = core_clock as i32;
                }
                
                if let Ok(mem_clock) = device.clock_info(Clock::Memory) {
                    data.mem_clock = mem_clock as i32;
                }
                
                if let Ok(fan) = device.fan_speed(0) {
                    data.fan_rpm = fan as i32;
                }
            }
        }
        
        // Všetko ostatné z Edge enviromentu sa pridáva externe alebo simuluje tu, 
        // napr HWMon CPU. Toto je mock priority pre MQTT ukážku z roadmapy.
        // Tieto budeme updatovať neskôr cez MQTT.
        
        data
    }
}
