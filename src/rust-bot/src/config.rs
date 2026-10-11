//! Configuration for GAMESA Rust Bot

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub log_path: PathBuf,
    pub ipc_socket: String,
    pub trust_key_path: Option<PathBuf>,
    pub thermal_limits: ThermalLimits,
    pub policy_config: PolicyConfig,
    pub feature_flags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalLimits {
    pub cpu_warning: f64,
    pub cpu_critical: f64,
    pub gpu_warning: f64,
    pub gpu_critical: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyConfig {
    pub enable_boost: bool,
    pub enable_throttle: bool,
    pub min_fps_target: f64,
    pub power_budget_w: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            log_path: PathBuf::from("/var/log/gamesa"),
            ipc_socket: "/tmp/gamesa.sock".into(),
            trust_key_path: None,
            thermal_limits: ThermalLimits::default(),
            policy_config: PolicyConfig::default(),
            feature_flags: vec!["thermal_prediction".into(), "genetic_presets".into()],
        }
    }
}

impl Default for ThermalLimits {
    fn default() -> Self {
        Self {
            cpu_warning: 85.0,
            cpu_critical: 95.0,
            gpu_warning: 80.0,
            gpu_critical: 90.0,
        }
    }
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            enable_boost: true,
            enable_throttle: true,
            min_fps_target: 60.0,
            power_budget_w: 150.0,
        }
    }
}

impl Config {
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        Ok(toml::from_str(&content)?)
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}
