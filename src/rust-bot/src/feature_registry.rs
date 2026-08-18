//! Feature Registry - Feature flags and toggles

use std::collections::HashMap;

pub struct FeatureRegistry {
    flags: HashMap<String, bool>,
}

impl FeatureRegistry {
    pub fn new() -> Self {
        let mut flags = HashMap::new();
        flags.insert("thermal_prediction".into(), true);
        flags.insert("genetic_presets".into(), true);
        flags.insert("micro_inference".into(), true);
        flags.insert("shared_memory_ipc".into(), true);
        flags.insert("zone_migration".into(), true);
        flags.insert("power_monitoring".into(), true);
        Self { flags }
    }

    pub fn is_enabled(&self, name: &str) -> bool {
        self.flags.get(name).copied().unwrap_or(false)
    }

    pub fn enable(&mut self, name: &str) {
        self.flags.insert(name.into(), true);
    }

    pub fn disable(&mut self, name: &str) {
        self.flags.insert(name.into(), false);
    }

    pub fn get_enabled(&self) -> Vec<String> {
        self.flags.iter()
            .filter(|(_, &v)| v)
            .map(|(k, _)| k.clone())
            .collect()
    }
}

impl Default for FeatureRegistry {
    fn default() -> Self { Self::new() }
}
