//! Policy Engine - Rule evaluation and decision making

use crate::{TelemetrySnapshot, Action, Domain, FeatureRegistry};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyResult {
    pub action: Action,
    pub domain: Domain,
    pub confidence: f64,
    pub valid: bool,
    pub needs_inference: bool,
    pub rule_matched: Option<String>,
}

pub struct PolicyEngine {
    rules: Vec<PolicyRule>,
}

#[derive(Debug, Clone)]
struct PolicyRule {
    name: String,
    domain: Domain,
    condition: Box<dyn Fn(&TelemetrySnapshot) -> bool + Send + Sync>,
    action: Box<dyn Fn(&TelemetrySnapshot) -> Action + Send + Sync>,
    priority: u8,
}

impl PolicyEngine {
    pub fn new() -> Self {
        let mut engine = Self { rules: Vec::new() };
        engine.register_default_rules();
        engine
    }

    fn register_default_rules(&mut self) {
        // Safety: Thermal critical
        self.rules.push(PolicyRule {
            name: "thermal_critical".into(),
            domain: Domain::Safety,
            condition: Box::new(|t| t.temp_cpu > 95.0 || t.temp_gpu > 90.0),
            action: Box::new(|t| Action::Throttle { zone_id: t.zone_id, target: 0.3 }),
            priority: 0,
        });

        // Thermal: Warning
        self.rules.push(PolicyRule {
            name: "thermal_warning".into(),
            domain: Domain::Thermal,
            condition: Box::new(|t| t.temp_cpu > 85.0 || t.temp_gpu > 80.0),
            action: Box::new(|t| Action::Throttle { zone_id: t.zone_id, target: 0.7 }),
            priority: 1,
        });

        // Performance: Low FPS
        self.rules.push(PolicyRule {
            name: "fps_boost".into(),
            domain: Domain::Performance,
            condition: Box::new(|t| t.fps() < 60.0 && t.thermal_headroom() > 0.3),
            action: Box::new(|t| Action::Boost { zone_id: t.zone_id, strength: 0.8 }),
            priority: 3,
        });

        // Performance: CPU bottleneck
        self.rules.push(PolicyRule {
            name: "cpu_bottleneck".into(),
            domain: Domain::Performance,
            condition: Box::new(|t| t.is_bottlenecked() == Some("cpu")),
            action: Box::new(|t| Action::Migrate { zone_id: t.zone_id, target_cores: 0xFF }),
            priority: 3,
        });

        // Power: Idle
        self.rules.push(PolicyRule {
            name: "idle_detected".into(),
            domain: Domain::Power,
            condition: Box::new(|t| t.cpu_util < 0.1 && t.gpu_util < 0.1),
            action: Box::new(|t| Action::Idle { zone_id: t.zone_id }),
            priority: 4,
        });
    }

    pub fn evaluate(&self, telemetry: &TelemetrySnapshot, features: &FeatureRegistry) -> PolicyResult {
        // Sort by priority (lower = higher priority)
        let mut matched_rules: Vec<_> = self.rules.iter()
            .filter(|r| (r.condition)(telemetry))
            .collect();
        matched_rules.sort_by_key(|r| r.priority);

        if let Some(rule) = matched_rules.first() {
            PolicyResult {
                action: (rule.action)(telemetry),
                domain: rule.domain,
                confidence: 0.8,
                valid: true,
                needs_inference: features.is_enabled("micro_inference"),
                rule_matched: Some(rule.name.clone()),
            }
        } else {
            PolicyResult {
                action: Action::NoOp,
                domain: Domain::Performance,
                confidence: 0.5,
                valid: true,
                needs_inference: false,
                rule_matched: None,
            }
        }
    }
}

impl Default for PolicyEngine {
    fn default() -> Self {
        Self::new()
    }
}
