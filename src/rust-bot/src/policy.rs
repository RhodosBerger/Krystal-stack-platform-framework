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

    /// Evaluates multiple counterfactual "what if" scenarios against current telemetry.
    /// Returns the winning optimal projection and all evaluated candidate projections.
    pub fn evaluate_what_if(
        &self,
        telemetry: &TelemetrySnapshot,
        scenarios: &[WhatIfScenario],
        _features: &FeatureRegistry,
    ) -> (WhatIfProjection, Vec<WhatIfProjection>) {
        let mut projections = Vec::with_capacity(scenarios.len());
        let current_fps = telemetry.fps();
        let current_latency = telemetry.frametime_ms;

        for sc in scenarios {
            let is_max_fan = sc.hypothesis.to_uppercase().contains("MAX_FAN")
                || sc.hypothesis.to_uppercase().contains("CONTINUOUS_MAX")
                || sc.hypothesis.to_uppercase().contains("TERMAL_NA_PLNO");

            let (proj_temp, proj_fps, proj_lat, approved, veto) = if is_max_fan {
                let temp = (telemetry.temp_cpu - 18.0).max(35.0);
                let fps = current_fps * 1.25;
                let lat = (current_latency * 0.78).max(3.5);
                (temp, fps, lat, true, None)
            } else {
                match sc.action {
                    Action::Boost { strength, .. } => {
                        let temp = telemetry.temp_cpu + (strength * 7.5);
                        let fps = current_fps * (1.0 + strength * 0.20);
                        let lat = (current_latency * 0.85).max(4.0);
                        if temp > 95.0 {
                            (temp, fps, lat, false, Some("ThermalLimitExceeded".into()))
                        } else {
                            (temp, fps, lat, true, None)
                        }
                    }
                    Action::Throttle { target, .. } => {
                        let temp = (telemetry.temp_cpu - (1.0 - target) * 15.0).max(40.0);
                        let fps = current_fps * target;
                        let lat = current_latency * (1.0 + (1.0 - target) * 0.4);
                        (temp, fps, lat, true, None)
                    }
                    Action::Migrate { .. } => {
                        let temp = (telemetry.temp_cpu - 3.0).max(40.0);
                        let fps = current_fps * 1.08;
                        let lat = (current_latency - 1.2).max(4.0);
                        (temp, fps, lat, true, None)
                    }
                    Action::ApplyPreset { .. } => {
                        let temp = (telemetry.temp_cpu - 4.0).max(40.0);
                        let fps = current_fps * 1.10;
                        let lat = (current_latency - 1.5).max(4.0);
                        (temp, fps, lat, true, None)
                    }
                    Action::Idle { .. } => {
                        let temp = (telemetry.temp_cpu - 20.0).max(35.0);
                        let fps = 30.0;
                        let lat = 33.3;
                        (temp, fps, lat, true, None)
                    }
                    Action::NoOp => {
                        (telemetry.temp_cpu, current_fps, current_latency, true, None)
                    }
                }
            };

            // Calculate composite utility score: higher FPS + lower temp + lower latency
            let utility = if approved {
                (proj_fps * 0.6) + ((100.0 - proj_temp) * 0.4) - (proj_lat * 0.5) + (sc.expected_gain * 10.0)
            } else {
                -1000.0
            };

            projections.push(WhatIfProjection {
                scenario: sc.clone(),
                projected_temp_cpu: (proj_temp * 10.0).round() / 10.0,
                projected_fps: (proj_fps * 10.0).round() / 10.0,
                projected_latency_ms: (proj_lat * 100.0).round() / 100.0,
                utility_score: (utility * 100.0).round() / 100.0,
                safety_approved: approved,
                veto_reason: veto,
            });
        }

        // Select approved projection with highest utility
        let best = projections.iter()
            .filter(|p| p.safety_approved)
            .max_by(|a, b| a.utility_score.partial_cmp(&b.utility_score).unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
            .unwrap_or_else(|| {
                // Safe NoOp fallback if all candidate scenarios are vetoed
                WhatIfProjection {
                    scenario: WhatIfScenario {
                        action: Action::NoOp,
                        hypothesis: "FallbackSafeNoOp".into(),
                        domain: Domain::Safety,
                        expected_gain: 0.0,
                    },
                    projected_temp_cpu: telemetry.temp_cpu,
                    projected_fps: current_fps,
                    projected_latency_ms: current_latency,
                    utility_score: 0.0,
                    safety_approved: true,
                    veto_reason: None,
                }
            });

        (best, projections)
    }
}

/// A speculative candidate scenario for counterfactual "what if" simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhatIfScenario {
    pub action: Action,
    pub hypothesis: String,
    pub domain: Domain,
    pub expected_gain: f64,
}

/// Projected system state metrics for a simulated counterfactual scenario.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhatIfProjection {
    pub scenario: WhatIfScenario,
    pub projected_temp_cpu: f64,
    pub projected_fps: f64,
    pub projected_latency_ms: f64,
    pub utility_score: f64,
    pub safety_approved: bool,
    pub veto_reason: Option<String>,
}

impl Default for PolicyEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_telemetry(temp_cpu: f64, frametime_ms: f64) -> TelemetrySnapshot {
        TelemetrySnapshot {
            timestamp: chrono::Utc::now(),
            zone_id: 1,
            frametime_ms,
            cpu_util: 0.65,
            gpu_util: 0.70,
            memory_util: 0.55,
            temp_cpu,
            temp_gpu: temp_cpu - 5.0,
            power_draw: 65.0,
            pe_core_mask: 0xFF,
        }
    }

    #[test]
    fn test_policy_rule_ladder_priority() {
        let engine = PolicyEngine::new();
        let features = FeatureRegistry::new();

        // 1. Critical thermal: should trigger thermal_critical throttle
        let crit = sample_telemetry(96.0, 16.6);
        let res_crit = engine.evaluate(&crit, &features);
        assert_eq!(res_crit.domain, Domain::Safety);
        assert_eq!(res_crit.rule_matched.as_deref(), Some("thermal_critical"));

        // 2. Normal low FPS with good headroom: should trigger fps_boost
        let normal = sample_telemetry(60.0, 20.0);
        let res_norm = engine.evaluate(&normal, &features);
        assert_eq!(res_norm.domain, Domain::Performance);
        assert_eq!(res_norm.rule_matched.as_deref(), Some("fps_boost"));
    }

    #[test]
    fn test_what_if_counterfactual_selection() {
        let engine = PolicyEngine::new();
        let features = FeatureRegistry::new();
        let t = sample_telemetry(70.0, 16.6);

        let scenarios = vec![
            WhatIfScenario {
                action: Action::Boost { zone_id: 1, strength: 0.8 },
                hypothesis: "HighPerformanceBoost".into(),
                domain: Domain::Performance,
                expected_gain: 0.5,
            },
            WhatIfScenario {
                action: Action::Throttle { zone_id: 1, target: 0.5 },
                hypothesis: "AggressiveThermalReduction".into(),
                domain: Domain::Thermal,
                expected_gain: 0.1,
            },
            WhatIfScenario {
                action: Action::NoOp,
                hypothesis: "StatusQuo".into(),
                domain: Domain::Performance,
                expected_gain: 0.0,
            },
        ];

        let (winner, all) = engine.evaluate_what_if(&t, &scenarios, &features);
        assert_eq!(all.len(), 3);
        assert!(winner.safety_approved);
        // Under normal temperature (70°C), boost has higher utility than throttle
        assert_eq!(winner.scenario.hypothesis, "HighPerformanceBoost");
        assert!(winner.projected_fps > 60.0);
    }

    #[test]
    fn test_what_if_safety_veto_under_extreme_thermal() {
        let engine = PolicyEngine::new();
        let features = FeatureRegistry::new();
        // Near critical temp 92°C: boost of 0.8 adds 6°C -> 98°C (> 95°C limit)
        let t_hot = sample_telemetry(92.0, 16.6);

        let scenarios = vec![
            WhatIfScenario {
                action: Action::Boost { zone_id: 1, strength: 0.8 },
                hypothesis: "DangerousOverclock".into(),
                domain: Domain::Performance,
                expected_gain: 0.5,
            },
            WhatIfScenario {
                action: Action::Throttle { zone_id: 1, target: 0.7 },
                hypothesis: "SafeThermalRelief".into(),
                domain: Domain::Thermal,
                expected_gain: 0.2,
            },
        ];

        let (winner, all) = engine.evaluate_what_if(&t_hot, &scenarios, &features);
        assert_eq!(all.len(), 2);
        // DangerousOverclock should be vetoed
        let overclock = all.iter().find(|p| p.scenario.hypothesis == "DangerousOverclock").unwrap();
        assert!(!overclock.safety_approved);
        assert_eq!(overclock.veto_reason.as_deref(), Some("ThermalLimitExceeded"));

        // Winner must be the safe scenario
        assert_eq!(winner.scenario.hypothesis, "SafeThermalRelief");
        assert!(winner.safety_approved);
    }

    #[test]
    fn test_what_if_continuous_max_fan_cooling() {
        let engine = PolicyEngine::new();
        let features = FeatureRegistry::new();
        let t_hot = sample_telemetry(90.0, 16.6);

        let scenarios = vec![
            WhatIfScenario {
                action: Action::Boost { zone_id: 1, strength: 0.9 },
                hypothesis: "CONTINUOUS_MAX_FAN_100".into(),
                domain: Domain::Thermal,
                expected_gain: 0.6,
            },
            WhatIfScenario {
                action: Action::Throttle { zone_id: 1, target: 0.5 },
                hypothesis: "HeavyThrottleDamp".into(),
                domain: Domain::Thermal,
                expected_gain: 0.1,
            },
        ];

        let (winner, all) = engine.evaluate_what_if(&t_hot, &scenarios, &features);
        assert_eq!(all.len(), 2);
        let max_fan_proj = all.iter().find(|p| p.scenario.hypothesis == "CONTINUOUS_MAX_FAN_100").unwrap();
        assert!(max_fan_proj.safety_approved);
        assert_eq!(max_fan_proj.projected_temp_cpu, 72.0); // 90.0 - 18.0
        assert_eq!(winner.scenario.hypothesis, "CONTINUOUS_MAX_FAN_100");
    }
}
