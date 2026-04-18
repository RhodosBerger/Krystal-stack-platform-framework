//! Example: Basic statistical governor for speed control
//!
//! This demonstrates a simple governor that adapts safe speed limits
//! based on observed vibration patterns.

use krystal_core::governor::{Governor, GovernorConfig, GovernorDecision, GovernorState};
use krystal_core::{Error, Result};
use serde_json::{json, Value};
use std::fmt::Debug;

/// Simple vibration-based speed governor
#[derive(Debug)]
struct VibrationGovernor {
    config: GovernorConfig,
    state: GovernorState,
    max_observed_vibration: f64,
    sample_count: u32,
}

impl VibrationGovernor {
    fn new(config: GovernorConfig) -> Self {
        Self {
            config,
            state: GovernorState::Initializing,
            max_observed_vibration: 0.0,
            sample_count: 0,
        }
    }
}

impl Governor for VibrationGovernor {
    fn name(&self) -> &str {
        "vibration_governor"
    }

    fn initialize(&mut self, config: GovernorConfig) -> Result<()> {
        self.config = config;
        self.state = GovernorState::Learning;
        println!("Vibration governor initialized");
        Ok(())
    }

    fn decide(&mut self, input: &Value, _context: &Value) -> Result<GovernorDecision> {
        let vibration = input
            .get("vibration")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| Error::ControlDecisionFailed("Missing vibration metric".into()))?;

        let speed = input
            .get("speed")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        // Simple heuristic: if vibration is high, reduce safe speed
        let safety_margin = 0.8; // 20% margin
        let safe_speed = (100.0 - vibration * 10.0) * safety_margin;

        let allowed = speed <= safe_speed;
        let confidence = 0.85; // Moderate confidence without sufficient samples

        let decision = GovernorDecision::new(allowed, confidence,
            format!("Vibration: {:.2}g, Speed: {:.1} RPM", vibration, speed))
            .with_limit(safe_speed);

        Ok(decision)
    }

    fn observe(&mut self, metrics: &Value) -> Result<()> {
        if let Some(vibration) = metrics.get("vibration").and_then(|v| v.as_f64()) {
            self.max_observed_vibration = self.max_observed_vibration.max(vibration);
            self.sample_count += 1;

            if self.sample_count > 100 {
                self.state = GovernorState::Active;
            }
        }
        Ok(())
    }

    fn state(&self) -> GovernorState {
        self.state
    }

    fn set_state(&mut self, state: GovernorState) -> Result<()> {
        self.state = state;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.max_observed_vibration = 0.0;
        self.sample_count = 0;
        self.state = GovernorState::Initializing;
        Ok(())
    }

    fn config(&self) -> &GovernorConfig {
        &self.config
    }

    fn explain(&self) -> String {
        format!(
            "Vibration Governor: max observed vibration {:.2}g over {} samples, state: {}",
            self.max_observed_vibration, self.sample_count, self.state
        )
    }
}

fn main() -> Result<()> {
    let config = GovernorConfig::new("vibration_governor")
        .with_threshold(0.85);

    let mut governor = VibrationGovernor::new(config);
    governor.initialize(GovernorConfig::new("vibration_governor"))?;

    // Simulate some normal operation
    for i in 0..10 {
        let metrics = json!({
            "vibration": 0.3 + (i as f64) * 0.01,  // Gradually increasing
            "speed": 50.0 + (i as f64) * 5.0,
        });

        governor.observe(&metrics)?;

        let decision = governor.decide(&metrics, &json!({}))?;
        println!(
            "Step {}: {} (confidence: {:.2}%, safe_limit: {:.1})",
            i,
            if decision.allowed { "ALLOWED" } else { "DENIED" },
            decision.confidence * 100.0,
            decision.safe_limit.unwrap_or(0.0)
        );
    }

    println!("\nGovernor explanation: {}", governor.explain());
    Ok(())
}
