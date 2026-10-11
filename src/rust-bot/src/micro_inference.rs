//! Micro Inference - Lightweight ML predictions

use crate::{TelemetrySnapshot, Action};

pub struct MicroInference {
    weights: Vec<f64>,
    threshold: f64,
}

impl MicroInference {
    pub fn new() -> Self {
        Self {
            weights: vec![0.3, 0.2, 0.15, -0.1, -0.1, 0.25, -0.05],
            threshold: 0.5,
        }
    }

    pub fn predict(&self, telemetry: &TelemetrySnapshot) -> Action {
        let features = [
            telemetry.cpu_util,
            telemetry.gpu_util,
            telemetry.fps() / 144.0,
            telemetry.temp_cpu / 100.0,
            telemetry.temp_gpu / 100.0,
            telemetry.thermal_headroom(),
            telemetry.power_draw / 200.0,
        ];

        let score: f64 = features.iter()
            .zip(self.weights.iter())
            .map(|(f, w)| f * w)
            .sum();

        if score > self.threshold {
            Action::Boost { zone_id: telemetry.zone_id, strength: score.min(1.0) }
        } else if score < -self.threshold {
            Action::Throttle { zone_id: telemetry.zone_id, target: (1.0 + score).max(0.3) }
        } else {
            Action::NoOp
        }
    }

    pub fn update_weights(&mut self, gradient: &[f64]) {
        let lr = 0.01;
        for (w, g) in self.weights.iter_mut().zip(gradient.iter()) {
            *w += lr * g;
        }
    }
}

impl Default for MicroInference {
    fn default() -> Self { Self::new() }
}
