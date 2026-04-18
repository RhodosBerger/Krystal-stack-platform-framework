//! Real-time metrics collection and analysis
//!
//! Metrics are the observational foundation for statistical governors.
//! They track system behavior and feed into decision-making.

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// A single metric observation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metric {
    pub name: String,
    pub value: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub tags: std::collections::HashMap<String, String>,
}

/// Statistics for a metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricStats {
    pub count: u64,
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub variance: f64,
    pub std_dev: f64,
}

/// Metrics collector for tracking system behavior
pub struct MetricsCollector {
    metrics: Arc<DashMap<String, Vec<Metric>>>,
    max_samples: usize,
}

impl MetricsCollector {
    pub fn new(max_samples: usize) -> Self {
        Self {
            metrics: Arc::new(DashMap::new()),
            max_samples,
        }
    }

    pub fn record(&self, metric: Metric) {
        let mut entry = self.metrics.entry(metric.name.clone()).or_insert_with(Vec::new);
        entry.push(metric);

        // Keep only recent samples
        if entry.len() > self.max_samples {
            entry.drain(0..entry.len() - self.max_samples);
        }
    }

    pub fn stats(&self, name: &str) -> Option<MetricStats> {
        self.metrics.get(name).and_then(|values| {
            if values.is_empty() {
                return None;
            }

            let values: Vec<f64> = values.iter().map(|m| m.value).collect();
            let count = values.len() as u64;
            let sum: f64 = values.iter().sum();
            let mean = sum / values.len() as f64;

            let variance = values.iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f64>() / values.len() as f64;
            let std_dev = variance.sqrt();

            Some(MetricStats {
                count,
                mean,
                min: values.iter().cloned().fold(f64::INFINITY, f64::min),
                max: values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                variance,
                std_dev,
            })
        })
    }

    pub fn clear(&self, name: &str) {
        self.metrics.remove(name);
    }

    pub fn clear_all(&self) {
        self.metrics.clear();
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new(10000)
    }
}
