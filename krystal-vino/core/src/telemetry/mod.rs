//! Telemetry and logging infrastructure
//!
//! Structured logging for observability and debugging

use tracing::{debug, info, warn};

/// Initialize telemetry (tracing subscriber)
pub fn init_telemetry() {
    tracing_subscriber::fmt()
        .with_target(true)
        .with_thread_ids(true)
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("Krystal Core telemetry initialized");
}

/// Log a governor decision
pub fn log_decision(governor: &str, allowed: bool, confidence: f64, reason: &str) {
    if allowed {
        info!(
            governor,
            confidence,
            reason,
            "Governor decision: ALLOWED"
        );
    } else {
        warn!(
            governor,
            confidence,
            reason,
            "Governor decision: DENIED"
        );
    }
}

/// Log a safety event
pub fn log_safety_event(event_type: &str, severity: &str, message: &str) {
    warn!(
        event_type,
        severity,
        message,
        "Safety event logged"
    );
}
