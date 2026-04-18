//! Error types for Krystal Core

use thiserror::Error;

/// Result type for Krystal Core operations
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur in Krystal Core
#[derive(Error, Debug)]
pub enum Error {
    #[error("Governor initialization failed: {0}")]
    GovernorInitializationError(String),

    #[error("Invalid governor configuration: {0}")]
    InvalidConfiguration(String),

    #[error("Control decision failed: {0}")]
    ControlDecisionFailed(String),

    #[error("Metrics collection failed: {0}")]
    MetricsError(String),

    #[error("Statistical computation failed: {0}")]
    StatisticalError(String),

    #[error("Runtime error: {0}")]
    RuntimeError(String),

    #[error("Safety violation: {0}")]
    SafetyViolation(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}
