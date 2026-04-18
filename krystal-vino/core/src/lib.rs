//! Krystal Core - High-performance real-time control engine for hardware optimization
//!
//! This library provides a statistically-informed governance system for real-time hardware control,
//! enabling probabilistic safety mechanisms instead of traditional deterministic governors.
//!
//! ## Architecture
//! - **Governors**: Trait-based safety control mechanisms backed by statistical models
//! - **Metrics**: Real-time telemetry collection and analysis
//! - **Extensions**: Pluggable governor implementations
//! - **Safety**: Bayesian reasoning for safety-critical decisions

pub mod error;
pub mod governor;
pub mod metrics;
pub mod runtime;
pub mod telemetry;

pub use error::{Error, Result};
pub use governor::{Governor, GovernorConfig, GovernorState};
pub use runtime::ControlRuntime;

/// Crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
