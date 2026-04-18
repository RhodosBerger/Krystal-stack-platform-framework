//! Governor trait definitions

use super::{GovernorConfig, GovernorDecision, GovernorState};
use crate::Result;
use serde_json::Value;
use std::fmt::Debug;

/// Core trait for all governors
///
/// A governor makes real-time safety decisions based on metrics and statistical models.
/// Implementations should be:
/// - Fast (microsecond-scale decisions)
/// - Stateful (track historical patterns)
/// - Safe (never allow unsafe decisions even under uncertainty)
pub trait Governor: Send + Sync + Debug {
    /// Unique name of this governor
    fn name(&self) -> &str;

    /// Initialize the governor with configuration
    fn initialize(&mut self, config: GovernorConfig) -> Result<()>;

    /// Make a control decision based on current input
    ///
    /// # Arguments
    /// * `input` - Current system metrics/state
    /// * `context` - Additional context (e.g., part material, tool info)
    ///
    /// # Returns
    /// A decision with confidence and recommended safe limits
    fn decide(&mut self, input: &Value, context: &Value) -> Result<GovernorDecision>;

    /// Update governor's statistical model with new observation
    ///
    /// Called periodically to refine the statistical model based on safe operation.
    fn observe(&mut self, metrics: &Value) -> Result<()>;

    /// Get current governor state
    fn state(&self) -> GovernorState;

    /// Set governor state
    fn set_state(&mut self, state: GovernorState) -> Result<()>;

    /// Reset governor to initial state
    fn reset(&mut self) -> Result<()>;

    /// Get governor configuration
    fn config(&self) -> &GovernorConfig;

    /// Validate input before decision-making
    ///
    /// Can be overridden for custom validation logic.
    fn validate_input(&self, input: &Value) -> Result<()> {
        // Default: accept all inputs
        Ok(())
    }

    /// Get human-readable explanation of current behavior
    fn explain(&self) -> String {
        format!("Governor '{}' - no specific explanation available", self.name())
    }
}

/// Trait for composable governor chains
///
/// Allows multiple governors to work together, each adding a layer of safety.
pub trait ComposableGovernor: Governor {
    /// Next governor in the chain
    fn next(&self) -> Option<&dyn Governor>;

    /// Set the next governor in the chain
    fn set_next(&mut self, governor: Box<dyn Governor>) -> Result<()>;
}
