//! Governor state management

use serde::{Deserialize, Serialize};

/// Current state of a governor
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GovernorState {
    /// Governor is initializing
    Initializing,
    /// Governor is active and making decisions
    Active,
    /// Governor is in learning mode (collecting data)
    Learning,
    /// Governor is in fail-safe mode (very conservative)
    FailSafe,
    /// Governor is disabled
    Disabled,
}

impl std::fmt::Display for GovernorState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GovernorState::Initializing => write!(f, "Initializing"),
            GovernorState::Active => write!(f, "Active"),
            GovernorState::Learning => write!(f, "Learning"),
            GovernorState::FailSafe => write!(f, "FailSafe"),
            GovernorState::Disabled => write!(f, "Disabled"),
        }
    }
}

impl Default for GovernorState {
    fn default() -> Self {
        GovernorState::Initializing
    }
}
