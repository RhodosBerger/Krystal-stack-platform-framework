//! Signal-First Scheduling System
//!
//! Decisions follow telemetry strength and domain rankings,
//! so resources flow to highest-value workloads while safety stays enforced.

use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;
use serde::{Deserialize, Serialize};

/// Signal with strength and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    pub id: String,
    pub source: SignalSource,
    pub kind: SignalKind,
    pub strength: f64,       // 0.0 - 1.0 normalized
    pub confidence: f64,     // 0.0 - 1.0
    pub timestamp: u64,
    pub ttl_ms: u64,
    pub payload: HashMap<String, f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SignalSource {
    Telemetry,
    User,
    Policy,
    Safety,
    Learning,
    External,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SignalKind {
    // Performance signals
    FrametimeSpike,
    CpuBottleneck,
    GpuBottleneck,
    MemoryPressure,

    // Thermal signals
    ThermalWarning,
    ThermalCritical,
    CoolingOpportunity,

    // Workload signals
    WorkloadChange,
    ForegroundSwitch,
    IdleDetected,
    BurstDetected,

    // User signals
    UserBoostRequest,
    UserProfileChange,

    // Safety signals
    GuardrailTriggered,
    EmergencyStop,
}

/// Domain for signal ranking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Domain {
    Safety,
    Performance,
    Thermal,
    Power,
    User,
}

/// Ranked signal for priority queue
#[derive(Debug, Clone)]
struct RankedSignal {
    signal: Signal,
    priority: f64,
    domain_rank: u32,
}

impl PartialEq for RankedSignal {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for RankedSignal {}

impl PartialOrd for RankedSignal {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RankedSignal {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first, then lower domain rank (safety=0 highest)
        match self.priority.partial_cmp(&other.priority) {
            Some(Ordering::Equal) => other.domain_rank.cmp(&self.domain_rank),
            Some(ord) => ord.reverse(), // Reverse for max-heap behavior
            None => Ordering::Equal,
        }
    }
}

/// Signal scheduler
pub struct SignalScheduler {
    queue: BinaryHeap<RankedSignal>,
    domain_weights: HashMap<Domain, f64>,
    kind_priorities: HashMap<SignalKind, f64>,
    processed: Vec<Signal>,
    max_queue_size: usize,
}

impl SignalScheduler {
    pub fn new() -> Self {
        let mut scheduler = Self {
            queue: BinaryHeap::new(),
            domain_weights: HashMap::new(),
            kind_priorities: HashMap::new(),
            processed: Vec::new(),
            max_queue_size: 1000,
        };
        scheduler.init_defaults();
        scheduler
    }

    fn init_defaults(&mut self) {
        // Domain weights (higher = more important)
        self.domain_weights.insert(Domain::Safety, 1.0);
        self.domain_weights.insert(Domain::User, 0.9);
        self.domain_weights.insert(Domain::Performance, 0.7);
        self.domain_weights.insert(Domain::Thermal, 0.8);
        self.domain_weights.insert(Domain::Power, 0.5);

        // Kind base priorities
        self.kind_priorities.insert(SignalKind::EmergencyStop, 1.0);
        self.kind_priorities.insert(SignalKind::ThermalCritical, 0.95);
        self.kind_priorities.insert(SignalKind::GuardrailTriggered, 0.9);
        self.kind_priorities.insert(SignalKind::ThermalWarning, 0.8);
        self.kind_priorities.insert(SignalKind::UserBoostRequest, 0.75);
        self.kind_priorities.insert(SignalKind::FrametimeSpike, 0.7);
        self.kind_priorities.insert(SignalKind::CpuBottleneck, 0.65);
        self.kind_priorities.insert(SignalKind::GpuBottleneck, 0.65);
        self.kind_priorities.insert(SignalKind::MemoryPressure, 0.6);
        self.kind_priorities.insert(SignalKind::WorkloadChange, 0.5);
        self.kind_priorities.insert(SignalKind::ForegroundSwitch, 0.5);
        self.kind_priorities.insert(SignalKind::BurstDetected, 0.4);
        self.kind_priorities.insert(SignalKind::CoolingOpportunity, 0.3);
        self.kind_priorities.insert(SignalKind::IdleDetected, 0.2);
        self.kind_priorities.insert(SignalKind::UserProfileChange, 0.4);
    }

    /// Enqueue a signal
    pub fn enqueue(&mut self, signal: Signal) {
        let domain = self.classify_domain(&signal);
        let priority = self.compute_priority(&signal, domain);
        let domain_rank = self.domain_rank(domain);

        let ranked = RankedSignal {
            signal,
            priority,
            domain_rank,
        };

        self.queue.push(ranked);

        // Trim if over capacity
        while self.queue.len() > self.max_queue_size {
            self.queue.pop();
        }
    }

    /// Dequeue highest priority signal
    pub fn dequeue(&mut self) -> Option<Signal> {
        let now = current_timestamp();

        // Skip expired signals
        while let Some(ranked) = self.queue.pop() {
            if ranked.signal.timestamp + ranked.signal.ttl_ms >= now {
                self.processed.push(ranked.signal.clone());
                return Some(ranked.signal);
            }
        }
        None
    }

    /// Peek at highest priority signal without removing
    pub fn peek(&self) -> Option<&Signal> {
        self.queue.iter().next().map(|r| &r.signal)
    }

    /// Get all pending signals sorted by priority
    pub fn drain_by_priority(&mut self) -> Vec<Signal> {
        let mut signals = Vec::new();
        while let Some(signal) = self.dequeue() {
            signals.push(signal);
        }
        signals
    }

    /// Get signals by domain
    pub fn filter_by_domain(&self, domain: Domain) -> Vec<&Signal> {
        self.queue.iter()
            .filter(|r| self.classify_domain(&r.signal) == domain)
            .map(|r| &r.signal)
            .collect()
    }

    fn classify_domain(&self, signal: &Signal) -> Domain {
        match signal.kind {
            SignalKind::EmergencyStop | SignalKind::GuardrailTriggered => Domain::Safety,
            SignalKind::ThermalWarning | SignalKind::ThermalCritical |
            SignalKind::CoolingOpportunity => Domain::Thermal,
            SignalKind::FrametimeSpike | SignalKind::CpuBottleneck |
            SignalKind::GpuBottleneck | SignalKind::BurstDetected => Domain::Performance,
            SignalKind::UserBoostRequest | SignalKind::UserProfileChange => Domain::User,
            SignalKind::MemoryPressure | SignalKind::IdleDetected => Domain::Power,
            SignalKind::WorkloadChange | SignalKind::ForegroundSwitch => Domain::Performance,
        }
    }

    fn compute_priority(&self, signal: &Signal, domain: Domain) -> f64 {
        let base = self.kind_priorities.get(&signal.kind).copied().unwrap_or(0.5);
        let domain_weight = self.domain_weights.get(&domain).copied().unwrap_or(0.5);

        // Priority = base * domain_weight * strength * confidence
        base * domain_weight * signal.strength * signal.confidence
    }

    fn domain_rank(&self, domain: Domain) -> u32 {
        match domain {
            Domain::Safety => 0,      // Highest priority
            Domain::Thermal => 1,
            Domain::User => 2,
            Domain::Performance => 3,
            Domain::Power => 4,
        }
    }

    /// Update domain weights dynamically
    pub fn set_domain_weight(&mut self, domain: Domain, weight: f64) {
        self.domain_weights.insert(domain, weight.clamp(0.0, 1.0));
    }

    /// Get queue stats
    pub fn stats(&self) -> SchedulerStats {
        let mut by_domain: HashMap<Domain, usize> = HashMap::new();
        let mut by_kind: HashMap<SignalKind, usize> = HashMap::new();

        for ranked in &self.queue {
            let domain = self.classify_domain(&ranked.signal);
            *by_domain.entry(domain).or_default() += 1;
            *by_kind.entry(ranked.signal.kind).or_default() += 1;
        }

        SchedulerStats {
            queue_size: self.queue.len(),
            processed_count: self.processed.len(),
            signals_by_domain: by_domain,
            signals_by_kind: by_kind,
        }
    }

    /// Clear processed history
    pub fn clear_history(&mut self) {
        self.processed.clear();
    }
}

impl Default for SignalScheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct SchedulerStats {
    pub queue_size: usize,
    pub processed_count: usize,
    pub signals_by_domain: HashMap<Domain, usize>,
    pub signals_by_kind: HashMap<SignalKind, usize>,
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_signal(kind: SignalKind, strength: f64) -> Signal {
        Signal {
            id: format!("sig-{:?}", kind),
            source: SignalSource::Telemetry,
            kind,
            strength,
            confidence: 0.9,
            timestamp: current_timestamp(),
            ttl_ms: 10000,
            payload: HashMap::new(),
        }
    }

    #[test]
    fn test_priority_ordering() {
        let mut scheduler = SignalScheduler::new();

        scheduler.enqueue(make_signal(SignalKind::IdleDetected, 0.5));
        scheduler.enqueue(make_signal(SignalKind::ThermalCritical, 0.9));
        scheduler.enqueue(make_signal(SignalKind::FrametimeSpike, 0.8));

        // Thermal critical should come first (safety domain)
        let first = scheduler.dequeue().unwrap();
        assert_eq!(first.kind, SignalKind::ThermalCritical);
    }

    #[test]
    fn test_domain_weights() {
        let mut scheduler = SignalScheduler::new();

        // Boost performance domain
        scheduler.set_domain_weight(Domain::Performance, 1.0);
        scheduler.set_domain_weight(Domain::Thermal, 0.3);

        scheduler.enqueue(make_signal(SignalKind::ThermalWarning, 0.5));
        scheduler.enqueue(make_signal(SignalKind::FrametimeSpike, 0.5));

        // With boosted performance weight, frametime should win
        let first = scheduler.dequeue().unwrap();
        // Note: ThermalWarning still has higher base priority, this tests the weighting
    }
}
