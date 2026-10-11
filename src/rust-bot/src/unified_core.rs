//! Unified Core - Rust-level multi-tier integration
//!
//! Mirrors Python's unified_system.py with native performance.
//! Handles real-time telemetry, decisions, and hardware control.

use std::collections::VecDeque;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

// ============================================================
// Level 0: Hardware Telemetry
// ============================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareTelemetry {
    pub cpu_util: f64,
    pub cpu_temp: f64,
    pub cpu_freq_mhz: u32,
    pub gpu_util: f64,
    pub gpu_temp: f64,
    pub gpu_clock_mhz: u32,
    pub gpu_power_watts: f64,
    pub vram_used_mb: u64,
    pub vram_total_mb: u64,
    pub memory_util: f64,
    pub timestamp_ms: u64,
}

impl Default for HardwareTelemetry {
    fn default() -> Self {
        Self {
            cpu_util: 0.3,
            cpu_temp: 55.0,
            cpu_freq_mhz: 3500,
            gpu_util: 0.4,
            gpu_temp: 60.0,
            gpu_clock_mhz: 1800,
            gpu_power_watts: 150.0,
            vram_used_mb: 2048,
            vram_total_mb: 8192,
            memory_util: 0.4,
            timestamp_ms: 0,
        }
    }
}

impl HardwareTelemetry {
    pub fn thermal_headroom(&self) -> f64 {
        ((85.0 - self.gpu_temp) / 85.0).max(0.0)
    }

    pub fn power_headroom(&self) -> f64 {
        (1.0 - self.gpu_power_watts / 250.0).max(0.0)
    }

    pub fn vram_util(&self) -> f64 {
        self.vram_used_mb as f64 / self.vram_total_mb as f64
    }

    pub fn to_state_vector(&self) -> [f64; 8] {
        [
            self.cpu_util,
            self.gpu_util,
            self.gpu_temp / 100.0,
            self.cpu_temp / 100.0,
            self.vram_util(),
            self.memory_util,
            self.gpu_power_watts / 250.0,
            self.gpu_temp.max(self.cpu_temp) / 100.0,
        ]
    }
}

// ============================================================
// Level 1: Signal Processing
// ============================================================

#[derive(Debug, Clone)]
pub struct PidController {
    kp: f64,
    ki: f64,
    kd: f64,
    integral: f64,
    prev_error: f64,
    output_min: f64,
    output_max: f64,
}

impl PidController {
    pub fn new(kp: f64, ki: f64, kd: f64) -> Self {
        Self {
            kp,
            ki,
            kd,
            integral: 0.0,
            prev_error: 0.0,
            output_min: -1.0,
            output_max: 1.0,
        }
    }

    pub fn update(&mut self, error: f64, dt: f64) -> f64 {
        self.integral += error * dt;
        self.integral = self.integral.clamp(-10.0, 10.0);

        let derivative = if dt > 0.0 {
            (error - self.prev_error) / dt
        } else {
            0.0
        };

        self.prev_error = error;

        let output = self.kp * error + self.ki * self.integral + self.kd * derivative;
        output.clamp(self.output_min, self.output_max)
    }

    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_error = 0.0;
    }
}

#[derive(Debug, Clone)]
pub struct ControlSignals {
    pub thermal_control: f64,
    pub performance_control: f64,
    pub thermal_headroom: f64,
    pub power_headroom: f64,
    pub memory_pressure: f64,
}

pub struct SignalProcessor {
    thermal_pid: PidController,
    perf_pid: PidController,
    thermal_target: f64,
    fps_target: f64,
    last_update: Instant,
}

impl SignalProcessor {
    pub fn new() -> Self {
        Self {
            thermal_pid: PidController::new(0.5, 0.1, 0.2),
            perf_pid: PidController::new(0.3, 0.05, 0.1),
            thermal_target: 75.0,
            fps_target: 60.0,
            last_update: Instant::now(),
        }
    }

    pub fn process(&mut self, telemetry: &HardwareTelemetry) -> ControlSignals {
        let now = Instant::now();
        let dt = now.duration_since(self.last_update).as_secs_f64();
        self.last_update = now;

        let thermal_error = self.thermal_target - telemetry.gpu_temp;
        let thermal_control = self.thermal_pid.update(thermal_error, dt);

        let estimated_fps = 144.0 * (1.0 - telemetry.gpu_util * 0.5);
        let fps_error = self.fps_target - estimated_fps;
        let perf_control = self.perf_pid.update(fps_error, dt);

        ControlSignals {
            thermal_control,
            performance_control: perf_control,
            thermal_headroom: telemetry.thermal_headroom(),
            power_headroom: telemetry.power_headroom(),
            memory_pressure: telemetry.vram_util(),
        }
    }
}

impl Default for SignalProcessor {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// Level 2: Learning (Simplified TD)
// ============================================================

#[derive(Debug, Clone)]
pub struct Experience {
    pub state: [f64; 8],
    pub action: [f64; 4],
    pub reward: f64,
    pub next_state: [f64; 8],
    pub td_error: f64,
}

pub struct TdLearner {
    q_weights: [[f64; 4]; 8],
    learning_rate: f64,
    discount: f64,
    experience_buffer: VecDeque<Experience>,
}

impl TdLearner {
    pub fn new() -> Self {
        Self {
            q_weights: [[0.1; 4]; 8],
            learning_rate: 0.01,
            discount: 0.95,
            experience_buffer: VecDeque::with_capacity(10000),
        }
    }

    pub fn get_q_values(&self, state: &[f64; 8]) -> [f64; 4] {
        let mut q = [0.0; 4];
        for (i, &s) in state.iter().enumerate() {
            for (j, q_j) in q.iter_mut().enumerate() {
                *q_j += s * self.q_weights[i][j];
            }
        }
        q
    }

    pub fn update(&mut self, state: [f64; 8], action: [f64; 4],
                  reward: f64, next_state: [f64; 8]) -> f64 {
        let q_current = self.get_q_values(&state);
        let q_next = self.get_q_values(&next_state);

        let max_q_next = q_next.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let action_idx = action.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let td_error = reward + self.discount * max_q_next - q_current[action_idx];

        // Update weights
        for (i, &s) in state.iter().enumerate() {
            self.q_weights[i][action_idx] += self.learning_rate * td_error * s;
        }

        // Store experience
        if self.experience_buffer.len() >= 10000 {
            self.experience_buffer.pop_front();
        }
        self.experience_buffer.push_back(Experience {
            state,
            action,
            reward,
            next_state,
            td_error,
        });

        td_error
    }

    pub fn decide(&self, state: &[f64; 8], exploration_rate: f64) -> ([f64; 4], f64) {
        if rand::random::<f64>() < exploration_rate {
            // Explore
            let action: [f64; 4] = [
                rand::random(),
                rand::random(),
                rand::random(),
                rand::random(),
            ];
            (action, 0.5)
        } else {
            // Exploit
            let q = self.get_q_values(state);
            let sum: f64 = q.iter().map(|x| x.abs()).sum();
            let action = if sum > 0.0 {
                [q[0] / sum, q[1] / sum, q[2] / sum, q[3] / sum]
            } else {
                [0.25; 4]
            };
            (action, 0.8)
        }
    }
}

impl Default for TdLearner {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// Level 3: Phase & Emergence (Simplified)
// ============================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    Solid,
    Liquid,
    Gas,
    Plasma,
    Critical,
}

pub struct PhaseEngine {
    pub phase: Phase,
    pub temperature: f64,
    pub order_parameter: f64,
    critical_temp: f64,
    transitions: u32,
}

impl PhaseEngine {
    pub fn new() -> Self {
        Self {
            phase: Phase::Solid,
            temperature: 0.3,
            order_parameter: 1.0,
            critical_temp: 0.7,
            transitions: 0,
        }
    }

    pub fn update(&mut self, gradient: f64, stability: f64) -> Phase {
        let old_phase = self.phase;

        self.temperature = 0.9 * self.temperature + 0.1 * gradient.abs();
        self.order_parameter = (1.0 - self.temperature / self.critical_temp).max(0.0);

        self.phase = if self.temperature < 0.3 {
            Phase::Solid
        } else if self.temperature < self.critical_temp {
            Phase::Liquid
        } else if (self.temperature - self.critical_temp).abs() < 0.05 {
            Phase::Critical
        } else if self.temperature < 0.9 {
            Phase::Gas
        } else {
            Phase::Plasma
        };

        if old_phase != self.phase {
            self.transitions += 1;
        }

        self.phase
    }

    pub fn exploration_rate(&self) -> f64 {
        match self.phase {
            Phase::Solid => 0.01,
            Phase::Liquid => 0.1,
            Phase::Critical => 0.5,
            Phase::Gas => 0.3,
            Phase::Plasma => 0.8,
        }
    }
}

impl Default for PhaseEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// Level 4: Preset Generation
// ============================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwarePreset {
    pub name: String,
    pub cpu_freq_mhz: u32,
    pub gpu_clock_mhz: u32,
    pub power_limit_watts: u32,
    pub fan_speed_rpm: u32,
    pub quality_score: f64,
}

pub struct PresetGenerator {
    presets: Vec<HardwarePreset>,
    generation_count: u32,
}

impl PresetGenerator {
    pub fn new() -> Self {
        Self {
            presets: Vec::new(),
            generation_count: 0,
        }
    }

    pub fn generate(&mut self, signals: &ControlSignals) -> HardwarePreset {
        self.generation_count += 1;

        // Generate based on current signals
        let thermal_factor = signals.thermal_headroom;
        let power_factor = signals.power_headroom;

        let preset = HardwarePreset {
            name: format!("gen_{}", self.generation_count),
            cpu_freq_mhz: (3000.0 + thermal_factor * 1500.0) as u32,
            gpu_clock_mhz: (1500.0 + thermal_factor * 800.0) as u32,
            power_limit_watts: (100.0 + power_factor * 150.0) as u32,
            fan_speed_rpm: (1000.0 + (1.0 - thermal_factor) * 2000.0) as u32,
            quality_score: thermal_factor * 0.5 + power_factor * 0.5,
        };

        self.presets.push(preset.clone());
        preset
    }
}

impl Default for PresetGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// Unified Core
// ============================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SystemMode {
    Init,
    Learning,
    Optimizing,
    Stable,
    Generating,
    Emergency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleResult {
    pub cycle: u64,
    pub mode: SystemMode,
    pub phase: Phase,
    pub reward: f64,
    pub td_error: f64,
    pub preset: Option<String>,
    pub telemetry_summary: TelemetrySummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetrySummary {
    pub gpu_temp: f64,
    pub gpu_util: f64,
    pub thermal_headroom: f64,
}

pub struct UnifiedCore {
    signal_processor: SignalProcessor,
    learner: TdLearner,
    phase_engine: PhaseEngine,
    preset_generator: PresetGenerator,
    mode: SystemMode,
    cycle_count: u64,
    reward_history: VecDeque<f64>,
    active_preset: Option<HardwarePreset>,
}

impl UnifiedCore {
    pub fn new() -> Self {
        Self {
            signal_processor: SignalProcessor::new(),
            learner: TdLearner::new(),
            phase_engine: PhaseEngine::new(),
            preset_generator: PresetGenerator::new(),
            mode: SystemMode::Init,
            cycle_count: 0,
            reward_history: VecDeque::with_capacity(1000),
            active_preset: None,
        }
    }

    pub fn tick(&mut self, telemetry: HardwareTelemetry) -> CycleResult {
        self.cycle_count += 1;

        // Level 1: Process signals
        let signals = self.signal_processor.process(&telemetry);

        // Level 2: Learn and decide
        let state = telemetry.to_state_vector();
        let exploration = self.phase_engine.exploration_rate();
        let (action, confidence) = self.learner.decide(&state, exploration);

        // Level 3: Update phase
        let gradient = if self.reward_history.len() > 1 {
            let last = *self.reward_history.back().unwrap_or(&0.5);
            let prev = *self.reward_history.get(self.reward_history.len() - 2).unwrap_or(&0.5);
            last - prev
        } else {
            0.0
        };
        let phase = self.phase_engine.update(gradient, confidence);

        // Compute reward
        let reward = self.compute_reward(&telemetry, &signals);
        self.reward_history.push_back(reward);
        if self.reward_history.len() > 1000 {
            self.reward_history.pop_front();
        }

        // Learn
        let next_state = state; // In real impl, would read new telemetry
        let td_error = self.learner.update(state, action, reward, next_state);

        // Level 4: Generate preset if needed
        let preset_name = if confidence < 0.5 || self.mode == SystemMode::Generating {
            let preset = self.preset_generator.generate(&signals);
            let name = preset.name.clone();
            self.active_preset = Some(preset);
            Some(name)
        } else {
            None
        };

        // Update mode
        self.update_mode(&telemetry, reward);

        CycleResult {
            cycle: self.cycle_count,
            mode: self.mode,
            phase,
            reward,
            td_error,
            preset: preset_name,
            telemetry_summary: TelemetrySummary {
                gpu_temp: telemetry.gpu_temp,
                gpu_util: telemetry.gpu_util,
                thermal_headroom: signals.thermal_headroom,
            },
        }
    }

    fn compute_reward(&self, telemetry: &HardwareTelemetry, signals: &ControlSignals) -> f64 {
        let thermal_reward = signals.thermal_headroom * 0.3;
        let perf_reward = telemetry.gpu_util * 0.3;
        let efficiency = telemetry.gpu_util / (telemetry.gpu_power_watts / 250.0 + 0.1);
        let efficiency_reward = efficiency.min(1.0) * 0.2;
        let stability_reward = self.compute_stability() * 0.2;

        (thermal_reward + perf_reward + efficiency_reward + stability_reward).clamp(0.0, 1.0)
    }

    fn compute_stability(&self) -> f64 {
        if self.reward_history.len() < 10 {
            return 0.5;
        }
        let recent: Vec<f64> = self.reward_history.iter().rev().take(10).cloned().collect();
        let mean: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
        let variance: f64 = recent.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / recent.len() as f64;
        1.0 - variance.sqrt()
    }

    fn update_mode(&mut self, telemetry: &HardwareTelemetry, reward: f64) {
        self.mode = if telemetry.gpu_temp > 85.0 {
            SystemMode::Emergency
        } else if reward < 0.3 {
            SystemMode::Generating
        } else if reward < 0.6 {
            SystemMode::Optimizing
        } else if self.cycle_count < 100 {
            SystemMode::Learning
        } else {
            SystemMode::Stable
        };
    }

    pub fn get_mode(&self) -> SystemMode {
        self.mode
    }

    pub fn get_cycle_count(&self) -> u64 {
        self.cycle_count
    }

    pub fn get_active_preset(&self) -> Option<&HardwarePreset> {
        self.active_preset.as_ref()
    }
}

impl Default for UnifiedCore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_core_cycle() {
        let mut core = UnifiedCore::new();
        let telemetry = HardwareTelemetry::default();

        let result = core.tick(telemetry);

        assert_eq!(result.cycle, 1);
        assert!(result.reward >= 0.0 && result.reward <= 1.0);
    }

    #[test]
    fn test_multiple_cycles() {
        let mut core = UnifiedCore::new();

        for _ in 0..100 {
            let telemetry = HardwareTelemetry::default();
            let _ = core.tick(telemetry);
        }

        assert_eq!(core.get_cycle_count(), 100);
    }
}
