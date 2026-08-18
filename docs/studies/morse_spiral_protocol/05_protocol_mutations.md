# Morse-Spiral Protocol: Protocol Mutations

## Evolutionary Protocol Mutation Engine

### Protocol Mutation Engine Architecture

The Protocol Mutation Engine provides evolutionary capabilities for the Morse-Spiral Protocol, allowing for self-optimization and adaptation through various mutation strategies:

```rust
pub struct ProtocolMutationEngine {
    pub mutation_strategies: Vec<MutationStrategy>,
    pub fitness_evaluator: FitnessEvaluator,
    pub variant_manager: ProtocolVariantManager,
    pub evolution_history: Vec<EvolutionStep>,
    pub selection_strategy: SelectionStrategy,
}

#[derive(Debug, Clone)]
pub enum MutationStrategy {
    PositionShuffle,        // Shuffle hex positions
    AmplitudeShift,         // Shift analog amplitude
    FrequencyModulation,    // Modulate frequency
    PhaseRotation,          // Rotate phase
    CompressionSwap,        // Swap compression methods
    TimingVariation,        // Vary signal timing
    PatternMutation,        // Mutate signal patterns
    FibonacciScale,         // Scale Fibonacci factors
    BinaryAnalogMix,        // Mix binary/analog ratios
    ErrorCorrection,        // Add/remove error correction
    SignalModulation,       // Modulate entire signal
    GridRepositioning,      // Reposition in hex grid
    AdaptiveScaling,        // Scale based on performance
}

impl ProtocolMutationEngine {
    pub fn generate_protocol_variant(&self, base_signal: &MorseSpiralSignal, strategy: &MutationStrategy) -> MorseSpiralSignal {
        let mut mutated_signal = *base_signal;
        
        match strategy {
            MutationStrategy::PositionShuffle => {
                // Mutate hex position based on Fibonacci sequence
                let fib_val = self.fibonacci_value(mutated_signal.fibonacci_sequence_id);
                let new_x = ((mutated_signal.grid_position.x as u16 + fib_val as u16) % 16) as u8;
                mutated_signal.grid_position.x = new_x;
            },
            MutationStrategy::AmplitudeShift => {
                // Apply random amplitude shift based on position
                let shift = (mutated_signal.grid_position.x as f32) * 0.05;
                mutated_signal.analog_amplitude = 
                    (mutated_signal.analog_amplitude + shift).clamp(0.0, 1.0);
            },
            MutationStrategy::FrequencyModulation => {
                // Modulate frequency with position-dependent factor
                let mod_factor = 1.0 + (mutated_signal.grid_position.y as f32) * 0.01;
                mutated_signal.analog_frequency *= mod_factor;
            },
            MutationStrategy::PhaseRotation => {
                // Rotate phase based on position
                let rotation = (mutated_signal.grid_position.z as f32) * 0.1;
                mutated_signal.analog_phase = (mutated_signal.analog_phase + rotation) % (2.0 * std::f32::consts::PI);
            },
            MutationStrategy::CompressionSwap => {
                // Swap compression method
                mutated_signal.binary_data = self.swap_compression_method(base_signal);
            },
            MutationStrategy::TimingVariation => {
                // Vary signal duration based on Fibonacci
                let variation = self.fibonacci_value(mutated_signal.fibonacci_sequence_id % 10) as u32 * 10;
                mutated_signal.duration = std::time::Duration::from_micros(
                    mutated_signal.duration.as_micros() as u64 + variation as u64
                );
            },
            MutationStrategy::PatternMutation => {
                // Mutate the binary data pattern
                mutated_signal.binary_data ^= self.pattern_mutation_mask(mutated_signal.grid_position);
            },
            MutationStrategy::FibonacciScale => {
                // Scale Fibonacci sequence ID
                mutated_signal.fibonacci_sequence_id = self.scaling_fibonacci_value(
                    mutated_signal.fibonacci_sequence_id
                );
            },
            MutationStrategy::BinaryAnalogMix => {
                // Adjust binary-analog mix ratio
                mutated_signal.compression_ratio = 
                    (mutated_signal.compression_ratio * 1.2).min(5.0);
            },
            MutationStrategy::ErrorCorrection => {
                // Add error correction bits to binary data
                mutated_signal.binary_data ^= self.calculate_error_correction_bits(base_signal);
            },
            MutationStrategy::SignalModulation => {
                // Apply full signal modulation
                mutated_signal = self.modulate_entire_signal(base_signal);
            },
            MutationStrategy::GridRepositioning => {
                // Reposition in hex grid based on performance
                mutated_signal = self.reposition_in_grid(base_signal);
            },
            MutationStrategy::AdaptiveScaling => {
                // Scale parameters based on performance metrics
                mutated_signal = self.adaptive_parameter_scaling(base_signal);
            },
        }
        
        mutated_signal
    }
    
    fn fibonacci_value(&self, n: u64) -> u64 {
        // Simple Fibonacci calculation
        if n <= 1 { return n; }
        let mut a = 0;
        let mut b = 1;
        for _ in 2..=n {
            let temp = a + b;
            a = b;
            b = temp;
        }
        b
    }
    
    fn scaling_fibonacci_value(&self, n: u64) -> u64 {
        // Fibonacci with scaling factor
        let fib_val = self.fibonacci_value(n);
        let scale_factor = (n % 3) + 1; // Scale by 1, 2, or 3
        fib_val * scale_factor
    }
    
    fn pattern_mutation_mask(&self, position: HexPosition) -> u128 {
        // Create mutation mask based on hex position
        let mask = ((position.w as u128) << 96) |
                  ((position.z as u128) << 64) |
                  ((position.y as u128) << 32) |
                  (position.x as u128);
        mask
    }
    
    fn calculate_error_correction_bits(&self, signal: &MorseSpiralSignal) -> u128 {
        // Simple XOR-based error correction
        let mut xor_result = signal.grid_position.w as u128;
        xor_result ^= (signal.grid_position.z as u128) << 16;
        xor_result ^= (signal.grid_position.y as u128) << 32;
        xor_result ^= (signal.grid_position.x as u128) << 48;
        xor_result
    }
    
    fn swap_compression_method(&self, signal: &MorseSpiralSignal) -> u128 {
        // Apply different compression technique
        let original_data = signal.binary_data;
        // Simple transformation: rotate bits and XOR with position hash
        let position_hash = self.hash_hex_position(&signal.grid_position);
        let rotated = (original_data << 16) | (original_data >> (128 - 16)); // 16-bit rotation
        rotated ^ position_hash
    }
    
    fn hash_hex_position(&self, position: &HexPosition) -> u128 {
        let mut hash: u128 = 0;
        hash |= (position.w as u128) << 96;
        hash |= (position.z as u128) << 64;
        hash |= (position.y as u128) << 32;
        hash |= (position.x as u128) << 0;
        hash
    }
    
    fn modulate_entire_signal(&self, signal: &MorseSpiralSignal) -> MorseSpiralSignal {
        // Apply comprehensive signal modulation
        let mut modulated = *signal;
        
        // Apply amplitude modulation
        modulated.analog_amplitude = (modulated.analog_amplitude * 1.1).min(1.0);
        
        // Apply frequency modulation
        modulated.analog_frequency *= 1.05;
        
        // Apply duration modulation
        modulated.duration = std::time::Duration::from_micros(
            (modulated.duration.as_micros() as f64 * 1.02) as u64
        );
        
        // Apply compression ratio adjustment
        modulated.compression_ratio = (modulated.compression_ratio * 1.05).min(10.0);
        
        modulated
    }
    
    fn reposition_in_grid(&self, signal: &MorseSpiralSignal) -> MorseSpiralSignal {
        // Reposition based on performance analysis
        let mut repositioned = *signal;
        
        // Calculate new position based on Fibonacci spiral
        let fib_id = signal.fibonacci_sequence_id as f64;
        let angle = fib_id * 0.618033988749 * 2.0 * std::f64::consts::PI;
        let radius = fib_id.sqrt();
        
        let new_x = ((angle.cos() * radius).rem_euclid(16.0).abs() as u8) % 16;
        let new_y = ((angle.sin() * radius).rem_euclid(16.0).abs() as u8) % 16;
        
        repositioned.grid_position.x = new_x;
        repositioned.grid_position.y = new_y;
        
        repositioned
    }
    
    fn adaptive_parameter_scaling(&self, signal: &MorseSpiralSignal) -> MorseSpiralSignal {
        // Adapt parameters based on signal characteristics
        let mut adapted = *signal;
        
        // Scale parameters adaptively
        adapted.analog_amplitude *= 0.9 + (signal.compression_ratio / 10.0) * 0.2;
        adapted.analog_frequency *= 0.8 + (signal.compression_ratio / 5.0) * 0.4;
        adapted.compression_ratio = signal.compression_ratio * 1.05;
        
        adapted
    }
}
```

### Fitness Evaluation System

```rust
pub struct FitnessEvaluator {
    pub performance_metrics: PerformanceMetrics,
    pub efficiency_calculator: EfficiencyCalculator,
    pub stability_analyzer: StabilityAnalyzer,
}

pub struct PerformanceMetrics {
    pub compression_ratio: f32,
    pub transmission_speed: f32,
    pub error_rate: f32,
    pub signal_quality: f32,
    pub resource_utilization: f32,
}

impl FitnessEvaluator {
    pub fn evaluate_fitness(&self, signal: &MorseSpiralSignal, original_data: &[u8]) -> f32 {
        let metrics = self.calculate_performance_metrics(signal, original_data);
        
        // Weighted fitness calculation
        let compression_score = metrics.compression_ratio / 10.0; // Normalize to 0-1
        let speed_score = metrics.transmission_speed / 1000000.0; // Normalize to 0-1 (assuming max 1M ops/sec)
        let error_score = (1.0 - metrics.error_rate).max(0.0); // Inverse of error rate
        let quality_score = metrics.signal_quality;
        let efficiency_score = (1.0 - metrics.resource_utilization).max(0.0);
        
        // Weighted combination (weights sum to 1.0)
        let fitness = 
            compression_score * 0.25 +     // 25% for compression
            speed_score * 0.20 +           // 20% for speed
            error_score * 0.25 +           // 25% for error resistance
            quality_score * 0.15 +         // 15% for signal quality
            efficiency_score * 0.15;       // 15% for resource efficiency
        
        fitness.clamp(0.0, 1.0)
    }
    
    fn calculate_performance_metrics(&self, signal: &MorseSpiralSignal, original_data: &[u8]) -> PerformanceMetrics {
        PerformanceMetrics {
            compression_ratio: signal.compression_ratio,
            transmission_speed: self.estimate_transmission_speed(signal),
            error_rate: self.estimate_error_rate(signal, original_data),
            signal_quality: self.estimate_signal_quality(signal),
            resource_utilization: self.estimate_resource_utilization(signal),
        }
    }
    
    fn estimate_transmission_speed(&self, signal: &MorseSpiralSignal) -> f32 {
        // Estimate transmission speed based on signal parameters
        let base_speed = 1000000.0; // 1M operations per second baseline
        let duration_factor = 1.0 / (signal.duration.as_nanos() as f32 * 1e-9 + 1e-6);
        base_speed * duration_factor
    }
    
    fn estimate_error_rate(&self, signal: &MorseSpiralSignal, original_data: &[u8]) -> f32 {
        // Estimate error rate based on signal properties
        let amplitude_stability = 1.0 - signal.analog_amplitude.abs(); // Higher amplitude = lower error
        let frequency_stability = 1.0 / (signal.analog_frequency / 1000.0 + 1.0); // Higher freq = more errors
        let phase_stability = signal.analog_phase.sin().abs(); // Phase errors
        
        // Combine stability factors
        (amplitude_stability + frequency_stability + phase_stability) / 3.0
    }
    
    fn estimate_signal_quality(&self, signal: &MorseSpiralSignal) -> f32 {
        // Calculate signal quality based on analog parameter ranges
        let amplitude_quality = signal.analog_amplitude.clamp(0.0, 1.0);
        let frequency_quality = (signal.analog_frequency / 10000.0).clamp(0.0, 1.0); // Normalize to 0-10kHz
        let phase_quality = (signal.analog_phase / (2.0 * std::f32::consts::PI)).clamp(0.0, 1.0);
        
        (amplitude_quality + frequency_quality + phase_quality) / 3.0
    }
    
    fn estimate_resource_utilization(&self, signal: &MorseSpiralSignal) -> f32 {
        // Estimate resource utilization (higher = more resources used)
        let amplitude_util = signal.analog_amplitude; // Higher amplitude = more power
        let frequency_util = signal.analog_frequency / 10000.0; // Normalize frequency utilization
        let duration_util = signal.duration.as_micros() as f32 / 1000.0; // Normalize duration
        
        (amplitude_util + frequency_util + duration_util) / 3.0
    }
}

pub struct EfficiencyCalculator;

impl EfficiencyCalculator {
    pub fn calculate_compression_efficiency(&self, signal: &MorseSpiralSignal, original_data: &[u8]) -> f32 {
        let original_size = original_data.len() as f32;
        let compressed_size = signal.binary_data.count_ones() as f32 / 8.0; // Rough estimate
        
        if compressed_size > 0.0 {
            original_size / compressed_size
        } else {
            1.0 // No compression
        }
    }
}

pub struct StabilityAnalyzer;

impl StabilityAnalyzer {
    pub fn analyze_signal_stability(&self, signal: &MorseSpiralSignal) -> SignalStability {
        SignalStability {
            amplitude_stability: self.calculate_amplitude_stability(signal),
            frequency_stability: self.calculate_frequency_stability(signal),
            phase_stability: self.calculate_phase_stability(signal),
            overall_stability: self.calculate_overall_stability(signal),
        }
    }
    
    fn calculate_amplitude_stability(&self, signal: &MorseSpiralSignal) -> f32 {
        // Stability is highest when amplitude is in middle range (0.3-0.7)
        let target_range = (signal.analog_amplitude >= 0.3) && (signal.analog_amplitude <= 0.7);
        if target_range {
            1.0
        } else {
            1.0 - (signal.analog_amplitude - 0.5).abs() * 2.0 // Decreasing from middle
        }
    }
    
    fn calculate_frequency_stability(&self, signal: &MorseSpiralSignal) -> f32 {
        // Stability based on frequency range
        if signal.analog_frequency >= 1000.0 && signal.analog_frequency <= 5000.0 {
            1.0 // Optimal frequency range
        } else {
            0.5 // Reduced stability outside optimal range
        }
    }
    
    fn calculate_phase_stability(&self, signal: &MorseSpiralSignal) -> f32 {
        // Stability is highest at phase multiples of π/4
        let phase_normalized = signal.analog_phase % (std::f32::consts::PI / 4.0);
        let closest_multiple = (phase_normalized / (std::f32::consts::PI / 4.0)).round() * (std::f32::consts::PI / 4.0);
        1.0 - (closest_multiple - signal.analog_phase).abs() / (std::f32::consts::PI / 4.0)
    }
    
    fn calculate_overall_stability(&self, signal: &MorseSpiralSignal) -> f32 {
        let amp = self.calculate_amplitude_stability(signal);
        let freq = self.calculate_frequency_stability(signal);
        let phase = self.calculate_phase_stability(signal);
        
        (amp + freq + phase) / 3.0
    }
}

#[derive(Debug, Clone)]
pub struct SignalStability {
    pub amplitude_stability: f32,
    pub frequency_stability: f32,
    pub phase_stability: f32,
    pub overall_stability: f32,
}
```

### Protocol Variant Manager

```rust
pub struct ProtocolVariantManager {
    pub active_variants: Vec<ProtocolVariant>,
    pub fitness_history: Vec<(ProtocolVariant, f32)>,
    pub evolution_state: EvolutionState,
    pub diversity_preserver: DiversityPreserver,
    pub convergence_detector: ConvergenceDetector,
}

#[derive(Debug, Clone)]
pub struct ProtocolVariant {
    pub variant_id: String,
    pub base_signal: MorseSpiralSignal,
    pub mutation_history: Vec<MutationStrategy>,
    pub performance_score: f32,
    pub success_rate: f32,
    pub compression_ratio: f32,
    pub transmission_efficiency: f32,
    pub error_rate: f32,
    pub creation_timestamp: std::time::SystemTime,
    pub generation: u32,
}

pub struct EvolutionState {
    pub generation_count: u32,
    pub best_variant: Option<String>,
    pub diversity_score: f32,
    pub convergence_threshold: f32,
    pub population_size: usize,
    pub mutation_rate: f32,
    pub selection_pressure: f32,
}

impl ProtocolVariantManager {
    pub fn evolve_best_protocol(&mut self, signal: &MorseSpiralSignal) -> MorseSpiralSignal {
        let mut mutation_engine = ProtocolMutationEngine {
            mutation_strategies: vec![
                MutationStrategy::PositionShuffle,
                MutationStrategy::AmplitudeShift,
                MutationStrategy::FrequencyModulation,
                MutationStrategy::PhaseRotation,
                MutationStrategy::CompressionSwap,
            ],
            fitness_evaluator: FitnessEvaluator::new(),
            variant_manager: self.clone_for_threading(),
            evolution_history: vec![],
            selection_strategy: SelectionStrategy::Tournament,
        };
        
        // Generate multiple variants
        let variants: Vec<MorseSpiralSignal> = mutation_engine.mutation_strategies
            .iter()
            .map(|strategy| mutation_engine.generate_protocol_variant(signal, strategy))
            .collect();
        
        // Evaluate fitness of each variant
        let fitness_scores: Vec<f32> = variants
            .iter()
            .map(|var| self.evaluate_variant_fitness(var, signal))
            .collect();
        
        // Select best variant
        match fitness_scores.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)) {
            Some((best_idx, _score)) => variants[best_idx],
            None => *signal, // Return original if no variants
        }
    }
    
    fn clone_for_threading(&self) -> ProtocolVariantManager {
        // Create a thread-safe clone for mutation engine
        ProtocolVariantManager {
            active_variants: self.active_variants.clone(),
            fitness_history: self.fitness_history.clone(),
            evolution_state: self.evolution_state.clone(),
            diversity_preserver: self.diversity_preserver.clone(),
            convergence_detector: self.convergence_detector.clone(),
        }
    }
    
    fn evaluate_variant_fitness(&self, variant_signal: &MorseSpiralSignal, original: &MorseSpiralSignal) -> f32 {
        // Calculate fitness based on multiple factors
        let compression_improvement = (variant_signal.compression_ratio - original.compression_ratio).max(0.0);
        let transmission_efficiency = 1.0 / (variant_signal.duration.as_micros() as f32 * 0.001 + 0.01);
        let signal_quality = (variant_signal.analog_amplitude * variant_signal.analog_frequency / 1000.0).min(1.0);
        
        // Weighted combination
        compression_improvement * 0.4 + transmission_efficiency * 0.3 + signal_quality * 0.3
    }
    
    pub fn add_variant(&mut self, variant: ProtocolVariant) {
        self.active_variants.push(variant);
        
        // Maintain population size
        if self.active_variants.len() > self.evolution_state.population_size {
            self.maintain_population_diversity();
        }
    }
    
    fn maintain_population_diversity(&mut self) {
        // Remove least fit variants to maintain size
        self.active_variants.sort_by(|a, b| b.performance_score.partial_cmp(&a.performance_score).unwrap_or(std::cmp::Ordering::Equal));
        
        // Keep the best variants
        let keep_count = self.evolution_state.population_size;
        self.active_variants.truncate(keep_count);
    }
    
    pub fn get_best_variant(&self) -> Option<&MorseSpiralSignal> {
        self.active_variants
            .iter()
            .max_by(|a, b| a.performance_score.partial_cmp(&b.performance_score).unwrap_or(std::cmp::Ordering::Equal))
            .map(|variant| &variant.base_signal)
    }
}

#[derive(Debug, Clone)]
pub enum SelectionStrategy {
    Tournament,
    RouletteWheel,
    RankBased,
    Uniform,
}

pub struct DiversityPreserver {
    pub diversity_threshold: f32,
    pub diversity_measure: DiversityMeasure,
}

#[derive(Debug, Clone)]
pub enum DiversityMeasure {
    HammingDistance,
    EuclideanDistance,
    SignalCorrelation,
}

impl DiversityPreserver {
    pub fn preserve_diversity(&self, variants: &mut Vec<ProtocolVariant>) {
        // Remove similar variants to maintain diversity
        let mut to_remove = Vec::new();
        
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                if self.measure_diversity(&variants[i], &variants[j]) < self.diversity_threshold {
                    // Mark one for removal
                    to_remove.push(j);
                }
            }
        }
        
        // Remove marked variants
        to_remove.sort_unstable();
        to_remove.dedup();
        
        for &idx in to_remove.iter().rev() {
            if idx < variants.len() {
                variants.remove(idx);
            }
        }
    }
    
    fn measure_diversity(&self, variant1: &ProtocolVariant, variant2: &ProtocolVariant) -> f32 {
        match self.diversity_measure {
            DiversityMeasure::HammingDistance => self.hamming_distance(&variant1.base_signal, &variant2.base_signal),
            DiversityMeasure::EuclideanDistance => self.euclidean_distance(&variant1.base_signal, &variant2.base_signal),
            DiversityMeasure::SignalCorrelation => self.signal_correlation(&variant1.base_signal, &variant2.base_signal),
        }
    }
    
    fn hamming_distance(&self, signal1: &MorseSpiralSignal, signal2: &MorseSpiralSignal) -> f32 {
        // Calculate Hamming distance between binary components
        let diff = signal1.binary_data ^ signal2.binary_data;
        let differences = diff.count_ones();
        differences as f32 / 128.0 // Normalize to 0-1 (128 bits in u128)
    }
    
    fn euclidean_distance(&self, signal1: &MorseSpiralSignal, signal2: &MorseSpiralSignal) -> f32 {
        // Calculate Euclidean distance in signal parameter space
        let amp_diff = (signal1.analog_amplitude - signal2.analog_amplitude).powi(2);
        let freq_diff = ((signal1.analog_frequency - signal2.analog_frequency) / 1000.0).powi(2); // Normalize frequency
        let phase_diff = (signal1.analog_phase - signal2.analog_phase).powi(2);
        
        (amp_diff + freq_diff + phase_diff).sqrt()
    }
    
    fn signal_correlation(&self, signal1: &MorseSpiralSignal, signal2: &MorseSpiralSignal) -> f32 {
        // Calculate signal correlation
        let amp_corr = signal1.analog_amplitude * signal2.analog_amplitude;
        let freq_corr = (signal1.analog_frequency / signal2.analog_frequency).min(1.0);
        let phase_corr = (signal1.analog_phase.cos() * signal2.analog_phase.cos() + 
                         signal1.analog_phase.sin() * signal2.analog_phase.sin()).abs();
        
        (amp_corr + freq_corr + phase_corr) / 3.0
    }
}

pub struct ConvergenceDetector {
    pub convergence_threshold: f32,
    pub stagnation_generations: usize,
    pub history_window: usize,
}

impl ConvergenceDetector {
    pub fn detect_convergence(&self, fitness_history: &[f32]) -> bool {
        if fitness_history.len() < self.history_window {
            return false;
        }
        
        let recent_fitness: &[f32] = &fitness_history[fitness_history.len() - self.history_window..];
        
        // Calculate average fitness over recent generations
        let avg_recent = recent_fitness.iter().sum::<f32>() / recent_fitness.len() as f32;
        
        // Check for stagnation
        let max_recent = recent_fitness.iter().fold(0.0/0.0, |a, &b| a.max(b));
        let min_recent = recent_fitness.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        
        // If difference between max and min is small over many generations, we're converged
        let fitness_range = max_recent - min_recent;
        fitness_range < self.convergence_threshold
    }
}

#[derive(Debug, Clone)]
pub struct EvolutionStep {
    pub generation: u32,
    pub best_fitness: f32,
    pub average_fitness: f32,
    pub diversity_score: f32,
    pub timestamp: std::time::SystemTime,
    pub mutations_applied: Vec<MutationStrategy>,
}

impl Default for EvolutionState {
    fn default() -> Self {
        EvolutionState {
            generation_count: 0,
            best_variant: None,
            diversity_score: 0.5,
            convergence_threshold: 0.01,
            population_size: 50,
            mutation_rate: 0.1,
            selection_pressure: 1.5,
        }
    }
}

impl FitnessEvaluator {
    pub fn new() -> Self {
        FitnessEvaluator {
            performance_metrics: PerformanceMetrics {
                compression_ratio: 1.0,
                transmission_speed: 1000000.0,
                error_rate: 0.0,
                signal_quality: 0.5,
                resource_utilization: 0.5,
            },
            efficiency_calculator: EfficiencyCalculator {},
            stability_analyzer: StabilityAnalyzer {},
        }
    }
}
```

### Evolutionary Selection and Crossover

```rust
impl ProtocolVariantManager {
    pub fn select_parents(&self, tournament_size: usize) -> (Option<ProtocolVariant>, Option<ProtocolVariant>) {
        if self.active_variants.is_empty() {
            return (None, None);
        }
        
        // Tournament selection
        let parent1 = self.tournament_selection(tournament_size);
        let parent2 = self.tournament_selection(tournament_size);
        
        (parent1, parent2)
    }
    
    fn tournament_selection(&self, tournament_size: usize) -> Option<ProtocolVariant> {
        if self.active_variants.is_empty() {
            return None;
        }
        
        let mut competitors = Vec::new();
        let sample_size = tournament_size.min(self.active_variants.len());
        
        // Randomly select competitors from the population
        for _ in 0..sample_size {
            let idx = rand::random::<usize>() % self.active_variants.len();
            competitors.push(self.active_variants[idx].clone());
        }
        
        // Select the best competitor
        competitors
            .into_iter()
            .max_by(|a, b| a.performance_score.partial_cmp(&b.performance_score).unwrap_or(std::cmp::Ordering::Equal))
    }
    
    pub fn crossover(&self, parent1: &ProtocolVariant, parent2: &ProtocolVariant) -> MorseSpiralSignal {
        // Perform crossover between two parent signals
        let mut offspring = parent1.base_signal;
        
        // Mix analog parameters from both parents
        offspring.analog_amplitude = (parent1.base_signal.analog_amplitude + parent2.base_signal.analog_amplitude) / 2.0;
        offspring.analog_frequency = (parent1.base_signal.analog_frequency + parent2.base_signal.analog_frequency) / 2.0;
        offspring.analog_phase = (parent1.base_signal.analog_phase + parent2.base_signal.analog_phase) / 2.0;
        
        // Mix binary data (simple bitwise operation)
        offspring.binary_data = parent1.base_signal.binary_data ^ parent2.base_signal.binary_data;
        
        // Average durations
        offspring.duration = std::time::Duration::from_micros(
            (parent1.base_signal.duration.as_micros() + parent2.base_signal.duration.as_micros()) as u64 / 2
        );
        
        // Average compression ratios
        offspring.compression_ratio = (parent1.base_signal.compression_ratio + parent2.base_signal.compression_ratio) / 2.0;
        
        offspring
    }
    
    pub fn adaptive_mutation_rate(&mut self) -> f32 {
        // Adjust mutation rate based on population diversity
        let diversity = self.calculate_population_diversity();
        
        // Lower mutation rate if population is diverse, higher if converging
        let base_rate = self.evolution_state.mutation_rate;
        let diversity_factor = 1.0 - diversity; // Lower diversity = higher mutation rate
        
        (base_rate * (1.0 + diversity_factor)).min(0.5) // Cap at 50%
    }
    
    fn calculate_population_diversity(&self) -> f32 {
        if self.active_variants.len() < 2 {
            return 1.0; // Max diversity for single variant
        }
        
        let mut total_distance = 0.0;
        let mut count = 0;
        
        for i in 0..self.active_variants.len() {
            for j in (i + 1)..self.active_variants.len() {
                let dist = self.calculate_signal_distance(
                    &self.active_variants[i].base_signal,
                    &self.active_variants[j].base_signal
                );
                total_distance += dist;
                count += 1;
            }
        }
        
        if count == 0 {
            return 1.0;
        }
        
        total_distance / count as f32
    }
    
    fn calculate_signal_distance(&self, signal1: &MorseSpiralSignal, signal2: &MorseSpiralSignal) -> f32 {
        // Calculate normalized distance between signals
        let amp_dist = (signal1.analog_amplitude - signal2.analog_amplitude).abs();
        let freq_dist = ((signal1.analog_frequency - signal2.analog_frequency) / 10000.0).abs(); // Normalize frequency
        let phase_dist = (signal1.analog_phase - signal2.analog_phase).abs() / (2.0 * std::f32::consts::PI);
        let duration_dist = ((signal1.duration.as_micros() as f32 - signal2.duration.as_micros() as f32) / 1000.0).abs();
        
        (amp_dist + freq_dist + phase_dist + duration_dist) / 4.0
    }
}

impl DiversityPreserver {
    pub fn calculate_population_diversity(&self, variants: &[ProtocolVariant]) -> f32 {
        if variants.len() < 2 {
            return 1.0;
        }
        
        let mut total_diversity = 0.0;
        let mut comparisons = 0;
        
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                let diversity = self.measure_diversity(&variants[i], &variants[j]);
                total_diversity += diversity;
                comparisons += 1;
            }
        }
        
        if comparisons == 0 {
            return 1.0;
        }
        
        total_diversity / comparisons as f32
    }
}
```

This protocol mutation system provides evolutionary capabilities for the Morse-Spiral Protocol, enabling it to adapt and optimize through various mutation strategies, fitness evaluation, and population management. The system maintains diversity while driving toward better performance metrics through evolutionary selection and crossover operations.