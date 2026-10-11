//! GAMESA Rust Bot - Main Entry Point

use gamesa_bot::{GamesaBot, Config, TelemetrySnapshot};
use chrono::Utc;
use std::time::Duration;

fn main() {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("GAMESA Rust Bot v0.1.0");
    println!("======================");

    // Load or create config
    let config = Config::default();
    println!("Config loaded: {:?}", config.log_path);

    // Create bot
    let mut bot = GamesaBot::new(config);
    println!("Bot initialized");

    // Demo: Process some telemetry
    let telemetry = TelemetrySnapshot {
        timestamp: Utc::now(),
        cpu_util: 0.75,
        gpu_util: 0.80,
        memory_util: 0.60,
        temp_cpu: 72.0,
        temp_gpu: 68.0,
        frametime_ms: 16.6,
        power_draw: 120.0,
        zone_id: 0,
        pe_core_mask: 0xFF,
    };

    println!("\nProcessing telemetry...");
    let decision = bot.process(telemetry);

    println!("\nTrusted Decision:");
    println!("  ID: {}", decision.id);
    println!("  Action: {:?}", decision.action);
    println!("  Domain: {:?}", decision.domain);
    println!("  Confidence: {:.2}", decision.confidence);
    println!("  Thermal Safe: {}", decision.validation.thermal_safe);
    println!("  Power Safe: {}", decision.validation.power_safe);

    // Get system state
    let state = bot.get_state();
    println!("\nSystem State:");
    println!("  CPU: {} MHz, {} P-cores, {} E-cores",
        state.cpu_state.frequency_mhz,
        state.cpu_state.p_cores_active,
        state.cpu_state.e_cores_active);
    println!("  GPU: {:.1}% util, {:.1}°C",
        state.gpu_state.utilization * 100.0,
        state.gpu_state.temperature);
    println!("  Grid: {}/{} cells active",
        state.grid_state.active_cells,
        state.grid_state.total_cells);
    println!("  Features: {:?}", state.feature_flags);

    println!("\nBot ready for IPC connections.");
}
