use std::sync::Arc;
use tokio::time::{sleep, Duration};
use chrono::{Local, Timelike};
use crate::{BitboardState, ob, governor};
use std::sync::atomic::Ordering;
use log::{info, warn};

#[derive(Debug, PartialEq)]
pub enum HybridMode {
    AutoDayNight,
    RenderFocus,
    MiningFocus,
}

pub async fn run_scheduler(state: Arc<BitboardState>) {
    let mut predictor = ob::TemperaturePredictor::new();
    let council = governor::ShadowCouncil::new();

    loop {
        if !state.running.load(Ordering::SeqCst) {
            break;
        }

        // Simulácia aktuálnych hodnôt telemetrie
        let current_telemetry = ob::ExtendedTelemetry {
            gpu_temp: 64.2,
            vram_temp: 86.0,
            power_percent: 78.0,
            util_percent: 92.0,
            idle_percent: 8.0,
            core_clock: 1450,
            mem_clock: 7100,
            fan_rpm: 1900,
            cpu_temp: 50.0,
            voltage: 0.85,
            ambient_temp: 23.0,
            hashrate: 55.4,
            rejected_shares: 0.1,
            render_queue_latency: 120.0,
            edge_task_priority: 0,
        };

        predictor.add_sample(current_telemetry.gpu_temp);

        // Security trigger - najvyššia priorita
        if predictor.rapid_increase() || current_telemetry.rejected_shares > 1.0 || current_telemetry.edge_task_priority > 5 {
            warn!("🚨 Bezpečnostný Trigger! Vypínam ťažbu a OC. Edge Task: {}, Skokové GPU teplo: {}", 
                  current_telemetry.edge_task_priority, predictor.rapid_increase());
            
            state.mining_active.store(false, Ordering::SeqCst);
            if state.overclock_active.load(Ordering::SeqCst) {
                ob::reset_overclock();
                state.overclock_active.store(false, Ordering::SeqCst);
            }
        } else {
            // Day/Night and Render vGPU Slicing
            let hour = Local::now().hour();
            let is_night = hour >= 22 || hour < 6; // Od 22:00 do 06:00

            let active_mode = {
                let m = state.hybrid_mode.lock().unwrap();
                match *m {
                    HybridMode::AutoDayNight => if is_night { HybridMode::MiningFocus } else { HybridMode::RenderFocus },
                    HybridMode::MiningFocus => HybridMode::MiningFocus,
                    HybridMode::RenderFocus => HybridMode::RenderFocus,
                }
            };

            match active_mode {
                HybridMode::MiningFocus => {
                    // 100% mining capacity via MPS allocation release
                    state.mining_active.store(true, Ordering::SeqCst);
                    state.render_active.store(false, Ordering::SeqCst);
                },
                HybridMode::RenderFocus => {
                    // Denný režim: 80% Idle je pre Render/Edge AI, 20% mining
                    state.mining_active.store(true, Ordering::SeqCst);
                    state.render_active.store(true, Ordering::SeqCst);
                },
                _ => {}
            }

            // Aplikácia Shadow Councilu pre Overclocking (Len ak sme v bezpeči z Power limitov)
            let future_temp_5m = predictor.predict_temp_5m();
            let allowed_ob = current_telemetry.power_percent < 80.0 && 
                             current_telemetry.gpu_temp < 68.0 &&
                             (active_mode == HybridMode::MiningFocus || current_telemetry.idle_percent > 85.0);

            if allowed_ob && council.evaluate(&current_telemetry, future_temp_5m) {
                if !state.overclock_active.load(Ordering::SeqCst) {
                    ob::apply_overclock();
                    state.overclock_active.store(true, Ordering::SeqCst);
                }
            } else if state.overclock_active.load(Ordering::SeqCst) {
                ob::reset_overclock();
                state.overclock_active.store(false, Ordering::SeqCst);
            }
        }

        sleep(Duration::from_secs(5)).await;
    }
}
