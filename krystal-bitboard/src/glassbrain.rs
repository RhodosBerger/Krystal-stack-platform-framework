use crate::BitboardState;
use std::sync::Arc;
use serde_json::json;
use tokio::time::{sleep, Duration};
use log::info;
use std::sync::atomic::Ordering;

/// Vygeneruje okamžitý snapshot "trajectory" pre administráciu.
pub fn generate_trajectory(state: Arc<BitboardState>) -> String {
    let mining_on = state.mining_active.load(Ordering::Relaxed);
    let oc_on = state.overclock_active.load(Ordering::Relaxed);
    
    let j = json!({
        "module": "krystal-bitboard",
        "active_services": {
            "vulkan_btc_miner": mining_on,
            "render_farm_node": state.render_active.load(Ordering::Relaxed),
            "shadow_council_oc": oc_on,
        },
        "trajectory": {
            "predicted_temp_5m": if oc_on { 68.2 } else { 62.5 },
            "hourly_profit_eur": if mining_on { 0.45 } else { 0.0 }, // Accountant mockup
            "downtime_risk_pct": if oc_on { 3.5 } else { 0.1 }
        }
    });
    j.to_string()
}

/// Spustí background worker pre pravidelný export / event streamovanie.
pub fn start_exporter(_state: Arc<BitboardState>) {
    info!("🧠 Glass Brain Trajectory Exporter zahájený.");
    tokio::spawn(async move {
        loop {
            // Predpríprava pre WebSockets alebo MQ
            sleep(Duration::from_secs(30)).await;
        }
    });
}
