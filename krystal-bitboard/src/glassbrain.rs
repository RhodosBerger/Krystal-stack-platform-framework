use crate::BitboardState;
use std::sync::Arc;
use serde_json::json;
use tokio::time::{sleep, Duration};
use log::info;
use std::sync::atomic::Ordering;

/// Vygeneruje okamžitý snapshot "trajectory" pre administráciu.
pub fn generate_trajectory(state: Arc<BitboardState>) -> String {
    let mining_on = state.mining_active.load(Ordering::Relaxed);
    let render_on = state.render_active.load(Ordering::Relaxed);
    let oc_on = state.overclock_active.load(Ordering::Relaxed);
    
    let mining_profit = if mining_on { 0.40 } else { 0.0 }; // Realistický mock na 3060
    let render_profit = if render_on { 1.35 } else { 0.0 }; // 0.05€ per frame * 30 FPS * network_factor
    
    let total_profit = mining_profit + render_profit;

    let j = json!({
        "module": "krystal-bitboard",
        "active_services": {
            "vulkan_btc_miner": mining_on,
            "render_farm_node": render_on,
            "shadow_council_oc": oc_on,
        },
        "financial_metrics_eur_per_hr": {
            "mining_revenue": mining_profit,
            "render_revenue": render_profit,
            "total_hybrid_revenue": total_profit
        },
        "trajectory": {
            "predicted_temp_5m": if oc_on { 68.2 } else { 62.5 },
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
