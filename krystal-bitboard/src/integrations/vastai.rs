use log::{info, warn};
use tokio::time::{sleep, Duration};
use crate::grid::MemoryGrid;
use std::sync::Arc;

/// Integrácia č. 1: Transformácia Krypto-Tlačiarní na Cloud-Native Datacentrá
/// (Napr. Vast.ai, Render Network)
///
/// Modul komunikuje cez REST API s cloudovými službami a získava okamžité bidy na GPU čas.
pub async fn start_cloud_render_listener(grid: Arc<MemoryGrid>) {
    info!("☁️ [Vast.ai / Cloud Bidding] Pripájam na cloud trh...");

    loop {
        // Simulujeme príchod lukratívnej požiadavky na 3D renderovanie.
        // Normálne by to bolo: let response = http_client.get("vast.ai/api/bids").await;
        
        sleep(Duration::from_secs(45)).await; // Očakáva prácu

        info!("🔔 [Cloud Bidding] Nový job prijatý! Odmena: 1.50 €/h za OctaneRender úlohu.");

        // Krystal-bitboard dynamicky presunie prioritu na Render Zónu 2 bez zahasenia Miningu (Zóna 1 na 10%).
        let mut zones = grid.zones.lock().unwrap();
        if let Some(miner_zone) = zones.iter_mut().find(|z| z.id == 1) {
            miner_zone.percent = 10.0; 
        }
        if let Some(render_zone) = zones.iter_mut().find(|z| z.id == 2) {
            render_zone.percent = 80.0;
        }
        drop(zones);
        grid.recalculate();

        info!("🟢 [Cloud Bidding] Zero-Latency Slicing úspešný: VRAM prealokovaná pre render posao.");
        
        sleep(Duration::from_secs(120)).await; // Job prebieha
        
        info!("✅ [Cloud Bidding] Render Job úspešne dokončený. Releasing VRAM...");
        // Governor ho sám prehodí naspäť po detekcii poklesu zisku z render zóny.
    }
}
