use std::sync::Arc;
use tokio::time::{sleep, Duration};
use log::{info, warn};
use chrono::{Local, Timelike};
use crate::grid::MemoryGrid;

pub async fn start_economic_governor(grid: Arc<MemoryGrid>) {
    info!("🏛️ Štartujem Economic Governor (Dual-Income: Mining vs. Render Profit Control)");

    loop {
        // --- DUAL-INCOME: Dynamické vyhodnocovanie trhu ---
        // Accountant simuluje zárobky na Edge node:
        let mining_profit_eur_per_hr = 0.40; // Aktuálny BTC profit
        
        // Zisky z renderingu (napr. z OctaneRender / Blender cloud network) 
        // Vypočítané z dostupných jobov a odmeny za frame zníženej o network letency.
        let render_frame_reward_eur = 0.05; 
        let frames_per_hr = 30.0;
        let base_render_profit = render_frame_reward_eur * frames_per_hr; // 1.50 €/h
        
        // Ak sa render latency zhorší (edge node timeout), reward klesá.
        let network_latency_penalty = 0.90; // 90% efektivita pripojenia
        let render_profit_eur_per_hr = base_render_profit * network_latency_penalty;

        let hr = Local::now().hour();
        let is_night = hr >= 22 || hr < 6;

        info!("💰 Trh: Mining={:.2} €/h | Render={:.2} €/h", mining_profit_eur_per_hr, render_profit_eur_per_hr);

        // --- Adaptívne Alokovanie Prostriedkov v zmysle zníženia latencií ---
        if render_profit_eur_per_hr > (mining_profit_eur_per_hr * 2.0) {
            // Render zarába viac ako dvojnásobne: Drastické zníženie latencie pre Render
            info!("📈 Render Profit dominuje! Púšťam 85% VRAM do Zóny 2 (Zníženie latencie pre rýchlejší zárobok).");
            grid.resize_zone(1, 0.0);  // Mining na pauzu, aby nerušil bandwidth
            grid.resize_zone(2, 85.0); // Render max bandwidth
            grid.resize_zone(3, 10.0); // Edge inference
            grid.resize_zone(4, 5.0);  // Buffer zostatok
        } else if mining_profit_eur_per_hr < 0.5 && !is_night {
            // Zisk padol pod minimum, okamžite alokuj na nulu a prepust vRAM!
            warn!("📉 Mining profit < 0.5 €/h. Zóne 1 nariaďujem 0% Grid Limit!");
            
            grid.resize_zone(1, 0.0);
            grid.resize_zone(2, 70.0); 
            grid.resize_zone(3, 25.0); // Prepneme VRAM viac na Edge AI
            grid.resize_zone(4, 5.0);
        } else if is_night {
            // Nocturnal režim (Lacnejší prúd = ťažba BTC letí)
            info!("🌙 Nocturnal Mód: ZLacnená elektrina, Mining focus.");
            grid.resize_zone(1, 80.0); // Mining ide na full kapacitu vRAM
            grid.resize_zone(2, 5.0);
            grid.resize_zone(3, 10.0);
            grid.resize_zone(4, 5.0);
        } else {
            // Vyvážený Hybridný Balance Focus pre maximálny P&L spread
            info!("⚖️ Dual-Earn Balance: Zbierame z oboch svetov paralelne bez straty priorít.");
            grid.resize_zone(1, 20.0); // Udržovanie Stratum syncu a jemný profit
            grid.resize_zone(2, 60.0); // Stabilný rendering
            grid.resize_zone(3, 15.0); // Edge
            grid.resize_zone(4, 5.0);
        }
        
        // Aplly the resizes to limits
        grid.recalculate();
        
        sleep(Duration::from_secs(30)).await;
    }
}
