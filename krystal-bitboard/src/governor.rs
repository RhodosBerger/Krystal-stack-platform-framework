use std::sync::Arc;
use tokio::time::{sleep, Duration};
use log::{info, warn};
use chrono::{Local, Timelike};
use crate::grid::MemoryGrid;

pub async fn start_economic_governor(grid: Arc<MemoryGrid>) {
    info!("🏛️ Štartujem Economic Governor (Mining Cost/Profit Control)");

    loop {
        // Simulácia ekonomickej role (Accountant)
        let mining_profit_eur_per_hr = 0.40; // Profit je menej ako 0.5 eur 
        
        let hr = Local::now().hour();
        let is_night = hr >= 22 || hr < 6;

        if mining_profit_eur_per_hr < 0.5 {
            // Zisk padol pod minimum, okamžite alokuj na nulu a prepust vRAM!
            warn!("📉 Baza Economic Governor: Mining profit < 0.5 €/h. Zóne 1 nariaďujem 0% Grid Limit!");
            
            grid.resize_zone(1, 0.0);
            grid.resize_zone(2, 80.0); // Render absorbuje 80% vRAM
            grid.resize_zone(3, 15.0); // Edge inference
            grid.resize_zone(4, 5.0);  // Buffer zostatok
        } else if is_night {
            // Nocturnal režim (Lacnejší prúd, voľný Edge)
            grid.resize_zone(1, 80.0); // Mining ide na full kapacitu vRAM
            grid.resize_zone(2, 5.0);
            grid.resize_zone(3, 10.0);
            grid.resize_zone(4, 5.0);
        } else {
            // Bežný Denný Hybrid Render / AI Focus
            grid.resize_zone(1, 10.0); // Podkladový mining (napr pool ping pong)
            grid.resize_zone(2, 70.0);
            grid.resize_zone(3, 15.0);
            grid.resize_zone(4, 5.0);
        }
        
        // Aplly the resizes to limits
        grid.recalculate();
        
        sleep(Duration::from_secs(25)).await;
    }
}
