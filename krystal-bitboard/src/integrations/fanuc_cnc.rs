use log::{info, warn};
use tokio::time::{sleep, Duration};
use std::sync::Arc;
use crate::grid::MemoryGrid;
use crate::ob::TemperaturePredictor;

/// Integrácia č. 3: Záchranná brzda pomocou Shadow Council
/// (FANUC CNC stroje a Gamesa Cortex V2 riadenie kvality)
///
/// Keď je Edge AI Zóna 3 ohrozená teplotou, tento modul prerušuje ostatné úlohy.
pub async fn start_fanuc_cnc_guardian(grid: Arc<MemoryGrid>) {
    info!("🏭 [FANUC CNC Guardian] Štartujem systém prežitia linky (Shadow Council Agent).");
    
    let mut mock_predictor = TemperaturePredictor::new();

    loop {
        sleep(Duration::from_secs(10)).await;
        
        // Predstavme si, že teplota haly (alebo server case-u) prudko stúpla.
        mock_predictor.add_sample(62.0);
        mock_predictor.add_sample(66.0); // +4 C skok!

        if mock_predictor.rapid_increase() {
            warn!("🚨 [FANUC CNC Guardian] Detekovaný prudký skok. Ochrana Edge AI inferencie (Zóna 3) aktivovaná!");
            warn!("📉 Okamžitý DOWNGRADE. Mining a Render sú suspendované na O MB.");

            let mut zones = grid.zones.lock().unwrap();
            
            // Hard limit: všetko ide do nuly okrem Edge zóny.
            for z in zones.iter_mut() {
                if z.id != 3 {
                     z.percent = 0.0;
                } else {
                     z.percent = 100.0; // FANUC OpenCV safety dostane plný VRAM bandwidth aby nespôsobilo haváriu.
                }
            }
            drop(zones);
            grid.recalculate();

            // Sme offline na inych úlohach na par minút
            sleep(Duration::from_secs(120)).await; 
            info!("♻️ [FANUC CNC Guardian] Termálna udalosť odznieva. Povoľujem Governorovi prevziať opäť kontrolu.");
        }
    }
}
