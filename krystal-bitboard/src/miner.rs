use std::sync::Arc;
use tokio::time::{sleep, Duration};
use log::info;
use crate::grid::MemoryGrid;

pub async fn start_miner_pipeline(grid: Arc<MemoryGrid>) {
    info!("⛏️ Štartujem Zónu 1: Vulkan SHA-256 Mining Pipeline");
    info!("🟢 Pripojené k stratum+tcp://pool.eu.ethermine.org:4444");

    loop {
        let alloc_size = 200; // Požadovaná alokácia mid-state cyklu (200 MB batch)
        
        // Block memory dispatch podľa Vulkano prístupu
        if grid.lock_memory(1, alloc_size) {
            // Simulácia samotného behu computera na karte...
            sleep(Duration::from_millis(1500)).await;
            
            // Releasing bufferov po vykonaní
            grid.free_memory(1, alloc_size);
        } else {
            // Nemáme RAM: Render alebo Edge nás obmedzil / governor znížil % na 0.
            sleep(Duration::from_secs(2)).await;
        }
    }
}
