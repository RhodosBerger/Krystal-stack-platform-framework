use std::sync::Arc;
use tokio::time::{sleep, Duration};
use log::info;
use crate::grid::MemoryGrid;

pub async fn start_render_pipeline(grid: Arc<MemoryGrid>) {
    info!("🎬 Štartujem Zónu 2: Blender Cycles / Vulkan Render Farm");

    loop {
        // Objemné dáta raytracing gridu: žiadame 500 MB.
        let chunk_size = 500; 
        
        if grid.lock_memory(2, chunk_size) {
            // Čas potrebný na vypočítanie chunk-u
            sleep(Duration::from_millis(2000)).await;
            
            grid.free_memory(2, chunk_size);
        } else {
            sleep(Duration::from_secs(1)).await;
        }
    }
}
