use std::sync::Arc;
use tokio::time::{sleep, Duration};
use log::info;
use crate::grid::MemoryGrid;

pub async fn start_edge_pipeline(grid: Arc<MemoryGrid>) {
    info!("🧠 Štartujem Zónu 3: Edge AI (ONNX inference, Voxel planning)");

    loop {
        // Predstavuje rýchlu operáciu (priorita > 8), ale málo memory náročnú
        let request_mem = 50; 
        
        if grid.lock_memory(3, request_mem) {
            // Inference execution v nanosekundách / jednotkách ms
            sleep(Duration::from_millis(50)).await;
            
            grid.free_memory(3, request_mem);
        } else {
            sleep(Duration::from_millis(200)).await;
        }
    }
}
