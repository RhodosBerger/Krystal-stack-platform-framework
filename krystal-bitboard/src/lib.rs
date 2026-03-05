pub mod grid;
pub mod miner;
pub mod render;
pub mod edge;
pub mod ob;
pub mod governor;
pub mod enterprise;

use std::sync::Arc;
use grid::MemoryGrid;

/// Inicializuje Memory Grid a všetky worker nodové príkazy.
pub fn start_grid() -> Arc<MemoryGrid> {
    // 8 GB (8192 MB) RTX 3060 vRAM Simulácia
    let grid = Arc::new(MemoryGrid::new(8192));
    
    // Zóna 1: Mining
    let grid_m = grid.clone();
    tokio::spawn(async move { miner::start_miner_pipeline(grid_m).await; });

    // Zóna 2: Render
    let grid_r = grid.clone();
    tokio::spawn(async move { render::start_render_pipeline(grid_r).await; });

    // Zóna 3: Edge AI
    let grid_e = grid.clone();
    tokio::spawn(async move { edge::start_edge_pipeline(grid_e).await; });

    // Zóna 4: Overclock Buffer
    let grid_o = grid.clone();
    tokio::spawn(async move { ob::start_telemetry(grid_o).await; });

    // Economic Governor pre realloc
    let grid_g = grid.clone();
    tokio::spawn(async move { governor::start_economic_governor(grid_g).await; });

    // --- Komerčná VRSTVA (Enterprise Upgrades) ---
    tokio::spawn(async move { enterprise::run_big_data_shipper().await; });
    tokio::spawn(async move { enterprise::run_central_admin().await; });
    tokio::spawn(async move { enterprise::run_decentralized_admin().await; });

    grid
}

/// Upraví vyhradené percentá pre zónu ručne.
pub fn resize_grid(grid: &Arc<MemoryGrid>, zone_id: u8, percent: f32) {
    grid.resize_zone(zone_id, percent);
    grid.recalculate();
}

/// JSON export per Glass Brain požiadavku.
pub fn get_grid_json(grid: &Arc<MemoryGrid>) -> String {
    let zones = grid.zones.lock().unwrap();
    let mining_pct = zones.iter().find(|z| z.id == 1).map(|z| z.percent).unwrap_or(0.0);
    
    let pred_msg = if mining_pct < 5.0 {
        "okamžité zotavenie: VRAM poskytnutá plne pre inference"
    } else {
        "za 5 min: realloc mining na 0 %, pretože render deadline"
    };

    let j = serde_json::json!({
        "module": "krystal-bitboard",
        "grid_state": {
            "total_vram_mb": grid.total_vram_mb,
            "zones": *zones
        },
        "predikcia": pred_msg
    });
    
    j.to_string()
}
