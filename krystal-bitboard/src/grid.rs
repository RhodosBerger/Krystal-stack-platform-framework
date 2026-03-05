use std::sync::Mutex;
use serde::{Deserialize, Serialize};
use log::{info, warn};

// Import z typov Vulkano je tu k dispozícii, ukážka pre napojenie:
// use vulkano::memory::allocator::{MemoryAllocateInfo};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Zone {
    pub id: u8,
    pub name: &'static str,
    pub percent: f32,
    pub size_mb: u64,
    pub used_mb: u64,
}

pub struct MemoryGrid {
    pub total_vram_mb: u64,
    pub zones: Mutex<Vec<Zone>>,
}

impl MemoryGrid {
    pub fn new(total_vram_mb: u64) -> Self {
        let zones = vec![
            Zone { id: 1, name: "Mining", percent: 30.0, size_mb: 0, used_mb: 0 },
            Zone { id: 2, name: "Render", percent: 50.0, size_mb: 0, used_mb: 0 },
            Zone { id: 3, name: "Edge AI", percent: 15.0, size_mb: 0, used_mb: 0 },
            Zone { id: 4, name: "OB/Buffer", percent: 5.0, size_mb: 0, used_mb: 0 },
        ];

        let grid = Self {
            total_vram_mb,
            zones: Mutex::new(zones),
        };
        grid.recalculate();
        grid
    }

    /// Prepočítava MB na základe definovaných % pre zóny
    pub fn recalculate(&self) {
        let mut zones = self.zones.lock().unwrap();
        for z in zones.iter_mut() {
            z.size_mb = (self.total_vram_mb as f64 * (z.percent as f64 / 100.0)) as u64;
        }
    }

    /// Mení percento zóny
    pub fn resize_zone(&self, zone_id: u8, percent: f32) {
        let mut zones = self.zones.lock().unwrap();
        if let Some(z) = zones.iter_mut().find(|z| z.id == zone_id) {
            z.percent = percent;
            info!("🔧 MemGrid Realloc: Zóna {} ({}) zmenená na {:.1} %", z.id, z.name, percent);
        }
    }
    
    /// VRAM Allocator: vracia true ak memory limit v zóne nebol prekročený. Zero-copy alokátor bez GC.
    pub fn lock_memory(&self, zone_id: u8, request_mb: u64) -> bool {
        let mut zones = self.zones.lock().unwrap();
        if let Some(z) = zones.iter_mut().find(|z| z.id == zone_id) {
            if z.used_mb + request_mb > z.size_mb {
                warn!("⚠️ Realloc Trigger! Fragmentácia v Zóne {} ({}). Použité: {}/{}, Žiadané: {}", 
                    z.id, z.name, z.used_mb, z.size_mb, request_mb);
                return false;
            }
            z.used_mb += request_mb;
            return true;
        }
        false
    }

    /// Dealokácia pamäte, po dispatchi príkazu shaderu.
    pub fn free_memory(&self, zone_id: u8, release_mb: u64) {
        let mut zones = self.zones.lock().unwrap();
        if let Some(z) = zones.iter_mut().find(|z| z.id == zone_id) {
            z.used_mb = z.used_mb.saturating_sub(release_mb);
        }
    }
}
