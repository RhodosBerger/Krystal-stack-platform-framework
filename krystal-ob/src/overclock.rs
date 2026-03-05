use crate::telemetry::TelemetryData;
use log::{info, warn};

pub struct ShadowCouncil;

impl ShadowCouncil {
    pub fn new() -> Self {
        Self
    }

    /// Vyhodnotí podmienky pretaktovania (Creator, Auditor, Accountant vzory)
    pub fn evaluate(&self, data: &TelemetryData, predicted_temp: f64) -> bool {
        // Creator (Návrh generátora):
        // Creator navrhuje pridať +50 MHz core a +100 MHz mem pre maximalizáciu edge performance a hashrate.
        
        // Auditor (Riziko check):
        // Auditor kontroluje, či predikcia teploty nevylezie nad 75 °C a nepribúdajú invalidné shares.
        let auditor_approves = predicted_temp < 75.0 && data.rejected_shares_percent < 1.0;

        // Accountant (Ekonomika):
        // Počíta hypotetický €/W zisk - či marginálny boost nie je drahší o 20% prúdu. 
        // Tu zjednodušene ak power utilization je pod 85%, sme ekonomicky ok.
        let accountant_approves = data.power_percent < 85.0;

        if !auditor_approves {
            warn!("Auditor zamietol návrh! Riziko prehriatia (Predikcia: {:.1} °C) alebo vysoký reject rate.", predicted_temp);
        }

        auditor_approves && accountant_approves
    }
}

/// Aplikuje boost cez Linux utiliy pre NVIDIA (nvidia-settings alebo nvml) / AMD (rocm-smi)
pub fn apply_overclock(core_offset: i32, mem_offset: i32) {
    info!("Aplikujem Shadow Council OC: Core +{} MHz, Mem +{} MHz", core_offset, mem_offset);
    // V produkcii napr: 
    // Command::new("nvidia-settings")
    //     .args(&["-a", &format!("[gpu:0]/GPUGraphicsClockOffset[3]={}", core_offset)])
    //     .output().unwrap();
}

/// Okamžitý downgrade v prípade kritického tasku z MQTT
pub fn reset_overclock() {
    info!("Kritická zmena statusu: Resetujem GPU takty na stock hodnoty (Downgrade)!");
    // Command::new("nvidia-smi").args(&["-rac"]).output();
}
