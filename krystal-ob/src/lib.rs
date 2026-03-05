pub mod governor;
pub mod overclock;
pub mod predict;
pub mod telemetry;
pub mod mqtt;
pub mod ipc;
pub mod api;

use std::sync::{Arc, Mutex};
use tokio::time::{sleep, Duration};
use telemetry::{TelemetryData, TelemetryCollector};
use governor::EconomicGovernor;
use overclock::ShadowCouncil;
use predict::TemperaturePredictor;
use mqtt::EdgeStatusState;
use serde_json::json;

pub struct ObMonitor {
    running: Arc<Mutex<bool>>,
    latest_telemetry: Arc<Mutex<Option<TelemetryData>>>,
    latest_prediction: Arc<Mutex<f64>>,
}

/// Spustí na pozadí asynchrónny monitor a Shadow Council agenta
pub fn start_ob_monitor() -> Arc<ObMonitor> {
    let running = Arc::new(Mutex::new(true));
    let latest_telemetry = Arc::new(Mutex::new(None));
    let latest_prediction = Arc::new(Mutex::new(0.0));

    let run_flag = running.clone();
    let tele_ref = latest_telemetry.clone();
    let pred_ref = latest_prediction.clone();

    let monitor = Arc::new(ObMonitor {
        running: running.clone(),
        latest_telemetry: latest_telemetry.clone(),
        latest_prediction: latest_prediction.clone(),
    });

    let edge_state = EdgeStatusState::new();
    let edge_state_worker = edge_state.clone();

    tokio::spawn(async move {
        mqtt::start_mqtt_listener(edge_state_worker.clone()).await;
        // Start IPC server pre komunikáciu s krystal-miner
        ipc::start_ipc_server("/tmp/krystal_ob.sock", edge_state_worker.clone()).await;
        
        let collector = TelemetryCollector::new(edge_state_worker);
        let mut predictor = TemperaturePredictor::new();
        let council = ShadowCouncil::new();
        let governor = EconomicGovernor::new();

        while *run_flag.lock().unwrap() {
            // 1. Zber vstupnej telemetrie z HW
            let current_telemetry = collector.collect_telemetry().await;
            
            // 2. Teplotná predikcia (Rolling Buffer, lineárna regresia)
            predictor.add_sample(current_telemetry.gpu_temp);
            let pred_temp = predictor.predict_in_5_min();
            *pred_ref.lock().unwrap() = pred_temp;

            // 3. Bezpečnostná brzda (+3°C za 10 sekúnd alebo vysoký reject rate)
            if predictor.has_rapid_increase() || current_telemetry.rejected_shares_percent > 1.0 {
                log::warn!("Aktivovaná bezpečnostná brzda! Okamžitý downgrade.");
                overclock::reset_overclock();
                sleep(Duration::from_secs(5)).await; // pauza
                continue;
            }

            // 4. Integrácia s Economic Governor
            if governor.is_overclock_allowed(&current_telemetry) {
                // 5. Overclock Agent (Shadow Council: Creator, Auditor, Accountant)
                if council.evaluate(&current_telemetry, pred_temp) {
                    overclock::apply_overclock(50, 100);
                }
            } else {
                // Ak Governor nepovolí (napr. kritická edge task), downgrade plynule
                overclock::reset_overclock();
            }

            *tele_ref.lock().unwrap() = Some(current_telemetry);
            sleep(Duration::from_secs(1)).await; // 1s interval telemetrie
        }
    });

    let monitor_ref = monitor.clone();
    tokio::spawn(async move {
        api::start_rest_api(monitor_ref).await;
    });

    monitor
}

/// Bezpečne ukončí event loop a resetuje takty
pub fn stop_ob_monitor(monitor: &ObMonitor) {
    if let Ok(mut r) = monitor.running.lock() {
        *r = false;
    }
    overclock::reset_overclock(); // downgrade na stock clocks
}

/// Export state JSON dát pre Glass Brain vizualizáciu
pub fn get_telemetry_json(monitor: &ObMonitor) -> String {
    let data = monitor.latest_telemetry.lock().unwrap();
    let pred = *monitor.latest_prediction.lock().unwrap();
    
    match &*data {
        Some(tele) => {
            let export = json!({
                "module": "krystal-ob",
                "telemetry": tele,
                "overclock_trajectory": {
                    "predicted_temp_5m": pred,
                    "target_core_offset": if tele.gpu_temp < 68.0 { 50 } else { 0 },
                    "target_mem_offset": if tele.gpu_temp < 68.0 { 100 } else { 0 }
                }
            });
            export.to_string()
        },
        None => "{}".to_string(),
    }
}
