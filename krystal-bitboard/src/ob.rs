use std::sync::Arc;
use std::collections::VecDeque;
use tokio::time::{sleep, Duration};
use log::info;
use crate::grid::MemoryGrid;

// Modul implementuje Lineárnu Regresiu z rolovej mapy 
pub struct TemperaturePredictor {
    pub history: VecDeque<f64>,
}

impl TemperaturePredictor {
    pub fn new() -> Self {
        Self { history: VecDeque::with_capacity(30) }
    }

    pub fn add_sample(&mut self, temp: f64) {
        if self.history.len() == 30 {
            self.history.pop_front();
        }
        self.history.push_back(temp);
    }

    pub fn rapid_increase(&self) -> bool {
        if self.history.len() >= 10 {
            let old_temp = self.history[self.history.len() - 10];
            let new_temp = *self.history.back().unwrap();
            (new_temp - old_temp) >= 3.0
        } else {
            false
        }
    }

    pub fn predict_temp_5m(&self) -> f64 {
        let n = usize::min(10, self.history.len());
        if n < 2 { return *self.history.back().unwrap_or(&0.0); }
        
        let skip = self.history.len() - n;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        for (i, &temp) in self.history.iter().skip(skip).enumerate() {
            let x = i as f64;
            let y = temp;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        let slope = (n as f64 * sum_xy - sum_x * sum_y) / (n as f64 * sum_xx - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n as f64;
        
        // 5 minút = 300 s presah
        intercept + slope * ((n as f64 - 1.0) + 300.0)
    }
}

pub async fn start_telemetry(grid: Arc<MemoryGrid>) {
    info!("⚡ Štartujem Zónu 4: Overclock Buffer & Telemetry Tracker");
    let mut predictor = TemperaturePredictor::new();

    loop {
        // Pre výpočet grafov a memory-mapping telemetrie potrebujeme lock
        if grid.lock_memory(4, 5) { // 5 MB spotreba
            
            // Zápis nového bloku - simulácia čítania teploty (mock 65 C)
            predictor.history.push_back(65.0);
            if predictor.history.len() > 30 {
                predictor.history.pop_front();
            }
            
            grid.free_memory(4, 5);
        }
        sleep(Duration::from_secs(1)).await;
    }
}
