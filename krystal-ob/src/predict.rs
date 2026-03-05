use std::collections::VecDeque;

pub struct TemperaturePredictor {
    history: VecDeque<f64>,
}

impl TemperaturePredictor {
    pub fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(30), // Rolling buffer 30 sekúnd
        }
    }

    /// Vloží najnovšiu teplotu (beží každú 1s v slučke)
    pub fn add_sample(&mut self, temp: f64) {
        if self.history.len() == 30 {
            self.history.pop_front();
        }
        self.history.push_back(temp);
    }

    /// Jednoduchá lineárna regresia (Najmenšie štvorce odchýlok) z do 10 posl. vzoriek v history
    pub fn predict_in_5_min(&self) -> f64 {
        let n = self.history.len() as f64;
        if n < 2.0 {
            return *self.history.back().unwrap_or(&0.0);
        }

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        let take_n = usize::min(10, self.history.len());
        let skip_n = self.history.len() - take_n;

        for (i, &temp) in self.history.iter().skip(skip_n).enumerate() {
            let x = i as f64;
            let y = temp;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        let fl_n = take_n as f64;
        let slope = (fl_n * sum_xy - sum_x * sum_y) / (fl_n * sum_xx - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / fl_n;

        // Vypočítame teplotu o 300 sekúnd (5 minút) do budúcnosti voči relácii k 1s indexu
        let current_t = fl_n - 1.0;
        let future_t = current_t + 300.0;

        intercept + slope * future_t
    }
    
    /// Rýchla bezpečnostná brzda: Ak je teplota vyššia o 3 °C ako pred 10 sekundami
    pub fn has_rapid_increase(&self) -> bool {
        if self.history.len() < 10 {
            return false;
        }
        let old_temp = self.history[self.history.len() - 10];
        let new_temp = self.history.back().unwrap();
        (new_temp - old_temp) >= 3.0
    }
}
