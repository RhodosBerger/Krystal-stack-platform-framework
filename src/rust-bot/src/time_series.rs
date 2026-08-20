//! Time Series Logger - Telemetry logging

use crate::TelemetrySnapshot;
use std::path::PathBuf;
use std::fs::{File, OpenOptions};
use std::io::Write;

pub struct TimeSeriesLogger {
    path: PathBuf,
    buffer: Vec<TelemetrySnapshot>,
    buffer_size: usize,
}

impl TimeSeriesLogger {
    pub fn new(path: &PathBuf) -> Self {
        Self {
            path: path.clone(),
            buffer: Vec::with_capacity(100),
            buffer_size: 100,
        }
    }

    pub fn log(&mut self, snapshot: &TelemetrySnapshot) {
        self.buffer.push(snapshot.clone());
        if self.buffer.len() >= self.buffer_size {
            self.flush();
        }
    }

    pub fn flush(&mut self) {
        if self.buffer.is_empty() { return; }

        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
        {
            for snapshot in &self.buffer {
                if let Ok(json) = serde_json::to_string(snapshot) {
                    let _ = writeln!(file, "{}", json);
                }
            }
        }
        self.buffer.clear();
    }
}

impl Drop for TimeSeriesLogger {
    fn drop(&mut self) {
        self.flush();
    }
}
