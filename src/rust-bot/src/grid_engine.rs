//! Grid Engine - Hex-based spatial partitioning

use crate::GridSummary;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HexCell {
    pub x: usize,
    pub y: usize,
    pub z: usize,
    pub signal_strength: f64,
    pub gpu_block_id: Option<u32>,
    pub core_affinity: u32,
    pub active: bool,
}

pub struct GridEngine {
    dimensions: (usize, usize, usize),
    cells: Vec<Vec<Vec<HexCell>>>,
}

impl GridEngine {
    pub fn new(x: usize, y: usize, z: usize) -> Self {
        let mut cells = Vec::with_capacity(x);
        for i in 0..x {
            let mut plane = Vec::with_capacity(y);
            for j in 0..y {
                let mut column = Vec::with_capacity(z);
                for k in 0..z {
                    column.push(HexCell {
                        x: i, y: j, z: k,
                        signal_strength: 0.0,
                        gpu_block_id: None,
                        core_affinity: 0xFF,
                        active: false,
                    });
                }
                plane.push(column);
            }
            cells.push(plane);
        }

        Self { dimensions: (x, y, z), cells }
    }

    pub fn get_cell(&self, x: usize, y: usize, z: usize) -> Option<&HexCell> {
        self.cells.get(x)?.get(y)?.get(z)
    }

    pub fn get_cell_mut(&mut self, x: usize, y: usize, z: usize) -> Option<&mut HexCell> {
        self.cells.get_mut(x)?.get_mut(y)?.get_mut(z)
    }

    pub fn activate(&mut self, x: usize, y: usize, z: usize, gpu_block: u32) -> bool {
        if let Some(cell) = self.get_cell_mut(x, y, z) {
            cell.active = true;
            cell.gpu_block_id = Some(gpu_block);
            true
        } else {
            false
        }
    }

    pub fn set_signal(&mut self, x: usize, y: usize, z: usize, strength: f64) {
        if let Some(cell) = self.get_cell_mut(x, y, z) {
            cell.signal_strength = strength.clamp(0.0, 1.0);
        }
    }

    pub fn get_summary(&self) -> GridSummary {
        let mut active = 0;
        let mut hottest: Option<(usize, usize, usize, f64)> = None;

        for (i, plane) in self.cells.iter().enumerate() {
            for (j, column) in plane.iter().enumerate() {
                for (k, cell) in column.iter().enumerate() {
                    if cell.active {
                        active += 1;
                        if hottest.map_or(true, |h| cell.signal_strength > h.3) {
                            hottest = Some((i, j, k, cell.signal_strength));
                        }
                    }
                }
            }
        }

        GridSummary {
            dimensions: self.dimensions,
            active_cells: active,
            total_cells: self.dimensions.0 * self.dimensions.1 * self.dimensions.2,
            hottest_cell: hottest.map(|(x, y, z, _)| (x, y, z)),
        }
    }

    pub fn rebalance(&mut self) {
        // Redistribute signal strength across active cells
        let active: Vec<_> = self.cells.iter().flatten().flatten()
            .filter(|c| c.active)
            .map(|c| c.signal_strength)
            .collect();

        if active.is_empty() { return; }

        let avg = active.iter().sum::<f64>() / active.len() as f64;

        for plane in &mut self.cells {
            for column in plane {
                for cell in column {
                    if cell.active {
                        // Smooth toward average
                        cell.signal_strength = cell.signal_strength * 0.7 + avg * 0.3;
                    }
                }
            }
        }
    }
}
