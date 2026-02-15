use pyo3::prelude::*;
use sysinfo::{System, SystemExt, ComponentExt};

#[pyclass]
struct PowerSafetyMonitor {
    sys: System,
}

#[pymethods]
impl PowerSafetyMonitor {
    #[new]
    fn new() -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();
        PowerSafetyMonitor { sys }
    }

    fn check_voltage_safety(&mut self) -> bool {
        self.sys.refresh_components();
        for component in self.sys.components() {
            // Placeholder: Real voltage checks depend on hardware sensors
            // Here we check temperature as a proxy for safety
            if component.temperature() > 85.0 {
                return false;
            }
        }
        true
    }

    fn get_safety_metrics(&mut self) -> String {
        self.sys.refresh_components();
        let mut metrics = String::new();
        for component in self.sys.components() {
             metrics.push_str(&format!("{}: {}°C\n", component.label(), component.temperature()));
        }
        metrics
    }
}

#[pymodule]
fn rust_planner(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PowerSafetyMonitor>()?;
    Ok(())
}

