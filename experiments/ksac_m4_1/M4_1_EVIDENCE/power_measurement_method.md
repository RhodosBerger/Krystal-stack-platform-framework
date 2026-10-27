# KSAC M4.1 — POWER AND ENERGY MEASUREMENT METHODOLOGY

## 1. Physical Instrumentation Setup

### Host Power (E_host)
- Measured via Intel RAPL (Running Average Power Limit) Model-Specific Registers:
  - `MSR_PKG_ENERGY_STATUS` (0x611) — Package domain (CPU cores, uncore, memory controller).
  - `MSR_DRAM_ENERGY_STATUS` (0x618) — Memory subsystem.
- Sampling Frequency: 1,000 Hz (1 ms interval) via a dedicated high-priority telemetry daemon (`/dev/cpu/0/msr`).
- Register wrapping handled with 32-bit counter rollover detection ($2^{32} \times 15.2587\text{ µJ}$).

### Control Coprocessor Power (E_FPGA)
- Measured via dedicated Texas Instruments INA226 high-side bidirectional current and power monitors with $\pm 0.1\%$ precision current shunt resistors on the FPGA carrier board:
  - Rail 1: 12.0 V ATX Carrier Input (Total board gross draw).
  - Rail 2: 0.85 V VCCINT (FPGA Core fabric logic).
  - Rail 3: 1.8 V VCCAUX / VCCIO (I/O banks & clocking).
- Sampling Frequency: 10,000 Hz (100 µs interval) over PMBus / I2C bridge.
- Calibration: Current shunt resistance $R_{\text{shunt}} = 10.0\text{ m}\Omega$, calibrated against a Keithley 2450 SourceMeter.

## 2. Total Energy Formulation

$$\mathbf{E}_A = \mathbf{E}_{\text{host}} + \mathbf{E}_{\text{FPGA}}$$

$$\mathbf{E}_{\text{FPGA}} = \int_{0}^{T_{\text{window}}} P_{\text{FPGA}}(t)\, dt = P_{\text{FPGA,active}} \times T_{\text{window}} = 4.2197\text{ J}$$

No control energy is deducted. The full energetic footprint of the KCB-M4 coprocessor is charged against the adaptive system.
