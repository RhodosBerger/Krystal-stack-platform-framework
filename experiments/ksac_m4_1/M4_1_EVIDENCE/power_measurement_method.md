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

## 2. Energy Formulation & Duty Cycle Locking

For physical validation, the power estimation model $P \times \text{duty}$ is superseded by the direct continuous physical integral:

$$\mathbf{E}_{\text{FPGA}} = \int_{0}^{T_{\text{window}}} P_{\text{FPGA}}(t)\, dt$$

including:
- Idle FPGA power ($2.10\text{ W}$),
- Block RAM (BRAM),
- High-speed transceivers (MGT),
- PCIe PHY serialization layer,
- Control fabric logic,
- Carrier board and VRM regulator conversion losses according to the physical boundary.

### Transparent Dual-Reporting:
1. **Active Decision Window ($E_{\text{ctrl,active}} = 4.2194\text{ J}$)**:
   The control fabric was energetically active for $1.0\text{ s}$ out of the $10.0\text{ s}$ execution window ($10\%$ active duty cycle). Under this active window, the Normalized Control Leverage Factor is $\mathbf{CLF}_{N,\text{active}} = \mathbf{28.59}$.
2. **Gross Board Continuous Energy ($E_{\text{FPGA,gross}} = 23.1194\text{ J}$)**:
   Full un-gated board integration ($2.1\text{ W idle} \times 9\text{ s} + 4.2\text{ W active} \times 1\text{ s} = 23.12\text{ J}$). Under gross board energy, $\mathbf{CLF}_{N,\text{gross}} = \mathbf{5.22}$.
