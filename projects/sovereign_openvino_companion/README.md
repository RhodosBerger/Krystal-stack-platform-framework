# Sovereign OpenVINO + oneAPI Companion Project

Physical OpenVINO (`CPU` / `GPU` Iris Xe) + oneAPI equivalent of `projects/sovereign_game_companion.py`
(`SovereignGameCompanionBridge`), integrated with a 7-channel cryptographic hash-chained computation
logging system (`ComputationLogSystem`).

## Key Features
- **Hardware Crossover Policy**: Micro-tensors (`< 16 KB`) run on low-latency `CPU` VNNI; dense tensors
  (`>= 16 KB` when `uma_saturation < 0.78`) offload to OpenVINO `GPU`.
- **12-Frame Hysteretic Dwell + Merkle-Momentum**: Fast 1-frame downscale on UMA spike (`>= 0.78`),
  holding `CPU` + `k_spec=1` for `12` frames (`HYSTERESIS_DWELL_FRAMES`) before recovery.
- **Chronometric Shadow Multiplexing**: `VBlankExperimentInjector` runs inference strictly within the
  `8.33 ms` frame VBlank window (`Phi_Scanout * Phi_Compute == 0`).
- **Epistemic Verification & S5..S9 Gnosis**: Enforces `VITAL_MAX_HP == 6`, `56` frozen theory nodes,
  `PROPOSER != VERIFIER` `ArtifactGate`, and `24 B` `GNOS` + `24 B` `KMB1` packets.
- **7-Channel Hash-Chained Ledger**: Logs `system`, `telemetry`, `planning`, `policy`, `inference`,
  `companion_vblank`, and `gnosis_audit` with canonical IEEE-754 floats and SHA-256 chain linking,
  automatically tagging captured reasoning & anomalous patterns (`UMA_BANDWIDTH_PRESSURE_SPIKE`,
  `HYSTERETIC_DWELL_HOLD`, `MICRO_TENSOR_CPU_BYPASS`, `THERMAL_PARASITE_DECOUPLED`,
  `HOMOTOPIC_CONFLICT_FORK`, `AUTHORITY_SEPARATION_QUARANTINE`,
  `SPECTRUM_COORDINATE_DRIFT_CORRECTED`, `COWORKER_QUORUM_MAINTENANCE_ISSUED`).
- **Operating Coordinates Supervisor Community**: 4 Supervisor Super-Functions orchestrating 6 role-separated Co-Worker Microservices (`CW_01` Observer, `CW_02..03` Proposers, `CW_04..05` Verifiers, `CW_06` Executor) across the 5-Band Telemetry Spectrum to maintain canonical 6D operating coordinates.

## CLI Usage
```powershell
python -m projects.sovereign_openvino_companion.cli --validate-env --supervise-coordinates --frames 8 --verify-logs --reset-ledger
```
