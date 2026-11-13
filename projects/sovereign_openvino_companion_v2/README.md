# Sovereign OpenVINO Companion v2 (Fine-Tuned & Adaptive Resource Spending)

Fine-tuned adaptive companion over `projects/sovereign_openvino_companion`:
- `BufferedComputationLogSystem`: SHA-256 hash-chained ledger with write-coalesced flushing (`flush_every=16`).
- `AdaptiveSpendingOptimizer`: EWMA latency-calibrated `adaptive_crossover_bytes`, predictive UMA velocity pre-braking (`mu_hat >= 0.72` -> `k_spec=4`), and early P5 parasite cutoff (`0.0180 mJ` vs `0.284 mJ`).
- `SensoryCoordinateMorphismEngine`: Bijective projection between 6D `OperatingCoordinatesVector` and 5D `SensoryManifoldCoordinates` with continuous morphing, P6 homotopic conflict isolation (`alpha perp beta`), and `20 B` `SMRP` encoding in slice `[108:128]`.
- `BinaryParameterAspectCompiler`: Compiles and round-trip verifies `128 B` zero-copy frames across `21:6:5` (`84:24:20`), `79:49` Golden (`phi`), and `90:38` Silver (`delta_S`) ratios + `24 B` `KMB1` lockstep packets via `ArtifactGate` (`PROPOSER != VERIFIER`).
- `StratifiedProcessMicroserviceFlowArchitect`: Audits Processes (`S0..S3`), Microservices (`CW_01..CW_06`), Stack-Defined Conclusions (`S9`), `FLOW_01..FLOW_06`, the `12 x 10 = 120`-cell Matrix Choice Palette, CPU Survival Governor metrics, and the Compute Motivation Core.

```powershell
# Session + Paired A/B Benchmark + Candidate Runtime Nodes
python -m projects.sovereign_openvino_companion_v2.cli --run-session --frames 16 --ab-benchmark 40 --candidate-nodes --reset-ledger

# Sensory Morphism + Binary Aspect-Ratio Compilation + Stack Flows & Matrix Governance
python -m projects.sovereign_openvino_companion_v2.cli --morph-sense --compile-binary-ratios --stack-flows --emit-bin-dir artifacts/binaries/v2 --reset-ledger
```

HTTP Endpoints on `krystal_kernel/sovereign_openvino_oneapi_app.py`:
- `GET /api/v1/runtime/sovereign_openvino_companion_v2`
- `GET /api/v1/runtime/sensory_binary_aspect_compiler`
- `POST /api/v1/runtime/sensory_binary_aspect_compiler`
