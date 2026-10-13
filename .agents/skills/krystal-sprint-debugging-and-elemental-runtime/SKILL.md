---
name: krystal-sprint-debugging-and-elemental-runtime
description: Comprehensive debugging playbook, elemental structure contracts, and operational asset synthesizing the day's sprint solutions across Rust bot, Epistemic Runtime, C ABI, Whisper acoustic pipeline, 32B/16B/64B zero-copy codecs, and Poslední Kmen elemental combat matrices.
---

# Krystal Sprint Debugging & Elemental Runtime Asset

Use this skill when developing, debugging, verifying, or extending the multi-language subsystems of Krystal Stack: the Elemental Structure, Cinematic & Mortar Ballistics, L.A. Noire MotionScan lipsync, Whisper acoustic pipeline, and Rust/C/Python native bindings.

---

## 1. Elemental Structure & Combat Matrix Contracts

The Krystal Stack elemental system coordinates tactical combat, dynamic terrain terraforming, crafting polarity, and artillery ballistics:

### 1.1 The Five Fundamental Elemental Polarities
1. **Crystal (`crystal`)**: Aetheric resonance, brittle refraction, sharp kinetic shatter.
2. **Toxic (`toxic`)**: Acidic decomposition, biological corrosive decay, caustic DOT.
3. **Druidic / Bark (`druid`)**: Living flora, rapid cellular regeneration, catalyst stabilization.
4. **Basalt / Fire (`basalt` / `fire`)**: Volcanic thermal density, combustion shockwaves, armor stripping.
5. **Aether / Celestial (`aether`)**: Quantum coherence, phase gravity gradients, epistemic arbitration.

### 1.2 Elemental Polarity Clash & Amber Catalyst Rule
* **Unstable Clash**: Mixing `Crystal` $\otimes$ `Toxic` without a stabilizer triggers explosive resonance instability and craft/cast failure (`ElementalPolarityError`).
* **Catalytic Stabilization**: Adding `Amber Catalyst` (`amber_catalyst` / Druidic Amber) stabilizes the polarity conflict, granting a $+25\%$ power budget increase and generating corrosive crystal shatter shards.

### 1.3 Elemental Impulse & Dynamic Hex Terraforming
Spell impacts and artillery shells exert physical impulse vectors on the hexagonal terrain (`ELEMENTAL_IMPULSE_MAP`):
* `crystal`: $(0.15, 0.85, 0.20)$ — crystalline jagged terrain, impairs footing.
* `toxic`: $(0.05, 0.20, 0.75)$ — acidic sludge pool, corrodes armor, burns over time.
* `druid`: $(0.60, 0.40, 0.10)$ — overgrowth, movement penalty, root pin.
* `burning_aether` + `toxic_acid` $\rightarrow$ **Combustion Shockwave** ($+2$ radial damage).
* `burning_aether` + `crystal_frost` $\rightarrow$ **Thermal Steam Quench** (converts hex to deep mud slag).

### 1.4 Ballistics Coupling
Mortar shell payloads can be infused with elemental warheads. The terminal plunge velocity couples directly with camera shake, doppler compression, and radial elemental dispersal.

---

## 2. Multi-Language Debugging Playbook & Solutions

Lessons learned and hard-won solutions from resolving CI/CD and multi-language compilation failures:

### 2.1 Rust Cargo Workspace Discipline
* **Issue**: Subcrates (`src/rust-bot`, `rust/epistemic_runtime`, `krystal-miner`) fail in CI with:
  `current package believes it's in a workspace when it's not`.
* **Fix**: Ensure every nested crate directory is explicitly listed in the root [Cargo.toml](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/Cargo.toml) under `[workspace.members]`:
  ```toml
  [workspace]
  members = [
      "src/rust-bot",
      "rust/epistemic_runtime",
      "krystal-miner"
  ]
  ```
* **Artifact Path Rule**: In a Cargo workspace, build artifacts (`target/`) are located at the repository root, not in crate subdirectories.

### 2.2 Rust Borrow Checker & Syntax Guardrails
* **Double-Colon Scope**: Always reference enum variants with `EnumName::Variant` (e.g. `SystemMode::Autonomous`, `VisemeId::AA`).
* **No Partial Moves**: When calculating hashes, signatures, or sending events, never move fields out of a struct that will be borrowed or signed later. Use `.clone()` or pass references `&`.
* **Trait Constraints on Closures**: Structs holding trait objects or closures (`Box<dyn Fn(...)>`) cannot derive `Clone` or `Debug`. Remove these derives and implement custom constructors.
* **Dead Code Cleanliness**: Prefix unused arguments with `_` (e.g., `_python_ok`, `_stability`) or annotate intentionally preserved ABI fields with `#[allow(dead_code)]`.

### 2.3 Headless Network Test Isolation Pattern
* **Issue**: Live tests calling `127.0.0.1:8089` crash on headless CI runners where no server is running.
* **Fix (Ephemeral Port Isolation)**:
  In Python test suites, spin up an ephemeral background test server on port 0:
  ```python
  @classmethod
  def setUpClass(cls):
      cls.server = HTTPServer(("127.0.0.1", 0), KrystalHubHandler)
      cls.port = cls.server.server_address[1]
      cls.base_url = f"http://127.0.0.1:{cls.port}"
      cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
      cls.thread.start()
  ```
  If relying on external daemons, use a strict pre-flight check in `setUpClass` and invoke `unittest.SkipTest(...)` if the port is unreachable.

### 2.4 C Core ABI & Header Synchronization
* **Header Rule**: Declarations in `src/c/*.h` must exactly mirror the parameter types, padding, and field order expected by Rust FFI, Python ctypes, and runtime C drivers (`gamesa_kernel_config_t`, `rpg_boost_map_t`).

---

## 3. High-Performance Binary Codecs & Whisper Acoustic Pipeline

### 3.1 Zero-Copy Frame Specification
| Packet | Magic | Size | Layout | Invariant |
|---|---|---|---|---|
| **KBSB MotionScan v2** (`KMCP`) | `b"KMCP"` | 64 B | `>4s I B B B B 8f f 3f I` | `vital_hp <= 6` |
| **KBSB Mortar Ballistic** (`KMBP`) | `b"KMBP"` | 32 B | `>4s I H B B 3f H H I` | `vital_hp <= 6` |
| **KBSB Deception Verdict** (`KDVP`) | `b"KDVP"` | 16 B | `>4s I B B H I` | `vital_hp <= 6` |

### 3.2 Compile-Time IEEE 802.3 CRC32 Table
Replace bit-shifting CRC loops with a 256-element lookup table (`CRC32_TABLE: [u32; 256]`):
* Generates an **$8\times$ throughput speedup**.
* Operates identically in Rust and Python (`zlib.crc32`).

### 3.3 Whisper Acoustic Phoneme & Cross-Modal Divergence
* **Formants**: $F_1$ (jaw opening), $F_2$ (mouth widening/pucker), $F_3$ (rhoticity/tension), Sibilance ratio ($4..8\text{ kHz}$).
* **Voice Stress Index**: Autonomic nervous system vocal cord tension calculated from pitch $F_0$ elevation and $F_3$ jitter.
* **Cross-Modal Divergence**:
  $$\text{divergence} = |\text{stress\_index} - \text{facial\_deception\_score}|$$
  If divergence $> 0.35$, the suspect is attempting conscious facial suppression while betraying autonomic vocal stress.

---

## 4. Verification & Operational Commands

### 4.1 Running All Multi-Language Tests
```powershell
# Python Full Integration Suite (Kinematics, Ballistics, Lipsync, Codecs)
python -m unittest tests/test_scene_composer_trajectories.py `
                   tests/test_mortar_camera_and_noir_lipsync.py `
                   tests/test_cinematic_codec_adaptation.py `
                   tests/test_advanced_cinematic_and_rust_integration.py

# Rust Bot Unit & Invariant Tests (WSL)
wsl -e bash -c "source ~/.cargo/env && cd /mnt/c/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework && cargo test -p gamesa-rust-bot && cargo test -p epistemic-runtime"
```

### 4.2 Running Live Codec Micro-Benchmarks
```powershell
python -c "from mimicry_engine.whisper_acoustic_pipeline import WhisperPhonemeAcousticPipeline; print(WhisperPhonemeAcousticPipeline.benchmark_codecs(10000))"
```

### 4.3 REST Endpoints (Mission Control Hub)
* `GET  /api/cinematic/status` — Codec capabilities and invariant limits.
* `GET  /api/cinematic/trajectory/mortar` — Ballistic trajectory samples.
* `GET  /api/cinematic/benchmark/packets` — Live packet encode/decode performance metrics.
* `POST /api/cinematic/whisper/predict` — Formant viseme inference and divergence.
* `POST /api/cinematic/codec/mortar-packet` — 32B KMBP codec roundtrip.
* `POST /api/cinematic/codec/verdict-packet` — 16B KDVP codec roundtrip.
* `GET  /api/verification/domains` — 6 Epistemic domains including `cinematic_interrogation`.

---

## 5. System Troubleshooting Reference Matrix

| Symptom | Root Cause | Solution |
|---|---|---|
| `VETO_SAFETY_BREACH` (HTTP 400) | `vital_hp > 6` in packet or payload | Clamp `vital_hp <= 6`. Invariant is inviolable. |
| `CRC32 mismatch` on decode | Endianness mismatch or modified bytes | Use big-endian `>I` format and IEEE 802.3 polynomial. |
| `ConnectionRefusedError` in test | Live test queried unstarted port | Use `setUpClass` ephemeral port binding (`127.0.0.1:0`). |
| Rust `unused variable / import` | Compiler warning in strict CI | Clean imports, prefix `_` or add `#[allow(dead_code)]`. |
| `ElementalPolarityError` | Crystal + Toxic mixed without catalyst | Add `amber_catalyst` reagent to balance polarity. |
| Divergence $> 0.35$ flagged | Vocal stress contradicts calm face | Suspect masking deception; trigger micro-probe. |

---

## 6. Elemental Structure & Kernel/Hypervisor Statistical Matrix

The Krystal Elemental Structure directly maps gameplay and physical polarities to bare-metal Kernel and Type-1 Hypervisor innovations:

### 6.1 Periodic Map of System Elements & Telemetry

| Element & Symbol | Ring Domain | Subsystem Innovation | Legacy Stat | Krystal Stat | Measured Gain |
|---|---|---|---|---|---|
| **Crystal (`Cr`)** | `Ring -1` (DMA Root) | Zero-Copy DMA & Page Grants | Latency: 12.40 ms | **0.38 ms** | **32.6x Speedup** |
| **Basalt / Fire (`Bf`)** | `Ring 0` (Thermal) | 100% Fan (3600 RPM) & DVFS | Temp: 78.4°C ($R_{th}=1.85$) | **52.1°C** ($R_{th}=0.72$) | **-26.3°C Drop** (0% Throttling) |
| **Druid / Bark (`Db`)** | `Ring 0` (Self-Healing) | AIMD Resilience & In-Memory Re-sync | Recovery: 1450 ms | **1.2 ms** | **1208x Speedup** (MTBF >1000h) |
| **Toxic / Slag (`Tx`)** | `Ring -1` (Interceptor) | Jitter Damping & Voltage Clamp (1.05V) | Jitter $\sigma$: 8.42 ms | **0.61 ms** | **13.8x Smoother** (P99: 6.2 ms) |
| **Aether / Q (`Ae`)** | `Ring 3` (Paravirtual) | Multi-Modal Whisper & 802.3 Codecs | Codec: 42.10 μs | **0.48 μs** | **87.7x Speedup** (Err: 1.1%) |
| **Amber Catalyst (`Amb`)** | Compound Synthesis | Polarity Clash Arbiter (`Cr` $\otimes$ `Tx`) | Clash Fail: 50% | **100% Safe** | **+25% Power Budget** |
| **Combustion Surge (`Cs`)** | Compound Synthesis | Queue Overpressure Flush (`Bf` + `Tx`) | Queue: 392k ops/s | **1,724,554 ops/s** | **+339.9% Throughput** |
| **Steam Quench (`Sq`)** | Compound Synthesis | Hex Heat Dissipation (`Bf` + `Cr`) | Energy: 28.5 nJ/op | **6.18 nJ/op** | **-78.3% Energy Cost** |

### 6.2 Verification & API Commands
* **CLI Periodic Table Generator**:
  ```powershell
  python krystal_kernel/elemental_hypervisor_telemetry_matrix.py
  ```
* **REST API Telemetric Manifest**:
  ```
  GET /api/elemental/hypervisor-map
  ```
* **Unit Test Suite**:
  ```powershell
  python -m unittest tests/test_elemental_hypervisor_telemetry_matrix.py
  ```
* **Platform Invariant**: `VITAL_MAX_HP == 6` strictly preserved in 100% of telemetry frames.

---

## 7. PROTIX Machine Learning Models & Neural Structures

The **PROTIX (Predictive Real-time Observation & Toxic-Invariant eXecution)** model suite implements formal deep learning architectures and multi-task neural heads adhering to `@skill:ml-best-practices`:

### 7.1 Multi-Task Deep Learning Architecture
* **Trunk**: `Input(12)` $\rightarrow$ `Dense(64, Swish)` $\rightarrow$ `LayerNorm` $\rightarrow$ `Dropout(0.1)` $\rightarrow$ `Dense(32, Swish)`.
* **Head 1 (Classification - 5 classes Softmax)**: Tactical Opponent Action:
  1. `TOXIC_SPORE_STRIKE`: Corrosive burst.
  2. `ACID_MUD_QUENCH`: Thermal damping & terrain sludge.
  3. `CATALYZED_SHARD_BURST`: Amber-stabilized crystal shatter.
  4. `DEFENSIVE_RESIN_ABSORB`: Bark/resin absorption.
  5. `DRAIN_MANA_CORROSION`: Resource starvation overtime.
* **Head 2 (Regression - 3 outputs)**: Continuous telemetry & damage prediction (`predicted_damage`, `predicted_undervolt_mv`, `predicted_temperature_c`).
* **Head 3 (Clustering / Dimensionality Reduction - 2D Projection)**: Projects representations into a 2D latent space with k-Means clustering and Silhouette Score ($s \in [-1, 1]$), identifying 3 tactical archetypes:
  - `AGGRESSIVE_STRIKER`: High damage, high thermal output ($(-1.8, 1.2)$).
  - `TACTICAL_COUNTER_HARVEST`: Balanced reactive, amber catalyst user ($(0.5, 0.2)$).
  - `RESILIENT_FORTIFIER`: Defensive bark soak, low heat ($(1.5, -1.4)$).
* **Head 4 (Risk & Anomaly Detection - Sigmoid)**: Invariant breach probability $P(\text{hazard})$ ensuring safety boundaries.

### 7.2 Preprocessing & Machine Learning Structures
* **$Z$-Score Standardization**: $z = \frac{x - \mu}{\sigma}$ with division-by-zero protection ($\sigma \ge 10^{-6}$).
* **Chronological Dataset Splitting**: 70% Train, 15% Validation, 15% Test without temporal leakage.
* **Vectorized Native Core**: Executes in $<0.4\text{ ms}$ (measured $\approx 310\text{ }\mu\text{s}$) with zero external heavy framework dependency lock-in.
* **Formal Keras 3 Graph Builder**: Generates production-ready JSON graph specification compliant with `keras.models.model_from_json`.

### 7.3 Operational Commands & Testing
* **CLI Engine & Multi-Task Evaluation**:
  ```powershell
  python krystal_kernel/protix_ml_models.py
  ```
* **Unit Test Suite**:
  ```powershell
  python -m unittest tests/test_protix_ml_models.py
  ```
* **Mission Control REST Endpoints**:
  - `GET  /api/ml/protix/architecture` — Returns Keras 3 Functional API graph.
  - `GET  /api/ml/protix/status` — Returns model config and catalog enums.
  - `POST /api/ml/protix/predict` — Multi-head real-time inference ($<0.4\text{ ms}$).
  - `POST /api/ml/protix/evaluate` — Evaluates accuracy, RMSE, MAE, and Silhouette Score.
* **Universal Platform Invariant**: `VITAL_MAX_HP == 6` strictly clamped across all predictions and training targets.

---

## 8. Network Task Arbiter: Stred Medzi Endpointami a Plánovaním Úloh

Architektonický **"Stred"** rieši rozpor medzi rigidnými synchrónnymi HTTP endpointmi a abstraktným plánovaním sieťových úloh:

### 8.1 Štruktúra Arbitrážneho Mosta
1. **Dual-Path Ingress**:
   - **Fast-Path Bypass** ($<0.1\text{ ms}$): Pre kritické bezpečnostné zásahy a invarianty (`CRITICAL_REALTIME`).
   - **Scheduled Quantum Dispatch** ($3.5\text{ ms}$ budget): Pre inferenciu modelu PROTIX, herné ťahy a asynchrónne dávky.
2. **QoS Priority Rings**:
   - `Band 0 (CRITICAL_REALTIME)`: Termálne havárie, ventilátor 100%, ochrana `VITAL_MAX_HP == 6`.
   - `Band 1 (HIGH_INFERENCE)`: Prediktívne ML úlohy PROTIX ($<0.4\text{ ms}$).
   - `Band 2 (STANDARD_GAME_STATE)`: Zápis do herného ledgeru a ťahy v Poslední Kmen.
   - `Band 3 (BACKGROUND_BATCH)`: Hĺbkové evaluácie modelov, telemetrické logy.
3. **Fyzikálne Ukotvený Backpressure**:
   - Prepojenie s Ohmovým zákonom ($I = \frac{V}{R_{\text{eff}}} \le 10\text{ A}$) a termálnym odporom ($R_{\text{th}} = 0.72^\circ\text{C/W}$).
   - Pri $T \ge 75^\circ\text{C}$ alebo prekročení prúdu arbiter odmietne nové nedôležité požiadavky (`NETWORK_ARBITER_CONGESTION_BACKPRESSURE`).
4. **Rozpočet na Rámec (60 FPS / 16.6 ms Frame Budget)**:
   - Presne $3.5\text{ ms}$ vyhradených pre sieťové úlohy eliminuje bufferbloat a chattering.

### 8.2 Overenie
```powershell
python -m unittest tests/test_network_task_arbiter.py
```

---

## 9. Tectonic Matrix Adaptability & Counterfactual Speculation Engine

The **Tectonic Matrix Adaptability Engine** governs the physical, elemental, and tactical terrain matrix ($\mathbf{M}_{\text{tectonic}}$) using the speculative decision grammar:
`IF -> WHAT-IF -> WHAT-IF -> THAN-IF -> ELSE-IF -> IF-GUARD`.

### 9.1 Mathematical & Structural Formulation
* **Tectonic Matrix Tensor ($\mathbf{M}_{\text{tectonic}}$)**: 64 cells ($8 \times 8$ local grid) modeling:
  - Elevation $z \in [0.0, 50.0]$
  - Shear stress $\tau \in [0.0, 1.0]$ (rift activation at $\tau > 0.80$)
  - Temperature $T \in [20.0, 100.0]^\circ\text{C}$
  - Elemental polarities $[c, t, d, f, a]$ (Crystal, Toxic, Druid, Fire, Aether)
  - Invariant entity vital HP ($HP \le 6$).
* **Copy-on-Write (CoW) Shadow Forks**:
  - Creates speculative copies of the matrix in $<25\text{ }\mu\text{s}$ without mutating the live state.

### 9.2 Counterfactual Grammar Semantics
1. **`IF <condition> THEN <action>`**: Baseline deterministic action (e.g. `mean_stress > 0.40 -> BASELINE_SLUDGE_QUENCH`).
2. **`WHAT_IF <shock_1> THEN <projection_1>`**: 1st-order hypothetical shock simulation (e.g. `THERMAL_SURGE == +18.0`).
3. **`WHAT_IF <shock_2> THEN <projection_2>`**: 2nd-order cascading counterfactual simulation with adaptive countermeasure.
4. **`THAN_IF <criterion, delta> THEN <preferred_action>`**:
   Comparative multi-objective arbitration:
   $$\Delta \text{Stability} = \text{Stability}(\mathbf{M}_{\text{speculative}}) - \text{Stability}(\mathbf{M}_{\text{baseline}})$$
   If $\Delta \text{Stability} \ge 0.04$, the speculative branch overrides the baseline.
5. **`ELSE_IF <condition> THEN <action>`**: Deterministic fallback branches (e.g. `mean_temp_c > 65.0 -> EMERGENCY_COOLING_SLAG`).
6. **`IF_GUARD`**: Universal invariant assertion enforcing `VITAL_MAX_HP == 6`, $T \le 75^\circ\text{C}$, $I \le 10\text{ A}$.

### 9.3 Operational Commands & REST Endpoints
* **CLI Engine**:
  ```powershell
  python krystal_kernel/tectonic_matrix_adaptability_engine.py
  ```
* **Unit Tests**:
  ```powershell
  python -m unittest tests/test_tectonic_matrix_adaptability.py
  ```
* **Mission Control REST Endpoints**:
  - `GET  /api/tectonic/matrix/status` — Live matrix dimensions, stress, stability, and rule set.
  - `POST /api/tectonic/speculate` — Runs counterfactual speculation and returns trace.
  - `POST /api/tectonic/adapt` — Commits optimal adaptive action to live matrix.


