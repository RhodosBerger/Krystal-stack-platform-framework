# Unified Backend-to-UI Access, Actual-State Documentation Standards & Multi-Sprint Prompt Corpus

- **Constitutional Invariant:** `VITAL_MAX_HP == 6` · Corpus `SHA-256 = 3e0cf19b56291ce27c13...`
- **Live Codebase Inventory:** `46` modules · `158` public functions · `44` classes · `63` bound terms (`0` unresolved) · `69` architecture `.md` specs

---

## 1. Best Choices in Our Derivative Integration Study (Pareto Decision Ranking)

| Rank | Choice ID | Composite Index | Gain | Why It Wins |
| :---: | :--- | :---: | :---: | :--- |
| `#1` | **`DF_01_192B_WNTC_KPKT_DATAGRAM`** | **`7.663`** | `5.04x` | Eliminates JSON DTO overhead across Windows <-> WSL2 in exactly 3 x 64B cache lines with dual CRC32 + SHA-256. |
| `#2` | **`DF_02_CONV_PACED_DOUBLE_TIME`** | **`6.018`** | `3.8x` | Prevents 1-tick noise thrashing while guaranteeing 0.00 ms VSync overrun at 120 Hz (k=2) and 0-frame halt at UMA >= 0.78. |
| `#3` | **`DF_05_U64_HEX_BITBOARD_FFI_MASK`** | **`5.875`** | `18.0x` | Fits the entire R=3 hex board (37 cells) inside a single 64-bit CPU register with zero heap allocations. |
| `#4` | **`DF_03_GODOT_ANIMATION_TRES_COMPILER`** | **`3.521`** | `4.2x` | Turns deterministic simulation & robotics traces directly into playable Godot 4 scenes without manual keyframing. |
| `#5` | **`DF_04_NEURAL_ASCII_SURVIVAL_HUD`** | **`2.861`** | `20.0x` | Cuts viewport bandwidth > 95% (< 300 B payload, < 0.5 ms CPU) when iGPU/UMA is saturated. |

---

## 2. Patterns Unifying Access Between Backend Modules & UI (`UAP_01..UAP_04`)

| Pattern ID | Principle | Contract |
| :--- | :--- | :--- |
| `UAP_01_PROJECTION_ENVELOPE_READ` | **Render(S) != S (Read-Only Signed Projection Envelope)** | Every backend module exposes UI state via a uniform envelope {vital_max_hp, state_sha256, uma_utilization, evidence_class, doc_anchor, view_tokens, payload}; UI never mutates authoritative state directly. |
| `UAP_02_INTENT_ACTION_STEP_GATE` | **~Valid(Action) => Delta_S == 0 (Governed Command Ingress)** | All UI clicks, DSL scripts, and CLI commands enter through StepExecutionGateProxy / ContractViolation with proposer != verifier before touching kernel state. |
| `UAP_03_STATE_AB_CSS_TOKEN_ADAPTER` | **Zero-Reallocation UMA-Adaptive Visual Tiering** | Backend emits discrete tier tokens (HOLOGRAM_FULL_0x3ff vs WIREFRAME_SHADOW_0x1f vs ASCII_SURVIVAL) mapped to CSS custom properties and BP_360..1920 sockets without DOM rebuilds. |
| `UAP_04_DOC_LINKED_CAPABILITY_MANIFEST` | **Self-Describing Backend Over Actual-State Documentation** | Every registered endpoint binds its live Python callable, AST signature, pytest anchor, and architecture Markdown path; startup/CI audits parity so docs never drift from live code. |

---

## 3. Best Practices for Opening the Backend Layer Over Actual-State Documentation (`BOP_01..BOP_05`)

| ID | Best Practice |
| :--- | :--- |
| `BOP_01_SINGLE_CAPABILITY_CATALOG` | Expose a single read-only `/api/v1/runtime/unified_gateway` manifest indexing all modules, UI views, and live `.md` anchors. |
| `BOP_02_DOC_STATE_PARITY_GATE` | Verify on every release that every `.md` architecture file referenced by the gateway exists on disk and matches live invariants (`HP==6`, `0` bare asserts). |
| `BOP_03_TYPED_ERROR_BOUNDARIES` | Map `ContractViolation` and `ValueError` deterministically to HTTP `400` JSON with `vital_max_hp: 6` and `delta_s: 0` — never leak raw stack traces or silently coerce invalid inputs. |
| `BOP_04_BINARY_INTERNAL_JSON_EDGE` | Keep internal module/hypervisor rings strictly binary (`64B WNTC`, `128B KPKT`, `32B .kopt`) and project to UTF-8 JSON/SVG only at the outer HTTP/UI view edge. |
| `BOP_05_EPISTEMIC_BADGING_IN_PAYLOAD` | Tag every backend field with `LOCAL_MEASURED`, `VERIFIED_WITH_SCOPE`, `DERIVED`, or `UNKNOWN` so neither UI nor downstream prompts over-claim. |

---

## 4. Multi-Sprint Long Sequence (`Nov 6 – Nov 17, 2026`) & Ready-to-Use Next Prompt Layer

### `SPRINT_W1_NOV06_TO_NOV08` (`5f62dad6 .. d5bfb828`) — 20-Layer Glyph-Trigonometric Coherent Runtime, Zero-GC Synthesis & Ubuntu GPU CI
- 17/20-layer glyph-acoustic-Bezier compiler & Joule amplitude engine
- Eliminated 2s LM Studio TCP stall & wired executable Rust/C++/Python synthesis on :8090
- Normalized GHCR Docker pipeline & enabled CrystalSec v3 / Iris Xe GPU test jobs

### `SPRINT_W2_NOV09_TO_NOV10` (`4bfe17d1 .. 7e477f3e`) — 5-Stratum S0..S4 Automaton, 16 Wire Parameters (84B/128B), Rubik S_48 & GA Recompiler
- Deterministic 8-bit intent automaton + Intel 11th-15th Gen ISA/socket adaptive kernel
- Grid-to-Logic Semantic Mirror & 4-Act Blender-Godot-Oxygen scenario
- Rubik's Cube G_Rubik subset S_48 choice mechanics & adaptive GA taxonomy optimizer

### `SPRINT_W3_NOV11_TO_NOV12` (`240317f8 .. cc7be983`) — Material-Phi (1.618) /studio Multimedia Hub, 3D Rubik + Marias Projection & S5..S9 Gnosis
- Procedural WAV soundtrack, SVG emblem, 3D Rubik renderer & 24B KMB1 Lockstep Multiplexer
- 4 Marias playing-card projection lenses (Render(S) = Pi_Marias(S) != S) & Blizzard tactile card manual
- S5..S9 Ontologic Gnosis strata with 24B GNOS binary descriptor & K8s/nftables zero-trust configs

### `SPRINT_W4_NOV13_TO_NOV14` (`0980c68e .. a95b9ae4`) — Sovereign OpenVINO Companion v1/v2, 64B WNTC WSL2-NT Bridge & Trigonometric Hybrid Platform
- Hash-chained ComputationLogSystem, 6-microservice Supervisor Community & Brier/ECE calibration
- 64B WNTC single-cache-line WSL2<->NT descriptor & 120-cell Hypervisor Matrix Choice Palette
- Z3 CrossDisplayMultiGenreParameterMatrix, StepExecutionGateProxy, adaptive rollback sync & Babylon 3D grid

### `SPRINT_W5_NOV16` (`82c28a59 .. 56ef041d`) — OpenVINO FMA/K=449 Proof, x86-64 X01..X05, Clash 37-Hex Arena, Godot/VR, AST Parser & 40.4x Priority Logger
- Proved Iris Xe sequential FMA bit-identity, localized OpenBLAS K=449 transition & achieved 126.4x AST speedup
- Measured X01..X05 (L2 55.1 GB/s, 14x/37.4x subnormal cliff, 5.04x 128B KPKT speedup, 13,362.7 arena commits/s)
- Built Goniometric Kernel DESC_01..03, Godot 10/20 layers, HENC micro-LLM, XLP_01..04 copilots, 3 dependency bridges, visual story generators & 0-bare-assert production gate

### `SPRINT_W6_NOV17_CURRENT` (`22b3d96a .. HEAD`) — Double-Time 120Hz/128B Bugfixes, Convolutional Hypervisor Interposer, C(6,2) Derivatives & Unified Gateway
- Fixed 5 Double-Time bugs: 120Hz k=2 clamping, non-overlapping 128B BUF_00/01, eta_crit=1.72, UMA>=0.78 halt & HTTP query params
- Added OpenVINO ConvolutionalHypervisorStepBridge (1x6x1x16 -> Conv1x3 -> StepExecutionGateProxy + 64B WNTC + 128B KPKT)
- Synthesized DF_01..DF_05 combinatorial derivatives, TH_01..TH_04 cross-module theories, Neurosymbolic KB & UnifiedBackendUiAccessGateway

### Synthesized Next-Layer Prompt Seed
```text
SYSTEM ANCHOR [KRYSTAL STACK PROMPT LAYER v2026.11.17]:
1. Lock constitutional invariants: VITAL_MAX_HP == 6, UMA < 0.78, PROPOSER != VERIFIER, ~Valid => Delta_S == 0, 0 bare asserts.
2. Route every operation window (1x6x1x16, canonicalize_float32) through ConvolutionalHypervisorStepBridge -> StepExecutionGateProxy.
3. Use binary cache-line multiples internally (64B WNTC, 128B KPKT, 192B Dual-OS, 32B .kopt) and expose UI strictly via UAP_01 Projection Envelopes.
4. Treat every new cross-module synthesis as PATTERN_CANDIDATE until an isolated CDT + pytest receipt promotes it to VERIFIED_WITH_SCOPE.
5. Stratify claims into LOCAL_MEASURED, VERIFIED_WITH_SCOPE, DERIVED, or UNKNOWN; never invoke OpenAI endpoints.
```
