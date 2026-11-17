# Conspectus of Verified Features vs. New Cross-Framework Bridging Inventions

> **Epistemic Mode (`AGENTS.md`):**
> - **Left Column (Conspected Features):** `LOCAL_MEASURED` / `VERIFIED_WITH_SCOPE` — empirically tested in `commits 82c28a59..0496bc4e`.
> - **Right Column (New Bridging Inventions `INV_01..INV_06`):** `EXPLORE · PATTERN_CANDIDATE` — structured hypotheses with explicit **Cheapest Discriminating Tests (CDT)** and **Falsifiers** awaiting promotion via `VERIFY`.
> - **Constitutional Anchors:** `VITAL_MAX_HP == 6` · `UMA < 0.78` · `PROPOSER != VERIFIER` · $\neg\text{Valid} \implies \Delta S = 0$.

---

## 1. Synthesis Matrix: Conspected Foundation $\longrightarrow$ New Bridging Inventions

| ID | Conspected Foundation (`VERIFIED` / `MEASURED`) | Sibling Repo Target | New Bridging Invention (`EXPLORE · PATTERN_CANDIDATE`) | Priority $\frac{R \times I \times F}{C}$ |
| :--- | :--- | :--- | :--- | :---: |
| **`INV_01`** | `ConvolutionalHypervisorStepBridge` (`1×6×1×16` OpenVINO Conv) + `128 B KPKT` (`5.04×` vs JSON) | `rust/krystal_sovereignty_bytecode` & `krystal-bitboard/src/edge.rs` | **Zero-GC Rust Bitboard & Bytecode C-ABI Ring (`KPKT <-> Rust SPSC`)**: Execute legal 37-hex bitboard moves and sovereignty bytecode verification in Rust over a shared 128 B C-ABI buffer. | **`138.6`** |
| **`INV_02`** | `OP_01_BACKBRANCH_UMA_FALLBACK` + `.kopt` Fresnel + State A/B Wireframes | `ascii_neural_compositor/neural_art_engine.py` | **Chronometric Neural-ASCII Survival Viewport**: When `UMA >= 0.78` persists for $> 24$ frames, auto-project the 3D arena into a `< 4 KB/frame` glyph/ASCII semantic HUD at guaranteed 120 Hz. | **`124.4`** |
| **`INV_03`** | `generate_hex_siege_visual_story` + `DESC_03_GODOT_LAYER_MASK` (`0x3ff`/`0x1f`) | `godot_project/scenes/` (`HoloStage.tscn`, `Compositor8KScalingArena.tscn`) | **Live Godot 4 Scene & `AnimationPlayer` Track Compiler**: Stream verified 128 B keyframes and `.kopt` IOR materials directly into existing `godot_project/scenes/*.tscn` stages. | **`115.5`** |
| **`INV_04`** | `64 B WNTC` (`WslNtCoexistenceBinaryBridge`) + `WSL2 RAM Quota` (`10 -> 8 GB`) | `wsl_bridge/bridge_daemon.py` (`/tmp/krystal_wsl.sock`) | **Live Dual-OS `WNTC + KPKT` Socket Multiplexer**: Replace JSON sync in `WSLBridgeDaemon` with packed `64 B WNTC + 128 B KPKT` frames and measure true cross-boundary round-trip latency. | **`109.2`** |
| **`INV_05`** | `x86_platform_findings.py` (`X01..X05`: L2 `55.1 GB/s`, `37.4×` subnormal cliff) | `cpp/hardware_probe/hardware_probe.cpp` | **Native C++ Hardware Probe DLL/SO Bridge**: Bind `hardware_probe.cpp` via `ctypes` to feed cycle-accurate TSC jitter & CPUID cache topology into `StepExecutionGateProxy`. | **`97.8`** |
| **`INV_06`** | `.kopt` Optical Codec + Priority Tabular Logger (`40.37×` compression, `ECE = 0.0517`) | `wordpress_plugin_krystal/` (`nabytok47_acf_oxygen_grid_bridge.php`) & `krystal_bot` | **Signed Binary Telemetry-to-Web/CMS Projection**: Push compact SHA-256 attested `.kopt` material presets & calibrated hardware profiles into Oxygen/WordPress and `krystal_bot` without cloud LLM calls. | **`84.0`** |

---

## 2. Detailed 6-Tuple Conspectus & Cheapest Discriminating Tests (CDT)

### `INV_01` · Zero-GC Rust Bitboard & Sovereignty Bytecode C-ABI Bridge
- **Mechanism:** Map the 37-cell axial hex board into a single `u64` bitmask (`37 <= 64` bits!) inside `krystal-bitboard` / `rust/krystal_sovereignty_bytecode`, exchanging `128 B` aligned frames directly with Python/OpenVINO without serialization.
- **Expected Effect:** Increase authoritative move validation from `13,362.7 commits/s` (Python) to `> 250,000 commits/s` (`> 18×`) with zero Python GC pause jitter.
- **Cheapest Discriminating Test (CDT):** Compile a minimal `cdylib` FFI function accepting `const uint8_t frame[128]` and benchmark 10,000 legal move checks + SHA-256 against `ClashGame.apply`.
- **Falsifier:** FFI boundary marshaling overhead exceeds Python `ClashGame.apply` (`0.075 ms`) or diverges on any legal move mask.

### `INV_02` · Chronometric Neural-ASCII Survival Viewport (`UMA >= 0.78` Tier-2 Fallback)
- **Mechanism:** Connect `ConvolutionalHypervisorStepBridge` to `ascii_neural_compositor` as a third degradation tier (`Full 3D 10-layer -> Wireframe 5-layer -> Neural-ASCII < 4 KB glyph grid`) during extreme thermal/UMA saturation.
- **Expected Effect:** Cut viewport memory traffic by `> 95 %` while keeping all 6 axial movement templates, enemy positions, and HP `0..6` human-readable at 120 Hz.
- **CDT:** Render one `ClashGame` state through `ascii_neural_compositor` glyph mapping and verify `< 1.0 ms` frame time + lossless unit/HP legibility.
- **Falsifier:** Glyph projection loses distinctness between friendly/enemy units or exceeds `2.0 ms` CPU time.

### `INV_03` · Live Godot 4 Scene & `AnimationPlayer` Track Compiler
- **Mechanism:** Bridge `dep_bridge_visual_story_generators.py` into `godot_project/scenes/HoloStage.tscn` and `Compositor8KScalingArena.tscn` by emitting native Godot 4 `[sub_resource type="Animation"]` position/rotation/layer-mask tracks.
- **Expected Effect:** Turn every deterministic simulation or sensor-conflict replay into a zero-code playable 3D cutscene inside the existing Godot project.
- **CDT:** Generate a `.tscn` animation resource from the 6-frame mortar siege generator and verify valid Godot 4 parser load & keyframe monotonically increasing timestamps.
- **Falsifier:** Malformed `.tscn` sub-resource references or layer-mask breach (`> 0x1f`) during `UMA >= 0.78` frames.

### `INV_04` · Live Dual-OS `WNTC (64B) + KPKT (128B)` WSL2 Daemon Multiplexer
- **Mechanism:** Upgrade `wsl_bridge/bridge_daemon.py` (`WSL_SYNC` currently using JSON dicts) to transmit concatenated `192 B` (`64 B WNTC + 128 B KPKT` = exactly `3 × 64 B` cache lines!) binary datagrams.
- **Expected Effect:** Resolve `PUZZLE_GAP_02_LIVE_WSL2_VSOCK_FD` (`sovereign_openvino_companion_v2`) and achieve `~5×` lower serialization CPU overhead across the Windows $\leftrightarrow$ WSL2 boundary.
- **CDT:** Loopback benchmark of 1,000 `192 B` `WNTC+KPKT` frames through `WSLBridgeDaemon` verifying `CRC32` + `SHA-256` and measuring `p50 / p95` latency vs. JSON.
- **Falsifier:** Byte-order/alignment mismatch or `< 2.0×` speedup over UTF-8 JSON.

### `INV_05` · Native C++ `hardware_probe` Shared-Library Telemetry Bridge
- **Mechanism:** Compile `cpp/hardware_probe/hardware_probe.cpp` into a lightweight shared module exposing invariant `RDTSC` jitter and cache-hierarchy descriptors directly to `ConvolutionalHypervisorStepBridge`.
- **Expected Effect:** Sub-microsecond hardware jitter sensing (`< 5 µs`) replacing coarse OS polling (`psutil`) inside the VBlank shadow window (`< 2.2 ms`).
- **CDT:** Compare 100 `psutil` calls vs. C++ probe calls for call overhead and `p99` variance.
- **Falsifier:** Compiler toolchain unavailability or probe overhead `> 0.1 ms`.

### `INV_06` · Signed Binary Telemetry-to-Oxygen/WordPress & `krystal_bot` Preset Sync
- **Mechanism:** Connect `priority_tabular_synthetic_logger.py` (`33.9 KB` calibrated trace) and `.kopt` material presets into `wordpress_plugin_krystal/nabytok47_acf_oxygen_grid_bridge.php` and `krystal_bot/config_cheatsheet_expert.py`.
- **Expected Effect:** Live 3D product/scene configurators on the web automatically pick the visitor's optimal rendering tier (`BP_360..1920`, `.kopt` Fresnel LOD) verified by cryptographic digest.
- **CDT:** Round-trip a `.kopt` preset + `CFG_01..05` recommendation into the ACF/Oxygen JSON/binary schema and verify digest match.
- **Falsifier:** Schema mismatch with `nabytok47_acf_oxygen_grid_bridge.php` field keys.
