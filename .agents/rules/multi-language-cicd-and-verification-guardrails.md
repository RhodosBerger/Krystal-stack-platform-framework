# Multi-Language CI/CD & Verification Guardrails (Krystal-Stack)

## Behavioral Rule
Apply to every pull request, multi-language crate, test suite, and CI/CD workflow in this repository.

### 1. Rust Cargo Workspace Discipline
- Every nested crate or subpackage in the repository (e.g., `src/rust-bot`, `rust/epistemic_runtime`, `krystal-miner`) MUST be explicitly registered in the root `Cargo.toml` under `[workspace.members]`.
- Never create a standalone or nested `Cargo.toml` without registering it in the root workspace manifest; doing so breaks cargo resolution in CI/CD with `current package believes it's in a workspace when it's not`.
- Remember that in a Cargo workspace, artifacts (`target/doc`, `target/release`, etc.) are written to the workspace root directory `target/`, NOT under individual subcrate directories.

### 2. Rust Syntax & Borrow Safety
- Enum variants must always be accessed via the standard double-colon scope operator `EnumName::Variant`.
- When verifying, hashing, or cryptographically signing telemetry or candidate action records, never partially move (`partial move`) fields out of an object that will subsequently be referenced or borrowed. Always borrow (`&`) or `.clone()`.
- Structs containing dynamic closures or trait objects (`Box<dyn Fn(...)>`) cannot implement `Clone` or `Debug`. Do NOT attach `#[derive(Clone, Debug)]` to such types.

### 3. Headless Network Test Isolation
- No integration test or unit test may assume that an HTTP server or daemon is active on `localhost:8089` (or any other port), unless the test suite itself starts a background daemon process.
- Any test suite querying network sockets must implement a preflight connectivity check in `setUpClass` with a strict timeout (<= 1.0s), and call `unittest.SkipTest(...)` if the endpoint is offline to guarantee deterministic green runs on headless CI runners.

### 4. C Core Engine ABI & Header Synchronization
- Header files in `src/c/*.h` must declare the exact API surface, types, and configurations used by demos, kernel drivers, and external language bindings (`gamesa_kernel_config_t`, `tbl_zone_config_t`, `rpg_recipe_config_t`, `rpg_boost_map_t`).
- Header declarations and C definitions must remain synchronized to prevent linker symbols or header mismatch failures during `cmake` and `ninja` builds.

### 5. CI/CD Artifact & Target Path Resolution
- When publishing documentation or compiled binaries in GitHub Actions workflows (`actions/upload-artifact@v4`), always ensure the staging directory exists before upload.
- Because Cargo workspace builds produce outputs in `target/` at the root, workflow steps must resolve artifacts from `target/` or use fallback staging directories (`doc-dist/`) rather than assuming subcrate-local `src/rust-bot/target/`.

### 6. Architectural Invariant Integrity
- Across all language boundaries (Python, Rust, C/C++, Janet, Godot GDScript), the fundamental system invariant `VITAL_MAX_HP == 6` must be strictly enforced.
- Metadata tags in exported scenes (`metadata/vital_max_hp = 6`) and actor structs must never exceed 6.

### 7. SceneComposer Kinematics & Trajectory Standards
- Dynamic camera paths in `mimicry_engine/scene_composer.py` must subclass `CameraTrajectory` and provide `get_camera_state(t) -> Tuple[Vec3, Vec3, float]`.
- Non-closed trajectories (such as `CinematicDollyZoomTrajectory` and `SpiralAscentTrajectory`) must clamp progress at `t == duration` when `loop=False`.
- Raymarching viewports must dynamically adapt camera focal length to instant FOV, and Godot 4.x scene exports must produce valid `Path3D` + `Curve3D` + `PathFollow3D` node structures when trajectories are active.
