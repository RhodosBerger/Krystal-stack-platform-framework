# Consistent Procedural Generation Patterns (Krystal-Stack)

All procedural generation engines, algorithms, shaders, and data schemas across the Krystal Stack platform framework must adhere to these non-negotiable architectural invariants:

1. **The Immutable 6 Max HP Vital Invariant:**
   - Every procedural entity—including characters, heroes, units, vehicles, defensive garrisons, citadels, and convoys—must have its hit points strictly clamped:
     $$\text{HP} \le 6 \quad \text{and} \quad \text{MaxHP} = 6$$
   - No procedural modifier, buff, or level scaler may elevate maximum health beyond 6. Damage and armor mitigation must operate within this discrete 6-point integer vital lattice.

2. **Axiomatic Determinism & Seed Hygiene:**
   - Every procedural generator must accept an optional deterministic integer or string `seed`.
   - When a `seed` is provided, output geometry, road networks, terrain heights, and spawn coordinates must be 100% reproducible across calls, platforms, and languages.
   - Never call bare unseeded global randomizers for persistent procedural assets. If no seed is provided, capture and expose the generated seed in the return payload for client replay.

3. **Harmonic Proportions & Mathematical Axioms:**
   - Dimension scaling, color gradients, and structural hierarchies must derive from explicit mathematical axioms:
     - **Golden Ratio:** $\phi = 1.61803398875$ and $\phi^{-1} \approx 0.61803398875$ for layout splits and harmonic color shifts.
     - **Whittaker Biome Phase Space:** Biomes must be parameterized continuously across Temperature $\mathcal{T} \in [-1, 1]$, Moisture $\mathcal{M} \in [-1, 1]$, and Techno-Anomaly $\mathcal{A} \in [0, 1]$.
     - **Continuous Smoothing:** Noise-driven elevations and shorelines must use $C^2$-continuous Hermite or sinusoidal interpolation to eliminate mesh creasing.

4. **Tripartite Schema Parity (Janet $\leftrightarrow$ Python $\leftrightarrow$ Java 21):**
   - Whenever a new procedural domain is created, its data model must be identically represented across:
     a. **Janet DSL** (`krystal_janet/<domain>.janet`): Canonical immutable data structures, maps, and functional generators.
     b. **Python Engine** (`krystal_web_hub/economic_engine/<domain>.py`): Strongly-typed `@dataclass` models, business logic, and JSON serialization.
     c. **Java 21 Transpiler Target**: Java records, `sealed interface` hierarchies, and parallel `CompletableFuture` / Virtual Thread pipelines.

5. **Safe Bounding & Numerical Clamping:**
   - Coordinate spaces must be explicitly bounded (e.g. Canvas $[0, 800] \times [0, 500]$, elevation $[0, 5000\text{ m}]$, normalized tensors $[0.0, 1.0]$).
   - Guard against zero-division in velocity, inverse distance weighting, and radius calculations:
     $$\text{safe\_div}(n, d) = \frac{n}{d + \epsilon} \quad (\epsilon = 10^{-6})$$

6. **Interactive Canvas & Zero-Placeholder Visual Policy:**
   - Every procedural engine exposed through the web hub must feature an interactive visual studio (HTML5 Canvas or Godot shader viewport) rendered in sleek dark mode with neon accents.
   - Never render broken image links, placeholder boxes, or empty rectangles. Use mathematical geometry, vector strokes, and procedural color blending directly on the canvas.
