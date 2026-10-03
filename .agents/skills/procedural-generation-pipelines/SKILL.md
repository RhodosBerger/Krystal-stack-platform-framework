---
name: procedural-generation-pipelines
description: End-to-end workflow cheatsheet for creating, validating, transpiling, and visualizing consistent procedural generation systems in Krystal Stack (terrain, cities, vehicles, islands, tactical arenas, and harmonic color tensors).
---

# Procedural Generation Pipelines (Krystal-Stack)

Use this skill when designing, implementing, expanding, or debugging procedural generation systems across the Krystal Stack framework.

## 1. Architectural Architecture & Core Invariants

Every procedural generator in Krystal Stack follows the 6-phase pipeline:

```
[Phase 1: Math & Lore Axioms]
       | (Phi 1.618, Whittaker Space, 6 Max HP Invariant)
       v
[Phase 2: Janet DSL Specification] (`krystal_janet/<domain>.janet`)
       | (Immutable tables, deterministic functional seed mappings)
       v
[Phase 3: Python Dataclass Engine] (`krystal_web_hub/economic_engine/<domain>.py`)
       | (Data validation, discrete physics, JSON serialization)
       v
[Phase 4: Modern Java 21 Transpiler] (`KrystalJavaTranspiler.generate_*`)
       | (Java records, sealed interfaces, parallel Virtual Thread pipelines)
       v
[Phase 5: Web Hub REST Endpoints] (`krystal_web_hub/krystal_engine_core.py`)
       | (Capped body, strict error handling, deterministic seed params)
       v
[Phase 6: Interactive HTML5 Canvas Studio] (`krystal_web_hub/static/<domain>_studio.html`)
       | (Top-down 2D canvas, neon aesthetic, real-time telemetry HUD)
       v
[Verification Suite] (Unit tests in `tests/test_<domain>.py` + Janet validator)
```

## 2. Standard Checklist for New Procedural Domains

When adding a procedural domain (e.g., street networks, island realms, celestial spires, vehicle cruisers):

1. **Verify Invariant Constants:**
   ```python
   GOLDEN_RATIO: float = 1.61803398875
   INV_GOLDEN_RATIO: float = 1.0 / GOLDEN_RATIO
   VITAL_MAX_HP: int = 6  # NEVER exceed 6
   ```

2. **Deterministic Seed Derivation:**
   ```python
   def generate_procedural_coords(seed: int = 42, count: int = 24) -> List[Dict[str, float]]:
       # Deterministic PRNG or trigonometric harmonic decomposition
       rng = random.Random(seed)
       points = []
       for i in range(count):
           angle = (i / float(count)) * 2.0 * math.pi
           noise = math.sin(angle * 3.0 + seed) * 20.0 + rng.uniform(-5.0, 5.0)
           r = 150.0 + noise
           points.append({"x": round(400.0 + r * math.cos(angle), 2),
                          "y": round(250.0 + r * math.sin(angle), 2)})
       return points
   ```

3. **Janet DSL Mirror (`krystal_janet/<domain>_engine.janet`):**
   - Define constants: `(def VITAL-MAX-HP 6)`, `(def GOLDEN-RATIO 1.6180339887)`.
   - Provide immutable dictionary schemas for all entities.
   - Verify with: `python -m krystal_janet.janet_bridge`.

4. **Java 21 Transpiler Parity:**
   - Map Python `@dataclass` directly to `public record DomainName(...)`.
   - Include compact constructor validation enforcing `vitalMaxHp <= VITAL_MAX_HP`:
     ```java
     public record ProceduralEntity(String id, int hp, int maxHp) {
         public ProceduralEntity {
             if (maxHp > 6) maxHp = 6;
             if (hp > maxHp) hp = maxHp;
         }
     }
     ```

5. **Web Hub Integration & Port Safety:**
   - Register endpoints in `krystal_web_hub/krystal_engine_core.py` under `/api/procedural/<domain>`.
   - Ensure the server is tested with timeout-guarded Python verification scripts (`timeout=5.0`).
   - Check port conflicts with `Get-NetTCPConnection -LocalPort <port>` before test launches.

6. **Interactive Studio Design Standards:**
   - Background: Dark noir charcoal (`#0f0f12` or `#18181b`).
   - Primary accents: Suave Pink Panther (`#ec4899`, `#f472b6`) or Cyber Neon (`#06b6d4`, `#10b981`).
   - HUD: Real-time telemetry displaying current seed, entity count, 6 Max HP bars (6 discrete heart/pip icons), and generation time in ms.

## 3. Common Troubleshooting & Debugging

| Symptom | Root Cause | Immediate Fix |
|---|---|---|
| **Geometry jumps on re-render** | Unseeded `Math.random()` in frontend canvas. | Pass the server's deterministic `seed` to the JavaScript draw routine and use a Mulberry32 or LCG PRNG in JS. |
| **HP bar shows 100/100 or 10/10** | Violation of the 6 Max HP vital invariant. | Clamp hit points to `min(hp, 6)` and render exactly 6 discrete pips/hearts in the HUD. |
| **Colors appear muddy or mismatched** | Ad-hoc RGB values without harmonic spacing. | Apply the Golden Ratio hue step: `hue = (base_hue + i * 137.508) % 360` for optimal perceptual separation. |
| **Janet bridge fails symbol audit** | Unbound dynamic symbols or missing core macros. | Run `python -m krystal_janet.janet_bridge` to inspect any unrecognized symbols. |
| **Transpiled Java compilation error** | Missing sealed interface permits or record syntax mismatch. | Ensure all permitted classes are listed in `permits` and defined in the same compilation unit. |
