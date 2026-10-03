# Krystal UI System — AI Style Contract v1.0

## 1. Design Identity & Mathematical Equation
```
Design Equation:
Dark cinematic portal + editorial fantasy typography + scientific simulation HUD + restrained neon telemetry
```
The application must feel like a **game-world operating system**, not a SaaS dashboard.

The visual hierarchy is strictly:
```
WORLD → FACTION → SYSTEM → ACTION → TELEMETRY
```
- The 3D arena is the **computational core**.
- The surrounding portal UI is the **narrative and navigation shell**.

---

## 2. Canonical Color System
| Token | Hex Value | Semantic Purpose |
| :--- | :--- | :--- |
| `--ks-black` | `#030406` | Arena / absolute background |
| `--ks-void` | `#070812` | Main application background |
| `--ks-navy` | `#101124` | Navigation / portal surfaces |
| `--ks-panel` | `#11131D` | Functional panels |
| `--ks-panel-soft` | `#171827` | Elevated content |
| `--ks-line` | `#292A38` | Structural borders (1px) |
| `--ks-text` | `#F1EFE8` | Primary text |
| `--ks-muted` | `#8D8997` | Secondary metadata |
| `--ks-gold` | `#D8AE4B` | World / navigation accent |
| `--ks-gold-hi` | `#F0D071` | Important fantasy CTA |
| `--ks-cyan` | `#58D8FF` | Crystal / system telemetry |
| `--ks-teal` | `#50EEE0` | Runtime state |
| `--ks-magenta` | `#FF307C` | Active ability / action |
| `--ks-green` | `#43D264` | Online / success state |
| `--ks-violet` | `#7548D8` | Spatial / debug geometry |

> **CRITICAL ACCENT RULE:**
> **Gold and Cyan must NEVER compete as equal accents.**
> - **Gold (`#D8AE4B`)** = World / Narrative layer.
> - **Cyan (`#58D8FF`)** = System / Computation layer.

### Arena Semantic Colors
- `green` (`#43D264`) = Status / Healthy
- `cyan` (`#58D8FF`) = World / Actor
- `violet` (`#7548D8`) = Bounds / Collider
- `magenta` (`#FF307C`) = Player action
- `red` (`#FF4757`) = Failure / Damage
- `yellow` (`#F59E0B`) = Warning

*Never use these randomly for decoration.*

---

## 3. Typography Architecture
Use exactly three typographic roles:
1. **Display Serif (monumental, high contrast):** Cormorant Garamond, Cinzel, or custom Roman serif (Tribe names, chapters, world titles).
2. **Condensed Sans (uppercase, tracked):** Bebas Neue, Oswald, or DIN Condensed (Navigation, buttons, labels).
3. **Monospace (technical):** JetBrains Mono, IBM Plex Mono (Telemetry, runtime, coordinates, system states).

### Scale & Spacing
- `WORLD TITLE`: 72–128px / 0.88 line-height
- `SECTION TITLE`: 42–72px
- `ENTITY TITLE`: 28–44px
- `UI LABEL`: 12–15px
- `BODY`: 16–19px
- `SYSTEM DATA`: 11–14px monospace
- `MICRO LABEL`: 10–12px / letter-spacing 0.20–0.32em

*Large serif typography must remain sparse. Never fill the interface with serif body text.*

---

## 4. Geometry & Spacing
- **Strictly rectilinear and architectural:**
  - `--radius-ui: 2px;`
  - `--radius-control: 4px;`
  - `--radius-terminal: 4px;`
- **1px structural borders.**
- **BANNED:**
  - NO generic rounded SaaS cards.
  - NO drop shadows or outer blur glows on structural panels.
  - NO glassmorphism / blurred card backgrounds.
  - NO pill cards.
- **Base 4px Spacing Unit:**
  - `4`: micro offset
  - `8`: inline spacing
  - `12`: compact control
  - `16`: standard content
  - `24`: component separation
  - `32`: panel padding
  - `48`: section spacing
  - `64`: major composition gap
  - `96`: cinematic whitespace
  *(Never generate arbitrary values like 27px, 43px, 71px).*

---

## 5. Dual Modes & Hybrid Presentation
1. **World / Portal Mode:** Cinematic fantasy, navy/black, gold, monumental editorial serif.
2. **Runtime / Arena Mode:** Diagnostic simulation layer, pure black, cyan/teal/magenta, monospace telemetry.
3. **Hybrid Mode:**
   - Dominant visual object (60–72% viewport width).
   - Contextual narrative/faction rail (28–40% viewport width).
   - Compact Edge HUD (<15% viewport area).

---

## 6. The 14 Canonical UI Primitives
Any AI generating Krystal UI must compose exclusively from these 14 primitives:
1. `KSNav`: Deep navy global header (76–96px) with gold active indicators.
2. `KSWorldHero`: Monumental title and asymmetric editorial negative space.
3. `KSEntitySelector`: Faction tabs (Crystal, Toxic, Druid).
4. `KSStatBar`: Segmented or sharp rectilinear rating bars.
5. `KSRadar`: Compact polygon radar chart.
6. `KSHudStatus`: Edge-anchored minimal status indicator.
7. `KSRuntimeMetric`: Metric box with explicit verification state tag (`[MEASURED]`, `[MODELED]`, `[TARGET]`).
8. `KSActionButton`: Monospace system action button (48–60px, 1px semantic border).
9. `KSWorldButton`: Gold fantasy CTA (56–72px, black text, uppercase condensed).
10. `KSTerminalPanel`: Monospace log and instruction feed.
11. `KSRenderViewport`: 3D procedural simulation canvas.
12. `KSBiomeIndicator`: Biome badge with continuous math metadata.
13. `KSChapterLabel`: Micro tracked uppercase eyebrow.
14. `KSArtifactCard`: Strict 1px bordered entity card.

---

## 7. Metric Stratification Taxonomy: MEASURED vs MODELED vs TARGET
When presenting performance, throughput, or latency metrics, always classify them into one of these three explicit states:
- **`[MEASURED]`**: Empirically benchmarked on active code (e.g. FastRingBuffer 1.724M pps vs Mutex 392k pps = 4.39×; CPU SDF 997k eval/s; terrain kernel 89.36 μs/sample; visual-entropy 1.07 ms/frame).
- **`[MODELED]`**: Mathematically extrapolated projection based on architectural modeling (e.g. AVX2 SIMD vectorization 6.8×; Iris Xe compute queue batching).
- **`[TARGET]`**: Roadmap performance target for unreleased backends (e.g. WebGPU 120 FPS; Neural SDF <0.28 ns/query; Rust SIMD 45×).

---

## 8. Generation Decision Order
Every agent creating or extending a screen must execute in this exact sequence:
1. **MODE** → World | Arena | Hybrid
2. **PRIMARY OBJECT** → What is visually dominant?
3. **INFORMATION HIERARCHY** → World → Entity → State → Action → Telemetry
4. **ACCENT DOMAIN** → Gold (Narrative) vs Cyan (Computation)
5. **EXISTING COMPONENTS** → Compose from the 14 KS primitives
6. **GRID** → Apply 4px base spacing
7. **MOTION** → Subtle state-relevant motion only (150–320ms)
8. **RESPONSIVE TRANSFORMATION** → Recompose rather than shrink
