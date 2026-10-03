# Krystal-Bootstrap: Procedural Instance Schemas & The 3D Web Engine Design System

**Author:** Krystal-Stack Research & Architecture Team  
**Date:** October 2026  
**Status:** Canonical Engineering Specification & Architecture Manifesto  

---

## 1. Executive Summary & The "Engine Bootstrap" Paradigm

In traditional web development, building user interfaces before 2011 required writing hundreds of lines of custom CSS, wrestling with browser inconsistencies, implementing grid math manually, and creating ad-hoc buttons and modal popups. **Twitter Bootstrap** transformed the entire industry by establishing:
1. **A Standardized Grid System** (`.container`, `.row`, `.col-md-6`)
2. **Modular UI Components** (`.btn`, `.card`, `.navbar`, `.badge`)
3. **Atomic Utility Classes** (`.p-3`, `.text-center`, `.bg-dark`)
4. **Responsive Breakpoints** (`sm`, `md`, `lg`, `xl`)

Today, **3D and procedural game engine development on the web is in the same fragmented pre-Bootstrap era**. Developers must write hundreds of lines of WebGL/Three.js boilerplate, raw WGSL/GLSL shaders, custom camera matrices, raymarching loops, and manual scene graph trees simply to render a procedural shape or procedural world in a web browser.

**Krystal-Bootstrap** introduces the world's first **3D Engine Design System & Web Component Framework**, providing:
- **Declarative HTML5 Web Components**: `<krystal-viewport>`, `<krystal-scene>`, `<krystal-grid>`, `<krystal-instance>`, `<krystal-modifier>`.
- **3D Spatial Layout Classes**: `.k-grid-radial`, `.k-grid-linear`, `.k-grid-hex`, `.k-grid-menger`.
- **Shader & Material Utility Tokens**: `.k-mat-cyber-neon`, `.k-mat-alchemical-gold`, `.k-mat-holographic`, `.k-mat-blueprint`.
- **Procedural Instance Generator Schemas**: Dynamic mathematical synthesis using Johan Gielis' 3D Superformula, alchemical polyhedra, and modular cybernetic spires.

```
+-----------------------------------------------------------------------------------+
|                        KRYSTAL-BOOTSTRAP COMPONENT HIERARCHY                      |
|                                                                                   |
|  <krystal-viewport class="k-viewport-16x9 k-fx-scanlines" cols="80" rows="30">    |
|    |                                                                              |
|    +--> <krystal-scene camera="orbital" lighting="cyberpunk">                     |
|           |                                                                       |
|           +--> <krystal-grid layout="radial" count="6" spacing="2.5">             |
|                  |                                                                |
|                  +--> <krystal-instance src="superformula:m=6"                    |
|                                         class="k-mat-cyber-neon">                 |
|                         |                                                         |
|                         +--> <krystal-modifier type="twist" rate="0.5" />         |
|                         +--> <krystal-modifier type="bevel" radius="0.08" />     |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

---

## 2. Dynamic Procedural Instance Generation

Instead of loading static, pre-baked 50 MB glTF/FBX mesh assets over the network, Krystal-Bootstrap uses **Procedural Instance Schemas** synthesized on the fly:

### A. The 3D Gielis Superformula Manifold
Johan Gielis' generalized superformula unifies circles, polygons, starfish, crystals, and natural shells into a single parametric equation:

$$r(\phi) = \left( \left| \frac{\cos\left(\frac{m \phi}{4}\right)}{a} \right|^{n_2} + \left| \frac{\sin\left(\frac{m \phi}{4}\right)}{b} \right|^{n_3} \right)^{-\frac{1}{n_1}}$$

By modulating parameters $(m, n_1, n_2, n_3)$, the generator in [procedural_generator.py](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/instances/procedural_generator.py) produces millions of distinct, mathematically rigorous 3D shapes on demand:
- $m = 6, n_1 = 1.0, n_2 = n_3 = 1.0 \implies$ Hexagonal Prism / Crystal
- $m = 5, n_1 = 0.5, n_2 = n_3 = 1.7 \implies$ Sacred Pentagonal Starfish
- $m = 8, n_1 = 0.2, n_2 = n_3 = 0.5 \implies$ Octahedral Spiked Mine

### B. Modular Cybernetic Data Spires
Synthesized through algorithmic stacking of cylindrical conduits, radial coolant radiators, and holographic interference crowns:

$$\Phi_{\text{spire}}(p) = \mathcal{S}_{\min}\left( \text{CoreCylinder}(p), \, \text{RadialArray}(\text{RadiatorFin}(p), N), \, k \right)$$

---

## 3. The 3D Web Component Architecture

Krystal-Bootstrap registers custom elements natively in the browser via `customElements.define()`:

```html
<!-- Complete 3D procedural stage in 8 lines of pure HTML -->
<link rel="stylesheet" href="/static/krystal-bootstrap.css">
<script src="/static/krystal-bootstrap.js" defer></script>

<krystal-viewport class="k-viewport-16x9 k-fx-scanlines"
                  mode="RECURSIVE_MIRROR"
                  cols="80" rows="30">
</krystal-viewport>
```

### Features:
1. **Zero External Dependencies**: Operates with zero npm modules, zero node_modules bloat, and pure web standards.
2. **Multi-Backend Fallback**: Supports real-time Server-Sent Events (SSE) ASCII streams, HTML5 Canvas 2D, and WebGL2/WebGPU shaders.
3. **Shadow DOM Encapsulation**: Styles and CRT scanline post-processing filters are isolated inside the element's shadow root.

---

## 4. Formal JSON Schema Standard

The Krystal-Bootstrap framework is backed by two formal JSON Schema definitions:
1. [krystal_instance.schema.json](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/schemas/krystal_instance.schema.json): Governs parametric definitions, symmetry groups, default continuous variables, and glyph mappings.
2. [krystal_engine_bootstrap.schema.json](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/schemas/krystal_engine_bootstrap.schema.json): Governs scene graphs, camera viewports, spatial layout grids, modifier stacks, and post-processing.

This schema ensures that **LLMs can generate, validate, and manipulate 3D web scenes with 100% syntactical guarantee**, eliminating hallucinated attribute names or broken scene graphs.

---

## 5. Architectural Invariants & Procedural Generation Standards

All procedural generation components, instances, and generators must conform to the platform-wide architectural rules:
- **Core Governance Rule**: [.agents/rules/consistent-procedural-generation-patterns.md](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/.agents/rules/consistent-procedural-generation-patterns.md)
  - **The 6 Max HP Vital Invariant**: Hit points for any generated entity, garrison, or vehicle are strictly $\text{HP} \le 6$ and $\text{MaxHP} = 6$.
  - **Axiomatic Determinism**: All generators must accept an optional deterministic `seed`.
  - **Harmonic Proportions**: Scaling and color tensors derive from the Golden Ratio ($\phi = 1.61803398875$).
  - **Tripartite Language Parity**: Identical schema representation across Janet DSL (`krystal_janet/`), Python `@dataclass`, and Java 21 `record`s.
- **Actionable Workflow Skill**: [.agents/skills/procedural-generation-pipelines/SKILL.md](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/.agents/skills/procedural-generation-pipelines/SKILL.md)

