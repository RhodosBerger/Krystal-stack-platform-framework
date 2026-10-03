# Krystal-Stack: Neural ASCII & Cognitive Engine Research Hub

Vítajte v centrálnej výskumnej dokumentácii pre novú generáciu **Neural ASCII Compositora**, **NPU akcelerácie**, **Vulkan Compute techník**, **Krystal-Lang geometrického bytekódu**, **otvoreného sveta (Open-World Procedural Engine)** a **kognitívneho riadenia cez Small Language Models (SLM)**.

Tento výskumný balík nadväzuje na pôvodný koncept [Active Optic Compositor (AOC)](../ASCII_COMPOSITOR_AI_ADAPTATION_STUDY.md) a rozvíja ho do revolučného, vysoko škálovateľného systému pre moderné herné, webové a distribuované výpočty.

---

## 📑 Prehľad výskumnej dokumentácie

| Dokument | Zameranie a kľúčové inovácie | Odkaz |
| :--- | :--- | :--- |
| **1. Cyklická Teória & Alternatívne Schémy Architektúry** | Teória cyklov v úspechoch GAMESA/Krystal-Stack, termodynamika výpočtov (Landauer limit), homeostatické kognitívne vlny ($\alpha, \beta, \gamma, \Omega$), porovnávacia štúdia 5 alternatívnych schém architektúry a zjednotený symplektický Hamiltonovský organizmus s Lyapunovovou stabilitou. | [CYCLIC_THEORY_AND_ALTERNATIVE_ARCHITECTURAL_SCHEMAS.md](./CYCLIC_THEORY_AND_ALTERNATIVE_ARCHITECTURAL_SCHEMAS.md) |
| **2. Deep Performance & Core Architecture** | Matematická Big-O analýza, profilovanie L1/L2/L3 cache, lock-free ring-buffer (`FastRingBuffer` 4.39x zrýchlenie), AVX2 SIMD dávkový raymarching, DirectML NPU INT8 neurálna destilácia manifolds a WebGPU compute. | [DEEP_PERFORMANCE_OPTIMIZATION_AND_CORE_ARCHITECTURE.md](./DEEP_PERFORMANCE_OPTIMIZATION_AND_CORE_ARCHITECTURE.md) |
| **3. Web Engine Bootstrap & Procedural Schemas** | Bootstrap analóg pre webové enginy, Gielis 3D Superformula $r(\phi)$, Cyber Spires, Alchemical Polyhedra, JSON Schémy pre inštancie a vlastné webové komponenty `<krystal-viewport>`, `<krystal-scene>`, `<krystal-grid>`. | [WEB_ENGINE_BOOTSTRAP_AND_PROCEDURAL_SCHEMAS.md](./WEB_ENGINE_BOOTSTRAP_AND_PROCEDURAL_SCHEMAS.md) |
| **4. Krystal-Lang: Geometric Bytecode & Topological VM** | Programovací jazyk transformujúci kód a fronty na geometrické tvary, syntéza bytekódu (`OP_ALLOC_QUEUE`, `OP_QUEUE_DISPATCH`, `OP_EMIT_SHAPE`), topologická virtuálna mašina s priepustnosťou cez 1.7M pps. | [KRYSTAL_LANG_GEOMETRIC_BYTECODE_AND_LLM_COMPILATION.md](./KRYSTAL_LANG_GEOMETRIC_BYTECODE_AND_LLM_COMPILATION.md) |
| **5. Janet Lisp Architecture & Transformation** | Transformácia princípov codebase do jazyka Janet, vláknové generátory (fibers), PEG prompt parser a funkcionálny CSG strom. | [JANET_TRANSFORMATION_AND_LISP_ARCHITECTURE.md](./JANET_TRANSFORMATION_AND_LISP_ARCHITECTURE.md) |
| **6. Open-World Procedural Rendering & Natural Compiler** | Nekonečný procedurálny terén $\mathcal{H}(x,z)$, multi-fractal fBm, termálna/hydraulická erózia, Whittaker biome phase space, prekladač ľudskej reči (SK/EN) do matematických AST a Godot 4.x shaderov. | [ADVANCED_OPENWORLD_PROCEDURAL_RENDERING.md](./ADVANCED_OPENWORLD_PROCEDURAL_RENDERING.md) |
| **7. Blender Core Modifier & Mimicry Compositor** | Digitálna mimikra reálnych objektov, procedurálny modifikačný stack (Bevel, Subsurf, Solidify, Lattice, Array, Displace) a skladanie herných kompozícií. | [BLENDER_CORE_MODIFIER_MIMICRY_ENGINE.md](./BLENDER_CORE_MODIFIER_MIMICRY_ENGINE.md) |
| **8. Rekurzívne AR Zrkadlenie & Antigravity Manifolds** | Rekurzívne zrkadlenie v projektívnej geometrii, Kleinove grupy, IFS fraktály a Antigravity prompt engine s databázou šablón. | [RECURSIVE_AR_MIRROR_AND_ANTIGRAVITY_MANIFOLDS.md](./RECURSIVE_AR_MIRROR_AND_ANTIGRAVITY_MANIFOLDS.md) |
| **9. Architektonický Blueprint Heterogénneho Triadu** | Heterogénny triad (CPU + GPU + NPU), Deferred ASCII Shading (G-Buffer), Screen-Space Ambient Occlusion (SSAO-A), Signed Distance Fields (SDF) glyph matching. | [DEEP_RESEARCH_NEURAL_ASCII_NPU_ENGINE.md](./DEEP_RESEARCH_NEURAL_ASCII_NPU_ENGINE.md) |
| **10. Tréning & Fine-Tuning Pipeline** | Architektúra `NanoCompositor-V3` (<1.8M parametrov), multi-task loss, FiLM telemetrická modulácia a INT8 NPU kvantizácia. | [TRAINING_AND_FINE_TUNING_PIPELINE.md](./TRAINING_AND_FINE_TUNING_PIPELINE.md) |
| **11. Herné techniky & Procedurálny Engine** | Vulkan GLSL Compute Shader (`ascii_rasterizer.comp`), procedurálny 3D raymarching v ASCII, Win32 Double-Buffered VT-100 driver (120+ FPS bez flickerovania). | [GAME_ENGINE_TECHNIQUES_AND_PROCEDURAL_RENDERING.md](./GAME_ENGINE_TECHNIQUES_AND_PROCEDURAL_RENDERING.md) |

---

## 🚀 Okamžite spustiteľný systém a Mission Control

V systéme beží v reálnom čase Localhost Mission Control Hub:
- **Port:** `http://localhost:8080/`
- **Spustenie:**
  ```powershell
  python start_localhost.py
  ```
- **Spustenie testov výkonnosti a benchmarku:**
  ```powershell
  python tests/benchmark_core_performance.py
  ```
- **Spustenie kompletnej sady testov (26 testov):**
  ```powershell
  python -m unittest discover tests -v
  ```

---
*Vytvorené a verifikované v rámci Krystal-Stack Platform Framework (2026).*

**Measured status & priorities:** see [PROJECT_AUDIT_AND_COMPUTE_PATTERNS.md](../PROJECT_AUDIT_AND_COMPUTE_PATTERNS.md) and the live API in [PROJECT_INTELLIGENCE_API.md](../PROJECT_INTELLIGENCE_API.md). Where a research document quotes a figure that is not backed by a measurement, the audit takes precedence.
