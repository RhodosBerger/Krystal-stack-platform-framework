# Krystal-Stack: Execution Architecture, Metrická Stratifikácia & AI Style Contract v1.0

**Dátum:** 3. Október 2026  
**Autori:** Dušan Kopecký & Krystal-Stack Core Engineering Group  
**Invariant Platformy:** `VITAL_MAX_HP = 6`  
**Zlatý Pomer:** $\phi = 1.61803398875$  
**Dokumentačný Kód:** `KRYSTAL-EXEC-UI-CONTRACT-2026`

---

## 1. Celkový Tok Výkonnostnej Architektúry (Execution Flow)

Architektúra Krystal-Stacku prepája spojitú procedurálnu matematiku, signed distance functions (SDF), fraktálny Brownov pohyb (fBm), topologický governor a heterogénne výpočtové backendy do uzavretého spätnoväzbového cyklu:

```mermaid
flowchart TD
    subgraph MathLayer["1. Spojitá Matematika Sveta (Continuous Math)"]
        M1["Analytické Manifoldy & Frakčné Derivácie"]
        M2["Voronoi Tenzory & Vztlakové Polia"]
        M3["Hamiltonián H = T + V"]
    end

    subgraph KernelLayer["2. SDF & fBm Kernely"]
        K1["6 Oktáv fBm Šum (Hash operácie)"]
        K2["Analytické Vzdialenostné Polia SDF(x)"]
        K3["Teplotná Erózia & Zosuvy (Talus θ_c = 35°)"]
    end

    subgraph GovernorLayer["3. Topologický Execution Governor"]
        G1["FastRingBuffer (1.724M pps, 4.39× vs Mutex)"]
        G2["Prioritné Fronty & Dynamic Throttle"]
        G3["Alokácia Rozpočtu Kvant (Q*, Barrier ξ)"]
    end

    subgraph Backends["4. Heterogénny Backend Dispatch"]
        B1["Python / C Referenčný Raymarcher (37–46 ms)"]
        B2["AVX2 / SIMD Vektorizácia (Modeled 6.8×)"]
        B3["Vulkan / WebGPU Compute (Target 120 FPS)"]
        B4["NPU DirectML Tenzorový Manifold (Target <0.28 ns)"]
    end

    subgraph Framebuffer["5. ASCII & TrueColor Framebuffer"]
        F1["Multi-Plane Kompozitor"]
        F2["Z-Buffer & Cel-Shading Obrysy"]
        F3["WebOS Desktop & 3D Viewport"]
    end

    subgraph Telemetry["6. Telemetria Vizuálnej Entropie (1.07 ms)"]
        T1["Shannonova Entropia S(f)"]
        T2["Hamiltoniánska Stabilita & Drift"]
        T3["LOD & Stride Dynamické Riadenie"]
    end

    MathLayer --> KernelLayer
    KernelLayer --> GovernorLayer
    GovernorLayer --> Backends
    Backends --> Framebuffer
    Framebuffer --> Telemetry
    Telemetry -- "Spätná Adaptívna Optimalizácia (Feedback Loop)" --> GovernorLayer
```

---

## 2. Stratifikácia Metrík: MEASURED vs MODELED vs TARGET

Pre zachovanie absolútnej inžinierskej integrity a zamedzenie miešania overených výsledkov s teoretickými plánmi zavádza Krystal-Stack tri striktné kategórie metrík:

```mermaid
graph TD
    classDef measured fill:#052e16,stroke:#22c55e,stroke-width:2px,color:#f0fdf4;
    classDef modeled fill:#451a03,stroke:#f59e0b,stroke-width:2px,color:#fef3c7;
    classDef target fill:#2e1065,stroke:#a855f7,stroke-width:2px,color:#faf5ff;

    subgraph Tier1["TIER 1: [MEASURED] Empiricky Namerané na Aktívnom Kóde"]
        M1["VM FastRingBuffer Throughput: 1,724,554 pps<br>(4.39× zrýchlenie oproti Mutexu 392,534 pps)"]:::measured
        M2["CPU SDF Raymarch Eval: 997,000 eval/s<br>(Python/C Emulácia: 37–46 ms/frame)"]:::measured
        M3["Terrain Kernel Sample Latency: 89.36 μs/sample<br>(11,190 vzoriek/s, 6 oktáv fBm / 28 krokov)"]:::measured
        M4["Visual Entropy Telemetry: 1.07 ms/frame<br>(Reálny beh monitora Shannonovej entropie)"]:::measured
    end

    subgraph Tier2["TIER 2: [MODELED] Architektonicky Simulované & Extrapolované"]
        MD1["AVX2 SIMD Terrain Vectorization: 13.14 μs/sample<br>(6.80× zrýchlenie cez 8-wide float registre)"]:::modeled
        MD2["Iris Xe Driver Bypass Call: 11.4 μs/call<br>(16.23× redukcia latencie voči WDDM stallu 185 μs)"]:::modeled
    end

    subgraph Tier3["TIER 3: [TARGET] Roadmap Ciele Budúcich Backendov"]
        TG1["NPU Neural SDF Manifold: < 0.28 ns/query<br>(3.5 × 10⁹ eval/s cez DirectML tenzorové jadrá)"]:::target
        TG2["WebGPU Compute Shader: 120 FPS<br>(8.33 ms rozpočet snímky v prehliadači)"]:::target
        TG3["Rust Native Compiled Raymarcher: 45× zrýchlenie<br>(Zostavenie s nulovou réžiou medzivrstvy)"]:::target
    end
```

### Prehľadová Tabuľka Výkonnostných Stavov

| Metrika / Komponent | Stav | Nameraná / Cieľová Hodnota | Baseline Hodnota | Zrýchlenie | Technický Komentár |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Topological VM Queue** | `[MEASURED]` | **1,724,554 pps** | 392,534 pps | **4.39×** | FastRingBuffer eliminuje zámky OS vlákien. |
| **Terrain Sample Latency** | `[MEASURED]` | **89.36 μs / vzorka** | 89.36 μs | 1.00× | 6 oktáv fBm pri 28 krokoch generuje vysokú réžiu hashovania. |
| **CPU SDF Raymarcher** | `[MEASURED]` | **997,000 eval/s** | 37.2 ms/frame | 1.00× | CPU referencia bez SPIR-V dispatchu. |
| **Telemetry Monitor** | `[MEASURED]` | **1.07 ms / frame** | 4.50 ms | **4.21×** | Telemetria vizuálnej entropie a stability. |
| **AVX2 SIMD Terrain** | `[MODELED]` | **13.14 μs / vzorka** | 89.36 μs | **6.80×** | Odhad paralelizácie hash generátora. |
| **Iris Xe Driver Bypass** | `[MODELED]` | **11.4 μs / volanie** | 185.0 μs | **16.23×** | Zníženie latencie odosielania paketov. |
| **NPU Neural SDF** | `[TARGET]` | **< 0.28 ns / dopyt** | 1003.0 ns | **3580×** | Cieľ tenzorového surogátu (DirectML). |
| **WebGPU Compute Pipeline**| `[TARGET]` | **120 FPS** | 24 FPS | **5.00×** | Natívny prehliadačový compute shader. |
| **Rust Native Raymarcher** | `[TARGET]` | **45× zrýchlenie** | 1.00× | **45.0×** | Kompilované LLVM jadro bez Python runtime. |

---

## 3. Krystal UI System — AI Style Contract v1.0

### 3.1 Dizajnová Rovnica
$$\text{Visual Identity} = \text{Dark Cinematic Portal} \times \text{Editorial Fantasy Typography} \times \text{Scientific Simulation HUD} \times \text{Restrained Neon Telemetry}$$

Rozhranie nesmie nikdy pripomínať generický SaaS dashboard. Ide o **operačný systém herného sveta** s jasne oddelenými prezentačnými vrstvami.

### 3.2 Dualita a Hybridný Režim (Hybrid Mode Topology)

```mermaid
graph TD
    subgraph WorldLayer["WORLD LAYER (Portál & Naratív)"]
        W1["Pozadie: Deep Navy #101124 / Void #070812"]
        W2["Akcent: Zlato #D8AE4B (Monumentálny naratív)"]
        W3["Typografia: Monumentálny Serif (Cinzel / Cormorant)"]
        W4["Asymetrická kompozícia, veľký negatívny priestor"]
    end

    subgraph ArenaLayer["ARENA LAYER (Výpočtové Jadro)"]
        A1["Pozadie: Pure Black #030406"]
        A2["Akcenty: Cyan #58D8FF / Teal #50EEE0 / Magenta #FF307C"]
        A3["Typografia: Monospace (JetBrains Mono)"]
        A4["Perspektívna mriežka, procedurálne wireframy, raymarch lúče"]
    end

    subgraph HybridMode["HYBRIDNÝ REŽIM (Poslední Kmen)"]
        H1["60–72% Šírky: Dominantná 3D Procedurálna Aréna"]
        H2["28–40% Šírky: Kontextuálny Frakčný & Naratívny Panel"]
        H3["< 15% Plochy: Kompaktný Okrajový HUD s [MEASURED] Značkami"]
    end

    WorldLayer --> HybridMode
    ArenaLayer --> HybridMode
```

### 3.3 Pravidlá Akcentov a Geometrie
1. **Zlato a Cyan si nesmú konkurovať ako rovnocenné farby:**
   - **Zlato (`--ks-gold` `#D8AE4B`)** patrí výhradne naratívnej a svetovej vrstve.
   - **Cyan (`--ks-cyan` `#58D8FF`)** patrí výhradne systémovej a výpočtovej vrstve.
2. **Architektonická pravouhlosť (No SaaS bloat):**
   - `--radius-ui: 2px;`
   - `--radius-control: 4px;`
   - Žiadne mäkké tiene (box-shadows), žiadne pilulkové karty, žiadny glassmorphism.
   - 1px jemné ohraničenia (`--ks-line` `#292A38`).
3. **Základná 4px mriežka:**
   - Vzdialenosti odvodené výhradne zo sady: `4`, `8`, `12`, `16`, `24`, `32`, `48`, `64`, `96`.

### 3.4 Katalóg 14 Kanonických Primitív
- `KSNav`: Globálna navigácia v hlbokom navy tóne so zlatým aktívnym indikátorom.
- `KSWorldHero`: Monumentálny nadpis s asymetrickým negatívnym priestorom.
- `KSEntitySelector`: Prepínač kmeňov (Crystal, Toxic, Druid).
- `KSStatBar`: Segmentovaný ukazovateľ atribútov s pravouhlými obdĺžnikmi.
- `KSRadar`: Kompaktný polygónový radarový graf schopností.
- `KSHudStatus`: Okrajový indikátor stavu spojenia a jadra.
- `KSRuntimeMetric`: Blok metriky s povinnou značkou `[MEASURED]`, `[MODELED]` alebo `[TARGET]`.
- `KSActionButton`: Systémové tlačidlo s 1px farebným rámom a monospace textom.
- `KSWorldButton`: Zlaté primárne svetové tlačidlo s čiernym písmom a vysokým trackingom.
- `KSTerminalPanel`: Panel technických protokolov a príkazového riadku.
- `KSRenderViewport`: 3D scéna s procedurálnym vykresľovaním.
- `KSBiomeIndicator`: Označenie biomu a spojitých matematických parametrov.
- `KSChapterLabel`: Mikro-kapitolový štítok s trackingom 0.28em.
- `KSArtifactCard`: Karta entít s 1px rámom a nulovým tieňovaním.

---

## 4. Platformový Invariant
V celom subsystéme platí nedotknuteľné pravidlo:
$$\mathbf{VITAL\_MAX\_HP} = 6$$
Žiadna entita, kmeň, výpočtové vlákno ani metrika nesmie prekročiť základný limit 6 bodov životnosti alebo stability.
