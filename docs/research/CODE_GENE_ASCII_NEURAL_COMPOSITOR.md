# Krystal-Stack: Code GENE Neural Compositor & Dual 3D Viewport Engine
**Fotorealistický ASCII Kompozitor, Miešanie Rastrov a Bounded 3D Priestor (Project GENE Imitation)**  
*Dokument: KS-RESEARCH-GENE-01 | Schválená architektúra | Dátum: 2026-10-08*  
*Autor: Dušan Kopecký & Krystal-Stack Research Council*

---

## 🌌 1. Strategický Kontext a Architektonická Motivácia

V predchádzajúcich iteráciách bol ASCII rendering chápaný predovšetkým ako telemetrický displej vizuálnej entropie a hrubá projekcia jednoduchých geometrických telies (guľa, torus, gyroid). 

Ako definovala nová strategická direktíva pre rozvoj codebase:
> *"Zobrazenie v ASCII kompozitore je v poriadku, ale my potrebujeme navrhnúť ASCII kompozitor tak, aby vyprodukoval fotorealistickú scénu za pomoci procedurálnych trikov, miešania rastrov a zjednotiť ASCII zobrazenie do pasáže, ktorú voláme 'Code GENE' imitácia, ktorá vlastne slúži na to, že zobrazuje ten 3D priestor a tú mriežku a má ohraničený priestor, ktorý môže vyrenderovať. My potrebujeme zaviesť ASCII neurál kompozitor tak, aby bol schopný produkovať grafiku ako Project GENE a zobrazovať to aj v ASCII aj v normálnom mesh zobrazení."*

Systém **Code GENE** (integrujúci princípy *Project Genie* z dielne Google DeepMind) rozširuje Krystal-Stack o novú paradigmu: **Action-Conditional Procedural World Synthesis** v striktne vymedzenom 3D priestore s duálnym zobrazovacím rozhraním.

---

## 📐 2. Bounded 3D Space (Ohraničený Priestor a Mriežka)

Namiesto nekonečného prázdneho priestoru zavádza Code GENE striktný fyzikálny a renderovací objem:

$$\mathcal{V}_{bounded} = [X_{min}, X_{max}] \times [Y_{min}, Y_{max}] \times [Z_{min}, Z_{max}]$$

- Predvolené dimenzie: $[-3.5, +3.5] \times [-2.2, +2.2] \times [-3.5, +3.5] \text{ m}$ (objem $\approx 215.6 \text{ m}^3$).
- **Bounding Box Wireframe Cage (12 hrán)**:
  - 4 podlahové hrany ($y = Y_{min}$),
  - 4 stropné hrany ($y = Y_{max}$),
  - 4 vertikálne pilierové hrany prepájajúce 8 rohov.
- **Taktická podlahová mriežka (Floor Coordinate Grid)**:
  - Pravouhlá koordinátna sieť na rovine $y = Y_{min}$ s krokom $\Delta = 1.0\text{ m}$.
  - V ASCII režime reprezentovaná priesečníkmi `┼` a spojnicami `·`.
  - V 3D Mesh režime renderovaná ako precízny linkový segmentový buffer.

Lúče (raymarching) vystupujúce z tohto vymedzeného priestoru sú striktne orezané, čím operátor presne vidí limity a hranice renderovacieho priestoru.

---

## 🎨 3. Procedurálne Triky a Miešanie Rastrov ("Miešanie Rastrov")

Pre dosiahnutie fotorealistického dojmu v obmedzenom ASCII médiu je implementovaný viacvrstvový rasterizér:

### A. Usporiadaný Bayer Dithering (Ordered Dithering 4×4 a 8×8)
Kvantizácia spojitého jasu $L \in [0.0, 1.0]$ do diskrétnych znakov bežne vytvára ostré prúžky (banding). Bayerova matica vnáša deterministický vysokofrekvenčný šum na základe obrazovkových súradníc $(x \pmod 4, y \pmod 4)$:

$$\mathbf{M}_{4 \times 4} = \frac{1}{16} \begin{bmatrix}
0 & 8 & 2 & 10 \\
12 & 4 & 14 & 6 \\
3 & 11 & 1 & 9 \\
15 & 7 & 13 & 5
\end{bmatrix}$$

$$L_{dithered} = \text{clamp}(L + (M_{x, y} - 0.5) \cdot k_{dither}, 0.0, 1.0)$$

Tým dochádza k jemnému optickému miešaniu znakov (dithering), ktoré ľudské oko vníma ako hladký fotorealistický prechod svetla a tieňa.

### B. Hierarchické sady znakov (Multi-Layer Glyph Sets)
1. **Gradientové bloky hustoty**: `' '`, `'░'`, `'▒'`, `'▓'`, `'█'`
2. **Sub-pixelové kvadrantové bloky**: `[' ', '▖', '▗', '▘', '▙', '▚', '▛', '▜', '▝', '▞', '▟', '█']`
3. **Smerové tangenciálne hrany (Sobel-like)**:
   Na tečných hranách a siluetách lúč vyhodnotí uhol normály v obrazovom priestore $\theta = \arctan2(N_y, N_x)$ a vyberie vhodný znak sklonu (`|`, `/`, `-`, `\`).
4. **Zrkadlové záblesky (Specular Glints)**:
   Tam, kde Blinn-Phongov komponent presiahne prah ($S > 0.78$), sú procedurálne osádzané svetelné záblesky (`✦`, `✧`, `✶`, `*`).
5. **Objemová hmla a Ambient Occlusion**:
   $$L_{final} = (D_{key} + D_{fill} + S_{spec} + R_{rim}) \cdot AO \cdot e^{-k_{fog} \cdot dist}$$

---

## 🧠 4. Project GENE Latent Dynamics Model

Po vzore **Project Genie** (DeepMind) engine nestavia na statickej geometrii, ale na **akčno-podmienenom generovaní stavov (Action-Conditional Generation)**:

```
[Operátor / Agent]
        │
        ▼ (Latent Action: 0..5)
┌──────────────────────────────────────────────┐
│       CodeGeneDynamicsModel                  │
│  • 0: IDLE (Objemová oscilácia, dych)        │
│  • 1: ORBIT_CAM (Rotácia zorného poľa +30°)  │
│  • 2: MORPH_TOPOLOGY (Kryštál ↔ Gyroid ↔     │
│       Citadela ↔ Torus)                      │
│  • 3: PULSE_DOPAMINE (Šoková energetická vlna)│
│  • 4: CRYSTAL_GROWTH (Hexagonálny rast)      │
│  • 5: RESCALE_BOUNDS (Expanzia/kompresia)    │
└──────────────────────┬───────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
┌──────────────────────┐    ┌──────────────────────┐
│  ASCII Raymarching   │    │  3D Mesh Synthesizer │
│  + Bayer Dithering   │    │  (Polygóny, Hrany)   │
└──────────────────────┘    └──────────────────────┘
```

---

## 🖥️ 5. Duálne Rozhranie: ASCII + Normálny 3D Mesh

Engine generuje z jedného matematického zdroja pravdy dva súbežné formáty:

1. **Režim ASCII (Fotorealistický procedurálny terminál)**:
   - Formát: textový buffer s voliteľnou 24-bit TrueColor ANSI syntaxou.
   - Poskytuje okamžitý prehľad o entropii scény, koherencii a hustote lúčov.
2. **Režim 3D Mesh (WebGL Three.js & Wavefront .OBJ)**:
   - Formát: zoznam vrcholov $[x, y, z]$, normál $[nx, ny, nz]$, UV mapovania $[u, v]$ a trojuholníkových plôch $[v_1, v_2, v_3]$.
   - Obsahuje kompletnú 12-hrannú drôtenú klietku (`cage_lines`) a podlahovú mriežku (`floor_grid_lines`).
   - Plná podpora rotácie myšou, zoomu, wireframe režimu a exportu do `.obj` pre import do herného enginu **Godot 4.x** alebo Blenderu.

---

## 🚀 6. Sprístupnené Rozhrania a Koncové Body

| Endpoint | Typ | Účel |
|---|---|---|
| `/code-gene` | Web UI | Interaktívne štúdio s duálnym split-screen zobrazením |
| `/api/code_gene/render` | GET | Poskytuje aktuálny ASCII frame a telemetriu ($E_{spatial}$, $Coherence$) |
| `/api/code_gene/mesh` | GET | Poskytuje 3D polygonálny mesh, wireframe klietku a súradnice v JSON |
| `/api/code_gene/state` | GET | Poskytuje stav dynamického modelu, rozmery klietky a nastavenia rastra |
| `/api/code_gene/action` | POST | Vykoná diskrétnu akciu Project GENE (0..5) a mutuje stav |
| `/api/code_gene/config` | POST | Aktualizuje Bayer dither maticu, sub-pixely a rozmery klietky |
| `/api/code_gene/export_obj` | POST | Vyexportuje polygonálny 3D model ako `.obj` do priečinka `godot_assets/` |
| `/api/stream` (Režim `CODE_GENE_IMITATION`) | SSE | Živý 30–60 FPS dátový prúd do Mission Control terminálu |

---

## 🏁 7. Záver

Implementáciou **Code GENE Neural Compositora** sme posunuli ASCII rendering z analytického nástroja na úroveň plnohodnotného generatívneho 3D syntetizátora. Spojenie **procedurálneho miešania rastrov**, **ohraničenej 3D mriežky** a **duálneho zobrazenia (ASCII + Mesh)** tvorí kľúčový základ pre budúcu AR a neurálnu generáciu herných svetov.
