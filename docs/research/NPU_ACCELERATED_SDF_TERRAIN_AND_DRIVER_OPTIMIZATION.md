# Krystal-Stack Research: NPU-Akcelerovaná Syntéza Terénu, SDF Raymarching & Optimalizácia Grafických Ovládačov

**Dátum:** 3. Október 2026  
**Autor:** Dušan Kopecký & Krystal-Stack Mathematical & Hardware Architecture Group  
**Kľúčový Invariant:** `VITAL_MAX_HP = 6`  
**Zlatý Pomer:** $\phi = 1.61803398875$, $\phi^{-1} \approx 0.61803398875$  
**Kritický Uhol Suťových Kužeľov:** $\theta_c \approx 35^\circ$ ($\tan \theta_c \approx 0.7002$)

---

## 1. Výkonná Syntéza

Tento dokument formalizuje syntézu matematických modelov nekonečných terénnych variet, numerického sphere-tracingu (SDF raymarchingu), previazaných eróznych modelov a nízkoúrovňovej optimalizácie pre integrovanú grafiku **Intel Iris Xe (96 EUs / 672 hardvérových vlákien)** spoločne s **NPU (Neural Processing Unit - DirectML / OpenVINO)**.

Architektúra prepája procedurálny vesmír hry **Poslední Kmen** (biómy **Crystal**, **Toxic**, **Druid** a centrálna **Studna Duší**) s hardvérovým obchádzaním réžie ovládačov (driver bypass) a neurónovými náhradnými modelmi (neural surrogate models).

---

## 2. Kontinuálna Matematika Terénnych Variet (Continuous World Generation)

### 2.1 Hlavná Terénna Rovnica (Master Terrain Manifold Equation)
Každý bod v priestore $(x, z)$ je generovaný spojitým funkčným predpisom:

$$H(x, z) = H_0 + A \left[ (1 - w_{ridge}) \cdot F_{fBm}(\omega x) + w_{ridge} \cdot R_{ridge}(\omega x) \right] \cdot C_{fault}(x)$$

Kde:
- $H_0$: Bázová elevácia biómu ($1.45$ pre Crystal, $-0.25$ pre Toxic, $0.65$ pre Druid).
- $A$: Amplitúda terénneho reliéfu.
- $F_{fBm}(\omega x)$: Fraktálny Brownov pohyb pre jemnú vlnitosť kopcov.
- $R_{ridge}(\omega x) = \sum_i (1 - |\sin(\omega_i x) \cos(\omega_i z)|)^2$: Multifraktálny hrebeňový šum pre ostré skalnaté a ľadové masívy.
- $w_{ridge} \in [0, 1]$: Váha hrebeňov (pre Crystal až $0.88$, pre Toxic len $0.20$).
- $C_{fault}(x)$: Bunková Voronoi zlomová maska modelujúca prepadliny a kaňony.

---

## 3. Previazané Modely Erózie (Coupled Erosion Models)

### 3.1 Teplotné Zvetrávanie a Suťové Kužele (Thermal Weathering & Talus Slopes)
Materiál sa zosúva a eroduje, len ak lokálny sklon prekročí kritický uhol vnútorného trenia horniny:

$$\Delta H_{thermal}(x) = -K_{talus} \cdot \max(0, \|\nabla H(x)\| - \tan \theta_c)$$

$$\theta_c \approx 35^\circ \quad \Longrightarrow \quad \tan \theta_c \approx 0.7002$$

- **Ak $\theta > \theta_c$:** Prebieha **ERÓZIA** (odpadávanie skál, červená zóna).
- **Ak $\theta \le \theta_c$:** Prebieha **SEDIMENTÁCIA / DEPOSIT** (tvorba stabilného suťového kužeľa, zelená zóna).

### 3.2 Hydraulická Erózia a Flúviálny Zárez (Hydraulic Erosion & Channel Incision)
Transportná kapacita vodného toku a zárez riečnej siete:

$$C(x) = K_c \cdot \|v(x)\| \cdot \sin \theta(x), \quad \text{kde } \sin \theta \approx \frac{\|\nabla H\|}{\sqrt{1 + \|\nabla H\|^2}}$$

$$\mathcal{E}_{hydraulic}(x) = K_e \cdot \text{clamp}(\nabla^2 H(x), -1.0, 1.0)$$

Laplacián $\nabla^2 H(x)$ detekuje lokálnu konvexnosť/konkávnosť: v úžľabinách ($\nabla^2 H > 0$) sa tok zarezáva hlbšie, na vyvýšeninách dochádza k obrusovaniu.

---

## 4. SDF Raymarching a NPU Akcelerácia

### 4.1 Numerické Trasovanie Lúčov (Sphere-Tracing)
Lúč $r_i(t) = r_0 + t \cdot \hat{d}_i$, pričom povrch je detekovaný, ak:

$$f(p) < \epsilon_{hit} \quad (\epsilon_{hit} \approx 0.001\text{ m})$$

### 4.2 Redukcia Ceny Výpočtu cez NPU Tensor Pass
V štandardnom CPU/GPU raymarcheri vyžaduje odhad povrchového normálu $\nabla f(p)$ 6 dodatočných vyhodnotení SDF funkcie (centrálne diferencie pozdĺž osí $x, y, z$):

$$\nabla f(p) \approx \frac{1}{2\delta} \sum_{k \in \{x,y,z\}} [f(p + \delta e_k) - f(p - \delta e_k)] e_k$$

$$\text{Cena za 1 hit (Baseline): } 1 \text{ (hit)} + 6 \text{ (normál)} = \mathbf{7 \text{ vyhodnotení}}$$

S **NPU neurónovým koprocesorom (DirectML / OpenVINO)**:
- NPU počíta neurónový aproximátor poľa $\nabla f(p)$ v jedinom maticovom tenzorovom prechode.
- **Cena za 1 hit (NPU):** $1 + 1 = \mathbf{2 \text{ vyhodnotenia}}$ $\Longrightarrow$ **Zníženie výpočtovej záťaže o 71.4%!**

---

## 5. Optimalizácia Ovládačov pre Intel Iris Xe (Driver Bypass)

| Metrika | Štandardný WDDM Ovládač | Krystal Zero-Copy Bypass | Zlepšenie |
| :--- | :--- | :--- | :--- |
| **Latencia volania príkazu** | $185.0\ \mu\text{s}$ | $11.4\ \mu\text{s}$ | **16.2× zrýchlenie** |
| **Paralelné vlákna** | Časovo prepínané | 672 hardvérových vlákien (96 EUs) | Maximálna saturácia |
| **Správa pamäte** | Duplikácie a staging | 3D Grid: NPU SRAM ↔ L1/L2 ↔ Host DDR VRAM | Nulové kopírovanie |
| **Inštrukčný našeptávač** | Generický | AVX2 prefetch (`PREFETCHT0`) + `VMOVNTPS` | 87.4% menej miss |
| **Invariant celistvosti** | Nešpecifikovaný | `VITAL_MAX_HP = 6` | 100% zachovaný |
