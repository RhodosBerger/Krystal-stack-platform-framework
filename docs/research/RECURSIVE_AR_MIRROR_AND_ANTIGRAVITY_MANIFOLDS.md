# Krystal-Stack // Pokročilé Zobrazovacie Techniky: Rekurzívne AR Zrkadlenie & Antigravity Prompt Engine

> **Dátum:** 1. Október 2026  
> **Autor:** Architektonický tím Krystal-Stack & Dušan Kopecký  
> **Klasifikácia:** Pokročilá vizualizačná špecifikácia, Vulkan API Integrácia & Kognitívna manipulácia templátov  

---

## 1. Exekutívne Zhrnutie

Tento dokument definuje implementáciu a matematické princípy **pokročilého testovacieho prostredia pre augmentovanú realitu (AR)** v rámci frameworku Krystal-Stack. Systém prepája:
1. **Rekurzívne geometrické zrkadlenie** založené na dihedrálnych grupách symetrií ($D_N$, Kaleidoskopické IFS skladanie, ne-euklidovské Calabi-Yau a Gyroidné minimálne povrchy).
2. **Vulkan API & Godot 4.x Pipeline**, kde modifikácie z kognitívneho rozhrania v reálnom čase rekurzívne generujú shaderové uniformy (Push Constants).
3. **Katalogizovanú databázu inštancií**: 7 geometrických mnohotvárností a 5 umeleckých AR presetov (35 unikátnych hybridných kombinácií).
4. **Antigravity Prompt Engine**: Prirodzený jazykový vstup (SK/EN), ktorý extrahuje geometrické princípy, počet zrkadlových rovín, hĺbku rekurzie a generuje perzistentné šablóny (Templates).

---

## 2. Princípy Geometrického Zrkadlenia & Dihedrálne Grupy $D_N$

V tradičnom renderovaní sa scény počítajú priamočiaro cez rasterizáciu trojuholníkov alebo raymarching. Krystal-Stack využíva **dihedrálne zrkadlenie invariantných rovín**, čo dramaticky znižuje výpočtovú náročnosť pri zachovaní vizuálnej komplexity.

```
                    ┌─────────────────────────┐
                    │  Vstupný lúč / Pixel UV │
                    └────────────┬────────────┘
                                 │
                                 ▼ Polar Transform: (r, θ)
                    ┌─────────────────────────┐
                    │  Dihedral Fold Sector   │
                    │   α = π / N (Folds)     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                     θ_mod = |(θ mod 2α) - α|
                                 │
                   ┌─────────────┴─────────────┐
                   ▼                           ▼
        Cartesian Fold:             Rekurzívny Odraz:
        fx = r · cos(θ_mod)         p' = p · φ^(1/N)
        fy = r · sin(θ_mod)         Fresnel: R(θ) útlm
```

### Matematická formulácia dihedrálneho skladania

Pre daný počet zrkadlových rovín $N \in [2, 16]$ definujeme uhlový sektor $\alpha$:
$$\alpha = \frac{\pi}{N}$$

Pre každý bod $(x, y)$ v normalizovanom priestore obrazovky prevedieme súradnice do polárnej sústavy:
$$r = \sqrt{x^2 + y^2}, \quad \theta = \operatorname{atan2}(y, x) + \omega t$$

Preloženie uhla cez zrkadlové osi dihedrálnej grupy $D_N$:
$$\theta_{\text{mod}} = \left| (\theta \pmod{2\alpha}) - \alpha \right|$$

Rekonštruovaný bod po zrkadlení:
$$x' = r \cos(\theta_{\text{mod}}), \quad y' = r \sin(\theta_{\text{mod}})$$

### Rekurzívna Fresnelova aproximácia (Schlick)

Pri $K$-násobnom rekurzívnom odraze svetelného lúča v zrkadlovom labyrinte dochádza k absorpcii energie podľa Fresnelovho koeficientu:
$$R(\theta) = R_0 + (1 - R_0)(1 - \cos\theta)^5$$

Kde $R_0$ je materiálová reflexia pri kolmom dopade (uniform `u_fresnel_factor`). Váha odrazu klesá exponenciálne:
$$W_k = W_0 \cdot \gamma^k \cdot R(\theta_k), \quad \gamma \approx 0.55$$

---

## 3. Katalóg Inštancií: Geometrické & Umelecké AR Entity

V priečinku `instances/instance_database.json` je formalizovaná databáza objektov a vizuálnych štýlov:

### 3.1 Geometrické Inštancie

| ID Inštancie | Názov | Kategória | Symetria | Matematická Formula / Vlastnosť |
| :--- | :--- | :--- | :--- | :--- |
| `PLATONIC_OCTAHEDRON` | Octahedral Dual Crystal | Platónske teleso | $O_h$ (48-násobná) | $\|x\| + \|y\| + \|z\| - R = 0$ |
| `PLATONIC_DODECAHEDRON` | Pentagonal Dodecahedron | Platónske teleso | $I_h$ (120-násobná) | Zlatý pomer $\phi = 1.61803$, 12 päťuholníkových faziet |
| `TORUS_KNOT_P3_Q5` | Trefoil Torus Knot T(3,5) | Topologická varieta | $C_{3v}$ | Parametrický uzol navinutý na toruse $3\times$ okolo osi, $5\times$ okolo jadra |
| `HYPERBOLIC_GYROID` | TPMS Minimal Gyroid | Hyperbolický povrch | $Ia\bar{3}d$ | $\sin x\cos y + \sin y\cos z + \sin z\cos x = 0$ |
| `MENGER_SPONGE_RECURSIVE` | Recursive Menger Fractal | Fraktálna varieta | $O_h$ (Sebapodobná) | Nekonečný prienik krížových dutín v mierke $3^{-N}$ |
| `KALEIDOSCOPIC_IFS` | Dihedral Kaleidoscopic IFS | Posvätná geometria | $D_6$ (Hexagonálna) | $p = \|p\|$, rotácia o $\frac{\pi}{N}$, rekurzívny sklad cez normály |
| `CALABI_YAU_CROSS_SECTION` | Calabi-Yau 6D Manifold | Superstrunová topológia | $SU(3)$ Holonómia | 3D stereografický priemet komplexnej variety $z_1^n + z_2^n = 1$ |

### 3.2 Umelecké Presety pre AR Zobrazenie

1. **`CYBERPUNK_NEON_AR`**: Žiarivý tyrkys (`#00F0FF`) + neónová magenta (`#FF0055`), husté blokové znaky ` ░▒▓█`, vysoký kontrast pre head-up displeje.
2. **`MONASTIC_ALCHEMICAL`**: Alchymistické zlato (`#FFC850`) + jantárový plameň (`#FF641E`), posvätné kruhy, znaky ` .✦✧✶✹✺᚛᚜⎔`.
3. **`HOLOGRAPHIC_QUANTUM`**: Kvantové laserové interferenčné línie, $632.8\text{ nm}$ He-Ne laserová dĺžka, stereoskopický anaglyf rozptyl.
4. **`BLUEPRINT_SCHEMATIC`**: Technický ISO výkres, CAD mriežka, smerové Sobelové hrany `|`, `/`, `-`, `\`.
5. **`BIOMECHANICAL_GIGER`**: Organická chitinová štruktúra, bio-rebrové prstence, tmavozelená fosforeskujúca paleta.

---

## 4. Antigravity Prompt Engine: Kognitívna Manipulácia & Templáty

Modul `antigravity_prompt_engine.py` funguje ako most medzi abstraktným zámerom operátora a striktnou matematikou GPU/Vulkan potrubia.

### Architektúra spracovania promptu

```
[Používateľský Prompt: SK / EN]
  │
  ├─► 1. Geometrická klasifikácia (Octahedron, Gyroid, Calabi-Yau, Torus Knot, Menger Sponge...)
  ├─► 2. Detekcia umeleckého štýlu (Alchemical, Cyberpunk, Hologram, Blueprint, Biomechanical...)
  ├─► 3. Extrakcia rádu symetrie (D_N: "6-fold", "8-uholník", "D12" -> mirror_folds: N)
  ├─► 4. Rekurzívna hĺbka ("depth 32", "hĺbka 24" -> recursion_limit)
  ├─► 5. Reflexné a chromatické modifikátory (Fresnel, Aberration)
  │
  ▼
[InstanceManager.compose_template()]
  │
  ├─► Generovanie JSON Šablóny (uložené v templates/TPL_*.json)
  ├─► Generovanie Vulkan GLSL Push Constants bloku
  └─► Live stream injekcia do Localhost Hubu (http://localhost:8080)
```

### Príklad generovaného Vulkan Push Constant bloku:

```glsl
// Auto-generated Vulkan GLSL Uniform Block by Antigravity Prompt Engine
// Template: Dihedral Kaleidoscopic IFS Fold × Sacred Alchemical Geometric Circles (D6)
layout(push_constant) uniform AntigravityARBlock {
    int   u_mirror_folds;          // = 6
    float u_fold_angle;            // = 0.523599 rad (30.0 deg)
    int   u_recursion_limit;       // = 32
    float u_fresnel_factor;        // = 0.880
    float u_chromatic_aberration;  // = 0.0030
    float u_scanline_density;      // = 90.0
    vec4  u_primary_color;         // = vec4(1.000, 0.784, 0.314, 1.0)
    vec4  u_accent_color;          // = vec4(1.000, 0.392, 0.118, 1.0)
} pushConstants;
```

---

## 5. Integrácia s Godot Engine 4.x

V priečinku `godot_project/` sú pripravené produkčné assety priamo prepojiteľné s Vulkan renderovacím jadrom:
- **`shaders/recursive_ar_mirror.gdshader`**: Implementuje rekurzívne dihedrálne skladanie priamo v screen-space fragmentovom shaderi s nulovým dopadom na geometrický pipeline.
- **`scenes/RecursiveMirrorARStage.tscn`**: 3D testovacia scéna so zlatým toroidom a dynamickou zrkadlovou vrstvou.
- **`scripts/KrystalHoloBridge.gd`**: Real-time HTTP mostík vysielajúci snímkovú frekvenciu, počet draw callov a geometrické objekty na server `http://127.0.0.1:8080/api/control`.

---

## 6. Verifikácia a Výsledky Testov

Testovací skript `tests/verify_advanced_ar_mirror.py` potvrdil 100% priechodnosť celého reťazca:
- **Test 1:** Načítanie databázy inštancií (7 geometrických, 5 umeleckých) – **PASSED**.
- **Test 2:** Matematická kompozícia dihedrálnych zrkadiel $D_8$ a generovanie 12 riadkového ASCII náhľadu – **PASSED**.
- **Test 3:** Antigravity Prompt Engine (spracovanie slovenských a anglických promptov, extrakcia kľúčových slov, GLSL blok) – **PASSED**.
- **Test 4:** Syntaktická integrita Godot 4.x Vulkan shadera – **PASSED**.
- **Test 5:** Live HTTP koncové body na `localhost:8080` (`/api/instances`, `/api/antigravity/prompt`, `/api/templates/compose`, `/api/active-template`, `/api/control`) – **PASSED**.

---

## 7. Záver

Implementované rozšírenie premieňa Krystal-Stack na plnohodnotný **kognitívny syntetizátor augmentovanej reality**. Používateľ môže pomocou prirodzeného jazyka okamžite meniť topologické vlastnosti scény, zrkadliť geometriu v dihedrálnych poliach $D_3 - D_{16}$ a cez Vulkan API sledovať adaptáciu vizuálnej entropie a ekonomického guvernéra v reálnom čase na `http://localhost:8080/`.
