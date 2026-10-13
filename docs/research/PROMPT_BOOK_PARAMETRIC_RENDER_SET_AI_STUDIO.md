# Krystal-Stack: Prompt Book v3 pre AI Studio — Nový Parametrický Render Set

**Autor:** Dušan Kopecký & Krystal-Stack Architecture & Cyber-Physical Systems Council  
**Klasifikácia:** Vývojové a promptovacie špecifikácie pre AI Studio / Antigravity IDE / Gemini Studio  
**Dátum:** Október 2026  
**Status:** RATIFIED SPECIFICATION v3.0 (Parametric Render Set & Typographic Dynamics)  
**Ústavný Invariant:** $\text{VITAL\_MAX\_HP} \equiv 6$ (Univerzálne presadzované na všetkých úrovniach)  
**Hardware Baseline:** Intel 11th Gen Tiger Lake Iris Xe UMA (Vulkan Compute, $\le 256\,\text{MB}$ RAM, $\ge 60\,\text{FPS}$)

---

## 1. Architektonická Vízia: Od Voľného Textu k Rigoróznym Parametrickým Setom

Klasické generatívne systémy pristupujú k tvorbe 3D prostredí formou „vibe-based“ textových reťazcov, čo vedie k nekonzistentným mierkam, prenikaniu geometrie (Z-fighting), prekročeniu videopamäte a absencii determinizmu.

V **Krystal-Stack** a prostredí **AI Studio (Google Antigravity / Gemini)** definujeme **Nový Parametrický Render Set (NPRS v3.0)** nie ako náhodný popis, ale ako **10-blokový typovaný deklaratívny kontrakt**. Každý render set v sebe integruje:
1. **Harmonickú kosínusovú topológiu** s radius derivátom a exaktnou detekciou previsov ($\xi(\theta) = \frac{|dr/d\theta|}{r(\theta)}$).
2. **Dihedrálne zrkadlenie ($D_n$)** so zachovaním Lipschitzovej izometrie ($L = 1.0$).
3. **Kinetickú typografiu** (5 módov textových animácií a 5x7 bitmap font) pre holografický HUD.
4. **Hardvérové mantinely Intel Iris Xe UMA** ($\le 256\,\text{MB}$, $\text{Z-fighting } \epsilon \ge 0.012\,\text{m}$, $\text{FPS } \ge 60$).
5. **Ústavný Invariant** $\text{VITAL\_MAX\_HP} = 6$.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│             KRYSTAL AI STUDIO: NOVÝ PARAMETRICKÝ RENDER SET (NPRS v3.0)                │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  [1. SceneIdentity]        ID scény, doména, naratívna úloha, cieľový počet uzlov     │
│  [2. WorldEnvironment]     Atmosféra, biómová mierka, priestorový hash cell_size_m     │
│  [3. ProceduralStructure]  SDF lúče, Dihedrálne záhyby D_n, Fraktálna dimenzia D_f     │
│  [4. HarmonicTopology]     Kosínusová obálka r(θ), Radius derivát dr/dθ, pomer xi(θ)   │
│  [5. TextureStack]         PBR drsnosť, metalicita, mikro-kavity, hrany alchýmie       │
│  [6. CinematicLighting]    Trojbodový pomer Key:Fill:Rim, volumetrická hmla, chromatika │
│  [7. TypographicDynamics]  5 kinetických módov (Matrix, Sine, Glitch, Decrypt, Voxel)  │
│  [8. CompositionCamera]    Zlatý rez Φ = 1.618, ohnisková vzdialenosť, hĺbka poľa      │
│  [9. HardwareLimits]       UMA strop <= 256 MB, Z-fighting ε >= 0.012m, 60 FPS budget  │
│  [10. InvariantEnforcer]   assert VITAL_MAX_HP == 6 (Striktná validačná brána)         │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Matematické Základy Nového Parametrického Render Setu

### 2.1 Harmonická Kosínusová Topológia a Radius Derivát
Globálna obálka parametrickej scény alebo terénu je v polárnej báze generovaná ortogonálnym rozvojom:
$$r(\theta) = R_0 + \sum_{k=1}^K A_k \cos(k \omega \theta + \phi_k)$$
kde:
* $R_0$ je nosný bázický polomer,
* $A_k$ je amplitúda $k$-tej harmonickej zložky (tlmená Zlatým rezom $\Phi = 1.6180339887$),
* $\omega$ je počet fundamentálnych symetrických lalokov,
* $\phi_k$ je fázový posun v radiánoch.

**Radius Derivát ($\frac{dr}{d\theta}$):**
$$\frac{dr}{d\theta} = -\sum_{k=1}^K A_k (k \omega) \sin(k \omega \theta + \phi_k)$$
*Poznámka:* Derivácia $\frac{dr}{d\theta}$ je fázovo posunutá o $+90^\circ$ ($\frac{\pi}{2}$) voči kosínusu. Na vrcholoch lalokov platí $\frac{dr}{d\theta} = 0$, kým maximálna strmosť nastáva pri prechode cez $R_0$.

**Metrika Strmosti a Detekcia Previsov ($\xi(\theta)$):**
$$\xi(\theta) = \frac{\left|\frac{dr}{d\theta}\right|}{r(\theta)}$$
* $\xi(\theta) \le 1.0$: Hladký terén reprezentovateľný výškovou mapou (2.5D Heightmap).
* $\xi(\theta) > 1.0$: Geometrický previs (cliff overhang) alebo podzemná jaskyňa, vyžadujúca plný 3D SDF raymarching.

### 2.2 Dihedrálne Skladanie ($D_n$) a Lipschitzova Izometria
Pre raymarching uzlov s dihedrálnou symetriou $D_n$ vykonáva shader skladanie bodu $\mathbf{x} \in \mathbb{R}^3$ cez $n$ reflexných rovín:
$$\mathbf{x}_{k+1} = \mathbf{x}_k - 2 \min\big(0, \mathbf{x}_k \cdot \mathbf{n}_k\big) \mathbf{n}_k, \quad \mathbf{n}_k = \left(\cos\frac{k\pi}{n}, \sin\frac{k\pi}{n}, 0\right)$$
Pretože reflexia je euklidovská ortogonálna transformácia, platí:
$$\|\mathbf{x}_{k+1} - \mathbf{y}_{k+1}\| = \|\mathbf{x}_k - \mathbf{y}_k\| \implies L = 1.0 \quad (\text{Lipschitzova konštanta je striktne } 1)$$

### 2.3 Rámcový Rozpočet a Z-Fighting Mantinel
Hostiteľský čip Intel Iris Xe UMA zdieľa systémovú pamäť LPDDR4x. Render sety presadzujú:
1. **Z-Fighting Epsilon:** Minimálna separácia prekrývajúcich sa plôch $\Delta z \ge 0.012\,\text{m}$ (12 mm).
2. **Pamäťový Strop:** Unified Memory $\le 256.0\,\text{MB}$.
3. **Časový Rámcový Budget ($16.6\,\text{ms}$ pre $60\,\text{FPS}$):**
   $$B_{\text{frame}} = B_{\text{sim}} (3.0\,\text{ms}) + B_{\text{raymarch}} (8.5\,\text{ms}) + B_{\text{text\_overlay}} (1.5\,\text{ms}) + B_{\text{hud}} (3.6\,\text{ms}) \le 16.6\,\text{ms}$$

---

## 3. Katalóg 5 Kanonických Parametrických Render Setov

Implementované a verifikované v [`krystal_kernel/parametric_render_set.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/parametric_render_set.py):

### Set 1: Crystalline Dihedral Manifold (`SET_01_CRYSTAL_POLYHEDRA_D6_D12`)
* **Doména:** Posvätná geometria, oktaéder a dvanásťsten, alchýmia, Vulkan raymarching.
* **Geometrické SDF:** `PLATONIC_OCTAHEDRON_GYROID`, dihedrálne záhyby $D_6$, rekurzívna hĺbka 24, fraktálna dimenzia 2.718.
* **Typografia:** `SINE_WAVE_FLOAT` s textom `"KRYSTAL EPISTEMIC CORE"`.
* **Optika:** Key/Fill/Rim = `(1.0, 0.4, 0.8)`, azúrová luminiscencia `#00f0ff`, hmla 0.035.
* **Inštancie:** 1024 cieľ / 4096 max, pamäť 128 MB.

### Set 2: Kinetic Typographic HUD & Decryptor (`SET_02_KINETIC_TYPOGRAPHIC_HUD`)
* **Doména:** Živá telemetria, kryptografický terminál, stav hrdinu.
* **Geometrické SDF:** `EXTRUDED_VOXEL_PANEL`, záhyby $D_4$, rekurzia 8.
* **Typografia:** `TYPEWRITER_DECRYPT` s textom `"AUTHORITATIVE ARBITER [HP: 6/6]"`, zelený fosfor `#10b981`.
* **Optika:** Key/Fill/Rim = `(1.2, 0.2, 0.6)`, ambient `#05070a`.
* **Inštancie:** 512 cieľ / 2048 max, pamäť 64 MB.

### Set 3: Sovereign Hex Arena 19 (`SET_03_SOVEREIGN_HEX_ARENA_19`)
* **Doména:** Taktická 19-hexová kyber-fyzikálna stratégia *Poslední Kmen*, zákopy sapérov, bojové lúče.
* **Geometrické SDF:** `HEXAGONAL_PRISM_PRIMITIVE`, záhyby $D_6$, rekurzia 16.
* **Typografia:** `GLITCH_CYBERPUNK` s textom `"POSLEDNÍ KMEN: TACTICAL ARENA"`.
* **Optika:** Key/Fill/Rim = `(0.9, 0.5, 0.7)`, fialová žiara `#8b5cf6`, Z-epsilon 0.018 m.
* **Inštancie:** 4096 cieľ / 8192 max, pamäť 256 MB.

### Set 4: Volumetric Overhang Biome & Deep Spires (`SET_04_OVERHANG_BIOME_CAVERN_8K`)
* **Doména:** Podzemné kryštálové jaskyne, masívne skalné previsy ($\xi(\theta) > 1.0$).
* **Geometrické SDF:** `APOLLONIAN_MANDELBOX_SDF`, záhyby $D_8$, rekurzia 32, dimenzia 2.85.
* **Typografia:** `VOXEL_3D_EXTRUDE` s textom `"CAVERN SECTOR XI > 1.0"`.
* **Optika:** Key/Fill/Rim = `(0.7, 0.6, 1.1)`, jantárový odtieň `#f59e0b`, hustá hmla 0.065.
* **Inštancie:** 6000 cieľ / 8192 max, pamäť 256 MB.

### Set 5: Cyberpunk Metropolis 8K Grid (`SET_05_CYBERPUNK_NEON_METROPOLIS`)
* **Doména:** Hustá ortogonálna mestská zástavba, mokré reflexné ulice, neónové billboardy.
* **Geometrické SDF:** `ORTHOGONAL_GRID_MEGABLOCK`, záhyby $D_4$, rekurzia 16.
* **Typografia:** `KINETIC_MATRIX` s textom `"NEO-KRYSTAL DOWNTOWN"`, neónová ružová `#f43f5e`.
* **Optika:** Key/Fill/Rim = `(1.1, 0.3, 0.9)`, drsnosť 0.12, emissive boost 2.2x.
* **Inštancie:** 8000 cieľ / 8192 max, pamäť 256 MB.

---

## 4. Master System Prompt pre AI Studio (Gemini / Antigravity)

Vložte tento prompt do systémovej inštrukcie v **Google AI Studio**, **Antigravity IDE** alebo **LM-Studio Cascade Workbench**:

```markdown
Ty si KRYSTAL ARCHITECT & PARAMETRIC RENDER SYNTHESIZER — špecializovaný generatívny engine pre Krystal-Stack a prostredie Google Antigravity.

TVOJA ÚLOHA:
Prekladať prirodzený jazyk (SK/EN) do exaktných, matematicky validných a hardvérovo limitovaných konfigurácií Nového Parametrického Render Setu (NPRS v3.0).

NEPREKROČITEĽNÉ ARCHITEKTONICKÉ PRAVIDLÁ (INVARIANTS):
1. VITAL_MAX_HP == 6: Každá scéna a HUD konfigurácia musí zachovávať hrdinský invariant max HP = 6. Nikdy negeneruj hodnoty presahujúce 6.
2. Z-FIGHTING DEFENSE: Prekrývajúce sa geometrické plochy musia mať odstup min_separation_epsilon_m >= 0.012 m (12 mm).
3. HARDWARE CEILING: Maximálna pamäťová alokácia pre Iris Xe UMA nesmie prekročiť 256.0 MB.
4. LIPSCHITZ ISOMETRY: Všetky dihedrálne skladania D_n (kde n je v rozsahu [2, 32]) musia zachovávať Lipschitzovu konštantu L = 1.0.
5. METRIKA PREVISOV:
   - Pomer xi(theta) = |dr/dtheta| / r(theta).
   - Ak xi <= 1.0, generuj štandardnú 2.5D výškovú mapu.
   - Ak xi > 1.0, explicitne deklaruj 'U_ENABLE_OVERHANGS': 1 a prepni na plný 3D SDF raymarching.
6. KINETICKÁ TYPOGRAFIA: Pre textový overlay vyberaj výhradne z 5 podporovaných módov:
   ['KINETIC_MATRIX', 'SINE_WAVE_FLOAT', 'GLITCH_CYBERPUNK', 'TYPEWRITER_DECRYPT', 'VOXEL_3D_EXTRUDE'].

FORMÁT VÝSTUPU:
Vždy vracaj striktný, strojovo čitateľný JSON podľa schémy NPRS v3.0, nasledovaný stručným matematickým zhrnutím.
```

---

## 5. Šablóny Úlohových Promptov (Task Prompt Blueprints)

### Blueprint A: Generovanie nového parametrického setu zo zadania
*Použite, keď chcete vytvoriť úplne novú vizuálnu scénu z naratívneho zadania.*

```markdown
[POŽIADAVKA PRE AI STUDIO]
Kontext: Vytvor nový parametrický render set pre taktickú zónu "Katakomby Posledného Kmeňa".
Naratívna úloha: Hráč vstupuje do hlbokých kryštálových chodieb s pulzujúcimi fialovými runami a výstražným terminálom.

Parametre na výpočet:
1. Zvoľ primárny SDF tvar s dihedrálnym skladaním D6 alebo D8.
2. Urči harmonické koeficienty pre r(theta) a over, či strmosť xi(theta) prekročí 1.0 (vytvorenie jaskynných previsov).
3. Nastav kinetickú typografiu terminálu (napr. TYPEWRITER_DECRYPT s textom hrdinu pri HP = 6/6).
4. Nastav trojbodové osvetlenie a volumetrickú hmlu.
5. Over pamäťový strop (max 256 MB) a Z-fighting epsilon (>= 0.015 m).

Generuj autoritatívny JSON objekt pre MCP nástroj 'antigravity-compose-template'.
```

### Blueprint B: Integrácia s Hybrid MCP Konnektorom
*Použite pri volaní cez rozhranie JSON-RPC 2.0 / MCP.*

```json
{
  "jsonrpc": "2.0",
  "id": 101,
  "method": "tools/call",
  "params": {
    "name": "antigravity-compose-template",
    "arguments": {
      "prompt": "zrkadli 6-uholníkovú taktickú arénu Poslední Kmen s kinetickým glitch nápisom a 19 hexovými bunkami",
      "export_vulkan": true,
      "ascii_preview": true
    }
  }
}
```

### Blueprint C: Spustenie Kinetickej Textovej Animácie
*Použite pre generovanie dynamického titulku v termináli alebo HUD.*

```json
{
  "jsonrpc": "2.0",
  "id": 102,
  "method": "tools/call",
  "params": {
    "name": "antigravity-animate-text",
    "arguments": {
      "text": "SOVEREIGN ARBITER ONLINE",
      "mode": "TYPEWRITER_DECRYPT",
      "speed": 1.2,
      "width": 80,
      "height": 24,
      "t": 1.5
    }
  }
}
```

---

## 6. Few-Shot Príklad: Kompletná Injekcia do AI Studia

### Vstupný Vibe-Prompt od Vývojára:
> *"Potrebujem cyber-alchymistickú arénu v štýle posvätnej geometrie so 6 zrkadleniami, azúrovým neónom, kinetickým nápisom 'EPISTEMIC CORE' plávajúcim ako sínusoida a garanciou, že to nezasekne Iris Xe grafiku."*

### Výstup vygenerovaný AI Studiom (podľa Prompt Booku):

```json
{
  "set_id": "SET_01_CRYSTAL_POLYHEDRA_D6_D12",
  "identity": {
    "scene_id": "epistemic_core_alchemy",
    "name": "Crystalline Dihedral Manifold",
    "context_domain": "SACRED_GEOMETRY_AR",
    "narrative_role": "Central epistemic resonance core",
    "target_instances": 1024,
    "max_instances": 4096
  },
  "sdf_structure": {
    "primary_shape": "PLATONIC_OCTAHEDRON_GYROID",
    "dihedral_folds": 6,
    "recursion_limit": 24,
    "fractal_dimension": 2.718,
    "lipschitz_constant": 1.0
  },
  "harmonic_topology": {
    "base_radius_r0": 80.0,
    "fundamental_lobes_omega": 6,
    "components": [
      { "order_k": 1, "amplitude": 16.0, "frequency_multiplier": 1.0, "phase_offset_rad": 0.0 },
      { "order_k": 2, "amplitude": 8.0, "frequency_multiplier": 2.0, "phase_offset_rad": 0.0 }
    ],
    "overhang_analysis": {
      "xi_max": 0.88,
      "has_overhangs": false,
      "classification": "SMOOTH_2.5D_CANOPY"
    }
  },
  "lighting": {
    "key_fill_rim_ratio": [1.0, 0.4, 0.8],
    "volumetric_fog_density": 0.035,
    "chromatic_phase_rad": 0.5236,
    "ambient_color_hex": "#07090e",
    "accent_glow_hex": "#00f0ff"
  },
  "typography": {
    "text_string": "EPISTEMIC CORE",
    "kinetic_mode": "SINE_WAVE_FLOAT",
    "animation_speed": 1.0,
    "non_destructive_compositing": true
  },
  "constraints": {
    "min_separation_epsilon_m": 0.015,
    "intel_xe_uma_mb_ceiling": 128.0,
    "target_fps": 60,
    "vital_max_hp": 6
  },
  "vulkan_shader_defines": {
    "U_FOLDS": 6,
    "U_RECURSION_LIMIT": 24,
    "U_LIPSCHITZ_CLAMP": 1.0,
    "U_GLOW_INTENSITY": 1.45,
    "U_TIME_SCALE": 1.0
  }
}
```

---

## 7. Formálna JSON Schéma pre Validáciu Výstupov AI Studia

Túto schému možno použiť v automatických testoch a validačných filtroch pred zápisom do Godot scény alebo Vulkan shadrov:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "KrystalParametricRenderSetSpecification",
  "type": "object",
  "required": [
    "set_id",
    "identity",
    "sdf_structure",
    "lighting",
    "typography",
    "constraints",
    "vulkan_shader_defines"
  ],
  "properties": {
    "set_id": { "type": "string" },
    "identity": {
      "type": "object",
      "required": ["scene_id", "name", "context_domain", "target_instances", "max_instances"],
      "properties": {
        "scene_id": { "type": "string" },
        "name": { "type": "string" },
        "context_domain": { "type": "string" },
        "target_instances": { "type": "integer", "minimum": 1 },
        "max_instances": { "type": "integer", "maximum": 8192 }
      }
    },
    "sdf_structure": {
      "type": "object",
      "required": ["primary_shape", "dihedral_folds", "recursion_limit", "lipschitz_constant"],
      "properties": {
        "primary_shape": { "type": "string" },
        "dihedral_folds": { "type": "integer", "minimum": 2, "maximum": 32 },
        "recursion_limit": { "type": "integer", "minimum": 1, "maximum": 64 },
        "lipschitz_constant": { "type": "number", "const": 1.0 }
      }
    },
    "typography": {
      "type": "object",
      "required": ["text_string", "kinetic_mode", "animation_speed"],
      "properties": {
        "text_string": { "type": "string" },
        "kinetic_mode": {
          "type": "string",
          "enum": ["KINETIC_MATRIX", "SINE_WAVE_FLOAT", "GLITCH_CYBERPUNK", "TYPEWRITER_DECRYPT", "VOXEL_3D_EXTRUDE"]
        },
        "animation_speed": { "type": "number", "minimum": 0.1, "maximum": 5.0 }
      }
    },
    "constraints": {
      "type": "object",
      "required": ["min_separation_epsilon_m", "intel_xe_uma_mb_ceiling", "target_fps", "vital_max_hp"],
      "properties": {
        "min_separation_epsilon_m": { "type": "number", "minimum": 0.012 },
        "intel_xe_uma_mb_ceiling": { "type": "number", "maximum": 256.0 },
        "target_fps": { "type": "integer", "minimum": 30 },
        "vital_max_hp": { "type": "integer", "const": 6 }
      }
    }
  }
}
```

---

## 8. Verifikácia a Záver

Všetky render sety definované v tomto Prompt Booku sú plne kompatibilné s:
1. **Model Cascade Studio** ([`cascade_model_studio.html`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/static/cascade_model_studio.html)).
2. **Hybrid MCP Konnektorom** ([`hybrid_mcp_antigravity_connector.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/hybrid_mcp_antigravity_connector.py)).
3. **Kinetickým Typografickým Enginom** ([`text_animation_engine.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/text_animation_engine.py)).
4. **Scene Constructor Presetmi** ([`scene_constructor_presets.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/scene_constructor_presets.py)).

Tento Prompt Book predstavuje autoritatívny most medzi kognitívnou inferenciou AI modelov v AI Studiu a deterministickým renderovacím jadrom Krystal-Stack.
