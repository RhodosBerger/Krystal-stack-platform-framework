# KRYSTAL-STACK: EXACT DESIGN MANUAL OF GAME ELEMENTS
## Architectural Specification, Visual Design Tokens, Card Anatomy, Spatial Hex Assets & HyperOS 4 HUD System

**Document Version:** 2.0.0  
**Status:** OFFICIAL DESIGN MANUAL & LIVING SPECIFICATION  
**Target Engines:** Godot 4.x Engine / Three.js r128 / Web Audio API / Krystal Engine Core  
**Design Paradigm:** Xiaomi HyperOS 4 / MIUI Fluid Glassmorphism + Poslední Kmen High-Fantasy Strategy  
**Master Manual Reference:** See [MASTER_GAME_COMPOSITION_AND_LEGEND_MANUAL.md](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/docs/design_system/MASTER_GAME_COMPOSITION_AND_LEGEND_MANUAL.md) for full Map Legend, Procedural Render Pipeline, Fetch Hydration, and Race/Specializations Matrix.  

---

## 1. Design Philosophy & Unified Visual Language

The Krystal-Stack visual and mechanical experience bridges two distinct design paradigms into a cohesive, high-performance tactical interface:

```
┌────────────────────────────────────────────────────────┐
│             KRYSTAL-STACK DESIGN PARADIGM             │
├───────────────────────────┬────────────────────────────┤
│   XIAOMI HYPEROS 4 / MIUI │  POSLEDNÍ KMEN / INNOGAMES │
│   LIQUID DESIGN SYSTEM    │  HIGH-FANTASY COMBAT LOGIC │
├───────────────────────────┼────────────────────────────┤
│ • Superellipse Squircles  │ • 6 HP Vital Invariant     │
│ • Gaussian Optical Blur   │ • Pointy-Topped Axial Hex  │
│ • Dynamic Island Capsule  │ • Double-Entry Mana Ledger │
│ • Spring Physics Curves   │ • 3 Tribal Archetypes      │
│ • Frosted Glass Overlays  │ • Bezier Projectile Flight │
└───────────────────────────┴────────────────────────────┘
```

### 1.1 The Three Invariant Mechanical Laws
Every game element documented in this manual strictly adheres to three architectural invariants:
1. **The Hero Vital Invariant ($\text{HP} \in [0, 6]$):**
   The maximum health of any hero unit cannot exceed $6$. Healing actions cap strictly at $6$, and dropping to $0$ immediately triggers tactical collapse.
2. **The Double-Entry Ledger Conservation Law:**
   No resource (Mana, Aether Crystal, Toxic Slime, Amber Rune) may be created or destroyed ex nihilo. Every expenditure debit must have a corresponding credit event in the cryptographic match ledger.
3. **The Axial Metric Spatial Invariant:**
   Distance between any two hex tiles $(q_1, r_1)$ and $(q_2, r_2)$ is defined rigorously by the axial L1-projection metric:
   $$D(H_1, H_2) = \frac{|\Delta q| + |\Delta q + \Delta r| + |\Delta r|}{2}, \quad \text{where } \Delta q = q_1 - q_2, \; \Delta r = r_1 - r_2$$

---

## 2. Master Design Tokens & Foundation Specification

The design system is parameterized through CSS variables, Three.js shaders, and Godot theme overrides.

### 2.1 Curvature & Geometry: Superellipse Squircles
HyperOS 4 eliminates sharp corners and standard circular fillets in favor of Lamé superellipses with exponent $n \approx 4.0$:
$$\left| \frac{x}{a} \right|^{4} + \left| \frac{y}{b} \right|^{4} = 1$$

In CSS and Godot Theme resources, these are mapped to progressive squircle tokens:
```css
:root {
    --hyper-squircle-xs: 6px;       /* Micro badges, status pills */
    --hyper-squircle-sm: 10px;      /* Buttons, prompt input boxes */
    --hyper-squircle-md: 14px;      /* Action cards, sector popups */
    --hyper-squircle-lg: 20px;      /* Dynamic Island, dialog containers */
    --hyper-squircle-xl: 28px;      /* Floating action dock, HUD panels */
    --hyper-squircle-pill: 9999px;  /* Filter pills, resource meters, capsules */
}
```

### 2.2 Gaussian Blur, Refraction & Optical Depth
All overlay containers utilize dual-layer frosted glass physics:
- **Optical Refraction:** `backdrop-filter: blur(28px) saturate(190%) contrast(105%);`
- **Surface Translucency:** `rgba(18, 22, 34, 0.72)` (Primary Glass), `rgba(26, 32, 48, 0.65)` (Elevated Surface)
- **Specular Top Highlight:** `inset 0 1px 0 rgba(255, 255, 255, 0.15)`
- **Atmospheric Ambient Drop Shadow:** `0 16px 40px rgba(0, 0, 0, 0.45)`

```css
.hyper-glass-panel {
    background: var(--hyper-glass);
    backdrop-filter: var(--hyper-blur);
    -webkit-backdrop-filter: var(--hyper-blur);
    border: 1px solid var(--hyper-glass-border);
    box-shadow: var(--hyper-shadow);
    border-radius: var(--hyper-squircle-lg);
}
```

### 2.3 Semantic Color Tokens & Tribal Palettes

| Token Name | Hex Code | RGB Values | Role & Application |
| :--- | :--- | :--- | :--- |
| `--bg-dark` | `#08090d` | `rgb(8, 9, 13)` | Master canvas & dark background |
| `--bg-viewport` | `#050608` | `rgb(5, 6, 8)` | 3D WebGL viewport canvas |
| `--text-main` | `#f0f3f8` | `rgb(240, 243, 248)`| Primary headings & card titles |
| `--text-muted` | `#8e95a5` | `rgb(142, 149, 165)`| Descriptions, meta stats, labels |
| `--text-subtle` | `#5a6275` | `rgb(90, 98, 117)` | Footers, grid coordinates, disabled |
| **`--cyan` (Krystal)** | `#66fcf1` | `rgb(102, 252, 241)`| Crystalline Tribe, in-range target reticle |
| **`--aether-blue`** | `#00f2fe` | `rgb(0, 242, 254)` | Pure mana energy, aether overclock |
| **`--toxic-green`** | `#39ff14` | `rgb(57, 255, 20)` | Toxic Tribe, acid slime, rooted status |
| **`--toxic-purple`**| `#9d4edd` | `rgb(157, 78, 221)` | Spore totem, poison mist, sabotage |
| **`--druid-gold`** | `#ffd700` | `rgb(255, 215, 0)` | Druid Tribe, nature bless, HP healing |
| **`--druid-amber`**| `#ff9100` | `rgb(255, 145, 0)` | Amber runes, oak wood, ancient bark |
| **`--alert-red`** | `#ff453a` | `rgb(255, 69, 58)` | Out-of-range alert, damage received |
| **`--focus-blue`** | `#64d2ff` | `rgb(100, 210, 255)`| Interactive focus ring, hover halos |

### 2.4 Typography Hierarchy
1. **Ancient Display / Tribal Titles:** `'Cinzel', serif` (Weights: 700, 900)  
   *Use case:* Game title, tribe crest banners, ultimate card names.
2. **UI & Card Content:** `'Outfit', 'Inter', sans-serif` (Weights: 400, 500, 600, 700)  
   *Use case:* Dynamic Island status, card descriptions, buttons, tooltips.
3. **Telemetry & Ledger Engine:** `'Fira Code', monospace` (Weights: 400, 600, 700)  
   *Use case:* Axial coordinates `[q, r]`, Mana values `3/10`, cryptographic hashes, distance counters.

---

## 3. Exact Anatomy of Action Cards

The Action Card is the core interactive token of the Krystal-Stack combat and economic system.

```
┌────────────────────────────────────────────────────────┐
│ [💎 3 MANA]                   [TIER 1] [KRYŠTÁL]  (1)  │
├────────────────────────────────────────────────────────┤
│                 KRYŠTÁLOVÝ METEOR                 (2)  │
├────────────────────────────────────────────────────────┤
│ ┌────────────────────────────────────────────────────┐ │
│ │                                                    │ │
│ │             3D HOLOGRAPHIC VIEWPORT                │ │
│ │          (Interactive Mesh & Shader)           (3) │ │
│ │                                                    │ │
│ └────────────────────────────────────────────────────┘ │
├────────────────────────────────────────────────────────┤
│  [🏹 RANGED: 1-4 HEX]       [💥 -2 HP]    [PARABOLIC](4)│
├────────────────────────────────────────────────────────┤
│  Zasiahne cieľ kryštálovým meteorom a spôsobí         │
│  2 body priameho zranenia. Dosah: 1-4 hexov.     (5)  │
├────────────────────────────────────────────────────────┤
│  [ 🎯 ZAMIERIŤ V 3D ]         [CD: 0 ROUNDS]      (6)  │
└────────────────────────────────────────────────────────┘
```

### 3.1 Card Component Breakdown
1. **Card Header Capsule:**
   - **Top-Left:** Mana cost badge with cyan glowing crystal icon.
   - **Top-Right:** Tribe badge (e.g., `[💎 KRYŠTÁL]`, `[🧪 JED]`, `[🌿 DRUID]`) + Tier indicator.
2. **Card Name & Type Title:**
   - Bold `'Cinzel'` font with tribal accent underglow.
3. **3D Holographic Mesh Viewport:**
   - 140px height window rendering real-time WebGL / Three.js geometry with rotation on card hover.
   - Particle dust emitter aligned to the card's elemental affiliation.
4. **Combat Attributes Strip:**
   - **Attack Type Badge:** `MELEE` (D=1), `RANGED` (D=1..4), `AOE` (D=1..3), `SELF` (D=0), `GLOBAL`.
   - **Primary Metric:** `HP Delta` (Damage `-2 HP`, Healing `+2 HP`), `Armor Delta` (`+2 Armor`).
   - **Trajectory Badge:** `Parabolic Arc`, `Linear Slash`, `Ground Crawl`, `Instant Flash`.
5. **Rules & Mechanical Description:**
   - Clear, concise rules text explaining targeting limitations, status effects, and synergies.
6. **Card Footer & Direct Aiming Trigger:**
   - **`[🎯 ZAMIERIŤ V 3D]` Button:** Switches camera to tactical aiming mode, enables the parabolic Bezier targeting arrow, and focuses the reticle on valid hex tiles.
   - **Combo Prerequisite Pill (If applicable):** Displays required status on target (e.g., `⚡ VYŽADUJE ROOTE`).

---

### 3.2 Canonical 9 Cards Specification Matrix

#### 1. Kryštálový Meteor (`crystal_meteor`)
- **Tribe:** Kryštálový Kmeň (Severné Štíty)
- **Cost:** 3 Mana
- **Attack Type:** `RANGED` | **Range:** 1 – 4 hexes
- **Trajectory:** Parabolic Arc ($H(D) = \min(3.8, 1.0 + 0.55 \cdot D)$)
- **Primary Effect:** $-2\text{ HP}$ direct damage to target
- **Status Effect:** `none`
- **3D Mesh Asset:** `crystal_shard.obj` | **Shader:** Cyan Glowing Facet (`#00ffff`)
- **Visual FX:** Trailing cyan mana dust (180 particles) + point impact flash

#### 2. Kryštálový Štít (`crystal_shield`)
- **Tribe:** Kryštálový Kmeň
- **Cost:** 2 Mana + 1 Aether Crystal
- **Attack Type:** `SELF` | **Range:** 0 hexes (Caster only)
- **Trajectory:** Instant Ground Flash
- **Primary Effect:** $+2\text{ Armor}$
- **Status Effect:** `shielded` (Duration: 2 rounds)
- **3D Mesh Asset:** `crystal_shield.obj` | **Shader:** Glass Crystalline (`#66fcf1`, opacity: 0.85)
- **Visual FX:** Expanding hexagonal forcefield barrier

#### 3. Rezonančný Pylón (`crystal_pylon`)
- **Tribe:** Kryštálový Kmeň
- **Cost:** 4 Mana + 1 Aether Crystal
- **Attack Type:** `SELF` / `BUILDING` | **Range:** 0 – 2 hexes (Friendly sector)
- **Trajectory:** Ground Drop
- **Primary Effect:** Places a permanent Mana Generator structure ($+1\text{ Mana/round}$)
- **Status Effect:** `mana_regen`
- **3D Mesh Asset:** `crystal_shard.obj` | **Shader:** Pure Aether Opalescence (`#e0ffff`)
- **Visual FX:** Concentric resonance rings emitting periodically

#### 4. Kyslý Sliz (`acid_slime`)
- **Tribe:** Jedovatý Kmeň (Pustina)
- **Cost:** 1 Mana + 1 Toxic Slime
- **Attack Type:** `RANGED` | **Range:** 1 – 3 hexes
- **Trajectory:** Parabolic Arc
- **Primary Effect:** $-1\text{ HP}$ damage
- **Status Effect:** `rooted` (Duration: 1 round — target cannot move or cast melee attacks)
- **3D Mesh Asset:** `acid_slime.obj` | **Shader:** Viscous Acid Green (`#39ff14`)
- **Visual FX:** Bubbling puddle with rising green toxic vapors

#### 5. Toxický Oblak (`toxic_cloud`)
- **Tribe:** Jedovatý Kmeň
- **Cost:** 4 Mana + 2 Toxic Slime
- **Attack Type:** `AOE` | **Range:** 1 – 3 hexes (Affects 1 hex radius)
- **Trajectory:** High Parabolic Drop
- **Primary Effect:** $-1\text{ HP}$ immediately, plus $-1\text{ HP}$ on round end for 3 rounds
- **Status Effect:** `poisoned`
- **3D Mesh Asset:** `toxic_totem.obj` | **Shader:** Organic Chitin (`#7fff00` / `#8a2be2`)
- **Visual FX:** Swirling purple and toxic green spore cloud (radius: 4.5m)

#### 6. Korene Zeme (`earth_roots`)
- **Tribe:** Druidi (Hlboký Les)
- **Cost:** 2 Mana + 1 Amber Rune
- **Attack Type:** `RANGED` | **Range:** 1 – 3 hexes
- **Trajectory:** Ground Crawl (Creeping root path along terrain surface)
- **Primary Effect:** $-1\text{ HP}$ damage
- **Status Effect:** `stunned` (Duration: 1 round — target skips next combat action)
- **3D Mesh Asset:** `earth_roots.obj` | **Shader:** Ancient Bark (`#8b4513`)
- **Visual FX:** Gnarled roots erupting through hex floor with floating green leaves

#### 7. Požehnanie Prírody (`nature_bless`)
- **Tribe:** Druidi
- **Cost:** 3 Mana + 1 Amber Rune
- **Attack Type:** `SELF` | **Range:** 0 hexes
- **Trajectory:** Instant Sanctuary Aura
- **Primary Effect:** $+2\text{ HP}$ healing (strictly capped at $6\text{ HP}$) + $+1\text{ Armor}$
- **Status Effect:** `regenerating`
- **3D Mesh Asset:** `druid_monolith.obj` | **Shader:** Mossy Granite with Golden Inlay (`#ffd700`)
- **Visual FX:** Vertical golden pillar of light with ascending life motes

#### 8. Úder Druidskej Palice (`druid_strike`)
- **Tribe:** Druidi
- **Cost:** 1 Mana
- **Attack Type:** `MELEE` | **Range:** Strictly 1 hex (Adjacent tile only)
- **Trajectory:** Fast Linear Slash ($H = 0.5\text{m}$)
- **Primary Effect:** $-1\text{ HP}$ physical blunt impact
- **Status Effect:** `none`
- **3D Mesh Asset:** `druid_monolith.obj` | **Shader:** Oak Rune Energy (`#ffd700`)
- **Visual FX:** Golden directional swipe blade arc

#### 9. Zuby Rozkladu (`decay_strike`)
- **Tribe:** Jedovatý Kmeň
- **Cost:** 2 Mana + 1 Toxic Slime
- **Attack Type:** `MELEE` | **Range:** Strictly 1 hex
- **Trajectory:** Low Linear Bite ($H = 0.5\text{m}$)
- **Primary Effect:** $-2\text{ HP}$ corrosive damage
- **Status Effect:** `poisoned` (Duration: 2 rounds)
- **3D Mesh Asset:** `acid_slime.obj` | **Shader:** Acidic Venom (`#7fff00`)
- **Visual FX:** Poisonous claw slash with venom splatter

---

### 3.3 Synergistic Combos & Escalation Ultimates

| Card / Ability ID | Name & Tribe | Tier | Mana & Resource Cost | Attack Type & Range | Combo Condition | Mechanical Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `corrosive_shatter_combo` | Korozívny Roztrieštenec (Kryštál) | 2 | 5 Mana + 2 Aether Crystals | `RANGED` (1–3 hex) | Requires target to be `rooted` | Smashes crystals on corroded foe; destroys all armor and deals $-4\text{ HP}$ damage ($1.5\times$ multiplier). |
| `primordial_bloom_combo` | Prvotvorný Rozkvet (Druid) | 2 | 4 Mana + 2 Amber Runes | `MELEE` (1–2 hex) | Requires target to be `stunned` | Drains 2 HP from foe, heals player $+2\text{ HP}$, grants $+2\text{ Armor}$. |
| `supernova_cataclysm` | Kryštálová Supernova (Kryštál) | 3 | 8 Mana + 4 Aether Crystals | `GLOBAL` (Board-wide) | Escalation: Round 3+ | $-5\text{ HP}$ damage to enemy, $+3\text{ Armor}$ to self, applies `blinded` status. |
| `pandemic_wave` | Pandemická Vlna (Jed) | 3 | 7 Mana + 4 Toxic Slime | `GLOBAL` (Board-wide) | Escalation: Round 3+ | $-3\text{ HP}$ instant damage, then $-2\text{ HP/round}$ for 4 rounds (`lethal_plague`). |
| `wrath_of_world_tree` | Hnev Stromu Sveta (Druid) | 3 | 8 Mana + 4 Amber Runes | `GLOBAL` (Board-wide) | Escalation: Round 3+ | Full heal to $6\text{ HP}$, grants $+4\text{ Armor}$, applies `stunned` to all enemy sectors. |

---

## 4. Anatomy of Spatial & Board Elements: The Hex World

Tactical combat takes place across a pointy-topped axial hex grid with discrete biomes and fortification sectors.

```
               /\
              /  \
             /    \
       .----+(q,r)+----.
      /      \    /      \
     /        \  /        \
    +----------\/----------+
```

### 4.1 Hex Geometry & World Space Projection
Given axial coordinates $(q, r)$ and outer hex radius $R = 1.5\text{m}$:
$$X = R \cdot \sqrt{3} \cdot \left( q + \frac{r}{2} \right)$$
$$Z = R \cdot \frac{3}{2} \cdot r$$
$$Y = 0.05 \cdot \sin(q \cdot 0.8) \cdot \cos(r \cdot 0.8) \quad \text{(Subtle terrain elevation)}$$

### 4.2 Terrain Biome Specifications

#### Biome A: Severné Kryštálové Štíty (Crystalline Peaks)
- **Primary Color:** `#0a192f` base with `#66fcf1` crystalline edge veins.
- **Surface Material:** Low-roughness refractive glass (`roughness: 0.15`, `metalness: 0.8`).
- **Environmental Shaders:** Procedural vertex displacement simulating sharp jagged crystal ridges.
- **Ambient Lighting:** Cyan omni-light (`energy: 1.4`) with floating ice particles.

#### Biome B: Jedovatá Pustina (Toxic Slime Marsh)
- **Primary Color:** `#0d1a0d` mud with `#39ff14` viscous fluid pools.
- **Surface Material:** High-specular organic fluid (`roughness: 0.05`, `metalness: 0.2`).
- **Environmental Shaders:** Animated time-based sine-wave bubbling on surface vertices.
- **Ambient Lighting:** Acid-green pulsing ground glow with rising toxic vapor quads.

#### Biome C: Posvätný Hlboký Les (Ancient Druid Forest)
- **Primary Color:** `#1a160c` rich loam with `#556b2f` mossy granite outcrops.
- **Surface Material:** High-roughness natural bark/moss (`roughness: 0.85`, `metalness: 0.05`).
- **Environmental Shaders:** Gnarled procedural root meshes weaving between hex boundaries.
- **Ambient Lighting:** Warm golden sunlight motes (`#ffd700`) drifting continuously.

#### Biome D: Neutrálna Citadela (Center Citadel)
- **Primary Color:** `#14171f` dark basalt stone with `#e0ffff` rune carvings.
- **Surface Material:** Chiseled stone brick with metallic rune inlays.
- **Environmental Shaders:** Triple fortification defense rings pulsing with barrier forcefields.

---

### 4.3 Sector Conquest & Fortification Anatomy

Every sector on the tactical map represents a contested economic control point:

```
┌────────────────────────────────────────────────────────┐
│  SECTOR: CITADELA STREDU              [NEUTRAL BASTION]│
├────────────────────────────────────────────────────────┤
│  FORTIFICATION HP: [████████████░░░░░░░░] 8 / 12       │
│  GARRISON: 2x Bastion Guard (⚔️ 3 | 🛡️ 3 | ❤️ 4)       │
│  YIELD: +2 Mana / +1 Aether Crystal per round          │
│  STATUS: ⚔️ UNDER SIEGE (Capture Progress: 45%)        │
├────────────────────────────────────────────────────────┤
│  [ 💥 VIESŤ ÚTOK ]    [ 🛡️ POSILNIŤ ]   [ 📊 ŠTATISTIKY ]│
└────────────────────────────────────────────────────────┘
```

#### Fortification States
1. **`UNCONTESTED`:** Sector is fully under owner's control. Fortification HP is at maximum. Standard resource yields flow into the owner's ledger each economy phase.
2. **`CONTESTED`:** Enemy units have entered the perimeter. Resource production is suspended (embargo status).
3. **`UNDER_SIEGE`:** Fortification structures have taken damage ($0 < \text{Fort HP} < \text{Max Fort HP}$). Garrison units actively engage attackers.
4. **`CAPTURED`:** Fortification HP reduced to $0$ and garrison destroyed. Sector flips ownership to the attacker, paying out an immediate plunder bounty.

---

## 5. Anatomy of the 3D Animated Targeting Arrow & Range Rules

The targeting arrow is the visual manifestation of the game's spatial validation engine, providing instantaneous perceptual feedback to the player.

```
                    P1 (Bezier Apex)
                         ▲
                        / \
                       /   \   Parabolic Trajectory
                      /     \
                     /       ▼
     P0 (Hero Hex) ──         ──▶ P2 (Target Hex)
     [q0, r0]                    [q2, r2]
     (Source)                    (Target Reticle)
```

### 5.1 Parabolic Bezier Mathematical Model
The curve trajectory is modeled as a 3D Quadratic Bezier spline:
$$P(t) = (1 - t)^{2} P_0 + 2(1 - t)t P_1 + t^{2} P_2, \quad t \in [0, 1]$$

Where:
- $P_0 = (X_{\text{source}}, Y_{\text{source}} + 0.35, Z_{\text{source}})$: Origin at hero's weapon hand/core.
- $P_2 = (X_{\text{target}}, Y_{\text{target}} + 0.20, Z_{\text{target}})$: Destination reticle on the ground.
- $P_1 = \left( \frac{X_0 + X_2}{2}, \; \max(Y_0, Y_2) + H(D), \; \frac{Z_0 + Z_2}{2} \right)$: Control point establishing dynamic apex height.

#### Dynamic Apex Elevation Function $H(D)$
$$H(D) = \begin{cases}
0.50\,\text{m}, & \text{if Attack Type is } \mathbf{MELEE} \text{ (low direct swing)} \\
0.20\,\text{m}, & \text{if Attack Type is } \mathbf{SELF} \text{ (immediate ground aura)} \\
\min(3.80\,\text{m}, \; 1.0 + 0.55 \cdot D), & \text{if Attack Type is } \mathbf{RANGED} \text{ (high ballistic arc)}
\end{cases}$$

### 5.2 Arrowhead Orientation & Alignment
The directional cone arrowhead (`THREE.ConeGeometry(0.3, 0.75, 16)`) is aligned precisely with the curve's end tangent vector:
$$T = \frac{d P(t)}{dt}\Big|_{t=1} = 2(P_2 - P_1)$$
$$\hat{T} = \frac{T}{\|T\|}$$
$$\mathbf{Quaternion} = \text{QuaternionFromVectors}\left( \mathbf{v}_{\text{forward}}=(0, 0, 1), \; \hat{T} \right)$$

### 5.3 Color State Feedback Logic
The visual presentation shifts dynamically based on range and resource validity:

```
┌─────────────────┬──────────────────┬─────────────────┬─────────────────┐
│ TARGET STATE    │ TUBE / ARROWHEAD │ RETICLE STATUS  │ DYNAMIC ISLAND  │
├─────────────────┼──────────────────┼─────────────────┼─────────────────┤
│ In Range & Valid│ Cyan (#66fcf1)   │ Rapid Rotation  │ "✓ V DOSAHU"    │
│                 │ Emissive: 0.85   │ Pulsing Scale   │ Green glow badge│
├─────────────────┼──────────────────┼─────────────────┼─────────────────┤
│ Out of Range    │ Crimson (#ff453a)│ Locked Static   │ "✕ MIMO DOSAHU" │
│ (D > Max/D < Min│ Emissive: 0.40   │ Red Alert Tint  │ Shake animation │
├─────────────────┼──────────────────┼─────────────────┼─────────────────┤
│ Insufficient    │ Amber (#ffb703)  │ Dashed Warning  │ "✕ NEDOSTATOK   │
│ Mana / Resource │ Emissive: 0.50   │ Ring            │ MANY"           │
└─────────────────┴──────────────────┴─────────────────┴─────────────────┘
```

### 5.4 Flight Animation & Impact Shockwave
When a card is cast:
1. **Projectile Interpolation:** A glowing sphere interpolates along $P(t)$ over duration:
   $$T_{\text{flight}} = 0.35 + 0.10 \cdot D\,\text{seconds}$$
2. **Particle Wake:** 40 trailing particles are emitted per second behind the projectile.
3. **Impact Shockwave:** Upon reaching $t=1$:
   - Ground shockwave ring expands radially ($r: 1.0\text{m} \to 3.5\text{m}$) over 400ms.
   - Opacity decays linearly to $0$.
   - Floating 3D combat text rises vertically ($Y: 0.5\text{m} \to 2.2\text{m}$) displaying `"-2 HP"`, `"+2 ARMOR"`, or `"ROOTED!"`.
   - Micro camera shake impulse ($\Delta \mathbf{cam} = \pm 0.08\,\text{m}$) applies for 120ms.

---

## 6. Anatomy of HUD & UI Components

The interface is driven by floating frosted-glass surfaces organized into four strategic quadrants.

```
┌────────────────────────────────────────────────────────────────────────┐
│ [💎 10/15] [❤️ 6/6] [🛡️ 0]     [   DYNAMIC ISLAND   ]    [ROUND 1: EKON]│
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│                                                                        │
│                          3D TACTICAL VIEWPORT                          │
│                                                                        │
│                                                                        │
├────────────────────────────────────────────────────────────────────────┤
│ [VŠETKY] [💎 KRYŠTÁL] [🧪 JED] [🌿 DRUID]        [ ⚔️ KONTROLA SEKTOROV ]│
│ ┌────────────────────────────────────────────────────────────────────┐ │
│ │  [CARD 1]      [CARD 2]      [CARD 3]      [CARD 4]      [CARD 5]  │ │
│ └────────────────────────────────────────────────────────────────────┘ │
│                     FLOATING ACTION CARDS DOCK                         │
└────────────────────────────────────────────────────────────────────────┘
```

### 6.1 Xiaomi Dynamic Island Status Capsule
Located at top-center of the viewport, this pill-shaped container smoothly morphs:
- **Dimensions:** Width $360\text{px} \to 540\text{px}$, Height $38\text{px} \to 54\text{px}$.
- **Ambient Mode:** Displays active round number, phase pill (`[EKONOMICKÁ FÁZA]`), and escalation level.
- **Aiming Mode:** Expands with cyan border glow, glowing targeting crosshair, ability name, attack type tag (`🏹 RANGED (1-4)`), real-time distance readouts, and an `[✕ ZRUŠIŤ]` escape button.

### 6.2 Floating Action Cards Dock
A frosted glass dock floating 18px above the bottom viewport edge:
- **Card Spacing & Layout:** Horizontal flexbox with `gap: 14px`, smooth scroll snap.
- **Card Hover Physics:** Card elevates `translateY(-10px)` with an expanded drop shadow (`0 20px 30px rgba(0,0,0,0.6)`) and a tribal colored border highlight.
- **Mana Cost Lockout:** If hero mana is less than card cost, card renders at $45\%$ opacity with a grayscale filter and a lock overlay.

### 6.3 Hero Vitals & Double-Entry Ledger Display
Located at top-left:
- **Mana Gauge:** Segmented glowing bar displaying current mana ($0$ to $15$).
- **Hero Health:** 6 distinct crystalline heart icons. Intact hearts pulse softly; damaged hearts turn dark gray with cracked vein shaders.
- **Armor Gauge:** Shield icon with numerical counter. Armor absorbs incoming damage before health is decrepted.
- **Ledger Feed Drawer:** Collapsible panel listing cryptographic hashes and double-entry debits/credits in real time.

---

## 7. Motion, Micro-Interactions & Procedural Audio Design

### 7.1 HyperOS Spring Physics Parameters
All UI movements and transitions use a damped harmonic oscillator:
$$m \frac{d^2 x}{dt^2} + c \frac{dx}{dt} + k x = 0$$

- **Stiffness ($k$):** $320\,\text{N/m}$
- **Damping ($c$):** $28\,\text{N}\cdot\text{s/m}$
- **Mass ($m$):** $1.0\,\text{kg}$
- **Damping Ratio ($\zeta$):** $\frac{c}{2\sqrt{km}} = \frac{28}{2\sqrt{320}} \approx 0.78$ (Slightly underdamped for responsive snap with zero jitter).
- **CSS Equivalent:** `cubic-bezier(0.20, 0.85, 0.25, 1.00)`

### 7.2 Procedural Web Audio Sound Synthesizer Specifications
To ensure zero third-party audio asset dependencies, all sound effects are synthesized live using the Web Audio API:

```
┌─────────────────┬────────────────────┬──────────────┬──────────────────┐
│ ACTION EVENT    │ OSCILLATOR TYPE    │ FREQ / CURVE │ ENVELOPE (ADSR)  │
├─────────────────┼────────────────────┼──────────────┼──────────────────┤
│ Card Select     │ Sine + Triangle    │ 440Hz → 880Hz│ A: 5ms, D: 80ms, │
│                 │                    │ (Arpeggio)   │ S: 0.1, R: 120ms │
├─────────────────┼────────────────────┼──────────────┼──────────────────┤
│ Projectile Fire │ Bandpass Filtered  │ 1200Hz →220Hz│ A: 10ms, D:180ms,│
│ (Whoosh)        │ White Noise        │ (Exponential)│ S: 0.0, R: 50ms  │
├─────────────────┼────────────────────┼──────────────┼──────────────────┤
│ Crystal Impact  │ Dual High Sine     │ 1760Hz+2640Hz│ A: 1ms, D: 350ms,│
│                 │ (Metallic Chime)   │ (Bell decay) │ S: 0.0, R: 300ms │
├─────────────────┼────────────────────┼──────────────┼──────────────────┤
│ Slime Splatter  │ Modulated Sawtooth │ 180Hz ~ 90Hz │ A: 15ms, D:220ms,│
│                 │ (Low-pass 400Hz)   │ (LFO 18Hz)   │ S: 0.2, R: 180ms │
├─────────────────┼────────────────────┼──────────────┼──────────────────┤
│ Melee Impact    │ Low Sine + Noise   │ 90Hz → 30Hz  │ A: 2ms, D: 140ms,│
│ (Blunt Punch)   │ (Bass thump)       │ (Punch punch)│ S: 0.0, R: 80ms  │
├─────────────────┼────────────────────┼──────────────┼──────────────────┤
│ Out of Range    │ Square (Low-pass)  │ 110Hz (Buzz) │ A: 5ms, D: 160ms,│
│ Warning Error   │                    │ (Two pulses) │ S: 0.0, R: 50ms  │
└─────────────────┴────────────────────┴──────────────┴──────────────────┘
```

---

## 8. Summary Table of Game Elements

| Element Category | Element Name | ID / Asset Code | Visual Style | Interaction Trigger |
| :--- | :--- | :--- | :--- | :--- |
| **Card (Ranged)** | Kryštálový Meteor | `crystal_meteor` | Cyan crystalline facet, trailing dust | Click card $\to$ 3D Reticle $\to$ Cast |
| **Card (Defensive)**| Kryštálový Štít | `crystal_shield` | Hexagonal refractive glass barrier | Click card $\to$ Cast on self |
| **Card (Building)** | Rezonančný Pylón | `crystal_pylon` | Opalescent aether spire with rings | Click card $\to$ Deploy on hex |
| **Card (Ranged CC)**| Kyslý Sliz | `acid_slime` | Bubbling neon green fluid pool | Click card $\to$ 3D Reticle $\to$ Cast |
| **Card (AOE)** | Toxický Oblak | `toxic_cloud` | Chitin spore column, purple vapor | Click card $\to$ Ground Splat $\to$ Cast |
| **Card (Ranged CC)**| Korene Zeme | `earth_roots` | Ancient gnarled oak roots & leaves | Click card $\to$ Ground Reticle $\to$ Cast |
| **Card (Heal)** | Požehnanie Prírody | `nature_bless` | Mossy granite monolith with gold aura | Click card $\to$ Cast on self |
| **Card (Melee)** | Úder Druidskej Palice| `druid_strike` | Amber curved slash wave ($D=1$) | Click card $\to$ Adjacent hex |
| **Card (Melee)** | Zuby Rozkladu | `decay_strike` | Acidic venom claw bite ($D=1$) | Click card $\to$ Adjacent hex |
| **Targeting** | 3D Bezier Arrow | `QuadraticBezier3` | Parabolic glowing tube with cone head | Hover hex while aiming |
| **Targeting** | Ground Reticle | `THREE.RingGeometry` | Concentric rotating segmented rings | Hover hex while aiming |
| **HUD** | Dynamic Island | `#dynamicIsland` | Morphing superellipse capsule | Global game state & aim status |
| **HUD** | Action Cards Dock | `#actionCardsDock` | Frosted glass bottom carousel | Card selection & filtering |
| **HUD** | Sector Combat Hub | `#sectorCombatModal`| Glass modal with fortification bars | Assault & garrison management |
| **Sector Node** | Center Citadel | `sector_center_citadel`| Basalt stone bastion with 3 rings | Hex click in tactical mode |
| **Sector Node** | Crystal Peaks | `sector_north_crystal`| Ice-blue spikes with mana conduit | Hex click in tactical mode |
| **Sector Node** | Toxic Marsh | `sector_south_toxic` | Slime pools with incubator spire | Hex click in tactical mode |
| **Sector Node** | Druid Grove | `sector_east_druid` | Ancient World Tree with amber runes | Hex click in tactical mode |
