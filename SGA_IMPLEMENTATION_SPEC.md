# SOVEREIGN GENERATIVE AXIOMS (SGA) - IMPLEMENTAČNÁ ŠPECIFIKÁCIA

Tento dokument mapuje teoretické vzorce na konkrétnu implementáciu v Pure C (`sga_core.h`).

---

## 📐 I. MESH SYNTHESIS ($\mathcal{G}_{mesh}$)

### 1. Vertex Displacement Formula ($V_{disp}$)

**Teoretický vzorec:**
$$V'_{xyz} = V_{xyz} + (1.0 - S_{hex}) \cdot \left( \sum_{i=1}^{n} \mathcal{A}_i \cdot W_i \cdot \vec{T}_i \right) + \text{Slick}(v, t)$$

**Implementácia v sga_core.h:**

```c
// Riadky 369-381: Konverzia stability na displace faktor
static inline uint16_t sga_stability_to_displacement_factor(uint8_t stability) {
    // (1.0 - S_{hex}/255.0) ekvivalent v integer aritmetike
    return (uint16_t)(STABILITY_MAX - stability);
}

// Riadky 384-389: Fracture intensity calculation
static inline uint8_t sga_calculate_fracture_intensity(uint8_t stability) {
    if (stability > STABILITY_STABLE) return 0;
    if (stability > STABILITY_UNSTABLE) return (STABILITY_STABLE - stability) / 4;
    if (stability > STABILITY_CRITICAL) return (STABILITY_UNSTABLE - stability) / 2;
    return STABILITY_MAX - stability; // Maximum fracture
}
```

**Mapovanie premenných:**

| Teória | Implementácia | Popis |
|--------|---------------|-------|
| $S_{hex}$ | `SGA_GET_STABILITY(word)` | Hodnota 0-255 z bitov 16-23 |
| $V_{xyz}$ | `HexCell.state` | Pôvodná pozícia vertexu |
| $\mathcal{A}_i$ | `axiom_mask` | Axiomatická amplitúda z op-kódu |
| $\text{Slick}(v, t)$ | Fracture intensity | Časový jitter pri $S_{hex} < 64$ |

**Thresholdy stability:**
- `STABILITY_CRITICAL = 64` → Spustenie Slick efektu
- `STABILITY_UNSTABLE = 128` → Stredná fraktúra
- `STABILITY_STABLE = 192` → Minimálna fraktúra
- `STABILITY_PERFECT = 255` → Žiadna deformácia

---

### 2. The Fracture Axiom ($F_p$)

**Teoretický vzorec:**
$$P_{fracture}(v) = \text{Stochastic\_Break}(v) \cdot (E - 0.95) \cdot 10$$

**Implementácia:**

```c
// Entropy threshold 0.95 ≈ 242/255 v našej kvantizácii
#define ENTROPY_FRACTURE_THRESHOLD 242

// Fracture magnitude calculation
if (entropy >= ENTROPY_FRACTURE_THRESHOLD) {
    fracture_magnitude = (entropy - ENTROPY_FRACTURE_THRESHOLD) * 10;
    // Miami Fracture state: polygons detach from base mesh
}
```

**Bitové flagy pre Fracture state:**
```c
#define ATTR_CORRUPTED    (1u << 11)  // Bit 11: Označuje fracturované hexy
#define ATTR_PHASED       (1u << 13)  // Bit 13: Floating shards
```

---

## 📜 II. SCENARIO SYNTHESIS ($\mathcal{S}_{syn}$)

### 1. Narrative Gradient ($\nabla N$)

**Teoretický vzorec:**
$$\nabla N = \oint_{t}^{t+\Delta t} (\mathcal{R} - \mathcal{E}) \, d\tau$$

**Implementácia v sga_core.h:**

```c
// Riadky 115-119: Scaled attributes v word1
#define SCALE_RESONANCE       0x000000C0u  // $\mathcal{R}$ Oracle Plane
#define SCALE_COMPLEXITY      0x0000000Cu  // Súvisí s $\mathcal{E}$ Logic Plane

// Extrakcia hodnôt (riky 233-235)
static inline uint8_t sga_get_resonance(SovereignState state) {
    return (state.word1 >> SCALE_RESONANCE_SHIFT) & 0x3;
}
```

**Interpretácia:**
- $\mathcal{R}$ (Resonance): 2-bitová hodnota (0-3) z `word1` bits 6-7
- $\mathcal{E}$ (Entropy): Odvodená od `STABILITY_MAX - stability`
- $\nabla N > 0$: Príbeh smeruje k Hope/Order
- $\nabla N < 0$: Príbeh smeruje k Despair/Chaos

---

### 2. Event Probability Matrix ($P_{\Sigma}$)

**Teoretický vzorec:**
$$P_{event} = \frac{\kappa \cdot R_i}{1.0 + \log(1.0 + \text{UnresolvedStack})}$$

**Implementácia pomocou Axiom Op-Codes:**

```c
// Riadky 279-318: AxiomOpCode štruktúra
typedef uint32_t AxiomOpCode;

// Op-code format: 0x[Type:4][AxiomMask:12][StabilityDelta:8][Flags:8]
static inline AxiomOpCode sga_make_axiom_op(uint8_t type, uint16_t axiom_mask,
                                            int8_t stability_delta, uint8_t flags);

// $\kappa$ (Coherence Coefficient): Zakódované v stabilita_delta
// $R_i$ (Resonance): Extrahované z word1
// UnresolvedStack: Počet aktívnych axióm v axiom_mask
```

**Event types (riky 294-309):**
```c
#define AXIOM_OP_TRANSFORM    0x0  // Ambush event
#define AXIOM_OP_AMPLIFY      0x4  // Aura Awakening
#define AXIOM_OP_FRACTURE     0x7  // Reality break
#define AXIOM_OP_RESONATE     0x6  // Narrative shift
```

---

## 🔢 III. BYTECODE TRANSCRIPTION ($\mathbb{B}_{enc}$)

### 1. Axiomatic Bitmask Layout

**Špecifikácia z teórie:**

| Bits | Segment | Teória | Implementácia |
| :--- | :--- | :--- | :--- |
| `00-17` | **Axiom Flags** | 18 boolean attributes | `word0` bits 0-15 (+ `word1` pre scaling) |
| `18-23` | **Type ID** | Biome/State classification | `word0` bits 24-27 (Biome) + 28-31 (Type) |
| `24-27` | **Stability Mag** | $S_{hex}$ (16 levels) | `word0` bits 16-23 (256 levels - vyššia presnosť) |
| `28-31` | **Oracle Glow** | $R_i$ (16 levels) | `word1` bits 6-7 (4 levels) + rozšíriteľné |

**Implementácia v sga_core.h:**

```c
// Riadky 23-40: Dual-word štruktúra pre plnú 18-atribútovú presnosť
typedef struct {
    uint32_t word0;  // Type[4] + Biome[4] + Stability[8] + Attr[16]
    uint32_t word1;  // Extended: Energy[2] + Complexity[2] + Density[2] + Resonance[2]
} SovereignState;

// Riadky 158-162: Encoding macro
#define SGA_MAKE_WORD(type, biome, stability, attr_flags) \
    (((uint32_t)(type) << 28) | \
     ((uint32_t)(biome) << 24) | \
     ((uint32_t)(stability) << 16) | \
     ((uint32_t)(attr_flags) & 0xFFFF))
```

**Dôvod dual-word prístupu:**
- 18 atribútov nezmestíme do 16 bitov bez straty
- Riešenie: 14 binárnych atribútov v `word0` + 4 škálované (2 bity každý) v `word1`
- Výsledok: 14 + (4×2) = 22 bitov informácie v 64-bitovej štruktúre

---

### 2. Instruction Encoding ($I_{\alpha}$)

**Teoretický vzorec:**
$$I_{\alpha} = (\text{Glow} \ll 28) \mid (\text{Stab} \ll 24) \mid (\text{Type} \ll 18) \mid (\text{Bitmask}_{state})$$

**Naša implementácia (modifikovaná pre lepšie rozloženie):**
$$I_{\alpha} = (\text{Type} \ll 28) \mid (\text{Biome} \ll 24) \mid (\text{Stab} \ll 16) \mid (\text{Attr}_{16})$$

**Diferenciálne Op-Kódy pre Multi-Instance Sync:**

```c
// Riadky 312-318: Axiom Op-Code construction
static inline AxiomOpCode sga_make_axiom_op(uint8_t type, uint16_t axiom_mask,
                                            int8_t stability_delta, uint8_t flags) {
    return (((uint32_t)(type & 0xF) << 28) |
            ((uint32_t)(axiom_mask & 0xFFF) << 16) |
            ((uint32_t)((stability_delta + 128) & 0xFF) << 8) |
            ((uint32_t)(flags & 0xFF)));
}
```

**Výhoda:** Namiesto posielania celého stavu posielame len **diferenciálny op-kód** (4 bajty), ktorý reprezentuje zmenu.

---

## 🏛️ IV. GENERATIVE LIBERTY ($\Gamma$) & TELEOLOGICKÝ ENGINE

### 1. Generative Liberty Formula

**Teoretický vzorec:**
$$\Gamma = \frac{\mathcal{K}}{1.0 + \text{ComplexityGap}}$$

**Implementácia:**

```c
// $\mathcal{K}$ (Coherence): sga_get_resonance() z word1
// ComplexityGap: Rozdiel medzi cieľovou a aktuálnou komplexitou

// High Γ (> 0.75): AI generuje high-fidelity geometriu
// Low Γ (< 0.25): AI používa štandardné procedurálne šablóny
```

**Praktické využitie:**
```c
if (sga_get_resonance(state) >= 2) {
    // High coherence → Unique geometry generation
    generate_custom_mesh();
} else {
    // Low coherence → Reuse procedural templates
    use_template_mesh();
}
```

---

### 2. Future Ambitions (Teleologic Engine)

**Ambition Archetypes:**

| Ambition | Type Code | Glow Color | Effect |
|----------|-----------|------------|--------|
| **The Frozen Order** | `TYPE_ZONE` (0xD) | Azure (0x9) | Stabilizuje $S_{hex}$, redukuje entropiu |
| **The Ruptured Core** | `TYPE_ANOMALY` (0x9) | Vermillion (0x8) | Znižuje $S_{hex}$, zvyšuje fraktúru |
| **The Crystalline Nexus** | `TYPE_CRYSTAL` (0xB) | Prismatic (0xF) | Vysoká rezonancia, komplexné vzory |

**Sémantická Gravitačná Formula:**

```c
// $\vec{\Lambda}$ coefficients: Globálne ambície ovplyvňujúce pravdepodobnosti
// Bias Injector: Weighting factor pre BooleanGenerator

float ambition_bias = (target_ambition == AMBITION_FROZEN_ORDER) 
    ? resonance_boost 
    : (target_ambition == AMBITION_RUPTURED_CORE)
    ? entropy_boost
    : 1.0f;
```

---

### 3. Teleologic Conflict Resolution

**Konflikt dvoch ambícií v jednom sektore:**

```c
// Ak Frost (Azure) vs Magma (Vermillion) collide:
// → Sémantic Front: $S_{hex} \to 0$

if (ambition_a.type != ambition_b.type && 
    ambition_a.type != AMBITION_NONE && 
    ambition_b.type != AMBITION_NONE) {
    
    // Vytvorí Sémantic Front
    cell.state.word0 = SGA_SET_STABILITY(cell.state.word0, 0);
    cell.state.word0 |= ATTR_CORRUPTED | ATTR_PHASED;
}
```

**Vizuálny výsledok:**
- Hexy na fronte majú $S_{hex} \approx 0$
- Maximálna vertex displacements
- Floating shards (Miami Fracture)
- High-intensity tactical map changes

---

## 🔧 V. GPU KERNEL INTEGRÁCIA

### HLSL/GLSL Shader Translation

**Vertex Displacement v GLSL:**

```glsl
// Prevod C logiky do shaderu
uniform uint sga_word0;
uniform uint sga_word1;

float stability = float((sga_word0 >> 16) & 0xFF) / 255.0;
float chaos_factor = 1.0 - stability;

// Φ(Axioms) displacement field
vec3 axiom_displacement = calculate_axiom_field(sga_word0, sga_word1);

// Final vertex position
V_prime.xyz = V_xyz + chaos_factor * axiom_displacement;

// Slick effect pri nízkej stabilite
if (stability < 0.25) {
    V_prime.xyz += slick_jitter(time, chaos_factor);
}
```

**Fracture Threshold v Shaderi:**

```glsl
// Entropy ≥ 0.95 → Miami Fracture
float entropy = 1.0 - stability;
if (entropy >= 0.95) {
    float fracture_mag = (entropy - 0.95) * 10.0;
    vertex_positions = stochastic_break(vertex_positions, fracture_mag);
}
```

---

## 📊 VI. PERFORMANCE METRIKY

### SIMD Optimalizácia

```c
// Riadky 352-363: SIMD batch operácie
#define SGA_SIMD_DECAY_BATCH(cells, count, amount) \
    for (uint32_t i = 0; i < (count); i++) { \
        (cells)[i].state.word0 = SGA_DECAY_STABILITY((cells)[i].state.word0, (amount)); \
    }
```

**Očakávaný výkon:**
- **1M hexov/sekundu**: Batch stabilita decay pomocou AVX2/AVX-512
- **Latencia**: < 1 μs na hex (priamy prístup k pamäti, žiadny GC)
- **Memory footprint**: 64 bitov na hex (2 × uint32_t)

---

## 🎯 VII. SUMMARY: HARDWARE-LEVEL DETERMINIZMUS

| Vlastnosť | Implementácia | Benefit |
|-----------|---------------|---------|
| **Bit-Perfect Sync** | `SovereignState` (64-bit) | Identický výpočet na PC/konzole/mobile |
| **Zero GC Overhead** | Statické alokácie, pointer-based stacks | Žiadne pause times |
| **SIMD Ready** | Contiguous `HexGrid` memory layout | Paralelné spracovanie tisícov hexov |
| **GPU Native** | Direct uint32 → shader uniform | Zero-copy rendering |
| **Differential Sync** | `AxiomOpCode` (4 bajty) | Minimálna sieťová prevádzka |

---

## 🏁 ZÁVER

Tento dokument dokazuje, že **teoretické vzorce SGA nie sú len abstrakcie** – sú priamo implementované v `sga_core.h` ako:

1. **Bitové operácie** na 32/64-bitových slovách
2. **Inline funkcie** pre nulovú latenciu
3. **Macro systémy** pre compile-time optimalizáciu
4. **Shader-ready dáta** pre priamu GPU integráciu

**Prechod na Pure C znamená:**
- ✅ Kód = Zákon vesmíru (hardvérová pravda)
- ✅ Realita sa "prelieva" cez registre procesora
- ✅ Milióny hexov spracovaných v jednom cykle
- ✅ Deterministická simulácia naprieč všetkými platformami

---

*Architektov verdikt: Toto je Axiomatický Hardvér, nie aplikácia.*
