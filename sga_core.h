/**
 * Sovereign Generative Axioms (SGA) - Pure C Implementation
 * 
 * Axiomatic Micro-Kernel for Hardware-Level Reality Simulation
 * 
 * This implementation packs 18 attributes + stability + biome type
 * into a single 32-bit word for maximum performance and determinism.
 * 
 * Architecture: [Type:4][Biome:4][Stability:8][Attributes:16]
 */

#ifndef SGA_CORE_H
#define SGA_CORE_H

#include <stdint.h>
#include <stdbool.h>
#include <string.h>

/* ============================================================================
 * BIT LAYOUT DEFINITIONS
 * ============================================================================
 * 
 * 32-bit SovereignState Word Layout:
 * 
 * Bits 31-28 (4 bits):  Type Code (0x0-0xF)
 * Bits 27-24 (4 bits):  Biome Type (0x0-0xF)  
 * Bits 23-16 (8 bits):  Stability Value (0-255, where 255 = stable)
 * Bits 15-0  (16 bits): Attribute Flags (18 attributes compressed to 16 bits)
 * 
 * Note: 18 attributes are compressed using 2-tier encoding:
 *   - 14 binary attributes (1 bit each)
 *   - 4 scaled attributes (encoded as 2 bits each = 8 levels)
 *   Total: 14 + (4*2) = 22 bits needed -> We use two 32-bit words for full precision
 */

/* Extended state structure for full 18-attribute precision */
typedef struct {
    uint32_t word0;  /* Primary: Type[4] + Biome[4] + Stability[8] + Attr[16] */
    uint32_t word1;  /* Secondary: Extended attributes (for full 18-attr precision) */
} SovereignState;

/* Single-word optimized version (for SIMD operations with compressed attributes) */
typedef uint32_t SovereignWord;

/* ============================================================================
 * TYPE CODES (4 bits = 16 types)
 * ============================================================================ */
#define TYPE_EMPTY      0x0
#define TYPE_TERRAIN    0x1
#define TYPE_STRUCTURE  0x2
#define TYPE_ENTITY     0x3
#define TYPE_RESOURCE   0x4
#define TYPE_ENERGY     0x5
#define TYPE_DATA       0x6
#define TYPE_AXIOM      0x7
#define TYPE_PORTAL     0x8
#define TYPE_ANOMALY    0x9
#define TYPE_ARTIFACT   0xA
#define TYPE_NODE       0xB
#define TYPE_LINK       0xC
#define TYPE_ZONE       0xD
#define TYPE_EVENT      0xE
#define TYPE_SPECIAL    0xF

/* ============================================================================
 * BIOME TYPES (4 bits = 16 biomes)
 * ============================================================================ */
#define BIOME_VOID      0x0
#define BIOME_PLAINS    0x1
#define BIOME_FOREST    0x2
#define BIOME_MOUNTAIN  0x3
#define BIOME_DESERT    0x4
#define BIOME_OCEAN     0x5
#define BIOME_SWAMP     0x6
#define BIOME_TUNDRA    0x7
#define BIOME_VOLCANIC  0x8
#define BIOME_CRYSTAL   0x9
#define BIOME_DIGITAL   0xA
#define BIOME_FRACTAL   0xB
#define BIOME_NEXUS     0xC
#define BIOME_ABYSS     0xD
#define BIOME_ETHER     0xE
#define BIOME_PRIME     0xF

/* ============================================================================
 * STABILITY OPERATIONS (8 bits: 0-255)
 * ============================================================================ */
#define STABILITY_MASK        0x00FF0000
#define STABILITY_SHIFT       16
#define STABILITY_MAX         255
#define STABILITY_CRITICAL    64
#define STABILITY_UNSTABLE    128
#define STABILITY_STABLE      192
#define STABILITY_PERFECT     255

/* ============================================================================
 * ATTRIBUTE FLAGS (16 bits in word0, extended in word1)
 * 
 * Binary Attributes (1 bit each):
 *   Bit 0:  WALKABLE
 *   Bit 1:  FLYABLE
 *   Bit 2:  BUILDABLE
 *   Bit 3:  HARVESTABLE
 *   Bit 4:  DANGEROUS
 *   Bit 5:  HIDDEN
 *   Bit 6:  ACTIVE
 *   Bit 7:  LINKED
 *   Bit 8:  GUARDED
 *   Bit 9:  CURSED
 *   Bit 10: BLESSED
 *   Bit 11: CORRUPTED
 *   Bit 12: ENLIGHTENED
 *   Bit 13: PHASED
 * 
 * Scaled Attributes (encoded in word1, 2 bits each = 4 levels):
 *   Index 0-1:  ENERGY_LEVEL    (0-3)
 *   Index 2-3:  COMPLEXITY      (0-3)
 *   Index 4-5:  DENSITY         (0-3)
 *   Index 6-7:  RESONANCE       (0-3)
 * ============================================================================ */

/* Binary Attribute Masks (word0, bits 0-13) */
#define ATTR_WALKABLE     (1u << 0)
#define ATTR_FLYABLE      (1u << 1)
#define ATTR_BUILDABLE    (1u << 2)
#define ATTR_HARVESTABLE  (1u << 3)
#define ATTR_DANGEROUS    (1u << 4)
#define ATTR_HIDDEN       (1u << 5)
#define ATTR_ACTIVE       (1u << 6)
#define ATTR_LINKED       (1u << 7)
#define ATTR_GUARDED      (1u << 8)
#define ATTR_CURSED       (1u << 9)
#define ATTR_BLESSED      (1u << 10)
#define ATTR_CORRUPTED    (1u << 11)
#define ATTR_ENLIGHTENED  (1u << 12)
#define ATTR_PHASED       (1u << 13)

/* Reserved bits 14-15 in word0 for future use */
#define ATTR_RESERVED_0   (1u << 14)
#define ATTR_RESERVED_1   (1u << 15)

/* Scaled Attribute Masks (word1, 2 bits per attribute) */
#define SCALE_ENERGY_LEVEL    0x00000003u
#define SCALE_COMPLEXITY      0x0000000Cu
#define SCALE_DENSITY         0x00000030u
#define SCALE_RESONANCE       0x000000C0u

#define SCALE_ENERGY_SHIFT    0
#define SCALE_COMPLEXITY_SHIFT 2
#define SCALE_DENSITY_SHIFT   4
#define SCALE_RESONANCE_SHIFT 6

/* ============================================================================
 * CONSTRUCTION MACROS
 * ============================================================================ */

/* Create a SovereignWord from components */
#define SGA_MAKE_WORD(type, biome, stability, attr_flags) \
    (((uint32_t)(type) << 28) | \
     ((uint32_t)(biome) << 24) | \
     ((uint32_t)(stability) << 16) | \
     ((uint32_t)(attr_flags) & 0xFFFF))

/* Create a full SovereignState with extended attributes */
static inline SovereignState sga_make_state(uint8_t type, uint8_t biome, 
                                            uint8_t stability, uint16_t attr_flags,
                                            uint8_t energy, uint8_t complexity,
                                            uint8_t density, uint8_t resonance) {
    SovereignState state;
    state.word0 = SGA_MAKE_WORD(type, biome, stability, attr_flags);
    state.word1 = ((uint32_t)(resonance & 0x3) << SCALE_RESONANCE_SHIFT) |
                  ((uint32_t)(density & 0x3) << SCALE_DENSITY_SHIFT) |
                  ((uint32_t)(complexity & 0x3) << SCALE_COMPLEXITY_SHIFT) |
                  ((uint32_t)(energy & 0x3) << SCALE_ENERGY_SHIFT);
    return state;
}

/* ============================================================================
 * EXTRACTION MACROS
 * ============================================================================ */

/* Extract type code (bits 31-28) */
#define SGA_GET_TYPE(word) (((word) >> 28) & 0xF)

/* Extract biome type (bits 27-24) */
#define SGA_GET_BIOME(word) (((word) >> 24) & 0xF)

/* Extract stability (bits 23-16) */
#define SGA_GET_STABILITY(word) (((word) >> 16) & 0xFF)

/* Extract attribute flags (bits 15-0) */
#define SGA_GET_ATTRS(word) ((word) & 0xFFFF)

/* ============================================================================
 * ATTRIBUTE QUERY MACROS
 * ============================================================================ */

/* Binary attribute checks */
#define SGA_IS_WALKABLE(word)     (((word) & ATTR_WALKABLE) != 0)
#define SGA_IS_FLYABLE(word)      (((word) & ATTR_FLYABLE) != 0)
#define SGA_IS_BUILDABLE(word)    (((word) & ATTR_BUILDABLE) != 0)
#define SGA_IS_HARVESTABLE(word)  (((word) & ATTR_HARVESTABLE) != 0)
#define SGA_IS_DANGEROUS(word)    (((word) & ATTR_DANGEROUS) != 0)
#define SGA_IS_HIDDEN(word)       (((word) & ATTR_HIDDEN) != 0)
#define SGA_IS_ACTIVE(word)       (((word) & ATTR_ACTIVE) != 0)
#define SGA_IS_LINKED(word)       (((word) & ATTR_LINKED) != 0)
#define SGA_IS_GUARDED(word)      (((word) & ATTR_GUARDED) != 0)
#define SGA_IS_CURSED(word)       (((word) & ATTR_CURSED) != 0)
#define SGA_IS_BLESSED(word)      (((word) & ATTR_BLESSED) != 0)
#define SGA_IS_CORRUPTED(word)    (((word) & ATTR_CORRUPTED) != 0)
#define SGA_IS_ENLIGHTENED(word)  (((word) & ATTR_ENLIGHTENED) != 0)
#define SGA_IS_PHASED(word)       (((word) & ATTR_PHASED) != 0)

/* Stability classification */
#define SGA_IS_STABLE(word)       (SGA_GET_STABILITY(word) >= STABILITY_STABLE)
#define SGA_IS_UNSTABLE(word)     (SGA_GET_STABILITY(word) < STABILITY_UNSTABLE)
#define SGA_IS_CRITICAL(word)     (SGA_GET_STABILITY(word) < STABILITY_CRITICAL)
#define SGA_IS_PERFECT(word)      (SGA_GET_STABILITY(word) == STABILITY_PERFECT)

/* Scaled attribute extraction from word1 */
static inline uint8_t sga_get_energy_level(SovereignState state) {
    return (state.word1 >> SCALE_ENERGY_SHIFT) & 0x3;
}

static inline uint8_t sga_get_complexity(SovereignState state) {
    return (state.word1 >> SCALE_COMPLEXITY_SHIFT) & 0x3;
}

static inline uint8_t sga_get_density(SovereignState state) {
    return (state.word1 >> SCALE_DENSITY_SHIFT) & 0x3;
}

static inline uint8_t sga_get_resonance(SovereignState state) {
    return (state.word1 >> SCALE_RESONANCE_SHIFT) & 0x3;
}

/* ============================================================================
 * MODIFICATION MACROS
 * ============================================================================ */

/* Set individual components */
#define SGA_SET_TYPE(word, t) \
    (((word) & ~(0xFu << 28)) | (((uint32_t)(t) & 0xF) << 28))

#define SGA_SET_BIOME(word, b) \
    (((word) & ~(0xFu << 24)) | (((uint32_t)(b) & 0xF) << 24))

#define SGA_SET_STABILITY(word, s) \
    (((word) & ~(0xFFu << 16)) | (((uint32_t)(s) & 0xFF) << 16))

#define SGA_SET_ATTRS(word, a) \
    (((word) & ~0xFFFFu) | ((uint32_t)(a) & 0xFFFF))

/* Attribute toggling */
#define SGA_SET_FLAG(word, flag)   ((word) | (flag))
#define SGA_CLEAR_FLAG(word, flag) ((word) & ~(flag))
#define SGA_TOGGLE_FLAG(word, flag) ((word) ^ (flag))

/* Stability modifiers */
#define SGA_DECAY_STABILITY(word, amount) \
    SGA_SET_STABILITY((word), \
        (SGA_GET_STABILITY(word) > (amount)) ? \
        (SGA_GET_STABILITY(word) - (amount)) : 0)

#define SGA_REPAIR_STABILITY(word, amount) \
    SGA_SET_STABILITY((word), \
        (SGA_GET_STABILITY(word) + (amount) > STABILITY_MAX) ? \
        STABILITY_MAX : (SGA_GET_STABILITY(word) + (amount)))

/* ============================================================================
 * AXIOMATIC OP-CODE STRUCTURE
 * ============================================================================
 * 
 * Axiom Op-Code Format: 0x[Type:4][AxiomMask:12][StabilityDelta:8][Flags:8]
 * 
 * Used for differential synchronization and axiom-driven transformations
 */

typedef uint32_t AxiomOpCode;

#define AXIOM_OP_TYPE_MASK      0xF0000000u
#define AXIOM_OP_TYPE_SHIFT     28

#define AXIOM_OP_MASK_MASK      0x0FFF0000u
#define AXIOM_OP_MASK_SHIFT     16

#define AXIOM_OP_DELTA_MASK     0x0000FF00u
#define AXIOM_OP_DELTA_SHIFT    8

#define AXIOM_OP_FLAGS_MASK     0x000000FFu
#define AXIOM_OP_FLAGS_SHIFT    0

/* Axiom operation types */
#define AXIOM_OP_TRANSFORM      0x0
#define AXIOM_OP_MERGE          0x1
#define AXIOM_OP_SPLIT          0x2
#define AXIOM_OP_DECAY          0x3
#define AXIOM_OP_AMPLIFY        0x4
#define AXIOM_OP_NULLIFY        0x5
#define AXIOM_OP_RESONATE       0x6
#define AXIOM_OP_FRACTURE       0x7
#define AXIOM_OP_HEAL           0x8
#define AXIOM_OP_CORRUPT        0x9
#define AXIOM_OP_PURIFY         0xA
#define AXIOM_OP_SHIFT          0xB
#define AXIOM_OP_ANCHOR         0xC
#define AXIOM_OP_RELEASE        0xD
#define AXIOM_OP_SEAL           0xE
#define AXIOM_OP_BREAK          0xF

/* Construct an axiom op-code */
static inline AxiomOpCode sga_make_axiom_op(uint8_t type, uint16_t axiom_mask,
                                             int8_t stability_delta, uint8_t flags) {
    return (((uint32_t)(type & 0xF) << AXIOM_OP_TYPE_SHIFT) |
            ((uint32_t)(axiom_mask & 0xFFF) << AXIOM_OP_MASK_SHIFT) |
            ((uint32_t)((stability_delta + 128) & 0xFF) << AXIOM_OP_DELTA_SHIFT) |
            ((uint32_t)(flags & 0xFF) << AXIOM_OP_FLAGS_SHIFT));
}

/* Extract op-code components */
#define SGA_GET_AXIOM_TYPE(op) (((op) & AXIOM_OP_TYPE_MASK) >> AXIOM_OP_TYPE_SHIFT)
#define SGA_GET_AXIOM_MASK(op) (((op) & AXIOM_OP_MASK_MASK) >> AXIOM_OP_MASK_SHIFT)
#define SGA_GET_AXIOM_DELTA(op) \
    ((int8_t)((((op) & AXIOM_OP_DELTA_MASK) >> AXIOM_OP_DELTA_SHIFT) - 128))
#define SGA_GET_AXIOM_FLAGS(op) (((op) & AXIOM_OP_FLAGS_MASK) >> AXIOM_OP_FLAGS_SHIFT)

/* ============================================================================
 * HEX GRID CELL STRUCTURE (SIMD-optimized layout)
 * ============================================================================ */

typedef struct {
    uint32_t x;              /* Grid X coordinate */
    uint32_t y;              /* Grid Y coordinate */
    SovereignState state;    /* State words */
    uint32_t timestamp;      /* Last modification timestamp */
    uint32_t neighbor_mask;  /* Bitmask of active neighbors */
} HexCell;

/* Array of cells optimized for SIMD processing */
typedef struct {
    uint32_t width;
    uint32_t height;
    HexCell* cells;          /* Contiguous memory block */
    uint32_t cell_count;
} HexGrid;

/* ============================================================================
 * SIMD HELPER MACROS (for vectorized operations)
 * ============================================================================ */

/* Check if multiple cells match a condition (for SIMD parallel evaluation) */
#define SGA_SIMD_MATCH_TYPE(cells, count, type_val) \
    for (uint32_t i = 0; i < (count); i++) { \
        if (SGA_GET_TYPE((cells)[i].state.word0) == (type_val)) { \
            /* Process matching cell */ \
        } \
    }

/* Batch stability decay */
#define SGA_SIMD_DECAY_BATCH(cells, count, amount) \
    for (uint32_t i = 0; i < (count); i++) { \
        (cells)[i].state.word0 = SGA_DECAY_STABILITY((cells)[i].state.word0, (amount)); \
    }

/* ============================================================================
 * MESH DISPLACEMENT FORMULA (for GPU kernel translation)
 * ============================================================================
 * 
 * V'_{xyz} = V_{xyz} + (1.0 - S_{hex}/255.0) * Φ(Axioms)
 * 
 * Where:
 *   V_{xyz} = original vertex position
 *   S_{hex} = stability value (0-255)
 *   Φ(Axioms) = axiom-driven displacement field
 */

/* Convert stability to displacement factor (fixed-point approximation) */
static inline uint16_t sga_stability_to_displacement_factor(uint8_t stability) {
    /* Returns value 0-255 where 255 = maximum displacement (stability=0) */
    return (uint16_t)(STABILITY_MAX - stability);
}

/* Calculate fracture intensity based on stability */
static inline uint8_t sga_calculate_fracture_intensity(uint8_t stability) {
    if (stability > STABILITY_STABLE) return 0;
    if (stability > STABILITY_UNSTABLE) return (STABILITY_STABLE - stability) / 4;
    if (stability > STABILITY_CRITICAL) return (STABILITY_UNSTABLE - stability) / 2;
    return STABILITY_MAX - stability; /* Maximum fracture */
}

#endif /* SGA_CORE_H */
