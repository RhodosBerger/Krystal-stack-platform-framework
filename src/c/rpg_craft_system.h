/**
 * RPG Craft System - Performance as Craft Engine
 *
 * Manages performance tuning through RPG-style crafting metaphors:
 * - Recipes: Predefined optimization combinations
 * - Inventories: Available boosts/presets
 * - Pattern Planners: Timer/cron/conditional scheduling
 * - Boost Maps: Zone-to-boost relationships
 * - Skill Trees: Unlockable optimization paths
 *
 * Directly wired into Thread Boost Layer to align crafted presets with active zones.
 */

#ifndef GAMESA_RPG_CRAFT_SYSTEM_H
#define GAMESA_RPG_CRAFT_SYSTEM_H

#include <stdint.h>
#include <stdbool.h>
#include "thread_boost_layer.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========== Configuration ========== */

#define GAMESA_MAX_RECIPES      128
#define GAMESA_MAX_INGREDIENTS  16
#define GAMESA_MAX_INVENTORY    256
#define GAMESA_MAX_PATTERNS     64
#define GAMESA_MAX_SKILLS       32

/* ========== Recipe System ========== */

/** Ingredient types for crafting */
typedef enum {
    INGREDIENT_CPU_BOOST,       /* CPU frequency/cores */
    INGREDIENT_GPU_BOOST,       /* GPU clock/power */
    INGREDIENT_MEMORY_TIER,     /* Memory allocation tier */
    INGREDIENT_THERMAL_LIMIT,   /* Temperature threshold */
    INGREDIENT_POWER_BUDGET,    /* Power envelope */
    INGREDIENT_LATENCY_TARGET,  /* Target frametime */
    INGREDIENT_ZONE_PRIORITY,   /* Zone signal strength */
} gamesa_ingredient_type_t;

/** Single ingredient in a recipe */
typedef struct {
    gamesa_ingredient_type_t type;
    float value;                /* Ingredient strength/amount */
    uint8_t quality;            /* 1-5 quality tier */
} gamesa_ingredient_t;

/** Crafting recipe */
typedef struct {
    uint32_t recipe_id;
    char name[64];
    gamesa_ingredient_t ingredients[GAMESA_MAX_INGREDIENTS];
    uint8_t ingredient_count;
    uint8_t difficulty;         /* Craft difficulty (affects success) */
    uint8_t tier;               /* Output quality tier */
    uint32_t cooldown_ms;       /* Cooldown before reuse */
    uint64_t last_craft_ns;
} gamesa_recipe_t;

/** Craft result (applied preset) */
typedef struct {
    uint32_t preset_id;
    uint32_t recipe_id;
    float effectiveness;        /* 0.0-1.0 based on craft quality */
    uint64_t duration_ns;       /* How long preset is active */
    uint64_t created_ns;
    bool active;
} gamesa_craft_result_t;

/* ========== Inventory System ========== */

/** Inventory item (available boosts) */
typedef struct {
    uint32_t item_id;
    gamesa_ingredient_type_t type;
    float value;
    uint8_t quality;
    uint32_t quantity;          /* Stack count */
    bool consumable;            /* Used up on craft */
} gamesa_inventory_item_t;

/** Player/system inventory */
typedef struct {
    gamesa_inventory_item_t items[GAMESA_MAX_INVENTORY];
    uint32_t item_count;
    uint32_t capacity;
} gamesa_inventory_t;

/* ========== Pattern Planners ========== */

/** Pattern trigger types */
typedef enum {
    PATTERN_TIMER,              /* Interval-based */
    PATTERN_CRON,               /* Cron expression */
    PATTERN_CONDITIONAL,        /* Telemetry condition */
    PATTERN_EVENT,              /* External event trigger */
} gamesa_pattern_type_t;

/** Condition for conditional patterns */
typedef struct {
    char variable[32];          /* Telemetry var name */
    char op;                    /* '<', '>', '=', '!' */
    float threshold;
} gamesa_condition_t;

/** Pattern planner entry */
typedef struct {
    uint32_t pattern_id;
    gamesa_pattern_type_t type;
    union {
        uint32_t interval_ms;           /* For TIMER */
        char cron_expr[32];             /* For CRON */
        gamesa_condition_t condition;   /* For CONDITIONAL */
        uint32_t event_id;              /* For EVENT */
    } trigger;
    uint32_t recipe_id;         /* Recipe to execute */
    uint32_t target_zone;       /* Zone to apply (0=all) */
    bool enabled;
    uint64_t last_trigger_ns;
} gamesa_pattern_t;

/* ========== Boost Maps ========== */

/** Boost map entry (zone → preset mapping) */
typedef struct {
    uint32_t zone_id;
    uint32_t preset_id;
    float weight;               /* Blend weight for multiple presets */
    uint8_t priority;
} gamesa_boost_entry_t;

/** Voltage curve point for driver generation */
typedef struct {
    uint32_t frequency_mhz;
    uint32_t voltage_mv;
} gamesa_voltage_point_t;

/** Generated driver preset */
typedef struct {
    uint32_t driver_id;
    char name[64];
    gamesa_voltage_point_t curve[16];
    uint8_t curve_points;
    float power_limit_w;
    float thermal_limit_c;
} gamesa_driver_preset_t;

/* ========== Skill Trees ========== */

/** Skill node in optimization tree */
typedef struct {
    uint32_t skill_id;
    char name[64];
    char description[256];
    uint8_t tier;               /* Skill tier/level */
    uint32_t prerequisites[4];  /* Required skill IDs */
    uint8_t prereq_count;
    bool unlocked;
    /* Bonus effects when unlocked */
    float cpu_boost_bonus;
    float gpu_boost_bonus;
    float thermal_headroom_bonus;
    float efficiency_bonus;
} gamesa_skill_t;

/** Skill tree */
typedef struct {
    gamesa_skill_t skills[GAMESA_MAX_SKILLS];
    uint32_t skill_count;
    uint32_t skill_points;      /* Available points to spend */
    uint32_t total_xp;          /* Experience from successful optimizations */
} gamesa_skill_tree_t;

/* ========== API Functions ========== */

/** Initialize craft system */
int gamesa_craft_init(void);

/** Shutdown craft system */
void gamesa_craft_shutdown(void);

/* Recipe Management */
int gamesa_recipe_register(const gamesa_recipe_t* recipe);
gamesa_recipe_t* gamesa_recipe_get(uint32_t recipe_id);
int gamesa_recipe_craft(uint32_t recipe_id, gamesa_craft_result_t* out);

/* Inventory */
int gamesa_inventory_add(const gamesa_inventory_item_t* item);
int gamesa_inventory_remove(uint32_t item_id, uint32_t quantity);
gamesa_inventory_t* gamesa_inventory_get(void);

/* Pattern Planners */
int gamesa_pattern_register(const gamesa_pattern_t* pattern);
int gamesa_pattern_enable(uint32_t pattern_id, bool enabled);
int gamesa_pattern_tick(uint64_t current_ns);  /* Process patterns */

/* Boost Maps */
int gamesa_boost_map_set(uint32_t zone_id, uint32_t preset_id, float weight);
int gamesa_boost_map_apply(uint32_t zone_id);

/* Driver Generation */
int gamesa_driver_generate(
    const gamesa_recipe_t* base_recipe,
    gamesa_driver_preset_t* out
);
int gamesa_driver_apply(uint32_t driver_id);

/* Skill Trees */
int gamesa_skill_unlock(uint32_t skill_id);
int gamesa_skill_add_xp(uint32_t xp);
gamesa_skill_tree_t* gamesa_skill_tree_get(void);
float gamesa_skill_get_bonus(gamesa_ingredient_type_t type);

/* Integration with Thread Boost Layer */
int gamesa_craft_apply_to_zone(uint32_t zone_id, uint32_t preset_id);
int gamesa_craft_sync_zones(void);  /* Sync all boost maps to zones */

#ifdef __cplusplus
}
#endif

#endif /* GAMESA_RPG_CRAFT_SYSTEM_H */
