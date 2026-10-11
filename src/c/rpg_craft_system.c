/**
 * RPG Craft System - Performance as Recipes
 */

#include "rpg_craft_system.h"
#include <stdlib.h>
#include <string.h>

static rpg_recipe_t g_recipes[MAX_RECIPES];
static uint32_t g_recipe_count = 0;
static rpg_boost_map_t g_boost_maps[MAX_BOOST_MAPS];
static uint32_t g_boost_map_count = 0;

int rpg_craft_init(void) {
    memset(g_recipes, 0, sizeof(g_recipes));
    memset(g_boost_maps, 0, sizeof(g_boost_maps));
    g_recipe_count = 0;
    g_boost_map_count = 0;
    return 0;
}

void rpg_craft_shutdown(void) {
    g_recipe_count = 0;
    g_boost_map_count = 0;
}

uint32_t rpg_recipe_create(rpg_recipe_config_t *config) {
    if (g_recipe_count >= MAX_RECIPES || !config) return (uint32_t)-1;

    rpg_recipe_t *recipe = &g_recipes[g_recipe_count];
    recipe->recipe_id = g_recipe_count;
    strncpy(recipe->name, config->name, sizeof(recipe->name) - 1);
    recipe->rarity = config->rarity;
    recipe->ingredient_count = config->ingredient_count;

    for (uint32_t i = 0; i < config->ingredient_count && i < MAX_INGREDIENTS; i++) {
        recipe->ingredients[i] = config->ingredients[i];
    }

    return g_recipe_count++;
}

rpg_preset_t rpg_recipe_craft(uint32_t recipe_id, rpg_craft_params_t *params) {
    rpg_preset_t preset = {0};
    if (recipe_id >= g_recipe_count) return preset;

    rpg_recipe_t *recipe = &g_recipes[recipe_id];
    preset.recipe_id = recipe_id;

    // Combine ingredients based on params
    float quality = params ? params->quality_target : 0.8f;
    preset.cpu_clock_offset = (int)(50 * quality);
    preset.gpu_clock_offset = (int)(100 * quality);
    preset.power_limit_pct = 80 + (int)(20 * quality);
    preset.fan_curve_id = quality > 0.9f ? 2 : 1;
    preset.thermal_target = 80 - (int)(10 * (1.0f - quality));

    return preset;
}

uint32_t rpg_boost_map_create(const char *name) {
    if (g_boost_map_count >= MAX_BOOST_MAPS || !name) return (uint32_t)-1;

    rpg_boost_map_t *map = &g_boost_maps[g_boost_map_count];
    map->map_id = g_boost_map_count;
    strncpy(map->name, name, sizeof(map->name) - 1);

    return g_boost_map_count++;
}

int rpg_boost_map_add_zone(uint32_t map_id, uint32_t zone_id, rpg_zone_params_t *params) {
    if (map_id >= g_boost_map_count || !params) return -1;

    rpg_boost_map_t *map = &g_boost_maps[map_id];
    if (map->zone_count >= 16) return -1;

    map->zones[map->zone_count].zone_id = zone_id;
    map->zones[map->zone_count].boost_factor = params->boost_factor;
    map->zones[map->zone_count].priority = params->priority;
    map->zone_count++;

    return 0;
}

int rpg_boost_map_apply(uint32_t map_id) {
    if (map_id >= g_boost_map_count) return -1;
    // Would apply to thread_boost_layer zones
    return 0;
}

rpg_craft_stats_t rpg_get_stats(void) {
    rpg_craft_stats_t stats = {0};
    stats.total_recipes = g_recipe_count;
    stats.total_boost_maps = g_boost_map_count;
    stats.active_presets = 0;  // Would track applied presets
    return stats;
}
