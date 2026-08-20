/**
 * System Generator Demo - Validation & Benchmarking
 */

#include <stdio.h>
#include <stdlib.h>
#include "../gamesa_kernel.h"
#include "../rpg_craft_system.h"

int main(int argc, char **argv) {
    printf("GAMESA System Generator Demo\n");
    printf("=============================\n\n");

    // Initialize systems
    gamesa_kernel_config_t kconfig = { .max_components = 32, .max_rivers = 16 };
    gamesa_kernel_init(&kconfig);
    rpg_craft_init();

    // Create performance recipes
    printf("Creating performance recipes...\n");

    rpg_ingredient_t balanced_ing[] = {
        { .type = RPG_ING_CPU_BOOST, .amount = 50, .quality = 0.8f },
        { .type = RPG_ING_GPU_CLOCK, .amount = 100, .quality = 0.8f },
        { .type = RPG_ING_POWER_LIMIT, .amount = 80, .quality = 0.9f }
    };
    rpg_recipe_config_t balanced_cfg = {
        .name = "Balanced Gaming",
        .rarity = RPG_RARITY_COMMON,
        .ingredient_count = 3,
        .ingredients = balanced_ing
    };
    uint32_t recipe_balanced = rpg_recipe_create(&balanced_cfg);

    rpg_ingredient_t perf_ing[] = {
        { .type = RPG_ING_CPU_BOOST, .amount = 100, .quality = 0.95f },
        { .type = RPG_ING_GPU_CLOCK, .amount = 200, .quality = 0.95f },
        { .type = RPG_ING_POWER_LIMIT, .amount = 100, .quality = 1.0f },
        { .type = RPG_ING_FAN_CURVE, .amount = 2, .quality = 0.9f }
    };
    rpg_recipe_config_t perf_cfg = {
        .name = "Max Performance",
        .rarity = RPG_RARITY_RARE,
        .ingredient_count = 4,
        .ingredients = perf_ing
    };
    uint32_t recipe_perf = rpg_recipe_create(&perf_cfg);

    printf("  Created: Balanced Gaming (id=%u)\n", recipe_balanced);
    printf("  Created: Max Performance (id=%u)\n", recipe_perf);

    // Craft presets
    printf("\nCrafting presets...\n");
    rpg_craft_params_t params = { .quality_target = 0.85f };
    rpg_preset_t preset1 = rpg_recipe_craft(recipe_balanced, &params);
    printf("  Balanced preset: CPU+%d GPU+%d Power=%d%%\n",
           preset1.cpu_clock_offset, preset1.gpu_clock_offset, preset1.power_limit_pct);

    params.quality_target = 0.95f;
    rpg_preset_t preset2 = rpg_recipe_craft(recipe_perf, &params);
    printf("  Performance preset: CPU+%d GPU+%d Power=%d%%\n",
           preset2.cpu_clock_offset, preset2.gpu_clock_offset, preset2.power_limit_pct);

    // Create boost maps
    printf("\nCreating boost maps...\n");
    uint32_t map_game = rpg_boost_map_create("gaming_profile");
    rpg_zone_params_t zp = { .boost_factor = 1.2f, .priority = 10 };
    rpg_boost_map_add_zone(map_game, 0, &zp);
    zp.boost_factor = 1.0f;
    zp.priority = 5;
    rpg_boost_map_add_zone(map_game, 1, &zp);

    printf("  Gaming profile map created (id=%u)\n", map_game);

    // Stats
    rpg_craft_stats_t stats = rpg_get_stats();
    printf("\nSystem Stats:\n");
    printf("  Total recipes: %u\n", stats.total_recipes);
    printf("  Total boost maps: %u\n", stats.total_boost_maps);

    // Cleanup
    rpg_craft_shutdown();
    gamesa_kernel_shutdown();

    printf("\nDemo complete.\n");
    return 0;
}
