/**
 * Memory Tier Demo - NUMA/Tiered Placement Demonstration
 */

#include <stdio.h>
#include <stdlib.h>
#include "../gamesa_kernel.h"
#include "../thread_boost_layer.h"

int main(int argc, char **argv) {
    printf("GAMESA Memory Tier Demo\n");
    printf("========================\n\n");

    // Initialize kernel
    gamesa_kernel_config_t config = {
        .max_components = 16,
        .max_rivers = 8
    };
    if (gamesa_kernel_init(&config) != 0) {
        fprintf(stderr, "Failed to initialize kernel\n");
        return 1;
    }

    // Create memory tier components
    gamesa_component_t *tier0 = gamesa_component_create("tier0_fast", GAMESA_COMPONENT_MEMORY);
    gamesa_component_t *tier1 = gamesa_component_create("tier1_bulk", GAMESA_COMPONENT_MEMORY);
    gamesa_component_t *tier2 = gamesa_component_create("tier2_archive", GAMESA_COMPONENT_MEMORY);

    if (!tier0 || !tier1 || !tier2) {
        fprintf(stderr, "Failed to create memory tiers\n");
        gamesa_kernel_shutdown();
        return 1;
    }

    printf("Memory Tiers Created:\n");
    printf("  Tier 0 (Fast):    id=%u\n", tier0->id);
    printf("  Tier 1 (Bulk):    id=%u\n", tier1->id);
    printf("  Tier 2 (Archive): id=%u\n", tier2->id);

    // Initialize thread boost layer
    tbl_init();

    // Create zones for each tier
    tbl_zone_config_t zone_cfg = {
        .p_core_mask = 0x0F,
        .e_core_mask = 0xF0,
        .gpu_block_start = 0,
        .gpu_block_count = 4,
        .priority = 10
    };

    uint32_t zone0 = tbl_zone_create(&zone_cfg);
    zone_cfg.priority = 5;
    uint32_t zone1 = tbl_zone_create(&zone_cfg);
    zone_cfg.priority = 1;
    uint32_t zone2 = tbl_zone_create(&zone_cfg);

    printf("\nBoost Zones Created:\n");
    printf("  Zone 0: priority=10 (hot path)\n");
    printf("  Zone 1: priority=5  (warm path)\n");
    printf("  Zone 2: priority=1  (cold path)\n");

    // Simulate tier migration
    printf("\nSimulating memory tier migration...\n");
    for (int i = 0; i < 5; i++) {
        tbl_zone_telemetry_t telem = tbl_zone_get_telemetry(zone0);
        printf("  Cycle %d: Zone 0 CPU=%.1f%% GPU=%.1f%% Temp=%.1fC\n",
               i, telem.cpu_util * 100, telem.gpu_util * 100, telem.temp_cpu);
    }

    // Cleanup
    gamesa_component_destroy(tier0);
    gamesa_component_destroy(tier1);
    gamesa_component_destroy(tier2);
    tbl_shutdown();
    gamesa_kernel_shutdown();

    printf("\nDemo complete.\n");
    return 0;
}
