/**
 * Thread Boost Layer - Zone/Core/GPU Mapping
 */

#include "thread_boost_layer.h"
#include <stdlib.h>
#include <string.h>

static boost_zone_t g_zones[MAX_BOOST_ZONES];
static uint32_t g_zone_count = 0;

int tbl_init(void) {
    memset(g_zones, 0, sizeof(g_zones));
    g_zone_count = 0;
    return 0;
}

void tbl_shutdown(void) {
    g_zone_count = 0;
}

uint32_t tbl_zone_create(tbl_zone_config_t *config) {
    if (g_zone_count >= MAX_BOOST_ZONES || !config) return (uint32_t)-1;

    boost_zone_t *zone = &g_zones[g_zone_count];
    zone->zone_id = g_zone_count;
    zone->p_core_mask = config->p_core_mask;
    zone->e_core_mask = config->e_core_mask;
    zone->gpu_block_start = config->gpu_block_start;
    zone->gpu_block_count = config->gpu_block_count;
    zone->priority = config->priority;
    zone->active = 1;

    return g_zone_count++;
}

int tbl_zone_boost(uint32_t zone_id, tbl_boost_profile_t *profile) {
    if (zone_id >= g_zone_count || !profile) return -1;

    boost_zone_t *zone = &g_zones[zone_id];
    zone->current_boost = profile->clock_multiplier;
    zone->thermal_budget = profile->thermal_budget;
    zone->power_budget = profile->power_budget;

    return 0;
}

int tbl_zone_throttle(uint32_t zone_id, float target) {
    if (zone_id >= g_zone_count) return -1;

    boost_zone_t *zone = &g_zones[zone_id];
    zone->current_boost *= target;
    if (zone->current_boost < 0.5f) zone->current_boost = 0.5f;

    return 0;
}

tbl_zone_telemetry_t tbl_zone_get_telemetry(uint32_t zone_id) {
    tbl_zone_telemetry_t telem = {0};
    if (zone_id >= g_zone_count) return telem;

    boost_zone_t *zone = &g_zones[zone_id];
    telem.zone_id = zone_id;
    telem.cpu_util = 0.5f;      // Would read from /proc
    telem.gpu_util = 0.5f;      // Would read from sysfs
    telem.temp_cpu = 60.0f;
    telem.temp_gpu = 55.0f;
    telem.power_draw = zone->power_budget * 0.8f;
    telem.active = zone->active;

    return telem;
}

int tbl_rebalance_under_pressure(float thermal_pressure) {
    for (uint32_t i = 0; i < g_zone_count; i++) {
        if (thermal_pressure > 0.8f) {
            g_zones[i].current_boost *= 0.9f;
        }
    }
    return 0;
}
