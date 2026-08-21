/**
 * Thread Boost Layer - Zone-Based Scheduler for GPU/CPU Coordination
 *
 * Core C runtime that binds spatial grid slices to:
 * - GPU memory blocks (VRAM allocation)
 * - P/E core masks (Performance/Efficiency scheduling)
 * - Signal strengths (priority from Guardian)
 * - Live rebalancing based on telemetry
 *
 * This is the conduit between Guardian priorities and low-level execution.
 */

#ifndef GAMESA_THREAD_BOOST_LAYER_H
#define GAMESA_THREAD_BOOST_LAYER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========== Configuration ========== */

#define GAMESA_MAX_ZONES        64
#define GAMESA_MAX_GRID_DIM     8
#define GAMESA_MAX_CORES        32
#define GAMESA_MAX_GPU_BLOCKS   256

/* ========== Core Types ========== */

/** 3D grid position for zone mapping */
typedef struct {
    uint8_t x;
    uint8_t y;
    uint8_t z;
} gamesa_grid_pos_t;

/** P/E core mask for hybrid CPU scheduling */
typedef struct {
    uint32_t p_cores;       /* Performance core bitmask */
    uint32_t e_cores;       /* Efficiency core bitmask */
    uint8_t  priority;      /* 0-255 scheduling priority */
} gamesa_core_mask_t;

/** GPU memory block allocation */
typedef struct {
    uint32_t block_id;
    uint64_t offset;        /* Offset in VRAM */
    uint64_t size;          /* Block size in bytes */
    uint8_t  tier;          /* Memory tier (0=fastest) */
    bool     pinned;        /* Pinned allocation */
} gamesa_gpu_block_t;

/** Zone definition - binds grid to resources */
typedef struct {
    uint32_t            zone_id;
    gamesa_grid_pos_t   grid_pos;
    gamesa_core_mask_t  core_mask;
    gamesa_gpu_block_t  gpu_block;
    float               signal_strength;    /* 0.0-1.0 from Guardian */
    bool                active;
    uint64_t            last_update_ns;
} gamesa_zone_t;

/** Telemetry snapshot for IPC to Python Guardian */
typedef struct {
    uint64_t timestamp_ns;
    float    cpu_util;          /* 0.0-1.0 */
    float    gpu_util;          /* 0.0-1.0 */
    float    memory_util;       /* 0.0-1.0 */
    float    temp_cpu;          /* Celsius */
    float    temp_gpu;          /* Celsius */
    float    frametime_ms;
    float    power_draw_w;
    uint32_t active_zone_count;
    uint32_t pe_core_mask;      /* Current P/E distribution */
} gamesa_telemetry_t;

/** Signal from Guardian for zone adjustment */
typedef struct {
    uint32_t zone_id;
    float    strength;          /* 0.0-1.0 priority */
    uint8_t  action;            /* BOOST, THROTTLE, MIGRATE, IDLE */
    uint32_t target_cores;      /* New core mask if action=MIGRATE */
} gamesa_signal_t;

/* Signal actions */
#define GAMESA_ACTION_BOOST     0x01
#define GAMESA_ACTION_THROTTLE  0x02
#define GAMESA_ACTION_MIGRATE   0x03
#define GAMESA_ACTION_IDLE      0x04

/* ========== Zone Grid Manager ========== */

/** Initialize the thread boost layer */
int gamesa_tbl_init(void);

/** Shutdown and cleanup */
void gamesa_tbl_shutdown(void);

/** Create a new zone at grid position */
int gamesa_zone_create(
    gamesa_grid_pos_t pos,
    gamesa_core_mask_t cores,
    uint64_t gpu_memory_size,
    gamesa_zone_t *out_zone
);

/** Destroy a zone */
int gamesa_zone_destroy(uint32_t zone_id);

/** Get zone by ID */
gamesa_zone_t* gamesa_zone_get(uint32_t zone_id);

/** Update zone signal strength (from Guardian) */
int gamesa_zone_set_signal(uint32_t zone_id, float strength);

/** Migrate zone to new cores */
int gamesa_zone_migrate(uint32_t zone_id, gamesa_core_mask_t new_cores);

/* ========== Scheduling ========== */

/** Rebalance all zones based on current signals */
int gamesa_rebalance(void);

/** Schedule work unit to best zone */
int gamesa_schedule_work(
    void (*work_fn)(void* ctx),
    void* ctx,
    uint32_t preferred_zone
);

/** Get current P/E core distribution */
gamesa_core_mask_t gamesa_get_core_distribution(void);

/** Set P/E core ratio (0.0 = all E, 1.0 = all P) */
int gamesa_set_pe_ratio(float ratio);

/* ========== Telemetry ========== */

/** Get current telemetry snapshot */
gamesa_telemetry_t gamesa_get_telemetry(void);

/** Register telemetry callback (for IPC to Python) */
typedef void (*gamesa_telemetry_cb)(const gamesa_telemetry_t* tel, void* ctx);
int gamesa_register_telemetry_callback(gamesa_telemetry_cb cb, void* ctx);

/* ========== IPC Bridge ========== */

/** Process incoming signal from Guardian */
int gamesa_process_signal(const gamesa_signal_t* signal);

/** Send telemetry to Guardian via IPC socket */
int gamesa_ipc_send_telemetry(int socket_fd);

/** Receive signals from Guardian via IPC socket */
int gamesa_ipc_recv_signals(int socket_fd);

/* ========== GPU Memory ========== */

/** Allocate GPU memory block */
int gamesa_gpu_alloc(uint64_t size, uint8_t tier, gamesa_gpu_block_t *out);

/** Free GPU memory block */
int gamesa_gpu_free(uint32_t block_id);

/** Get GPU memory stats */
typedef struct {
    uint64_t total_bytes;
    uint64_t used_bytes;
    uint64_t peak_bytes;
    uint32_t block_count;
} gamesa_gpu_stats_t;

gamesa_gpu_stats_t gamesa_gpu_get_stats(void);

#ifdef __cplusplus
}
#endif

#endif /* GAMESA_THREAD_BOOST_LAYER_H */
