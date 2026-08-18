/**
 * Autonomous World Generator - Procedural Content Creation
 *
 * Generates worlds, terrain, objects on-the-fly with memory-mapped
 * scheduling tied to thread boost zones.
 */

#ifndef GAMESA_AUTONOMOUS_WORLD_GENERATOR_H
#define GAMESA_AUTONOMOUS_WORLD_GENERATOR_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Configuration
 * ============================================================ */

#define AWG_MAX_CHUNKS      1024
#define AWG_MAX_LAYERS      8
#define AWG_MAX_GENERATORS  16
#define AWG_CHUNK_SIZE      64   // 64x64x64 voxels per chunk

/* ============================================================
 * Types
 * ============================================================ */

typedef enum {
    AWG_BIOME_PLAINS,
    AWG_BIOME_FOREST,
    AWG_BIOME_DESERT,
    AWG_BIOME_MOUNTAIN,
    AWG_BIOME_OCEAN,
    AWG_BIOME_TUNDRA,
    AWG_BIOME_VOLCANIC,
    AWG_BIOME_CUSTOM
} awg_biome_t;

typedef enum {
    AWG_LAYER_TERRAIN,
    AWG_LAYER_VEGETATION,
    AWG_LAYER_STRUCTURES,
    AWG_LAYER_ENTITIES,
    AWG_LAYER_LIGHTING,
    AWG_LAYER_PHYSICS,
    AWG_LAYER_AI_HINTS,
    AWG_LAYER_CUSTOM
} awg_layer_t;

typedef enum {
    AWG_GEN_PERLIN,         // Classic Perlin noise
    AWG_GEN_SIMPLEX,        // Simplex noise
    AWG_GEN_VORONOI,        // Voronoi diagrams
    AWG_GEN_CELLULAR,       // Cellular automata
    AWG_GEN_L_SYSTEM,       // L-system fractals
    AWG_GEN_WAVE_FUNCTION,  // Wave function collapse
    AWG_GEN_ML_VAE,         // VAE latent decode (OpenVINO)
    AWG_GEN_ML_GAN,         // GAN generation (OpenVINO)
    AWG_GEN_ML_DIFFUSION    // Diffusion model (OpenVINO)
} awg_generator_type_t;

typedef struct {
    int32_t x, y, z;
} awg_coord_t;

typedef struct {
    uint8_t material;
    uint8_t density;
    uint8_t metadata;
    uint8_t flags;
} awg_voxel_t;

typedef struct {
    awg_coord_t origin;
    uint32_t size;
    awg_biome_t biome;
    float lod_level;           // 0.0 = full detail, 1.0 = distant
    uint64_t seed;
    awg_voxel_t *data;         // size^3 voxels
    uint32_t boost_zone_id;    // Thread boost zone for generation
    uint32_t memory_tier;      // Memory tier placement
    bool dirty;
    bool generating;
    uint64_t generation_time_us;
} awg_chunk_t;

typedef struct {
    awg_generator_type_t type;
    float frequency;
    float amplitude;
    uint32_t octaves;
    float persistence;
    float lacunarity;
    void *ml_model_handle;    // OpenVINO model if ML type
} awg_layer_config_t;

typedef struct {
    uint64_t seed;
    float world_scale;
    uint32_t chunk_load_radius;
    uint32_t max_concurrent_gen;
    bool use_ml_generation;
    char ml_model_path[256];
    uint32_t target_boost_zone;
} awg_world_config_t;

typedef struct {
    uint32_t chunks_generated;
    uint32_t chunks_cached;
    uint64_t total_gen_time_us;
    uint64_t avg_gen_time_us;
    float memory_used_mb;
    uint32_t ml_inferences;
} awg_stats_t;

/* ============================================================
 * World Generator API
 * ============================================================ */

// Lifecycle
int awg_init(awg_world_config_t *config);
void awg_shutdown(void);

// Chunk management
awg_chunk_t* awg_chunk_request(awg_coord_t coord, float lod);
void awg_chunk_release(awg_chunk_t *chunk);
int awg_chunk_generate_async(awg_coord_t coord, float lod);
bool awg_chunk_is_ready(awg_coord_t coord);

// Layer configuration
int awg_layer_configure(awg_layer_t layer, awg_layer_config_t *config);
int awg_layer_set_generator(awg_layer_t layer, awg_generator_type_t gen);

// Biome control
int awg_biome_set_at(awg_coord_t coord, awg_biome_t biome);
awg_biome_t awg_biome_get_at(awg_coord_t coord);
int awg_biome_blend_radius(uint32_t radius);

// ML integration
int awg_ml_load_model(const char *model_path, awg_generator_type_t type);
int awg_ml_set_latent(float *latent_vector, uint32_t size);
int awg_ml_generate_chunk(awg_chunk_t *chunk);

// Thread boost integration
int awg_set_boost_zone(uint32_t zone_id);
int awg_set_memory_tier(uint32_t tier);
int awg_prioritize_generation(awg_coord_t *coords, uint32_t count);

// Statistics
awg_stats_t awg_get_stats(void);
void awg_reset_stats(void);

/* ============================================================
 * Noise Functions (internal, exposed for custom generators)
 * ============================================================ */

float awg_noise_perlin_2d(float x, float y, uint64_t seed);
float awg_noise_perlin_3d(float x, float y, float z, uint64_t seed);
float awg_noise_simplex_2d(float x, float y, uint64_t seed);
float awg_noise_simplex_3d(float x, float y, float z, uint64_t seed);
float awg_noise_fbm(float x, float y, float z, uint32_t octaves, float persistence);

#ifdef __cplusplus
}
#endif

#endif // GAMESA_AUTONOMOUS_WORLD_GENERATOR_H
