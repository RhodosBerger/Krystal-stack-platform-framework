/**
 * Cognitive Synthesis Engine - ML-Driven Content Creation
 *
 * Orchestrates generative models (VAE, GAN, Diffusion, LLM) for
 * autonomous content creation with feedback loops.
 */

#ifndef GAMESA_COGNITIVE_SYNTHESIS_ENGINE_H
#define GAMESA_COGNITIVE_SYNTHESIS_ENGINE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Configuration
 * ============================================================ */

#define CSE_MAX_MODELS       16
#define CSE_MAX_PIPELINES    8
#define CSE_LATENT_DIM       512
#define CSE_MAX_BATCH        32

/* ============================================================
 * Types
 * ============================================================ */

typedef enum {
    CSE_MODEL_VAE,           // Variational Autoencoder
    CSE_MODEL_GAN,           // Generative Adversarial Network
    CSE_MODEL_DIFFUSION,     // Diffusion Model
    CSE_MODEL_TRANSFORMER,   // Transformer/LLM
    CSE_MODEL_NERF,          // Neural Radiance Fields
    CSE_MODEL_HYBRID         // Multi-model ensemble
} cse_model_type_t;

typedef enum {
    CSE_OUTPUT_MESH,         // 3D mesh data
    CSE_OUTPUT_TEXTURE,      // 2D texture
    CSE_OUTPUT_VOXEL,        // Voxel grid
    CSE_OUTPUT_HEIGHTMAP,    // Terrain heightmap
    CSE_OUTPUT_POINTCLOUD,   // Point cloud
    CSE_OUTPUT_TEXT,         // Text/dialogue
    CSE_OUTPUT_AUDIO,        // Sound/music
    CSE_OUTPUT_BEHAVIOR      // AI behavior tree
} cse_output_type_t;

typedef enum {
    CSE_QUALITY_DRAFT,       // Fast, low quality
    CSE_QUALITY_PREVIEW,     // Medium speed/quality
    CSE_QUALITY_PRODUCTION,  // High quality
    CSE_QUALITY_ULTRA        // Maximum quality (slow)
} cse_quality_t;

typedef struct {
    cse_model_type_t type;
    char model_path[256];
    char name[64];
    uint32_t model_id;
    bool loaded;
    bool gpu_accelerated;
    uint32_t latent_dim;
    float inference_time_ms;
    void *openvino_handle;
} cse_model_t;

typedef struct {
    float *data;
    uint32_t dimensions;
    float temperature;       // Sampling temperature
    float guidance_scale;    // Classifier-free guidance
} cse_latent_t;

typedef struct {
    cse_output_type_t type;
    void *data;
    uint32_t data_size;
    uint32_t width, height, depth;
    float quality_score;     // Self-assessed quality
    uint64_t generation_id;
} cse_output_t;

typedef struct {
    char name[64];
    uint32_t pipeline_id;
    uint32_t model_ids[4];   // Up to 4 models in sequence
    uint32_t model_count;
    cse_output_type_t output_type;
    cse_quality_t quality;
    bool async_execution;
} cse_pipeline_t;

typedef struct {
    uint64_t generation_id;
    float reward;            // From guardian feedback
    cse_latent_t latent;     // Latent that produced this
    cse_output_type_t type;
} cse_experience_t;

typedef struct {
    uint32_t models_loaded;
    uint32_t pipelines_active;
    uint64_t total_generations;
    float avg_inference_ms;
    float avg_quality_score;
    float memory_used_mb;
    uint32_t experiences_stored;
} cse_stats_t;

/* ============================================================
 * Engine Lifecycle
 * ============================================================ */

int cse_init(void);
void cse_shutdown(void);

/* ============================================================
 * Model Management
 * ============================================================ */

// Load model from OpenVINO IR or ONNX
uint32_t cse_model_load(const char *path, cse_model_type_t type, const char *name);
int cse_model_unload(uint32_t model_id);
cse_model_t* cse_model_get(uint32_t model_id);

// Model configuration
int cse_model_set_device(uint32_t model_id, const char *device);  // "CPU", "GPU", "NPU"
int cse_model_set_precision(uint32_t model_id, const char *precision);  // "FP32", "FP16", "INT8"
int cse_model_warm_up(uint32_t model_id, uint32_t iterations);

/* ============================================================
 * Latent Space Operations
 * ============================================================ */

// Create/manipulate latent vectors
cse_latent_t* cse_latent_create(uint32_t dim);
void cse_latent_destroy(cse_latent_t *latent);
int cse_latent_randomize(cse_latent_t *latent, uint64_t seed);
int cse_latent_interpolate(cse_latent_t *a, cse_latent_t *b, float t, cse_latent_t *out);
int cse_latent_add_noise(cse_latent_t *latent, float scale);

// Guided generation
int cse_latent_from_text(const char *prompt, cse_latent_t *out);
int cse_latent_from_image(const void *image_data, uint32_t size, cse_latent_t *out);
int cse_latent_condition(cse_latent_t *latent, const char *condition);

/* ============================================================
 * Generation
 * ============================================================ */

// Single-shot generation
cse_output_t* cse_generate(uint32_t model_id, cse_latent_t *latent, cse_quality_t quality);
void cse_output_destroy(cse_output_t *output);

// Batch generation
int cse_generate_batch(uint32_t model_id, cse_latent_t **latents, uint32_t count,
                       cse_output_t **outputs);

// Async generation
uint64_t cse_generate_async(uint32_t model_id, cse_latent_t *latent, cse_quality_t quality);
bool cse_is_ready(uint64_t generation_id);
cse_output_t* cse_get_result(uint64_t generation_id);

/* ============================================================
 * Pipeline API
 * ============================================================ */

uint32_t cse_pipeline_create(const char *name, cse_output_type_t output_type);
int cse_pipeline_add_stage(uint32_t pipeline_id, uint32_t model_id);
int cse_pipeline_set_quality(uint32_t pipeline_id, cse_quality_t quality);
cse_output_t* cse_pipeline_execute(uint32_t pipeline_id, cse_latent_t *latent);
int cse_pipeline_destroy(uint32_t pipeline_id);

/* ============================================================
 * Learning & Feedback
 * ============================================================ */

// Log generation for experience replay
int cse_log_experience(uint64_t generation_id, float reward);
int cse_get_best_latents(cse_output_type_t type, cse_latent_t **latents, uint32_t count);

// Evolve latents based on rewards
int cse_evolve_latent(cse_latent_t *latent, float mutation_rate);
int cse_crossover_latents(cse_latent_t *a, cse_latent_t *b, cse_latent_t *child);

/* ============================================================
 * Integration
 * ============================================================ */

// Thread boost integration
int cse_set_boost_zone(uint32_t zone_id);
int cse_set_priority(uint32_t priority);

// Memory tier integration
int cse_set_memory_tier(uint32_t tier);
int cse_prefetch_model(uint32_t model_id);

// Statistics
cse_stats_t cse_get_stats(void);

#ifdef __cplusplus
}
#endif

#endif // GAMESA_COGNITIVE_SYNTHESIS_ENGINE_H
