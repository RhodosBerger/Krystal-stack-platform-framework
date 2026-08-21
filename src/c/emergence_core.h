/**
 * Emergence Core - Hardware-Level Emergent Behavior
 *
 * Implements attractor dynamics, phase transitions, and
 * collective optimization at the hardware control level.
 */

#ifndef GAMESA_EMERGENCE_CORE_H
#define GAMESA_EMERGENCE_CORE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Configuration
 * ============================================================ */

#define EMG_MAX_ATTRACTORS    32
#define EMG_STATE_DIMENSIONS  8
#define EMG_MAX_AGENTS        64
#define EMG_HISTORY_SIZE      1024

/* ============================================================
 * Attractor Dynamics
 * ============================================================ */

typedef struct {
    char name[32];
    float center[EMG_STATE_DIMENSIONS];
    float basin_radius;
    float stability;
    float energy;
    uint32_t visits;
} emg_attractor_t;

typedef struct {
    float position[EMG_STATE_DIMENSIONS];
    float velocity[EMG_STATE_DIMENSIONS];
    float energy;
    uint32_t attractor_id;  // Nearest attractor
} emg_state_t;

// Attractor landscape API
int emg_attractor_init(void);
void emg_attractor_shutdown(void);
uint32_t emg_attractor_add(const char *name, const float *center, float stability);
int emg_attractor_remove(uint32_t id);
emg_state_t emg_attractor_update(const float *current_state, float dt);
uint32_t emg_attractor_nearest(const float *state);

/* ============================================================
 * Phase Transitions
 * ============================================================ */

typedef enum {
    EMG_PHASE_SOLID,      // Locked optimization
    EMG_PHASE_LIQUID,     // Exploring
    EMG_PHASE_GAS,        // Random search
    EMG_PHASE_PLASMA,     // Creative breakthrough
    EMG_PHASE_CRITICAL    // Phase transition point
} emg_phase_t;

typedef struct {
    emg_phase_t phase;
    float temperature;
    float order_parameter;
    float critical_temp;
    uint32_t transitions;
} emg_phase_state_t;

// Phase transition API
int emg_phase_init(float critical_temp);
emg_phase_t emg_phase_update(float gradient, float stability);
int emg_phase_induce(emg_phase_t target);
float emg_phase_exploration_rate(void);
emg_phase_state_t emg_phase_get_state(void);

/* ============================================================
 * Collective Intelligence
 * ============================================================ */

typedef struct {
    uint32_t id;
    float position[EMG_STATE_DIMENSIONS];
    float velocity[EMG_STATE_DIMENSIONS];
    float best_position[EMG_STATE_DIMENSIONS];
    float best_score;
} emg_agent_t;

typedef struct {
    uint32_t agent_count;
    float global_best[EMG_STATE_DIMENSIONS];
    float global_best_score;
    float diversity;
} emg_swarm_state_t;

// Collective intelligence API
int emg_swarm_init(uint32_t n_agents);
void emg_swarm_shutdown(void);
int emg_swarm_update(float (*objective)(const float *state));
emg_swarm_state_t emg_swarm_get_state(void);
void emg_swarm_get_consensus(float *out);

/* ============================================================
 * Synapse Network
 * ============================================================ */

typedef struct {
    uint32_t source_id;
    uint32_t target_id;
    float weight;
    float plasticity;
    uint64_t last_activation;
} emg_synapse_t;

// Synapse network API
int emg_synapse_init(void);
void emg_synapse_shutdown(void);
int emg_synapse_connect(uint32_t source, uint32_t target, float weight);
int emg_synapse_activate(uint32_t node, float value);
int emg_synapse_hebbian_update(float reward);
float emg_synapse_path_strength(uint32_t source, uint32_t target);

/* ============================================================
 * Unified Emergence Engine
 * ============================================================ */

typedef struct {
    emg_state_t current_state;
    emg_phase_state_t phase;
    emg_swarm_state_t swarm;
    float emergent_score;
    uint32_t cycle_count;
} emg_engine_state_t;

typedef struct {
    float recommended_state[EMG_STATE_DIMENSIONS];
    emg_phase_t phase;
    uint32_t nearest_attractor;
    float exploration_rate;
    float swarm_diversity;
} emg_decision_t;

// Engine API
int emg_engine_init(void);
void emg_engine_shutdown(void);
emg_decision_t emg_engine_evolve(const float *telemetry,
                                  float (*objective)(const float *));
int emg_engine_learn(float reward);
emg_engine_state_t emg_engine_get_state(void);

/* ============================================================
 * Integration with Existing Systems
 * ============================================================ */

// Bridge to thread boost layer
int emg_bridge_boost_zone(uint32_t zone_id, const emg_decision_t *decision);

// Bridge to RPG craft system
int emg_bridge_craft_preset(const emg_decision_t *decision, uint32_t *preset_id);

// Bridge to kernel rivers
int emg_bridge_send_decision(const char *river_name, const emg_decision_t *decision);

// Callback for external objective functions
typedef float (*emg_objective_fn)(const float *state, void *user_data);
int emg_set_objective(emg_objective_fn fn, void *user_data);

#ifdef __cplusplus
}
#endif

#endif // GAMESA_EMERGENCE_CORE_H
