/**
 * GAMESA Kernel - Core Component & River Management
 */

#include "gamesa_kernel.h"
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

static gamesa_kernel_t *g_kernel = NULL;
static pthread_mutex_t g_kernel_lock = PTHREAD_MUTEX_INITIALIZER;

int gamesa_kernel_init(gamesa_kernel_config_t *config) {
    pthread_mutex_lock(&g_kernel_lock);

    if (g_kernel != NULL) {
        pthread_mutex_unlock(&g_kernel_lock);
        return -1;  // Already initialized
    }

    g_kernel = calloc(1, sizeof(gamesa_kernel_t));
    if (!g_kernel) {
        pthread_mutex_unlock(&g_kernel_lock);
        return -1;
    }

    g_kernel->max_components = config ? config->max_components : 64;
    g_kernel->max_rivers = config ? config->max_rivers : 32;
    g_kernel->components = calloc(g_kernel->max_components, sizeof(gamesa_component_t*));
    g_kernel->rivers = calloc(g_kernel->max_rivers, sizeof(gamesa_river_t*));

    pthread_mutex_unlock(&g_kernel_lock);
    return 0;
}

void gamesa_kernel_shutdown(void) {
    pthread_mutex_lock(&g_kernel_lock);

    if (g_kernel) {
        free(g_kernel->components);
        free(g_kernel->rivers);
        free(g_kernel);
        g_kernel = NULL;
    }

    pthread_mutex_unlock(&g_kernel_lock);
}

gamesa_component_t* gamesa_component_create(const char *name, gamesa_component_type_t type) {
    if (!g_kernel || !name) return NULL;

    gamesa_component_t *comp = calloc(1, sizeof(gamesa_component_t));
    if (!comp) return NULL;

    strncpy(comp->name, name, sizeof(comp->name) - 1);
    comp->type = type;
    comp->state = GAMESA_COMPONENT_CREATED;
    comp->id = g_kernel->component_count++;

    g_kernel->components[comp->id] = comp;
    return comp;
}

int gamesa_component_start(gamesa_component_t *comp) {
    if (!comp) return -1;
    comp->state = GAMESA_COMPONENT_RUNNING;
    return 0;
}

int gamesa_component_stop(gamesa_component_t *comp) {
    if (!comp) return -1;
    comp->state = GAMESA_COMPONENT_STOPPED;
    return 0;
}

void gamesa_component_destroy(gamesa_component_t *comp) {
    if (!comp) return;
    if (g_kernel && comp->id < g_kernel->max_components) {
        g_kernel->components[comp->id] = NULL;
    }
    free(comp);
}

gamesa_river_t* gamesa_river_create(const char *name, gamesa_river_config_t *config) {
    if (!g_kernel || !name) return NULL;

    gamesa_river_t *river = calloc(1, sizeof(gamesa_river_t));
    if (!river) return NULL;

    strncpy(river->name, name, sizeof(river->name) - 1);
    river->id = g_kernel->river_count++;

    if (config) {
        river->transport = config->transport;
        river->direction = config->direction;
        river->buffer_size = config->buffer_size;
    } else {
        river->transport = GAMESA_RIVER_SHM;
        river->direction = GAMESA_RIVER_BIDIRECTIONAL;
        river->buffer_size = 4096;
    }

    g_kernel->rivers[river->id] = river;
    return river;
}

int gamesa_river_send(gamesa_river_t *river, gamesa_river_msg_t *msg) {
    if (!river || !msg) return -1;
    river->msgs_sent++;
    return 0;
}

int gamesa_river_recv(gamesa_river_t *river, gamesa_river_msg_t *msg, int timeout_ms) {
    if (!river || !msg) return -1;
    river->msgs_received++;
    return 0;  // Would block/poll in real impl
}

void gamesa_river_destroy(gamesa_river_t *river) {
    if (!river) return;
    if (g_kernel && river->id < g_kernel->max_rivers) {
        g_kernel->rivers[river->id] = NULL;
    }
    free(river);
}
