/**
 * GAMESA Kernel - Foundation Layer for Component Orchestration
 *
 * Core responsibilities:
 * - Component lifecycle management (spawn, monitor, restart)
 * - River (data stream) creation and routing
 * - Resource arbitration and quota enforcement
 * - IPC multiplexing across Python/Rust/C boundaries
 * - Driver registration and hot-plug support
 */

#ifndef GAMESA_KERNEL_H
#define GAMESA_KERNEL_H

#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========== Configuration ========== */

#define GAMESA_MAX_COMPONENTS       64
#define GAMESA_MAX_RIVERS           128
#define GAMESA_MAX_DRIVERS          32
#define GAMESA_RIVER_BUFFER_SIZE    4096
#define GAMESA_KERNEL_VERSION       0x010000  /* 1.0.0 */

/* ========== Component System ========== */

/** Component types in the stack */
typedef enum {
    COMPONENT_GUARDIAN,         /* Python guardian layer */
    COMPONENT_RUST_BOT,         /* Rust trusted decision engine */
    COMPONENT_THREAD_BOOST,     /* C thread boost layer */
    COMPONENT_RPG_CRAFT,        /* C RPG craft system */
    COMPONENT_OPENVINO,         /* OpenVINO bridge */
    COMPONENT_MESA_DRIVER,      /* Mesa/Gallium driver */
    COMPONENT_DMA_MODULE,       /* Kernel DMA module */
    COMPONENT_TWEAKER_UI,       /* Tkinter UI */
    COMPONENT_CUSTOM,           /* User-defined */
    GAMESA_COMPONENT_MEMORY,    /* Memory tier component */
} gamesa_component_type_t;

/** Component state */
typedef enum {
    COMP_STATE_INIT,
    COMP_STATE_RUNNING,
    COMP_STATE_PAUSED,
    COMP_STATE_ERROR,
    COMP_STATE_STOPPED,
} gamesa_component_state_t;

#define GAMESA_COMPONENT_CREATED COMP_STATE_INIT
#define GAMESA_COMPONENT_RUNNING COMP_STATE_RUNNING
#define GAMESA_COMPONENT_STOPPED COMP_STATE_STOPPED

/** Component descriptor */
typedef struct {
    uint32_t                    id;
    char                        name[64];
    gamesa_component_type_t     type;
    gamesa_component_state_t    state;
    pid_t                       pid;            /* Process ID if external */
    pthread_t                   thread;         /* Thread ID if internal */
    uint64_t                    start_time_ns;
    uint64_t                    cpu_time_ns;
    uint32_t                    restart_count;
    void*                       context;        /* Component-specific data */
} gamesa_component_t;

/** Component callbacks */
typedef struct {
    int (*init)(gamesa_component_t* comp, void* config);
    int (*start)(gamesa_component_t* comp);
    int (*stop)(gamesa_component_t* comp);
    int (*health_check)(gamesa_component_t* comp);
    void (*cleanup)(gamesa_component_t* comp);
} gamesa_component_ops_t;

/* ========== River System (Data Streams) ========== */

/** River flow direction */
typedef enum {
    RIVER_UNIDIRECTIONAL,       /* One-way flow */
    RIVER_BIDIRECTIONAL,        /* Two-way flow */
} gamesa_river_direction_t;

#define GAMESA_RIVER_BIDIRECTIONAL RIVER_BIDIRECTIONAL

/** River transport type */
typedef enum {
    RIVER_TRANSPORT_SHMEM,      /* Shared memory (fastest) */
    RIVER_TRANSPORT_SOCKET,     /* Unix socket */
    RIVER_TRANSPORT_PIPE,       /* Named pipe */
    RIVER_TRANSPORT_QUEUE,      /* Message queue */
    RIVER_TRANSPORT_DMA,        /* DMA buffer (GPU) */
} gamesa_river_transport_t;

#define GAMESA_RIVER_SHM RIVER_TRANSPORT_SHMEM

/** River QoS settings */
typedef struct {
    uint32_t    max_latency_us;     /* Max acceptable latency */
    uint32_t    min_throughput_bps; /* Min throughput */
    uint8_t     priority;           /* 0-255, higher = more priority */
    bool        lossy;              /* Allow message drops */
    bool        ordered;            /* Guarantee ordering */
} gamesa_river_qos_t;

/** River descriptor */
typedef struct {
    uint32_t                    id;
    char                        name[64];
    uint32_t                    source_id;      /* Source component */
    uint32_t                    sink_id;        /* Destination component */
    gamesa_river_direction_t    direction;
    gamesa_river_transport_t    transport;
    gamesa_river_qos_t          qos;

    /* Runtime state */
    void*                       buffer;
    uint32_t                    buffer_size;
    uint64_t                    bytes_transferred;
    uint64_t                    messages_sent;
    uint64_t                    messages_dropped;
    uint64_t                    msgs_sent;
    uint64_t                    msgs_received;
    uint64_t                    last_activity_ns;
    bool                        active;
} gamesa_river_t;

/** River configuration */
typedef struct {
    gamesa_river_transport_t transport;
    gamesa_river_direction_t direction;
    uint32_t buffer_size;
} gamesa_river_config_t;

/** River message header */
typedef struct {
    uint32_t    magic;              /* 0xCAFE1234 */
    uint32_t    type;               /* Message type */
    uint32_t    size;               /* Payload size */
    uint64_t    timestamp_ns;
    uint32_t    source_id;
    uint32_t    sequence;
} gamesa_river_msg_t;

/* River message types */
#define RIVER_MSG_TELEMETRY     0x0001
#define RIVER_MSG_SIGNAL        0x0002
#define RIVER_MSG_DECISION      0x0003
#define RIVER_MSG_PRESET        0x0004
#define RIVER_MSG_EVENT         0x0005
#define RIVER_MSG_HEARTBEAT     0x00FF

/* ========== Driver System ========== */

/** Driver type */
typedef enum {
    DRIVER_GPU,
    DRIVER_CPU_FREQ,
    DRIVER_THERMAL,
    DRIVER_POWER,
    DRIVER_MEMORY,
    DRIVER_NETWORK,
} gamesa_driver_type_t;

/** Driver descriptor */
typedef struct {
    uint32_t                id;
    char                    name[64];
    gamesa_driver_type_t    type;
    char                    version[32];
    void*                   ops;            /* Driver operations */
    void*                   private_data;
    bool                    loaded;
} gamesa_driver_t;

/* ========== Kernel Core ========== */

/** Kernel configuration */
typedef struct {
    uint32_t max_components;
    uint32_t max_rivers;
} gamesa_kernel_config_t;

/** Kernel state */
typedef struct {
    uint32_t                max_components;
    uint32_t                max_rivers;
    uint32_t                component_count;
    uint32_t                river_count;
    gamesa_component_t**    components;
    gamesa_river_t**        rivers;
    pthread_mutex_t         component_lock;
    pthread_mutex_t         river_lock;

    /* Drivers */
    gamesa_driver_t         drivers[GAMESA_MAX_DRIVERS];
    uint32_t                driver_count;
    pthread_mutex_t         driver_lock;

    /* Runtime */
    bool                    running;
    uint64_t                boot_time_ns;
    pthread_t               scheduler_thread;
    pthread_t               monitor_thread;
} gamesa_kernel_t;

/* ========== Kernel API ========== */

/** Initialize kernel */
int gamesa_kernel_init(gamesa_kernel_config_t* config);

/** Shutdown kernel */
void gamesa_kernel_shutdown(void);

/** Component Management */
gamesa_component_t* gamesa_component_create(const char* name, gamesa_component_type_t type);
int gamesa_component_start(gamesa_component_t* comp);
int gamesa_component_stop(gamesa_component_t* comp);
void gamesa_component_destroy(gamesa_component_t* comp);

/** River Management */
gamesa_river_t* gamesa_river_create(const char* name, gamesa_river_config_t* config);
int gamesa_river_send(gamesa_river_t* river, gamesa_river_msg_t* msg);
int gamesa_river_recv(gamesa_river_t* river, gamesa_river_msg_t* msg, int timeout_ms);
void gamesa_river_destroy(gamesa_river_t* river);

/** River statistics */

/** Get river statistics */
typedef struct {
    uint64_t bytes_sent;
    uint64_t bytes_recv;
    uint64_t messages_sent;
    uint64_t messages_recv;
    uint64_t messages_dropped;
    double   avg_latency_us;
    double   throughput_bps;
} gamesa_river_stats_t;

int gamesa_river_stats(gamesa_kernel_t* kernel, uint32_t id, gamesa_river_stats_t* stats);

/* Driver Management */

/** Register driver */
int gamesa_driver_register(
    gamesa_kernel_t* kernel,
    const char* name,
    gamesa_driver_type_t type,
    void* ops,
    uint32_t* out_id
);

/** Load driver */
int gamesa_driver_load(gamesa_kernel_t* kernel, uint32_t id);

/** Unload driver */
int gamesa_driver_unload(gamesa_kernel_t* kernel, uint32_t id);

/* ========== Standard Rivers ========== */

/** Create standard telemetry river (C → Python/Rust) */
int gamesa_create_telemetry_river(gamesa_kernel_t* kernel, uint32_t source_id);

/** Create standard signal river (Python → C) */
int gamesa_create_signal_river(gamesa_kernel_t* kernel, uint32_t sink_id);

/** Create standard decision river (Rust → C) */
int gamesa_create_decision_river(gamesa_kernel_t* kernel, uint32_t sink_id);

/* ========== Utility Macros ========== */

#define GAMESA_RIVER_MSG_MAGIC  0xCAFE1234

#define GAMESA_COMPONENT_FOREACH(kernel, comp) \
    for (gamesa_component_t* comp = (kernel)->components; \
         comp < (kernel)->components + (kernel)->component_count; \
         comp++)

#define GAMESA_RIVER_FOREACH(kernel, river) \
    for (gamesa_river_t* river = (kernel)->rivers; \
         river < (kernel)->rivers + (kernel)->river_count; \
         river++)

#ifdef __cplusplus
}
#endif

#endif /* GAMESA_KERNEL_H */
