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
} gamesa_component_type_t;

/** Component state */
typedef enum {
    COMP_STATE_INIT,
    COMP_STATE_RUNNING,
    COMP_STATE_PAUSED,
    COMP_STATE_ERROR,
    COMP_STATE_STOPPED,
} gamesa_component_state_t;

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

/** River transport type */
typedef enum {
    RIVER_TRANSPORT_SHMEM,      /* Shared memory (fastest) */
    RIVER_TRANSPORT_SOCKET,     /* Unix socket */
    RIVER_TRANSPORT_PIPE,       /* Named pipe */
    RIVER_TRANSPORT_QUEUE,      /* Message queue */
    RIVER_TRANSPORT_DMA,        /* DMA buffer (GPU) */
} gamesa_river_transport_t;

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
    uint64_t                    bytes_transferred;
    uint64_t                    messages_sent;
    uint64_t                    messages_dropped;
    uint64_t                    last_activity_ns;
    bool                        active;
} gamesa_river_t;

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

/** Kernel state */
typedef struct {
    /* Components */
    gamesa_component_t      components[GAMESA_MAX_COMPONENTS];
    uint32_t                component_count;
    pthread_mutex_t         component_lock;

    /* Rivers */
    gamesa_river_t          rivers[GAMESA_MAX_RIVERS];
    uint32_t                river_count;
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
int gamesa_kernel_init(gamesa_kernel_t* kernel);

/** Shutdown kernel */
void gamesa_kernel_shutdown(gamesa_kernel_t* kernel);

/** Start kernel scheduler */
int gamesa_kernel_start(gamesa_kernel_t* kernel);

/** Stop kernel */
void gamesa_kernel_stop(gamesa_kernel_t* kernel);

/* Component Management */

/** Register component */
int gamesa_component_register(
    gamesa_kernel_t* kernel,
    const char* name,
    gamesa_component_type_t type,
    const gamesa_component_ops_t* ops,
    void* config,
    uint32_t* out_id
);

/** Start component */
int gamesa_component_start(gamesa_kernel_t* kernel, uint32_t id);

/** Stop component */
int gamesa_component_stop(gamesa_kernel_t* kernel, uint32_t id);

/** Get component by ID */
gamesa_component_t* gamesa_component_get(gamesa_kernel_t* kernel, uint32_t id);

/** Get component by name */
gamesa_component_t* gamesa_component_find(gamesa_kernel_t* kernel, const char* name);

/* River Management */

/** Create river between components */
int gamesa_river_create(
    gamesa_kernel_t* kernel,
    const char* name,
    uint32_t source_id,
    uint32_t sink_id,
    gamesa_river_transport_t transport,
    const gamesa_river_qos_t* qos,
    uint32_t* out_id
);

/** Destroy river */
int gamesa_river_destroy(gamesa_kernel_t* kernel, uint32_t id);

/** Send message on river */
int gamesa_river_send(
    gamesa_kernel_t* kernel,
    uint32_t river_id,
    uint32_t msg_type,
    const void* data,
    uint32_t size
);

/** Receive message from river */
int gamesa_river_recv(
    gamesa_kernel_t* kernel,
    uint32_t river_id,
    void* buffer,
    uint32_t* size,
    uint32_t timeout_ms
);

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
