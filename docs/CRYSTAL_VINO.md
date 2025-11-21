# GAMESA Crystal-Vino Runtime Architecture

**Version**: 1.0
**Codename**: Cross-Forex

## Overview

Crystal-Vino implements a high-frequency trading floor for hardware resources. Instead of static rules, AI agents trade compute, thermal, and precision resources in real-time via a centralized socket bus.

```
┌─────────────────────────────────────────────────────────────┐
│                      GAMESAD (Daemon)                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │               CRYSTAL_SOCKETD (Exchange)              │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │            MARKET TICKER (16-50ms)             │  │  │
│  │  │  hex_compute | hex_memory | thermal_headroom   │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│         ┌─────────────────┼─────────────────┐               │
│         │                 │                 │               │
│         ▼                 ▼                 ▼               │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│  │  IRIS XE    │   │   SILICON   │   │   NEURAL    │       │
│  │  TRADER     │   │   TRADER    │   │   TRADER    │       │
│  │  (GPU)      │   │   (CPU)     │   │   (NPU)     │       │
│  └─────────────┘   └─────────────┘   └─────────────┘       │
│         │                 │                 │               │
│         └─────────────────┼─────────────────┘               │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              GUARDIAN/HEX ENGINE                      │  │
│  │         (Central Bank - Order Clearing)               │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Protocol (`crystal_protocol.py`)

GAMESA_JSON_V1 message definitions:

| Message Type | Direction | Purpose |
|--------------|-----------|---------|
| `MARKET_TICKER` | Broadcast | Telemetry state every 16-50ms |
| `TRADE_ORDER` | Agent → Exchange | Request resource adjustment |
| `DIRECTIVE` | Guardian → Agent | Approval/denial response |

**Commodities (Currencies)**:
- `HEX_COMPUTE` - CPU/GPU utilization (0x00-0xFF)
- `HEX_MEMORY` - Memory pressure
- `HEX_IO` - I/O bandwidth
- `THERMAL_HEADROOM_GPU/CPU` - Degrees until throttling
- `PRECISION_BUDGET` - FP32/FP16/INT8 allowance
- `LATENCY_BUDGET` - Frame time budget

### 2. Socket Control Unit (`crystal_socketd.py`)

The central nervous system:

```python
from crystal_socketd import CrystalSocketd, SocketMode

socketd = CrystalSocketd(mode=SocketMode.MEMORY)
socketd.start()

# Register agents
socketd.register_agent("AGENT_IRIS_XE", "GPU")

# Run market tick
result = socketd.tick({"gpu_temp": 75, "cpu_util": 0.7})
```

**Functions**:
- Ingests telemetry from kernel/Vulkan/OS
- Broadcasts market ticker to all agents
- Arbitrates trade orders through Guardian
- Issues directives to agents

### 3. Trading Agents (`crystal_agents.py`)

| Agent | Input | Decision | Output |
|-------|-------|----------|--------|
| **Iris Xe** | High `hex_compute` | "Can't afford FP32" | Switch to FP16 (XMX) |
| **Silicon** | High `hex_io` | "GPU starving" | Prefetch RAM to L3 |
| **Neural** | GAME_MODE scenario | "Low priority" | Reduce batch size |
| **Cache** | High `hex_memory` | "Anticipate DMA" | Queue prefetch |

**Example Trade**:
```python
order = TradeOrder(
    source="AGENT_IRIS_XE",
    action="REQUEST_FP16_OVERRIDE",
    bid=TradeBid(
        reason="HEX_COMPUTE_HIGH",
        est_thermal_saving=2.0,
        priority=7
    )
)
```

### 4. Guardian/Hex Engine (`guardian_hex.py`)

Central bank functions:

**Hex Depth (Interest Rates)**:
| Level | Value | Effect |
|-------|-------|--------|
| MINIMAL | 0x10 | Low restriction |
| MODERATE | 0x50 | Normal trading |
| HIGH | 0x80 | Cautious approvals |
| EXTREME | 0xC0 | Emergency mode |
| MAXIMUM | 0xFF | Halt most trading |

**Order Clearing Logic**:
```python
guardian = GuardianHexEngine()

# Arbitrate order
result = guardian.clear_order(order, commodities)
# result.approved, result.reason, result.risk_score
```

**Regulations**:
- Thermal limits (GPU > 90°C = DENY_ALL)
- Power limits (> 250W = restrictions)
- Latency floors

### 5. Daemon (`gamesad.py`)

Main runtime orchestrator:

```python
from gamesad import GamesaDaemon, RuntimeConfig

daemon = GamesaDaemon(RuntimeConfig(
    tick_interval_ms=16,
    guardian_mode=GuardianMode.NORMAL
))

daemon.start()
daemon.set_scenario(Scenario.GAME_COMBAT)

# Run ticks
for i in range(100):
    result = daemon.tick({"gpu_temp": 70, "gpu_util": 0.8})

daemon.stop()
```

### 6. Driver Pack (`driver_pack.py`)

Deployment orchestration:

```python
pack = DriverPack()

# Validate environment
validation = pack.validate_environment()
# Checks: cargo, cmake, python, gcc

# Deploy full stack
success, results = pack.deploy()
```

## Protocol Examples

### Market Ticker
```json
{
  "type": "MARKET_TICKER",
  "ts": 17100100,
  "cycle": 42,
  "commodities": {
    "hex_compute": 143,
    "hex_memory": 42,
    "thermal_headroom_gpu": 12.5
  },
  "state": {
    "scenario": "GAME_COMBAT",
    "market_status": "OPEN"
  }
}
```

### Trade Order
```json
{
  "type": "TRADE_ORDER",
  "order_id": "ord_a1b2c3d4",
  "source": "AGENT_IRIS_XE",
  "action": "REQUEST_FP16_OVERRIDE",
  "bid": {
    "reason": "HEX_COMPUTE_HIGH",
    "est_thermal_saving": 2.0,
    "priority": 7
  }
}
```

### Clearing Directive
```json
{
  "type": "DIRECTIVE",
  "target": "AGENT_IRIS_XE",
  "permit_id": "ord_a1b2c3d4",
  "status": "APPROVED",
  "params": {
    "duration_ms": 5000,
    "force_precision": "FP16"
  }
}
```

## Runtime Flow

```
1. Telemetry Ingestion
   └─► CPU/GPU temp, utilization, power

2. Market Ticker Broadcast (16ms)
   └─► All agents receive current prices

3. Agent Analysis
   └─► Each agent evaluates trading opportunity

4. Trade Orders
   └─► Agents submit orders to exchange

5. Guardian Arbitration
   └─► Orders cleared based on risk/regulations

6. Directive Issuance
   └─► Approved agents receive execution permits

7. Agent Execution
   └─► FP16 switch, prefetch, thread park, etc.
```

## Scenarios

| Scenario | Guardian Mode | Typical Actions |
|----------|---------------|-----------------|
| `IDLE` | NORMAL | Minimal trading |
| `GAME_COMBAT` | AGGRESSIVE | FP16 preference, boost allowed |
| `ML_TRAINING` | CONSERVATIVE | Batch reduction, thermal caution |
| Emergency | EMERGENCY | Only thermal-saving orders |

## File Structure

```
src/python/
├── crystal_protocol.py    # GAMESA_JSON_V1 definitions
├── crystal_socketd.py     # Market exchange
├── crystal_agents.py      # Trading agents
├── guardian_hex.py        # Central bank
├── gamesad.py            # Runtime daemon
└── driver_pack.py        # Deployment orchestration
```
