# KRYSTAL-STACK: COMPLETE DEVOPS & CONTINUOUS DELIVERY PLAN
## Autonomous Daemon Orchestration, Automated Test Gates, Godot Headless Pipeline & MCP Monitoring

**Document Version:** 1.0.0  
**Classification:** DevOps & Systems Engineering Blueprint  
**Framework:** Krystal-Stack Platform Framework  
**Target Environments:** Windows 11 (Vulkan / Python 3.14), Godot 4.x Headless, MCP Servers  
**Active Scheduled Cron:** `task-3835` (`*/30 * * * *`)  
**Date:** 2026-10-03  

---

## 1. DevOps System Architecture & Overview

The Krystal-Stack DevOps architecture coordinates multiple continuous-running subsystems without third-party cloud lock-in:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   KRYSTAL-STACK DEVOPS ARCHITECTURE                    │
├───────────────────────────┬────────────────────────────────────────────┤
│   CONTINUOUS INTEGRATION  │  ORCHESTRATION & RUNTIMES                  │
│   • 189+ Unit/Integ Tests │  • Port 8089: krystal_engine_core.py       │
│   • 100% Pass Enforced    │  • Port 8080: krystal_web_hub (Static)     │
│   • Double-Entry Auditing │  • Task task-3835: Recurring DevOps Cron   │
├───────────────────────────┼────────────────────────────────────────────┤
│   GODOT 4 EXPORT PIPELINE │  MCP ECOSYSTEM & LIVE TELEMETRY            │
│   • Headless .TSCN Valid  │  • krystal_mcp_server.py (9 Tools)         │
│   • AST Compaction Engine │  • Dual-Plane Bridge to WordPress/Oxygen   │
│   • Vulkan HW Benchmark   │  • Zero-Downtime Hot-Reload & Port Rebind  │
└───────────────────────────┴────────────────────────────────────────────┘
```

---

## 2. Phase 1: Local Daemon Orchestration & Zero-Downtime Socket Rebinding

### 2.1 The Windows Socket Zombie Problem
On Windows NT systems, killing a parent Python daemon often leaves background socket bindings alive in the `TIME_WAIT` or zombie state, preventing immediate restart with `OSError: [WinError 10048] Only one usage of each socket address is normally permitted`.

### 2.2 Orchestration Automation
The DevOps pipeline utilizes atomic PowerShell process termination prior to re-launch:
```powershell
# DevOps automated clean socket rebind script
$targetPort = 8089
$pids = Get-NetTCPConnection -LocalPort $targetPort -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
foreach ($p in $pids) {
    if ($p -gt 0) {
        Write-Host "[DEVOPS] Terminating zombie process on port $targetPort: PID $p"
        Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
    }
}
Start-Sleep -Milliseconds 800
# Launch daemon with unbuffered stdout
python -u -m krystal_web_hub.krystal_engine_core
```

---

## 3. Phase 2: Autonomous Continuous Integration Gates (CI)

Every code modification triggers the mandatory test discovery gate. A single failure halts deployment.

### 3.1 Test Suite Matrix (100% Pass Rule)

| Test Suite File | Domain Covered | Assertions / Tests |
| :--- | :--- | :--- |
| `tests/test_range_and_animation_system.py` | Axial distance, Melee vs Ranged, Bezier trajectories | 11 tests |
| `tests/test_krystal_mcp_architecture.py` | Native MCP server tools, declarative AST, Oxygen bridge | 10 tests |
| `tests/test_sector_fight_system.py` | Sector assault, garrison damage, annexation | 9 tests |
| `tests/test_economic_framework.py` | Double-entry dual-earn ledger, resource caps | 8 tests |
| `tests/test_krystal_engine_and_3d_builder.py`| 9-card registry, Godot .TSCN generation, assets | 12 tests |
| `tests/test_godot_theoretical_formulas.py` | 6 new mathematical formulas (SDF, Bezier drag, Fusion) | 12 tests |
| `tests/test_*.py` (Remaining 10 suites) | Vulkan compute, cyclic kernel, bot RAG, invoice CRUD | 127 tests |
| **TOTAL VERIFIED SUITE** | **Platform-Wide Coverage** | **189+ Tests (100% Pass)** |

---

## 4. Phase 3: Godot 4 Headless Export & AST Scene Verification

The engine generates declarative `.tscn` Godot scenes in real time. The CI pipeline validates these scenes headlessly:
1. **Grammar & Lexical Check:** Ensures every generated `.tscn` has valid section headers (`[gd_scene]`, `[ext_resource]`, `[node]`).
2. **Resource Integrity:** Asserts that referenced mesh assets (`crystal_shard.obj`, `acid_slime.obj`, `earth_roots.obj`) exist and have non-zero file sizes.
3. **AST Compaction Ratio:** Enforces that compacted ASTs achieve $\ge 8\times$ size reduction compared to uncompressed JSON.

---

## 5. Phase 4: MCP Protocol Continuous Deployment & Health Monitoring

The Krystal MCP Server (`krystal_mcp_server.py`) exposes 9 native tools to external IDEs and agents:
- `krystal-get-instructions`
- `krystal-discover-primitives`
- `krystal-declarative-to-scene`
- `krystal-insert-theme-tokens`
- `krystal-bind-ledger-data`
- `krystal-set-spatial-conditions`
- `krystal-preview-scene`
- `krystal-edit-node`
- `krystal-bridge-to-oxygen`

### 5.1 Continuous Health Verification
The scheduled task `task-3835` executes every 30 minutes:
1. Pings HTTP endpoint `GET http://127.0.0.1:8089/api/health`.
2. Validates that `cards_count >= 9` and `active_subscribers >= 0`.
3. Runs the test suite via `python -m unittest discover tests`.
4. Emits a high-priority telemetry notice if any invariant is breached.

---

## 6. Phase 5: Production Rollback & Fail-Safe Strategy

In the event of an unhandled runtime exception or test suite regression:
1. **Automatic Fallback to In-Memory Snapshot:** `ACTIVE_ECONOMIC_MATCH` and `GAME_STATE` maintain a rollback stack of the last 10 turns.
2. **Re-initialization Endpoint:** `POST /api/economy/reset` re-arms the match to pristine conditions without requiring server restart.
3. **Log Rotation:** Logs in `.system_generated/tasks/` are trimmed at 10MB to prevent storage bloat.
