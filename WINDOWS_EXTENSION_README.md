# Windows Extension for GAMESA/KrystalStack Framework

This extension enhances the GAMESA framework with Windows-specific capabilities, including registry optimization, hardware monitoring, performance counters, and advanced process management. It implements an economic resource trading system specifically tailored for Windows environments.

## Features

- **Registry Management**: Advanced Windows registry optimization and backup capabilities
- **WMI Integration**: Hardware monitoring through Windows Management Instrumentation
- **Service Management**: Windows service control and optimization
- **Performance Counters**: Access to Windows performance monitoring data
- **Process Management**: Process priority and thread affinity control
- **Economic Resource Trading**: Economic model for trading Windows system resources
- **3D Grid Memory Integration**: Memory allocation based on 3D coordinate system
- **Cross-Forex Resource Market**: Treat Windows resources as tradable assets

## Architecture

The Windows extension follows the same dual-layer architecture as the main GAMESA framework:

```
┌─────────────────────────────────────────────────────────────┐
│                     GUARDIAN LAYER (Python)                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ Windows Manager │  │  Registry Mgr   │  │ WMI Manager │ │
│  │                 │  │                 │  │             │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
│         │                       │                    │      │
│  ┌─────────────────────────────────────────────────────────┤
│  │              Windows Resource Manager                   │
│  │  (Registry, Services, Performance Counters, Processes)  │
│  └─────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────┘
```

## Windows-Specific Resource Types

The extension introduces Windows-specific resource types that can be traded economically:

- `PROCESS_PRIORITY`: Process priority levels (Idle, Normal, High, Realtime)
- `THREAD_AFFINITY`: CPU core affinity for threads
- `PERFORMANCE_COUNTER`: Access to Windows performance counters
- `SERVICE_CONTROL`: Windows service management permissions
- `REGISTRY_QUOTA`: Registry space allocation

## Installation

The extension requires the following optional dependencies for full functionality:
- `wmi`: For Windows Management Instrumentation access
- `pywin32`: For Windows service management

Install with:
```bash
pip install wmi pywin32
```

## Usage

The Windows extension integrates seamlessly with the existing GAMESA framework:

```python
from windows_extension import WindowsExtensionManager, WindowsOptimizationAgent

# Initialize the Windows extension
windows_ext = WindowsExtensionManager()

# Create an optimization agent
agent = WindowsOptimizationAgent("MyGameOptimizer")

# Optimize a specific process
agent.optimize_performance_critical_process("game.exe", priority=WindowsPriority.ABOVE_NORMAL)

# Run continuous optimization
agent.run_continuous_optimization(interval_seconds=5.0)
```

## Economic Model

The extension implements an economic model for Windows resources:

- Each resource type has a market price based on availability
- Agents bid for resources using internal credits
- Resource allocation decisions consider cost, benefit, and risk
- Cooldown mechanisms prevent resource exhaustion

## Compatibility

The Windows extension maintains full compatibility with the existing GAMESA framework while adding Windows-specific capabilities. All existing functionality continues to work as expected.

## Safety Features

- Registry change backups before modification
- Process safety checks to prevent system instability
- Resource allocation limits to prevent exhaustion
- Service management with appropriate permissions checks