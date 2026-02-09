# Nightmare Training Module

The Nightmare Training module implements the offline learning protocol for the FANUC RISE v2.1 Advanced CNC Copilot system. This module enables the system to learn from failure scenarios during machine idle time, improving its safety response and resilience without risking physical hardware.

## Architecture

The Nightmare Training system consists of three main components:

### 1. Adversary Component (`adversary.py`)
The Adversary injects synthetic failures into clean telemetry logs to create training scenarios. It implements various failure types:

- **Spindle Load Spike**: Simulates tool breakage scenarios with sudden load increases up to 200%
- **Thermal Runaway**: Models coolant failure with temperature rising at 10°C/sec
- **Phantom Trauma**: Creates high vibration with low load scenarios (sensor drift)
- **Vibration Anomaly**: Simulates bearing wear or imbalance issues
- **Coolant Failure**: Tests temperature vs. coolant flow relationships
- **Tool Breakage**: Combines multiple failure symptoms simultaneously

### 2. Dreamer Component (`dreamer.py`)
The Dreamer runs the Shadow Council (Auditor & Dopamine Engine) against modified telemetry data during the "Dream State". It:

- Processes each data point through the Shadow Council evaluation
- Analyzes failure detection and response effectiveness
- Tracks kill switch triggers and cortisol spikes
- Determines if responses were preemptive (before theoretical failure)
- Identifies missed failures requiring policy updates
- Updates dopamine policies based on learning insights

### 3. Orchestrator Component (`orchestrator.py`)
The Orchestrator coordinates the entire Nightmare Training process:

- Loads historical telemetry data (REM Cycle)
- Generates failure scenarios to inject
- Runs simulation loops with the Dreamer
- Consolidates learning from multiple scenarios
- Updates policies based on consolidated insights
- Schedules training sessions for multiple machines

## Usage

### Running Nightmare Training

```bash
python -m cms.nightmare_training --machine-id 1 --duration 2.0 --probability 0.7 --verbose
```

Options:
- `--machine-id`: Specific machine ID to train on (default: all machines)
- `--duration`: Hours of historical data to replay (default: 1.0)
- `--probability`: Probability of injecting failures (0.0 to 1.0, default: 0.7)
- `--idle-check`: Check if machine is idle before training
- `--verbose`: Enable verbose logging

### Programmatic Usage

```python
from cms.nightmare_training.orchestrator import NightmareTrainingOrchestrator
from cms.app_factory import create_app
from cms.models import get_session_local, create_database_engine
from cms.repositories.telemetry_repository import TelemetryRepository

# Initialize components
engine = create_database_engine()
db_session = get_session_local(engine)()

telemetry_repo = TelemetryRepository(db_session)
# ... initialize other components as shown in __main__.py

# Create orchestrator
orchestrator = NightmareTrainingOrchestrator(shadow_council, telemetry_repo)

# Run training session
results = orchestrator.run_nightmare_training_session(
    machine_id=1,
    duration_hours=2.0,
    failure_probability=0.7
)
```

## Key Features

### 1. REM Cycle (Data Replay)
- Loads high-resolution telemetry logs from TimescaleDB
- Reconstructs exact physical state of the machine during previous shifts
- Maintains temporal relationships and sequences

### 2. Adversary (Fault Injection)
- Systematically injects "Chaos" into clean logs
- Multiple failure scenarios per session
- Configurable severity and probability

### 3. Dreamer (Simulation Loop)
- Runs Shadow Council against modified timelines
- Tests if system triggers "Kill Switch" or "Cortisol Spike" before theoretical failure
- Measures response effectiveness

### 4. Memory Consolidation (Policy Update)
- Updates dopamine_policy.json based on missed failures
- Increases sensitivity to precursor patterns that were overlooked
- Improves future response capabilities

## Benefits

1. **Risk-Free Learning**: Systems learn from catastrophic scenarios without physical hardware risk
2. **Improved Response Time**: Faster recognition of dangerous patterns
3. **Enhanced Safety**: Better anticipation of failure conditions
4. **Reduced False Positives**: Refinement of detection thresholds
5. **Proactive Protection**: Shift from reactive safety to proactive resilience

## Integration with Shadow Council

The Nightmare Training module works closely with the Shadow Council governance pattern:

- **Creator Agent**: Helps generate optimization strategies based on learned trauma
- **Auditor Agent**: Validates responses against physics constraints using Death Penalty function
- **Accountant Agent**: Evaluates economic impact of learned responses

## Theoretical Foundation

Based on biological memory consolidation principles where the brain replays experiences during sleep to strengthen important memories and forget irrelevant details. In the manufacturing context, this translates to:

- Replaying past operations during idle time
- Injecting failure scenarios to test response readiness
- Consolidating learning into improved safety policies
- Strengthening recognition of dangerous patterns

## Output Files

- `nightmare_training.log`: Training session logs
- `dopamine_policy.json`: Updated policy files based on learning
- Training statistics and performance metrics

## Safety Considerations

- All training occurs during machine idle time
- No impact on production operations
- Policies are validated before deployment
- Gradual sensitivity adjustments to avoid excessive false alarms