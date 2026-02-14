# Open Mechanics: Insider Preview (v1.4.0)

Welcome to the **Insider Preview** of the Open Mechanics "Cortex Engine".
This release includes the full suite of Cognitive Manufacturing tools:

1.  **Cortex Engine**: The unified orchestrator.
2.  **Hex Trace**: The "Black Box" flight recorder.
3.  **Dopamine Engine**: The OpenVINO-powered Neural Safety system.
4.  **Evolutionary Discovery**: The Genetic Algorithm optimizer.

## Quick Start (Demo)

We have included a stress-test demo script to validate the system's "Consciousness" under load.

### 1. Run the Cortex Server
The system runs as a background service (or you can run it manually):
```bash
python3 /opt/advanced_cnc_copilot/main.py
```

### 2. Run the Stress Test (Requires Sysbench)
This script simulates a heavy manufacturing load while querying the Cortex for its internal state.
```bash
/opt/advanced_cnc_copilot/backend/benchmarks/stress_test_cortex.sh
```

**What to look for:**
- **Hex Trace**: You will see a `Trace: <HEX STRING>` in the output. This is the cryptographic proof of the machine's state.
- **Spectrum**: The "Covalent" state of the machine.
- **Latency**: Even under 100% CPU load, the Cortex should respond in <50ms.

## Features to Test

### Neural Interdiction
The system is equipped with `phantom_net.onnx`. It will automatically block "Phantom Trauma" events (where the machine *feels* stress but is physically safe).
- **Verify**: Check logs for "Phantom Trauma Detected" during operation.

### Evolutionary Optimization
The system "dreams" of better paths.
- **Trigger**: Send a POST request to `/cortex/evolve` with a sample profile.
- **Result**: The system will return a mutated profile with higher efficiency.

## Feedback
Please report any "Stability Breaks" (Covalent Bond failures) to the architecture team.
