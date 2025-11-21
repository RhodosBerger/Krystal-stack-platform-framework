"""
GAMESA Replay System - Recording and Playback of System State

Provides:
- State recording during runtime
- Deterministic replay for debugging
- Time-travel debugging
- Scenario simulation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum, auto
from collections import deque
import json
import time
import copy


class RecordingMode(Enum):
    """Recording modes."""
    FULL = auto()        # Record everything
    METRICS = auto()     # Only metrics
    DECISIONS = auto()   # Only decisions/actions
    EVENTS = auto()      # Only events


@dataclass
class StateSnapshot:
    """Snapshot of system state at a point in time."""
    cycle: int
    timestamp: float
    telemetry: Dict[str, float]
    decisions: Dict[str, Any]
    events: List[Dict]
    metrics: Dict[str, float]
    checksum: str = ""

    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        """Compute state checksum for verification."""
        data = f"{self.cycle}:{self.timestamp}:{sorted(self.telemetry.items())}"
        return hex(hash(data) & 0xFFFFFFFF)[2:]


@dataclass
class Recording:
    """A complete recording session."""
    recording_id: str
    name: str
    mode: RecordingMode
    snapshots: List[StateSnapshot] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Get recording duration."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    @property
    def cycle_count(self) -> int:
        """Get number of recorded cycles."""
        return len(self.snapshots)


class Recorder:
    """
    Records system state for later playback.
    """

    def __init__(self, mode: RecordingMode = RecordingMode.FULL,
                 max_snapshots: int = 100000):
        self.mode = mode
        self.max_snapshots = max_snapshots
        self._recording: Optional[Recording] = None
        self._is_recording = False
        self._recordings: Dict[str, Recording] = {}
        self._counter = 0

    def start_recording(self, name: str = "", metadata: Dict = None) -> str:
        """Start a new recording session."""
        self._counter += 1
        rec_id = f"rec_{self._counter:04d}_{int(time.time())}"

        self._recording = Recording(
            recording_id=rec_id,
            name=name or rec_id,
            mode=self.mode,
            metadata=metadata or {}
        )
        self._is_recording = True
        return rec_id

    def stop_recording(self) -> Optional[Recording]:
        """Stop current recording and return it."""
        if not self._recording:
            return None

        self._recording.end_time = time.time()
        self._recordings[self._recording.recording_id] = self._recording
        recording = self._recording
        self._recording = None
        self._is_recording = False
        return recording

    def record_state(self, cycle: int, telemetry: Dict, decisions: Dict,
                     events: List[Dict] = None, metrics: Dict = None):
        """Record a state snapshot."""
        if not self._is_recording or not self._recording:
            return

        if len(self._recording.snapshots) >= self.max_snapshots:
            # Rolling buffer - remove oldest
            self._recording.snapshots.pop(0)

        snapshot = StateSnapshot(
            cycle=cycle,
            timestamp=time.time(),
            telemetry=copy.deepcopy(telemetry) if self.mode != RecordingMode.EVENTS else {},
            decisions=copy.deepcopy(decisions) if self.mode in [RecordingMode.FULL, RecordingMode.DECISIONS] else {},
            events=copy.deepcopy(events or []) if self.mode in [RecordingMode.FULL, RecordingMode.EVENTS] else [],
            metrics=copy.deepcopy(metrics or {}) if self.mode in [RecordingMode.FULL, RecordingMode.METRICS] else {}
        )
        self._recording.snapshots.append(snapshot)

    def get_recording(self, rec_id: str) -> Optional[Recording]:
        """Get a recording by ID."""
        return self._recordings.get(rec_id)

    def list_recordings(self) -> List[Dict]:
        """List all recordings."""
        return [
            {
                "id": r.recording_id,
                "name": r.name,
                "cycles": r.cycle_count,
                "duration": r.duration,
                "mode": r.mode.name
            }
            for r in self._recordings.values()
        ]

    def export_recording(self, rec_id: str) -> str:
        """Export recording to JSON."""
        rec = self._recordings.get(rec_id)
        if not rec:
            return "{}"

        data = {
            "id": rec.recording_id,
            "name": rec.name,
            "mode": rec.mode.name,
            "start_time": rec.start_time,
            "end_time": rec.end_time,
            "metadata": rec.metadata,
            "snapshots": [
                {
                    "cycle": s.cycle,
                    "timestamp": s.timestamp,
                    "telemetry": s.telemetry,
                    "decisions": s.decisions,
                    "events": s.events,
                    "metrics": s.metrics,
                    "checksum": s.checksum
                }
                for s in rec.snapshots
            ]
        }
        return json.dumps(data, indent=2)

    def import_recording(self, json_str: str) -> Optional[str]:
        """Import recording from JSON."""
        try:
            data = json.loads(json_str)
            rec = Recording(
                recording_id=data["id"],
                name=data["name"],
                mode=RecordingMode[data["mode"]],
                start_time=data["start_time"],
                end_time=data.get("end_time"),
                metadata=data.get("metadata", {})
            )

            for s in data.get("snapshots", []):
                snap = StateSnapshot(
                    cycle=s["cycle"],
                    timestamp=s["timestamp"],
                    telemetry=s["telemetry"],
                    decisions=s["decisions"],
                    events=s.get("events", []),
                    metrics=s.get("metrics", {}),
                    checksum=s.get("checksum", "")
                )
                rec.snapshots.append(snap)

            self._recordings[rec.recording_id] = rec
            return rec.recording_id
        except Exception:
            return None


class ReplayEngine:
    """
    Replays recorded sessions for analysis and debugging.
    """

    def __init__(self, recording: Recording):
        self.recording = recording
        self._position = 0
        self._speed = 1.0
        self._paused = True
        self._callbacks: Dict[str, Callable[[StateSnapshot], None]] = {}

    @property
    def current_cycle(self) -> int:
        """Get current replay cycle."""
        if self._position < len(self.recording.snapshots):
            return self.recording.snapshots[self._position].cycle
        return -1

    @property
    def progress(self) -> float:
        """Get replay progress (0-1)."""
        if not self.recording.snapshots:
            return 0.0
        return self._position / len(self.recording.snapshots)

    def set_speed(self, speed: float):
        """Set replay speed multiplier."""
        self._speed = max(0.1, min(10.0, speed))

    def play(self):
        """Start/resume replay."""
        self._paused = False

    def pause(self):
        """Pause replay."""
        self._paused = True

    def seek(self, cycle: int) -> bool:
        """Seek to specific cycle."""
        for i, snap in enumerate(self.recording.snapshots):
            if snap.cycle >= cycle:
                self._position = i
                return True
        return False

    def seek_time(self, timestamp: float) -> bool:
        """Seek to specific timestamp."""
        for i, snap in enumerate(self.recording.snapshots):
            if snap.timestamp >= timestamp:
                self._position = i
                return True
        return False

    def step(self) -> Optional[StateSnapshot]:
        """Step forward one cycle."""
        if self._position >= len(self.recording.snapshots):
            return None

        snapshot = self.recording.snapshots[self._position]
        self._position += 1

        # Call registered callbacks
        for callback in self._callbacks.values():
            callback(snapshot)

        return snapshot

    def step_back(self) -> Optional[StateSnapshot]:
        """Step backward one cycle."""
        if self._position <= 0:
            return None

        self._position -= 1
        return self.recording.snapshots[self._position]

    def get_snapshot(self, cycle: int) -> Optional[StateSnapshot]:
        """Get snapshot for specific cycle."""
        for snap in self.recording.snapshots:
            if snap.cycle == cycle:
                return snap
        return None

    def get_range(self, start_cycle: int, end_cycle: int) -> List[StateSnapshot]:
        """Get snapshots in cycle range."""
        return [s for s in self.recording.snapshots
                if start_cycle <= s.cycle <= end_cycle]

    def register_callback(self, name: str, callback: Callable[[StateSnapshot], None]):
        """Register callback for replay events."""
        self._callbacks[name] = callback

    def compare_cycles(self, cycle1: int, cycle2: int) -> Dict[str, Tuple]:
        """Compare two cycles and return differences."""
        snap1 = self.get_snapshot(cycle1)
        snap2 = self.get_snapshot(cycle2)

        if not snap1 or not snap2:
            return {}

        diffs = {}

        # Compare telemetry
        for key in set(snap1.telemetry.keys()) | set(snap2.telemetry.keys()):
            v1 = snap1.telemetry.get(key)
            v2 = snap2.telemetry.get(key)
            if v1 != v2:
                diffs[f"telemetry.{key}"] = (v1, v2)

        # Compare metrics
        for key in set(snap1.metrics.keys()) | set(snap2.metrics.keys()):
            v1 = snap1.metrics.get(key)
            v2 = snap2.metrics.get(key)
            if v1 != v2:
                diffs[f"metrics.{key}"] = (v1, v2)

        return diffs

    def find_anomalies(self, metric: str, threshold: float = 2.0) -> List[Dict]:
        """Find cycles where metric changed significantly."""
        anomalies = []
        prev_value = None

        for snap in self.recording.snapshots:
            value = snap.telemetry.get(metric) or snap.metrics.get(metric)
            if value is None:
                continue

            if prev_value is not None:
                change = abs(value - prev_value)
                if change > threshold:
                    anomalies.append({
                        "cycle": snap.cycle,
                        "metric": metric,
                        "prev": prev_value,
                        "current": value,
                        "change": change
                    })

            prev_value = value

        return anomalies


class Simulator:
    """
    Runs simulations using recorded data as starting point.
    """

    def __init__(self, recording: Recording):
        self.recording = recording
        self._modifications: Dict[int, Dict] = {}

    def modify_at_cycle(self, cycle: int, modifications: Dict):
        """Schedule modifications to inject at a cycle."""
        self._modifications[cycle] = modifications

    def run_simulation(self, from_cycle: int, steps: int,
                       step_fn: Callable[[Dict], Dict]) -> List[StateSnapshot]:
        """
        Run simulation from a cycle.

        Args:
            from_cycle: Starting cycle
            steps: Number of steps to simulate
            step_fn: Function that takes current state and returns next state

        Returns:
            List of simulated snapshots
        """
        # Find starting snapshot
        start_snap = None
        for snap in self.recording.snapshots:
            if snap.cycle >= from_cycle:
                start_snap = snap
                break

        if not start_snap:
            return []

        results = []
        current_state = {
            "telemetry": copy.deepcopy(start_snap.telemetry),
            "decisions": copy.deepcopy(start_snap.decisions),
            "metrics": copy.deepcopy(start_snap.metrics)
        }

        for i in range(steps):
            cycle = from_cycle + i

            # Apply modifications
            if cycle in self._modifications:
                for key, value in self._modifications[cycle].items():
                    if "." in key:
                        category, field = key.split(".", 1)
                        if category in current_state:
                            current_state[category][field] = value
                    else:
                        current_state["telemetry"][key] = value

            # Run simulation step
            current_state = step_fn(current_state)

            # Create snapshot
            snap = StateSnapshot(
                cycle=cycle,
                timestamp=time.time(),
                telemetry=copy.deepcopy(current_state.get("telemetry", {})),
                decisions=copy.deepcopy(current_state.get("decisions", {})),
                events=[],
                metrics=copy.deepcopy(current_state.get("metrics", {}))
            )
            results.append(snap)

        return results


# ============================================================
# DEMO
# ============================================================

def demo():
    """Demonstrate replay system."""
    print("=== GAMESA Replay System Demo ===\n")

    import random

    # Create recorder
    recorder = Recorder(mode=RecordingMode.FULL)

    # Record some data
    rec_id = recorder.start_recording("demo_session", {"version": "1.0"})
    print(f"Started recording: {rec_id}")

    for i in range(50):
        telemetry = {
            "cpu_util": 0.5 + random.gauss(0, 0.1),
            "gpu_temp": 65 + random.gauss(0, 5),
            "fps": 60 + random.gauss(0, 5)
        }
        decisions = {"action": [random.random() for _ in range(3)]}
        metrics = {"reward": random.random()}

        recorder.record_state(i, telemetry, decisions, metrics=metrics)

    recording = recorder.stop_recording()
    print(f"Recorded {recording.cycle_count} cycles over {recording.duration:.2f}s\n")

    # Create replay engine
    engine = ReplayEngine(recording)

    print("--- Replay ---")
    engine.seek(10)
    snap = engine.step()
    print(f"Cycle {snap.cycle}: fps={snap.telemetry['fps']:.1f}")

    snap = engine.step()
    print(f"Cycle {snap.cycle}: fps={snap.telemetry['fps']:.1f}")

    # Compare cycles
    print("\n--- Cycle Comparison ---")
    diffs = engine.compare_cycles(10, 40)
    for key, (v1, v2) in list(diffs.items())[:3]:
        print(f"  {key}: {v1:.2f} -> {v2:.2f}")

    # Find anomalies
    print("\n--- Anomaly Detection ---")
    anomalies = engine.find_anomalies("gpu_temp", threshold=8)
    for a in anomalies[:3]:
        print(f"  Cycle {a['cycle']}: temp changed by {a['change']:.1f}")

    # Simulation
    print("\n--- Simulation ---")
    simulator = Simulator(recording)
    simulator.modify_at_cycle(25, {"gpu_temp": 90})  # Inject thermal spike

    def sim_step(state):
        state["telemetry"]["gpu_temp"] = state["telemetry"].get("gpu_temp", 65) * 0.95 + 65 * 0.05
        return state

    sim_results = simulator.run_simulation(20, 10, sim_step)
    for snap in sim_results[::3]:
        print(f"  Simulated cycle {snap.cycle}: temp={snap.telemetry['gpu_temp']:.1f}")

    # Export
    print(f"\n--- Export ---")
    exported = recorder.export_recording(rec_id)
    print(f"Exported {len(exported)} bytes of JSON")


if __name__ == "__main__":
    demo()
