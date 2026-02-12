import os
import time
from typing import Dict, Any

from .config import RuntimeConfig
from .evolution import EvolutionaryTuner
from .grid_memory import GridMemory3D
from .inference import OpenVinoInferenceEngine
from .logging_system import JsonLogger
from .planning_economics import EconomicPlanner
from .runtime_policy import RuntimePolicyApplier
from .scheduler import TimerScheduler
from .telemetry import TelemetryCollector
from .windows_api import WindowsApiAdapter


class OVOOrchestrator:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.telemetry = TelemetryCollector()
        self.scheduler = TimerScheduler()
        self.grid = GridMemory3D(config.grid_x, config.grid_y, config.grid_z)
        self.tuner = EvolutionaryTuner(config.population_size, config.mutation_rate)
        self.inference = OpenVinoInferenceEngine()
        log_dir = os.getenv("OVO_LOG_DIR", "logs")
        self.logger = JsonLogger(log_path=f"{log_dir}/runtime_log.jsonl")
        self.planner = EconomicPlanner(base_interval=config.interval_seconds)
        self.windows_api = WindowsApiAdapter()
        self.policy_applier = RuntimePolicyApplier()

        self.telemetry_task_id = self.scheduler.add_periodic_task("telemetry", config.interval_seconds)
        self.inference_task_id = self.scheduler.add_periodic_task("inference", config.interval_seconds)

    def run(self) -> None:
        last_result = None
        i = 0
        while self.config.cycles <= 0 or i < self.config.cycles:
            now = time.time()
            due = self.scheduler.due_tasks(now)

            snapshot = None
            result = None
            plan = None
            win_state = self.windows_api.sample()
            self.logger.event("system", {"cycle": i, **win_state.as_dict()})

            for task in due:
                if task.task_id == self.telemetry_task_id:
                    snapshot = self.telemetry.sample()
                    self.logger.event("telemetry", snapshot.as_dict())

                if task.task_id == self.inference_task_id:
                    if snapshot is None:
                        snapshot = self.telemetry.sample()
                    plan = self.planner.decide(snapshot, last_result)
                    self.logger.event("planning", {"cycle": i, **plan.as_dict()})

                    params = self.tuner.step(reward_signal=plan.reward_signal)
                    threads = max(1, min(params["threads"], plan.threads_cap))
                    streams = max(1, min(params["streams"], plan.streams_cap))
                    policy = self.policy_applier.apply(threads=threads, streams=streams, mode=plan.mode)
                    self.logger.event("policy", {"cycle": i, **policy.as_dict()})
                    result = self.inference.run(threads=threads, streams=streams)
                    last_result = result
                    self.logger.event(
                        "inference",
                        {
                            "cycle": i,
                            "threads": threads,
                            "streams": streams,
                            "mode": plan.mode,
                            "economics_score": plan.economics_score,
                            **result.as_dict(),
                        },
                    )

            grid_payload: Dict[str, Any] = {
                "cycle": i,
                "snapshot": snapshot.as_dict() if snapshot else None,
                "inference": result.as_dict() if result else None,
                "plan": plan.as_dict() if plan else None,
                "windows_api": win_state.as_dict(),
            }
            coord = self.grid.auto_place(i, grid_payload)
            stats = self.grid.stats()
            self.logger.event(
                "grid_update",
                {
                    "cycle": i,
                    "coord": coord,
                    "used_cells": stats.used_cells,
                    "capacity": stats.capacity,
                },
            )

            sleep_for = plan.target_interval if plan else self.config.interval_seconds
            time.sleep(max(0.01, sleep_for))
            i += 1
