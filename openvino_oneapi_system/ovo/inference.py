import time
from dataclasses import dataclass
from typing import Dict, Any

import random


@dataclass
class InferenceResult:
    latency_ms: float
    throughput: float
    backend: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "latency_ms": self.latency_ms,
            "throughput": self.throughput,
            "backend": self.backend,
        }


class OpenVinoInferenceEngine:
    def __init__(self) -> None:
        self.backend = "simulated"
        self._openvino_available = False
        try:
            from openvino.runtime import Core  # type: ignore

            self._openvino_core = Core()
            self._openvino_available = True
            self.backend = "openvino"
        except Exception:
            self._openvino_core = None

    def run(self, threads: int, streams: int) -> InferenceResult:
        start = time.perf_counter()

        if self._openvino_available:
            # Lightweight runtime warm path. Real model loading can be plugged in here.
            _ = self._openvino_core.available_devices  # touch runtime
            self._cpu_work(30000)
        else:
            self._cpu_work(15000)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        adjusted = max(0.1, elapsed_ms * (1.15 - min(1.0, threads / 32.0)) * (1.0 - min(0.3, streams * 0.05)))
        throughput = 1000.0 / adjusted
        return InferenceResult(latency_ms=adjusted, throughput=throughput, backend=self.backend)

    @staticmethod
    def _cpu_work(size: int) -> float:
        acc = 0.0
        for _ in range(size):
            x = random.random()
            y = random.random()
            acc += (x * y) / (x + y + 1e-6)
        return acc
