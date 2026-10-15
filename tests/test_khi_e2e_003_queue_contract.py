import unittest
from collections import deque
from dataclasses import dataclass
from typing import Iterable

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP


@dataclass(frozen=True)
class QueueRunResult:
    arrivals: int
    completions: int
    drops: int
    max_depth: int
    deadline_misses_125us: int
    deadline_misses_250us: int
    deadline_misses_500us: int
    max_age_us: float


class DeterministicQueueModel:
    """Deterministic single-server contract model for KHI-E2E-003.

    This does not replace the physical benchmark. It validates the classification
    logic for sustained 8 kHz ingestion, bounded queues, deadline misses and drops.
    """

    def __init__(self, arrival_period_us: float = 125.0, capacity: int = 4096):
        self.arrival_period_us = float(arrival_period_us)
        self.capacity = int(capacity)

    def run(self, service_times_us: Iterable[float]) -> QueueRunResult:
        queue = deque()
        server_free_at = 0.0
        arrivals = 0
        completions = 0
        drops = 0
        max_depth = 0
        miss125 = 0
        miss250 = 0
        miss500 = 0
        max_age = 0.0

        for index, service_us in enumerate(service_times_us):
            arrival_at = index * self.arrival_period_us
            arrivals += 1

            while queue and queue[0][1] <= arrival_at:
                _arrival, _done, age = queue.popleft()
                completions += 1
                miss125 += int(age > 125.0)
                miss250 += int(age > 250.0)
                miss500 += int(age > 500.0)
                max_age = max(max_age, age)

            if len(queue) >= self.capacity:
                queue.popleft()
                drops += 1

            start_at = max(arrival_at, server_free_at)
            done_at = start_at + float(service_us)
            age = done_at - arrival_at
            queue.append((arrival_at, done_at, age))
            server_free_at = done_at
            max_depth = max(max_depth, len(queue))

        while queue:
            _arrival, _done, age = queue.popleft()
            completions += 1
            miss125 += int(age > 125.0)
            miss250 += int(age > 250.0)
            miss500 += int(age > 500.0)
            max_age = max(max_age, age)

        return QueueRunResult(
            arrivals=arrivals,
            completions=completions,
            drops=drops,
            max_depth=max_depth,
            deadline_misses_125us=miss125,
            deadline_misses_250us=miss250,
            deadline_misses_500us=miss500,
            max_age_us=max_age,
        )


def analytical_mean_service_us(t_ingest_us: float, t_control_us: float, k: int) -> float:
    if k <= 0:
        raise ValueError("K must be >= 1")
    return float(t_ingest_us) + float(t_control_us) / float(k)


def traffic_intensity(arrival_hz: float, mean_service_us: float) -> float:
    service_hz = 1_000_000.0 / mean_service_us
    return float(arrival_hz) / service_hz


class TestKhiE2E003QueueContract(unittest.TestCase):
    def setUp(self):
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_8khz_quantum_is_125_microseconds(self):
        self.assertAlmostEqual(1_000_000.0 / 8000.0, 125.0)

    def test_k1_model_is_analytically_unstable_for_reported_means(self):
        service = analytical_mean_service_us(26.0, 135.0, 1)
        self.assertEqual(service, 161.0)
        self.assertGreater(traffic_intensity(8000.0, service), 1.0)

    def test_k2_model_is_analytically_stable_for_reported_means(self):
        service = analytical_mean_service_us(26.0, 135.0, 2)
        self.assertEqual(service, 93.5)
        self.assertLess(traffic_intensity(8000.0, service), 1.0)

    def test_k8_model_has_large_stability_margin(self):
        service = analytical_mean_service_us(26.0, 135.0, 8)
        self.assertAlmostEqual(service, 42.875)
        self.assertLess(traffic_intensity(8000.0, service), 0.5)

    def test_service_below_quantum_is_bounded_without_drops(self):
        model = DeterministicQueueModel(capacity=4096)
        result = model.run([100.0] * 20_000)
        self.assertEqual(result.drops, 0)
        self.assertEqual(result.arrivals, result.completions)
        self.assertLessEqual(result.max_depth, 1)
        self.assertEqual(result.deadline_misses_125us, 0)

    def test_service_above_quantum_accumulates_queue_age(self):
        model = DeterministicQueueModel(capacity=4096)
        result = model.run([140.0] * 2_000)
        self.assertGreater(result.max_depth, 1)
        self.assertGreater(result.deadline_misses_125us, 0)
        self.assertGreater(result.max_age_us, 125.0)

    def test_bounded_capacity_eventually_drops_when_overloaded(self):
        model = DeterministicQueueModel(capacity=32)
        result = model.run([200.0] * 2_000)
        self.assertGreater(result.drops, 0)

    def test_k_must_be_positive(self):
        with self.assertRaises(ValueError):
            analytical_mean_service_us(26.0, 135.0, 0)


if __name__ == "__main__":
    unittest.main()
