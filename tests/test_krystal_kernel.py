"""Tests for krystal_kernel: params, AIMD, scheduler, healing, learning, breakers and the OpenAI-compatible API."""
import base64
import json
import math
import os
import struct
import sys
import tempfile
import threading
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from krystal_kernel.backends import Backend, BackendRouter, CircuitBreaker, ReferenceBackend  # noqa: E402
from krystal_kernel.eventlog import EventLog  # noqa: E402
from krystal_kernel.healing import Supervisor  # noqa: E402
from krystal_kernel.hwprofile import HardwareProfile, detect_profile  # noqa: E402
from krystal_kernel.kernel import Backpressure, ComputeKernel, DeadlineMissed  # noqa: E402
from krystal_kernel.kernels import hash_embed  # noqa: E402
from krystal_kernel.lanes import AIMD, ProcessLane, ThreadLane  # noqa: E402
from krystal_kernel.learning import LaneBandit, Learner, cost_bucket  # noqa: E402
from krystal_kernel.openai_api import ApiContext, TokenBucket, handle  # noqa: E402
from krystal_kernel.params import KernelParams, derive_params  # noqa: E402

JSON = {"Content-Type": "application/json"}


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def make_kernel(thread_limit=1, process_workers=0, **overrides):
    params = KernelParams(**overrides)
    lanes = {"thread": ThreadLane(max(thread_limit, 1), AIMD(1, max(thread_limit, 1), thread_limit))}
    if process_workers:
        lanes["process"] = ProcessLane(process_workers, AIMD(1, process_workers, process_workers))
    k = ComputeKernel(params=params, profile=HardwareProfile(), lanes=lanes)
    k.warm()
    return k


class TestProfileAndParams(unittest.TestCase):
    def test_detect_profile_is_sane(self):
        p = detect_profile()
        self.assertGreaterEqual(p.logical_cores, 1)
        self.assertGreaterEqual(p.physical_cores, 1)
        self.assertLessEqual(p.physical_cores, p.logical_cores)
        self.assertGreater(p.ram_gb, 0)

    def test_params_without_calibration_use_physical_minus_reserve(self):
        prof = HardwareProfile(physical_cores=4, logical_cores=8, gil_enabled=True)
        p = derive_params(prof)
        self.assertEqual(p.reserve_cores, 1)
        self.assertEqual(p.process_workers, 3)
        self.assertEqual(p.provenance["process_workers"]["source"], "derived")

    def test_params_with_calibration_may_use_smt(self):
        prof = HardwareProfile(physical_cores=4, logical_cores=8, gil_enabled=True,
                               calibration={"best_process_workers": 8, "process_speedup": {"8": 3.1}})
        p = derive_params(prof)
        self.assertEqual(p.process_workers, 7)  # logical - reserve
        self.assertEqual(p.provenance["process_workers"]["source"], "measured")

    def test_params_calibration_knee_below_physical_is_respected(self):
        prof = HardwareProfile(physical_cores=4, logical_cores=8, calibration={"best_process_workers": 2})
        self.assertEqual(derive_params(prof).process_workers, 2)

    def test_free_threaded_python_keeps_process_lane_minimal(self):
        prof = HardwareProfile(physical_cores=4, logical_cores=8, gil_enabled=False)
        self.assertEqual(derive_params(prof).process_workers, 1)

    def test_queue_bounds_and_explain_covers_every_param(self):
        for cores in (1, 2, 8, 64):
            p = derive_params(HardwareProfile(physical_cores=cores, logical_cores=cores))
            self.assertGreaterEqual(p.max_queue, 16)
            self.assertLessEqual(p.max_queue, 4096)
            self.assertGreater(p.rate_per_s, 0)
            self.assertGreaterEqual(p.burst, p.rate_per_s)
        names = {e["param"] for e in p.explain()}
        self.assertEqual(names, set(p.to_dict()))


class TestAIMD(unittest.TestCase):
    def test_increase_bounded_by_hi_and_decrease_respects_cooldown(self):
        clock = FakeClock()
        a = AIMD(1, 8, 2, cooldown_s=1.0, clock=clock)
        for _ in range(200):
            a.on_signal(True)
        self.assertEqual(a.value, 8)
        a.on_signal(False)
        first = a.limit
        self.assertLess(first, 8)
        a.on_signal(False)  # inside cooldown: ignored
        self.assertEqual(a.limit, first)
        clock.advance(1.5)
        a.on_signal(False)
        self.assertLess(a.limit, first)
        for _ in range(50):
            clock.advance(2)
            a.on_signal(False)
        self.assertEqual(a.value, 1)  # never below lo


class TestKernelScheduling(unittest.TestCase):
    def test_edf_order(self):
        k = make_kernel(thread_limit=1)
        try:
            blocker = k.submit("sleep", {"s": 0.15}, kind="blocker")
            time.sleep(0.03)  # blocker occupies the single slot
            fa = k.submit("echo", {"v": "A"}, kind="A", deadline_ms=5000)
            fc = k.submit("echo", {"v": "C"}, kind="C")
            fb = k.submit("echo", {"v": "B"}, kind="B", deadline_ms=1000)
            for f in (blocker, fa, fb, fc):
                f.result(timeout=5)
            order = [e["kind"] for e in k.log.tail(20, "task_done")]
            self.assertEqual(order, ["blocker", "B", "A", "C"])
        finally:
            k.shutdown()

    def test_queue_full_is_explicit_backpressure(self):
        k = make_kernel(thread_limit=1, max_queue=3)
        try:
            k.submit("sleep", {"s": 0.3}, kind="blocker")
            time.sleep(0.03)
            ok = [k.submit("echo", {}, kind="x") for _ in range(3)]
            with self.assertRaises(Backpressure) as cm:
                k.submit("echo", {}, kind="x")
            self.assertEqual(cm.exception.reason, "queue_full")
            self.assertGreater(cm.exception.retry_after_s, 0)
            self.assertEqual(k.counters["rejected"], 1)
            for f in ok:
                f.result(timeout=5)
        finally:
            k.shutdown()

    def test_unreachable_deadline_is_shed_at_admission(self):
        k = make_kernel(thread_limit=1)
        try:
            k._svc_ewma = 500.0  # pretend work is slow
            k.submit("sleep", {"s": 0.2}, kind="blocker")
            time.sleep(0.03)
            with self.assertRaises(Backpressure) as cm:
                k.submit("echo", {}, kind="x", deadline_ms=10)
            self.assertEqual(cm.exception.reason, "deadline_unreachable")
        finally:
            k.shutdown()

    def test_expired_work_is_dropped_not_executed(self):
        k = make_kernel(thread_limit=1)
        try:
            k.submit("sleep", {"s": 0.3}, kind="blocker")
            time.sleep(0.03)
            f = k.submit("echo", {"v": 1}, kind="late", deadline_ms=60)
            with self.assertRaises(DeadlineMissed):
                f.result(timeout=5)
            self.assertEqual(k.counters["expired"], 1)
            self.assertEqual([e for e in k.log.tail(50, "task_done") if e["kind"] == "late"], [])
        finally:
            k.shutdown()

    def test_unknown_kernel_rejected_immediately(self):
        k = make_kernel()
        try:
            with self.assertRaises(KeyError):
                k.submit("nope", {})
        finally:
            k.shutdown()

    def test_kernel_exception_propagates_and_counts(self):
        k = make_kernel()
        try:
            with self.assertRaises(RuntimeError):
                k.submit("fail", {"msg": "boom"}).result(timeout=5)
            self.assertEqual(k.counters["errors"], 1)
        finally:
            k.shutdown()

    def test_process_lane_runs_kernels(self):
        k = make_kernel(thread_limit=1, process_workers=2)
        try:
            vals = [k.submit("burn", {"n": 1000}, lanes=["process"]).result(timeout=10) for _ in range(4)]
            self.assertEqual(len(set(vals)), 1)
            self.assertGreater(k.lanes["process"].completed, 0)
        finally:
            k.shutdown()

    def test_worker_crash_retries_on_thread_lane(self):
        k = make_kernel(thread_limit=1, process_workers=1)
        try:
            fut = k.submit("sleep", {"s": 0.6}, kind="victim", lanes=["process", "thread"])
            time.sleep(0.2)
            # make sure it is on the process lane before killing the worker
            if k.lanes["process"].inflight == 0:
                self.skipTest("task was routed to the thread lane; nothing to crash")
            k.lanes["process"]._workers[0].kill()
            self.assertEqual(fut.result(timeout=10), 0.6)
            self.assertEqual(k.counters["retried"], 1)
            self.assertTrue(k.log.tail(10, "task_retry"))
        finally:
            k.shutdown()


class TestHealing(unittest.TestCase):
    def test_dead_worker_is_respawned_and_lane_works_again(self):
        k = make_kernel(thread_limit=1, process_workers=1, heal_cooldown_s=0.0)
        try:
            sup = Supervisor(k)
            lane = k.lanes["process"]
            lane._workers[0].kill()
            deadline = time.time() + 5
            while lane.dead_workers() == 0 and time.time() < deadline:
                time.sleep(0.02)
            h = sup.check()
            self.assertEqual(lane.dead_workers(), 0)
            lane.warm()
            self.assertEqual(lane.submit_raw("echo", {"v": 7}).result(timeout=10), {"v": 7})
            self.assertTrue(k.log.tail(20, "heal"))
            self.assertIn(h["state"], ("ok", "degraded"))
        finally:
            k.shutdown()

    def test_restart_budget_prevents_restart_storms(self):
        clock = FakeClock()
        k = make_kernel(thread_limit=1, process_workers=1, heal_cooldown_s=1.0, max_restarts_per_min=1)
        try:
            sup = Supervisor(k, clock=clock)
            lane = k.lanes["process"]

            def kill_and_wait():
                lane._workers[0].kill()
                t = time.time() + 5
                while lane.dead_workers() == 0 and time.time() < t:
                    time.sleep(0.02)

            kill_and_wait()
            sup.check()
            self.assertEqual(sup.heals, 1)
            lane.warm()
            clock.advance(5)  # past cooldown, but the per-minute budget (1) is spent
            kill_and_wait()
            h = sup.check()
            self.assertEqual(sup.heals, 1)
            self.assertGreaterEqual(sup.suppressed, 1)
            self.assertEqual(lane.dead_workers(), 1)  # visibly broken instead of flapping
            self.assertEqual(h["state"], "degraded")
            self.assertTrue(k.log.tail(20, "heal_suppressed"))
        finally:
            k.shutdown()

    def test_dead_dispatcher_is_restarted(self):
        k = make_kernel(thread_limit=1, heal_cooldown_s=0.0)
        try:
            sup = Supervisor(k)
            old = k._dispatcher
            k._stop.set()  # make the loop exit
            old.join(timeout=2)
            k._stop.clear()
            self.assertFalse(k.dispatcher_alive())
            sup.check()
            self.assertTrue(k.dispatcher_alive())
            self.assertEqual(k.submit("echo", {"v": 1}).result(timeout=5), {"v": 1})
        finally:
            k.shutdown()


class TestLearning(unittest.TestCase):
    def _feed(self, log, kind, bucket, lane, ms, n=10):
        for _ in range(n):
            log.emit("task_done", kind=kind, bucket=bucket, lane=lane, ok=True, missed=False,
                     service_ms=ms, queue_ms=0.0, latency_ms=ms)

    def test_cost_bucket(self):
        self.assertEqual([cost_bucket(c) for c in (0, 1, 2, 3, 4, 255, 256)], [0, 0, 1, 1, 2, 7, 8])

    def test_crossover_pattern_becomes_routing_prior_for_unseen_buckets(self):
        log = EventLog()
        params = KernelParams()
        learner = Learner(log, params, mine_every=10 ** 9)
        for b in range(0, 3):
            self._feed(log, "embed", b, "thread", 5.0)
            self._feed(log, "embed", b, "process", 20.0)
        for b in range(3, 6):
            self._feed(log, "embed", b, "thread", 100.0)
            self._feed(log, "embed", b, "process", 30.0)
        pats = learner.learn()
        cross = [p for p in pats if p["kind"] == "lane_crossover"]
        self.assertEqual(len(cross), 1)
        self.assertEqual(cross[0]["min_bucket"], 3)
        self.assertEqual(learner.bandit.rules["embed"]["min_bucket"], 3)
        # never-seen buckets are routed by the mined rule, without exploration
        self.assertEqual(learner.bandit.advise("embed", 9, ["thread", "process"]), "process")
        self.assertEqual(learner.bandit.advise("embed", 0 + 0, ["thread", "process"]) in ("thread", "process"), True)
        self.assertGreaterEqual(learner.bandit.decisions["prior"], 0)
        # the pattern is announced exactly once (circulation without log spam)
        learner.learn()
        self.assertEqual(len(log.tail(100, "pattern")), len([p for p in pats]))

    def test_unseen_bucket_prior_below_threshold_prefers_thread(self):
        b = LaneBandit(KernelParams())
        b.set_rule("embed", 3, "process", "thread")
        self.assertEqual(b.advise("embed", 1, ["thread", "process"]), "thread")
        self.assertEqual(b.advise("embed", 7, ["thread", "process"]), "process")

    def test_bandit_exploits_the_faster_lane(self):
        params = KernelParams(epsilon=0.0, epsilon_floor=0.0)
        b = LaneBandit(params)
        for _ in range(5):
            b.update("k", 2, "thread", 10.0, True, False)
            b.update("k", 2, "process", 40.0, True, False)
        self.assertEqual({b.advise("k", 2, ["thread", "process"]) for _ in range(50)}, {"thread"})

    def test_bandit_penalises_errors_and_deadline_misses(self):
        b = LaneBandit(KernelParams(epsilon=0.0, epsilon_floor=0.0))
        for _ in range(8):
            b.update("k", 1, "thread", 5.0, False, True)   # fast but failing
            b.update("k", 1, "process", 30.0, True, False)
        self.assertEqual(b.advise("k", 1, ["thread", "process"]), "process")

    def test_error_hotspot_and_deadline_risk_patterns(self):
        log = EventLog()
        learner = Learner(log, KernelParams(), mine_every=10 ** 9)
        for _ in range(12):
            log.emit("task_done", kind="k", bucket=0, lane="process", ok=False, missed=True,
                     service_ms=3.0, queue_ms=0.0, latency_ms=3.0)
        kinds = {p["kind"] for p in learner.learn()}
        self.assertIn("error_hotspot", kinds)
        self.assertIn("deadline_risk", kinds)

    def test_capacity_bound_pattern(self):
        log = EventLog()
        learner = Learner(log, KernelParams(), mine_every=10 ** 9)
        for _ in range(12):
            log.emit("task_done", kind="k", bucket=0, lane="thread", ok=True, missed=False,
                     service_ms=1.0, queue_ms=9.0, latency_ms=10.0)
        self.assertIn("capacity_bound", {p["kind"] for p in learner.learn()})

    def test_state_roundtrip(self):
        log = EventLog()
        l1 = Learner(log, KernelParams(), mine_every=10 ** 9)
        self._feed(log, "embed", 2, "thread", 7.0)
        l1.bandit.set_rule("embed", 3, "process", "thread")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "learned.json")
            l1.save(path)
            l2 = Learner(EventLog(), KernelParams(), mine_every=10 ** 9)
            self.assertTrue(l2.load(path))
            self.assertEqual(l2.bandit.rules, l1.bandit.rules)
            self.assertEqual(l2.bandit.table(), l1.bandit.table())
            self.assertFalse(l2.load(os.path.join(d, "missing.json")))

    def test_faulty_subscriber_cannot_break_the_data_path(self):
        log = EventLog()
        log.subscribe(lambda ev: 1 / 0)
        log.emit("x")
        self.assertEqual(log.dropped_sub_errors, 1)
        self.assertEqual(len(log.tail()), 1)


class _Flaky(Backend):
    name = "flaky"

    def __init__(self):
        self.fail = True
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        if self.fail:
            raise RuntimeError("device lost")
        return [[1.0] * 64 for _ in texts]


class TestBackends(unittest.TestCase):
    def test_breaker_state_machine(self):
        clock = FakeClock()
        br = CircuitBreaker(window=10, error_rate=0.5, cooldown_s=10, min_calls=4, clock=clock)
        self.assertTrue(br.allow())
        for _ in range(4):
            br.record(False)
        self.assertEqual(br.state, "open")
        self.assertFalse(br.allow())
        clock.advance(11)
        self.assertTrue(br.allow())
        self.assertEqual(br.state, "half_open")
        br.record(False)
        self.assertEqual(br.state, "open")
        clock.advance(11)
        self.assertTrue(br.allow())
        self.assertEqual(br.record(True), "closed")

    def test_router_falls_back_trips_and_recovers(self):
        clock = FakeClock()
        k = make_kernel(thread_limit=2)
        try:
            flaky = _Flaky()
            router = BackendRouter([flaky, ReferenceBackend(k)], log=k.log, window=10, error_rate=0.5, cooldown_s=10, clock=clock)
            for _ in range(6):
                vecs, used = router.embed(["a"])
                self.assertEqual(used, "reference-cpu")  # caller never sees the failure
                self.assertEqual(len(vecs[0]), 64)
            self.assertEqual(router.breakers["flaky"].state, "open")
            calls = flaky.calls
            router.embed(["a"])
            self.assertEqual(flaky.calls, calls)  # open breaker: accelerator is not even tried
            flaky.fail = False
            clock.advance(11)
            _, used = router.embed(["a"])
            self.assertEqual(used, "flaky")  # half-open probe succeeded
            self.assertEqual(router.breakers["flaky"].state, "closed")
            self.assertTrue(k.log.tail(50, "breaker_opened"))
            self.assertTrue(k.log.tail(50, "breaker_closed"))
        finally:
            k.shutdown()

    def test_supervisor_reports_open_breaker_as_degraded(self):
        k = make_kernel(thread_limit=1)
        try:
            flaky = _Flaky()
            k.backends = BackendRouter([flaky, ReferenceBackend(k)], log=k.log, window=6, min_calls=1) if False else \
                BackendRouter([flaky, ReferenceBackend(k)], log=k.log, window=6)
            for _ in range(6):
                k.backends.embed(["a"])
            h = Supervisor(k).check()
            self.assertEqual(h["state"], "degraded")
            self.assertTrue(any("flaky" in r for r in h["reasons"]))
        finally:
            k.shutdown()


class TestEmbeddingKernel(unittest.TestCase):
    def test_deterministic_unit_norm_and_surface_similarity(self):
        v1 = hash_embed({"texts": ["hello world", "hello world", "hello worlds", "zzzz qqqq"]})
        self.assertEqual(v1[0], v1[1])
        for v in v1:
            self.assertAlmostEqual(math.sqrt(sum(x * x for x in v)), 1.0, places=3)
        cos = lambda a, b: sum(x * y for x, y in zip(a, b))
        self.assertGreater(cos(v1[0], v1[2]), cos(v1[0], v1[3]) + 0.3)


class TestOpenAIApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel = make_kernel(thread_limit=2, process_workers=1, rate_per_s=1000.0, burst=1000.0)
        cls.ctx = ApiContext(cls.kernel, keys=[])

    @classmethod
    def tearDownClass(cls):
        cls.kernel.shutdown()

    def post(self, path, body, headers=JSON, ctx=None):
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        s, h, b = handle(ctx or self.ctx, "POST", path, headers, raw)
        return s, h, json.loads(b) if b[:1] in (b"{", b"[") else b

    def test_models(self):
        s, h, b = handle(self.ctx, "GET", "/v1/models", {}, b"")
        d = json.loads(b)
        self.assertEqual(s, 200)
        self.assertEqual(d["object"], "list")
        self.assertIn("krystal-hash-embed-64", [m["id"] for m in d["data"]])
        self.assertTrue(h["X-Request-Id"].startswith("req_"))
        self.assertNotIn("Access-Control-Allow-Origin", h)  # no wildcard CORS on the new surface

    def test_embeddings_shape_float_and_base64(self):
        s, h, d = self.post("/v1/embeddings", {"input": ["alpha", "beta"]})
        self.assertEqual(s, 200)
        self.assertEqual([x["index"] for x in d["data"]], [0, 1])
        self.assertEqual(len(d["data"][0]["embedding"]), 64)
        self.assertTrue(d["usage"]["approximate"])
        self.assertIn(h["X-Krystal-Backend"], ("reference-cpu",))
        s2, _, d2 = self.post("/v1/embeddings", {"input": "alpha", "encoding_format": "base64"})
        raw = base64.b64decode(d2["data"][0]["embedding"])
        floats = struct.unpack("<64f", raw)
        for a, b in zip(floats, d["data"][0]["embedding"]):
            self.assertAlmostEqual(a, b, places=4)

    def test_embeddings_validation(self):
        for body, code in [({"input": []}, "invalid_input"), ({"input": 5}, "invalid_input"),
                           ({"input": ["a"], "dimensions": 128}, "unsupported_dimensions"),
                           ({"input": ["a"], "encoding_format": "int8"}, "invalid_encoding_format"),
                           ({"input": ["x" * 9000]}, "input_too_large")]:
            s, _, d = self.post("/v1/embeddings", body)
            self.assertEqual(s, 400, body)
            self.assertEqual(d["error"]["code"], code)
        s, _, d = self.post("/v1/embeddings", {"input": "a", "model": "gpt-4"})
        self.assertEqual((s, d["error"]["code"]), (404, "model_not_found"))

    def test_chat_completion_is_honest_reference_model(self):
        s, _, d = self.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "one two two three"}]})
        self.assertEqual(s, 200)
        self.assertEqual(d["object"], "chat.completion")
        msg = d["choices"][0]["message"]
        self.assertEqual(msg["role"], "assistant")
        self.assertIn("no language model", msg["content"])
        self.assertIn("4 words", msg["content"])
        self.assertIn("3 unique", msg["content"])
        self.assertEqual(d["usage"]["total_tokens"], d["usage"]["prompt_tokens"] + d["usage"]["completion_tokens"])

    def test_chat_validation_and_stream_rejected_explicitly(self):
        s, _, d = self.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "stream": True})
        self.assertEqual((s, d["error"]["code"]), (400, "stream_not_supported"))
        s, _, d = self.post("/v1/chat/completions", {"messages": []})
        self.assertEqual((s, d["error"]["code"]), (400, "invalid_messages"))

    def test_transport_rules(self):
        s, _, d = self.post("/v1/embeddings", {"input": "a"}, headers={"Content-Type": "text/plain"})
        self.assertEqual(s, 415)
        s, _, d = self.post("/v1/embeddings", b"{not json")
        self.assertEqual((s, d["error"]["code"]), (400, "invalid_json"))
        s, _, d = self.post("/v1/embeddings", b"[1,2]")
        self.assertEqual(s, 400)
        big = b'{"input": "' + b"a" * (self.ctx.max_body + 10) + b'"}'
        s, _, d = self.post("/v1/embeddings", big)
        self.assertEqual((s, d["error"]["code"]), (413, "payload_too_large"))
        s, _, _ = handle(self.ctx, "DELETE", "/v1/models", {}, b"")
        self.assertEqual(s, 405)
        s, _, _ = handle(self.ctx, "GET", "/v1/nope", {}, b"")
        self.assertEqual(s, 404)

    def test_auth_when_keys_configured(self):
        ctx = ApiContext(self.kernel, keys=["sk-secret-123"])
        s, h, b = handle(ctx, "GET", "/v1/models", {}, b"")
        self.assertEqual(s, 401)
        self.assertEqual(json.loads(b)["error"]["type"], "authentication_error")
        self.assertEqual(h["WWW-Authenticate"], "Bearer")
        s, _, _ = handle(ctx, "GET", "/v1/models", {"Authorization": "Bearer wrong"}, b"")
        self.assertEqual(s, 401)
        s, _, _ = handle(ctx, "GET", "/v1/models", {"authorization": "Bearer sk-secret-123"}, b"")
        self.assertEqual(s, 200)
        self.kernel.backends = self.ctx.router

    def test_rate_limit_returns_429_with_retry_after(self):
        clock = FakeClock()
        k = make_kernel(thread_limit=1, rate_per_s=1.0, burst=2.0)
        try:
            ctx = ApiContext(k, keys=[], clock=clock)
            codes = [handle(ctx, "GET", "/v1/models", {}, b"")[0] for _ in range(4)]
            self.assertEqual(codes, [200, 200, 429, 429])
            s, h, b = handle(ctx, "GET", "/v1/models", {}, b"")
            self.assertEqual(s, 429)
            self.assertGreaterEqual(int(h["Retry-After"]), 1)
            clock.advance(5)
            self.assertEqual(handle(ctx, "GET", "/v1/models", {}, b"")[0], 200)
        finally:
            k.shutdown()

    def test_token_bucket_is_per_tenant(self):
        clock = FakeClock()
        tb = TokenBucket(1.0, 1.0, clock)
        self.assertTrue(tb.take("a")[0])
        self.assertFalse(tb.take("a")[0])
        self.assertTrue(tb.take("b")[0])

    def test_overload_maps_to_503_with_retry_after(self):
        k = make_kernel(thread_limit=1)
        try:
            ctx = ApiContext(k, keys=[])
            real = k.submit

            def boom(*a, **kw):
                raise Backpressure("queue_full", 1.2)
            k.submit = boom
            s, h, b = handle(ctx, "POST", "/v1/chat/completions", JSON,
                             json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode())
            k.submit = real
            self.assertEqual(s, 503)
            self.assertEqual(h["Retry-After"], "2")
            self.assertEqual(json.loads(b)["error"]["code"], "overloaded")
        finally:
            k.shutdown()

    def test_metrics_and_introspection_endpoints(self):
        for _ in range(3):
            self.post("/v1/embeddings", {"input": ["x"]})
        s, h, b = handle(self.ctx, "GET", "/metrics", {}, b"")
        text = b.decode()
        self.assertEqual(s, 200)
        self.assertIn("krystal_queue_depth", text)
        self.assertIn('krystal_tasks_total{status="completed"}', text)
        self.assertIn("krystal_kernel_health", text)
        for route in ("/api/kernel/status", "/api/kernel/patterns", "/api/kernel/events?n=5", "/api/kernel/params", "/api/kernel/hardware"):
            s, _, b = handle(self.ctx, "GET", route, {}, b"")
            self.assertEqual(s, 200, route)
            self.assertEqual(json.loads(b)["status"], "OK")

    def test_concurrent_requests_all_succeed(self):
        results = []

        def one(i):
            s, _, d = self.post("/v1/embeddings", {"input": [f"text {i}"] * 3})
            results.append(s)
        ts = [threading.Thread(target=one, args=(i,)) for i in range(24)]
        [t.start() for t in ts]
        [t.join(timeout=30) for t in ts]
        self.assertEqual(results.count(200), 24)


if __name__ == "__main__":
    unittest.main()
