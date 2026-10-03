"""Tests for krystal_bot. Synthetic data below exists only inside tests, to exercise the acceptance gates."""
import base64
import json
import os
import random
import shutil
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from krystal_bot import bytecode as BC
from krystal_bot import composer as CP
from krystal_bot import donor as DN
from krystal_bot import morse as MO
from krystal_bot import risk as RK
from krystal_bot.api import handle
from krystal_bot.llm_gateway import (GatewayDenied, LLMConfig, LLMGateway, analyze_binary_log, redact, safe_eval_math)
from krystal_bot.mesh import Mesh, MeshAuth, MeshNode, is_loopback_url, signed_post
from krystal_bot.pending import PendingEvaluator
from krystal_bot.quota import Quota, QuotaManager
from krystal_bot.rag import RagIndex, chunk_markdown, fold, ingest_events
from krystal_bot.service import BotService
from krystal_bot.skills import Gateway, Pairing, Skill, parse_manifest


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


# ---------------------------------------------------------------------- RAG
class TestRag(unittest.TestCase):
    def test_accent_folding_and_ranking(self):
        ix = RagIndex()
        ix.add("a.md", "doc", "Plánovač používa frontu s termínmi a spätným tlakom.")
        ix.add("b.md", "doc", "Pixel art paleta a dithering pre krajinky.")
        hits = ix.search("planovac termin")
        self.assertTrue(hits and hits[0]["source"] == "a.md")
        self.assertEqual(fold("Žltý kôň"), "zlty kon")

    def test_idempotent_and_eviction_prefers_logs(self):
        ix = RagIndex(max_docs=3)
        d = ix.add("doc.md", "doc", "keep me around please")
        self.assertEqual(d, ix.add("doc.md", "doc", "keep me around please"))
        for i in range(5):
            ix.add("log", "event", f"noise event number {i}")
        self.assertEqual(len(ix), 3)
        self.assertTrue(any(x["kind"] == "doc" for x in ix.search("keep around")))
        self.assertGreater(ix.evicted, 0)

    def test_unrelated_query_returns_nothing(self):
        ix = RagIndex()
        ix.add("a", "doc", "kernel scheduler deadline")
        self.assertEqual(ix.search("zzzzqqqq xxxjjj"), [])

    def test_chunking_keeps_code_fences_and_headings(self):
        md = "# Title\nintro text that is long enough to keep\n\n## Code\n```\n# not a heading\nx = 1\n```\nafter the fence text here ok\n"
        ch = chunk_markdown(md)
        self.assertTrue(any(h == "Code" for h, _ in ch))
        self.assertFalse(any(h == "not a heading" for h, _ in ch))

    def test_persistence_roundtrip_and_atomic_save(self):
        d = tempfile.mkdtemp()
        try:
            p = os.path.join(d, "i.jsonl")
            ix = RagIndex(p)
            ix.add("s", "doc", "persist this sentence about quotas")
            ix.save()
            self.assertFalse(os.path.exists(p + ".tmp"))
            self.assertTrue(RagIndex(p).search("persist quotas"))
        finally:
            shutil.rmtree(d)

    def test_event_ingest_summarises_task_done(self):
        evs = [{"seq": i, "t": i, "type": "task_done", "lane": "thread", "ok": True, "latency_ms": 5 + i % 3} for i in range(250)]
        evs.append({"seq": 999, "t": 999, "type": "heal", "lane": "process", "action": "restart"})
        ix = RagIndex()
        n = ingest_events(ix, evs, window=100)
        self.assertEqual(n, 4)  # 3 summaries + 1 heal
        self.assertTrue(ix.search("heal restart"))


# ---------------------------------------------------------------------- quotas
class TestQuota(unittest.TestCase):
    def test_window_denial_retry_and_refund(self):
        c = Clock()
        q = Quota("x", 10, 60, c)
        self.assertEqual(q.try_consume(6), (True, 0.0))
        ok, retry = q.try_consume(6)
        self.assertFalse(ok)
        self.assertAlmostEqual(retry, 60.0, places=3)
        c.t += 61
        self.assertTrue(q.try_consume(10)[0])
        q.refund(4)
        self.assertAlmostEqual(q.remaining(), 4.0)
        self.assertEqual(q.try_consume(11), (False, float("inf")))

    def test_persistence(self):
        d = tempfile.mkdtemp()
        try:
            c = Clock()
            m = QuotaManager(os.path.join(d, "q.json"), clock=c)
            m.define("tok", 100, 3600)
            m.try_consume("tok", 70)
            m.save()
            m2 = QuotaManager(os.path.join(d, "q.json"), clock=c)
            q = m2.define("tok", 100, 3600)
            self.assertAlmostEqual(q.used(), 70.0)
        finally:
            shutil.rmtree(d)


# ---------------------------------------------------------------------- LLM gateway
def fake_transport(record):
    def t(url, headers, body, timeout):
        record.append((url, headers, body))
        return 200, {}, json.dumps({"model": "m", "choices": [{"message": {"content": "answer"}}], "usage": {"total_tokens": 100}}).encode()
    return t


class TestLLM(unittest.TestCase):
    def mk(self, **kw):
        rec = []
        cfg = LLMConfig(enabled=True, model="m", base_url="http://127.0.0.1:1/v1", **kw)
        return LLMGateway(cfg, transport=fake_transport(rec), profile_fn=lambda: {"logical_cores": 8, "ram_gb": 24}, clock=Clock()), rec

    def setUp(self):
        self._dbg = os.environ.pop("KRYSTAL_DEBUG", None)

    def tearDown(self):
        os.environ.pop("KRYSTAL_DEBUG", None)
        if self._dbg is not None:
            os.environ["KRYSTAL_DEBUG"] = self._dbg

    def test_denied_by_default_and_on_weak_device(self):
        g = LLMGateway(LLMConfig(), profile_fn=lambda: {"logical_cores": 64, "ram_gb": 256})
        with self.assertRaises(GatewayDenied):
            g.chat([{"role": "user", "content": "hi"}])
        g, rec = self.mk()
        with self.assertRaises(GatewayDenied) as cm:
            g.chat([{"role": "user", "content": "hi"}])
        self.assertIn("debug", str(cm.exception))
        self.assertEqual(rec, [])  # nothing was sent

    def test_debug_mode_or_big_device_allows(self):
        g, rec = self.mk(debug=True)
        self.assertEqual(g.chat([{"role": "user", "content": "hi"}])["text"], "answer")
        g2 = LLMGateway(LLMConfig(enabled=True, model="m", base_url="http://127.0.0.1:1/v1"), transport=fake_transport([]),
                        profile_fn=lambda: {"logical_cores": 32, "ram_gb": 64})
        self.assertTrue(g2.policy()["allowed"])

    def test_remote_needs_explicit_allow(self):
        cfg = LLMConfig(enabled=True, model="m", debug=True, base_url="https://api.example.com/v1")
        g = LLMGateway(cfg, transport=fake_transport([]))
        self.assertFalse(g.policy()["allowed"])
        cfg.allow_remote = True
        self.assertTrue(LLMGateway(cfg, transport=fake_transport([])).policy()["allowed"])

    def test_quota_exhaustion_and_usage_refund(self):
        g, _ = self.mk(debug=True, requests_per_min=2, tokens_per_day=100000)
        g.chat([{"role": "user", "content": "a"}])
        g.chat([{"role": "user", "content": "b"}])
        with self.assertRaises(GatewayDenied) as cm:
            g.chat([{"role": "user", "content": "c"}])
        self.assertEqual(cm.exception.code, "rate_limited")
        g2, _ = self.mk(debug=True, tokens_per_day=2000)
        g2.chat([{"role": "user", "content": "a"}])
        self.assertLess(g2.quotas.get("llm_tokens").used(), 200)  # reservation refunded down to reported usage (100)

    def test_secrets_never_leave(self):
        os.environ["TEST_KEY_ENV"] = "sk-SECRETSECRETSECRET1234"
        try:
            g, rec = self.mk(debug=True, api_key_env="TEST_KEY_ENV")
            g.chat([{"role": "user", "content": "my key is sk-ABCDEFGHIJKLMNOPQRSTUV and mail me at a@b.com in C:\\Users\\dusan\\x"}])
            body = rec[0][2].decode()
            self.assertNotIn("sk-ABCDEFGHIJKLMNOPQRSTUV", body)
            self.assertNotIn("a@b.com", body)
            self.assertNotIn("dusan", body)
            self.assertEqual(rec[0][1]["Authorization"], "Bearer sk-SECRETSECRETSECRET1234")  # sent as header only
            self.assertNotIn("SECRETSECRET", json.dumps(g.cfg.public()))
        finally:
            del os.environ["TEST_KEY_ENV"]
        self.assertIn("[REDACTED", redact("password=hunter2"))

    def test_breaker_and_prompt_limit(self):
        def bad(url, headers, body, timeout):
            return 500, {}, b"x"
        g = LLMGateway(LLMConfig(enabled=True, model="m", debug=True, base_url="http://127.0.0.1:1/v1", breaker_failures=2), transport=bad, clock=Clock())
        for _ in range(2):
            with self.assertRaises(GatewayDenied):
                g.chat([{"role": "user", "content": "x"}])
        with self.assertRaises(GatewayDenied) as cm:
            g.chat([{"role": "user", "content": "x"}])
        self.assertEqual(cm.exception.code, "breaker_open")
        g2, _ = self.mk(debug=True, max_prompt_chars=10)
        with self.assertRaises(GatewayDenied) as cm:
            g2.chat([{"role": "user", "content": "x" * 50}])
        self.assertEqual(cm.exception.code, "prompt_too_large")

    def test_safe_math(self):
        self.assertAlmostEqual(safe_eval_math("2^10 + sqrt(16) * pi"), 1024 + 4 * 3.141592653589793)
        for bad in ("__import__('os').system('x')", "().__class__", "9**9**9", "1/0", "open('f')", "a+1"):
            with self.assertRaises(ValueError, msg=bad):
                safe_eval_math(bad)

    def test_binary_analysis_finds_record_size(self):
        rec = b"\xAA\xBB\x00\x10\x01\x02\x03\x04"
        data = b"".join(rec[:4] + bytes([i % 256]) + rec[5:] for i in range(200))
        a = analyze_binary_log(data)
        self.assertEqual(a["likely_record_size"] % 8, 0)
        self.assertFalse(a["looks_compressed_or_encrypted"])

    def test_document_math_sends_ground_truth(self):
        g, rec = self.mk(debug=True)
        r = g.document_math("3*7")
        self.assertEqual(r["facts"]["numeric_value"], 21)
        self.assertIn("Unverified", r["markdown"])
        self.assertIn("21", rec[0][2].decode())


# ---------------------------------------------------------------------- bytecode, composer, morse
class TestBytecode(unittest.TestCase):
    def test_roundtrip_determinism(self):
        p = BC.compile_config({"seed": 5, "techniques": ["dither", "outline", "grid", "hud"]})
        q = BC.Program.from_bytes(p.to_bytes())
        self.assertEqual(p, q)
        self.assertEqual(BC.run(p).fingerprint(), BC.run(q).fingerprint())
        self.assertNotEqual(BC.run(p).fingerprint(), BC.run(BC.compile_config({"seed": 6})).fingerprint())

    def test_tamper_and_invalid(self):
        data = bytearray(BC.compile_config({}).to_bytes())
        data[10] ^= 0xFF
        with self.assertRaises(BC.BytecodeError):
            BC.Program.from_bytes(bytes(data))
        with self.assertRaises(BC.BytecodeError):
            BC.Program.from_bytes(b"nope")
        with self.assertRaises(BC.BytecodeError):
            BC.Program().emit("DISC", 1, 2, 3, 99)  # role out of range
        with self.assertRaises(BC.BytecodeError):
            BC.Program([(0x77, ())]).validate()

    def test_config_validation(self):
        for bad in ({"width": 9999}, {"scene": "x"}, {"palette": "none"}, {"techniques": ["laser"]}, {"bogus": 1},
                    {"compute": {"trigger": "never"}}, {"hud": {"metrics": [1, 2, 3, 4, 5]}}, {"seed": -1}):
            with self.assertRaises(BC.BytecodeError, msg=str(bad)):
                BC.normalize_config(bad)

    def test_replicate_crossover_always_valid(self):
        rng = random.Random(1)
        a, b = BC.compile_config({"seed": 1}), BC.compile_config({"scene": "still_life", "seed": 2})
        for _ in range(30):
            kid = BC.replicate(a, rng, 0.3)["child"]
            BC.run(kid)
            BC.Program.from_bytes(kid.to_bytes())
            BC.run(BC.crossover(a, b, rng))
        self.assertTrue(any(not BC.replicate(a, random.Random(i), 0.3)["identical"] for i in range(10)))

    def test_all_palettes_and_scenes_render_png(self):
        for pal in CP.PALETTE_IDS:
            for scene in ("landscape", "still_life"):
                cv = BC.run(BC.compile_config({"palette": pal, "scene": scene, "width": 40, "height": 30}))
                png = cv.to_png(2)
                self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
                self.assertIn("<svg", cv.to_svg(2))

    def test_output_size_is_capped(self):
        cv = CP.Canvas(255, 255)
        png = cv.to_png(16)
        self.assertLess(len(png), 400000)


class TestMorse(unittest.TestCase):
    def test_text_program_tensor_roundtrips(self):
        self.assertEqual(MO.decode_text(MO.encode_text("Krystal 47")), "KRYSTAL47")
        p = BC.compile_config({"seed": 3, "scene": "still_life"})
        m = MO.encode_program(p)
        self.assertEqual(MO.decode_program(m), p)
        t = MO.morse_to_tensor(m, 20000)
        self.assertEqual(len(t), 20000)
        self.assertEqual(MO.decode_program(MO.tensor_to_morse(t)), p)
        with self.assertRaises(ValueError):
            MO.morse_to_tensor(m, 10)

    def test_bad_morse_rejected(self):
        for bad in ("", "-.-. / ....", "...... ....."):
            with self.assertRaises(BC.BytecodeError):
                MO.decode_program(bad)

    def test_predictor_learns_structure_but_not_noise(self):
        periodic = "TTPTTMTTPTTM" * 60
        m = MO.ProcessPredictor()
        ev = m.evaluate(periodic)
        self.assertTrue(ev["beats_baseline"], ev)
        rng = random.Random(0)
        noise = "".join(rng.choice("TP") for _ in range(800))
        self.assertFalse(MO.ProcessPredictor().evaluate(noise)["beats_baseline"])
        self.assertEqual(MO.ProcessPredictor().evaluate("TTT")["status"], "insufficient_data")

    def test_event_symbols(self):
        self.assertEqual(MO.event_symbol({"type": "task_done", "ok": True, "lane": "process"}), "P")
        self.assertEqual(MO.event_symbol({"type": "task_done", "ok": False}), "X")
        self.assertEqual(MO.event_symbol({"type": "task_done", "ok": True, "missed": True}), "M")


# ---------------------------------------------------------------------- risk + pending
def synth_events(n, signal=True, seed=1):
    rng = random.Random(seed)
    evs = []
    for i in range(n):
        dl = rng.choice([50, 100, 200, 400])
        q = rng.uniform(0, dl * 1.5)
        svc = rng.uniform(1, 40)
        bad = (q + svc > dl) if signal else (rng.random() < 0.3)
        evs.append({"seq": i, "type": "task_done", "kind": "embed", "bucket": rng.randint(0, 6), "lane": rng.choice(["thread", "process"]),
                    "ok": True, "missed": bad, "deadline_ms": dl, "queue_ms": q, "service_ms": svc, "latency_ms": q + svc, "retried": False})
    return evs


class TestRisk(unittest.TestCase):
    def test_features_do_not_leak_service_time(self):
        ev = synth_events(1)[0]
        f1 = RK.featurize(ev)
        ev2 = dict(ev, service_ms=9999.0, latency_ms=ev["queue_ms"] + 9999.0)
        self.assertEqual(f1, RK.featurize(ev2))
        self.assertIsNone(RK.featurize({"type": "heal"}))

    def test_insufficient_data_uses_heuristic(self):
        d = tempfile.mkdtemp()
        try:
            m = RK.RiskModel(os.path.join(d, "r.json"))
            rep = m.fit(synth_events(40))
            self.assertEqual(rep["status"], "insufficient_data")
            self.assertFalse(m.accepted)
            self.assertEqual(m.backend, "heuristic")
        finally:
            shutil.rmtree(d)

    def test_model_accepted_with_signal_rejected_on_noise(self):
        d = tempfile.mkdtemp()
        try:
            m = RK.RiskModel(os.path.join(d, "r.json"))
            rep = m.fit(synth_events(1500, True), prefer="stdlib")
            self.assertEqual(rep["status"], "trained")
            self.assertIsNotNone(rep["stdlib_auc"])
            m2 = RK.RiskModel(os.path.join(d, "r2.json"))
            rep2 = m2.fit(synth_events(1500, False), prefer="stdlib")
            self.assertFalse(rep2["accepted"], rep2)   # noise labels must not pass the gate
            self.assertEqual(m2.backend, "heuristic")
            self.assertEqual(len(RK.RiskModel(os.path.join(d, "r.json")).predict([RK.featurize(synth_events(1)[0])])), 1)
        finally:
            shutil.rmtree(d)

    def test_auc_basic(self):
        self.assertEqual(RK.auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertEqual(RK.auc([0, 1], [0.5, 0.5]), 0.5)
        self.assertIsNone(RK.auc([1, 1], [0.1, 0.2]))


class TestPending(unittest.TestCase):
    def test_quota_bounds_each_cycle_and_overflow_is_counted(self):
        d = tempfile.mkdtemp()
        try:
            c = Clock()
            qm = QuotaManager(clock=c)
            flagged = []
            pe = PendingEvaluator(RK.RiskModel(os.path.join(d, "r.json")), qm, flagged.append, max_pending=50, batch=20, records_per_min=30, flag_threshold=0.9)
            for ev in synth_events(80):
                pe.submit(ev)
            self.assertEqual(pe.counters["dropped_overflow"], 30)
            r1 = pe.cycle()
            r2 = pe.cycle()
            self.assertEqual((r1["processed"], r2["processed"]), (20, 10))   # 30/min quota, not 40
            r3 = pe.cycle()
            self.assertEqual(r3["processed"], 0)
            self.assertTrue(r3.get("deferred_by_quota"))
            c.t += 61
            self.assertGreater(pe.cycle()["processed"], 0)
            self.assertTrue(all(f["risk"] >= 0.9 for f in flagged))
            self.assertFalse(pe.submit({"type": "heal"}))
        finally:
            shutil.rmtree(d)

    def test_riskiest_first(self):
        d = tempfile.mkdtemp()
        try:
            qm = QuotaManager()
            seen = []
            pe = PendingEvaluator(RK.RiskModel(os.path.join(d, "r.json")), qm, seen.append, batch=5, records_per_min=100, flag_threshold=0.0)
            evs = synth_events(40)
            for e in evs:
                pe.submit(e)
            pe.cycle()
            top = sorted((RK.heuristic_risk(RK.featurize(e)) for e in evs), reverse=True)[:5]
            self.assertEqual(sorted((f["risk"] for f in seen), reverse=True), [round(x, 4) for x in top])
        finally:
            shutil.rmtree(d)


# ---------------------------------------------------------------------- donor
class TestDonor(unittest.TestCase):
    def test_rfc8032_vectors(self):
        vec = [("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60", "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a", "",
                "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"),
               ("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb", "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c", "72",
                "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00")]
        for sk, pk, msg, sig in vec:
            sk, pk, msg, sig = map(bytes.fromhex, (sk, pk, msg, sig))
            self.assertEqual(DN.public_key(sk), pk)
            self.assertEqual(DN.sign(sk, msg), sig)
            self.assertTrue(DN.verify(pk, msg, sig))
            self.assertFalse(DN.verify(pk, msg + b"!", sig))

    def test_token_lifecycle(self):
        sk = os.urandom(32)
        pk = DN.public_key(sk)
        t = DN.issue_token(sk, "ada", 1, now=1000.0)
        self.assertTrue(DN.check_token(t, [pk], now=1000.0)["valid"])
        self.assertEqual(DN.check_token(t, [pk], now=1000.0 + 2 * 86400)["reason"], "expired")
        self.assertFalse(DN.check_token(t, [DN.public_key(os.urandom(32))], now=1000.0)["valid"])
        self.assertEqual(DN.check_token(t, [], now=1000.0)["reason"], "no issuer public key configured")
        head, body, sig = t.split(".")
        forged = head + "." + DN._b64e(DN._b64d(body).replace(b"ada", b"eve")) + "." + sig
        self.assertFalse(DN.check_token(forged, [pk], now=1000.0)["valid"])
        for junk in ("", "a.b", "KD1.!!.!!", "x" * 5000):
            self.assertFalse(DN.check_token(junk, [pk])["valid"])


# ---------------------------------------------------------------------- mesh
class TestMesh(unittest.TestCase):
    def test_loopback_only(self):
        for u in ("http://127.0.0.1:80", "http://localhost:9", "http://[::1]:5"):
            self.assertTrue(is_loopback_url(u), u)
        for u in ("https://127.0.0.1", "http://10.0.0.5", "http://example.com", "http://127.0.0.1@evil.com", "ftp://127.0.0.1"):
            self.assertFalse(is_loopback_url(u), u)
        auth = MeshAuth(os.urandom(32))
        with self.assertRaises(ValueError):
            signed_post(auth, "http://example.com", "/mesh/hop", {})
        with self.assertRaises(ValueError):
            Mesh(auth).register("x", "http://192.168.1.2:80")

    def test_cycle_signatures_replay_and_eviction(self):
        auth = MeshAuth(os.urandom(32))
        nodes = [MeshNode(auth, f"n{i}", telemetry=lambda i=i: {"load": i}) for i in range(3)]
        for n in nodes:
            n.start()
        try:
            mesh = Mesh(auth, timeout=2.0)
            for n in nodes:
                mesh.register(n.name, n.url, n.node_id)
            r = mesh.cycle()
            self.assertTrue(r["ok"], r)
            self.assertEqual([h["telemetry"]["load"] for h in r["hops"]], [0, 1, 2])
            self.assertTrue(all(h["signature_valid"] for h in r["hops"]))
            # replay: re-sending identical headers must be rejected
            body = b'{"cycle_id":"x","hop_index":0}'
            hdr = auth.sign_request("POST", "/mesh/hop", body)
            self.assertTrue(auth.verify_request("POST", "/mesh/hop", hdr, body)[0])
            self.assertEqual(auth.verify_request("POST", "/mesh/hop", hdr, body), (False, "replayed nonce"))
            # a node with the wrong secret fails validation, then is evicted after 3 cycles
            rogue = MeshNode(MeshAuth(os.urandom(32)), "rogue")
            rogue.start()
            try:
                mesh.register("rogue", rogue.url, rogue.node_id)
                for _ in range(3):
                    rr = mesh.cycle()
                    self.assertFalse(rr["ok"])
                self.assertNotIn(rogue.node_id, [p["node_id"] for p in mesh.peers()])
            finally:
                rogue.stop()
            nodes[2].stop()
            for _ in range(3):
                mesh.cycle()
            self.assertEqual(len(mesh.peers()), 2)
        finally:
            for n in nodes:
                n.stop()

    def test_stale_timestamp_rejected(self):
        c = Clock(5000.0)
        auth = MeshAuth(os.urandom(32), c)
        hdr = auth.sign_request("POST", "/p", b"")
        c.t += 120
        self.assertEqual(auth.verify_request("POST", "/p", hdr, b""), (False, "timestamp outside window"))

    def test_register_probe_adopts_real_id_and_rejects_wrong_secret(self):
        auth = MeshAuth(os.urandom(32))
        good, rogue = MeshNode(auth, "good"), MeshNode(MeshAuth(os.urandom(32)), "rogue")
        good.start()
        rogue.start()
        try:
            mesh = Mesh(auth)
            self.assertEqual(mesh.register("good", good.url), good.node_id)
            with self.assertRaises(ValueError):
                mesh.register("rogue", rogue.url)
            with self.assertRaises(ValueError):
                mesh.register("dead", "http://127.0.0.1:9")
        finally:
            good.stop()
            rogue.stop()


# ---------------------------------------------------------------------- skills
MANIFEST = "---\nname: art-studio\ndescription: Draw things\ntriggers: [/draw, /art2]\nrequires_donor: true\n---\nbody"


class TestSkills(unittest.TestCase):
    def test_manifest_parse_and_rejection(self):
        s = parse_manifest(MANIFEST)
        self.assertEqual((s.name, s.triggers, s.requires_donor), ("art-studio", ["/draw", "/art2"], True))
        for bad in ("no front matter", "---\nname: Bad Name!\n---\n", "---\ndescription: x\n---\n"):
            with self.assertRaises(ValueError):
                parse_manifest(bad)
        s2 = parse_manifest("---\nname: ok\ntriggers: [/ok, rm -rf /, /x y]\n---\n")
        self.assertEqual(s2.triggers, ["/ok"])

    def test_policy_pairing_and_containment(self):
        donor = {"v": False}
        g = Gateway(Pairing(), is_donor=lambda: donor["v"])
        g.register(Skill("hello", "hi", ["/hello"], channels=["extension", "mesh"]), lambda a, c: f"hello {a}")
        g.register(Skill("vip", "donor", ["/vip"], requires_donor=True), lambda a, c: "secret")
        g.register(Skill("boom", "fails", ["/boom"]), lambda a, c: 1 / 0)
        self.assertEqual(g.dispatch("/hello x", "extension")["reply"], "hello x")
        self.assertTrue(g.dispatch("/hello x", "web").get("denied"))            # channel not enabled
        self.assertTrue(g.dispatch("/hello x", "mesh", "node1").get("denied"))  # unpaired sender
        code = g.pairing.request("node1")
        self.assertEqual(g.pairing.approve(code), "node1")
        self.assertIsNone(g.pairing.approve(code))                              # one-time
        self.assertTrue(g.dispatch("/hello x", "mesh", "node1")["ok"])
        self.assertTrue(g.dispatch("/vip", "extension").get("denied"))
        donor["v"] = True
        self.assertEqual(g.dispatch("/vip", "extension")["reply"], "secret")
        r = g.dispatch("/boom", "extension")
        self.assertFalse(r["ok"])
        self.assertIn("ZeroDivisionError", r["reply"])
        self.assertIn("Unknown command", g.dispatch("/nope")["reply"])

    def test_pairing_expiry_and_cap(self):
        c = Clock()
        p = Pairing(ttl_s=10, max_pending=2, clock=c)
        code = p.request("a")
        c.t += 11
        self.assertIsNone(p.approve(code))
        p.request("a")
        p.request("b")
        with self.assertRaises(ValueError):
            p.request("c")


# ---------------------------------------------------------------------- service + API
class TestServiceApi(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.root, "config"))
        os.makedirs(os.path.join(self.root, "docs"))
        with open(os.path.join(self.root, "docs", "kernel.md"), "w", encoding="utf-8") as f:
            f.write("# Scheduler\nThe kernel scheduler uses earliest deadline first ordering and back-pressure.\n")
        self.sk = os.urandom(32)
        with open(os.path.join(self.root, "config", "donor_pubkeys.json"), "w") as f:
            json.dump([DN.public_key(self.sk).hex()], f)
        self.svc = BotService(self.root, use_keras=False)
        self._env = os.environ.pop("KRYSTAL_OPERATOR_KEY", None)

    def tearDown(self):
        self.svc.shutdown()
        shutil.rmtree(self.root, ignore_errors=True)
        os.environ.pop("KRYSTAL_OPERATOR_KEY", None)
        if self._env:
            os.environ["KRYSTAL_OPERATOR_KEY"] = self._env

    def call(self, method, path, obj=None, headers=None, raw=None):
        h = {"Content-Type": "application/json"}
        h.update(headers or {})
        body = raw if raw is not None else (json.dumps(obj).encode() if obj is not None else b"")
        s, rh, b = handle(self.svc, method, path, h, body)
        return s, rh, (json.loads(b) if b[:1] in b"{[" else b)

    def test_message_flow_search_and_palette(self):
        self.assertEqual(self.call("POST", "/api/bot/reindex", {"events": False})[0], 200)
        s, _, j = self.call("POST", "/api/bot/message", {"text": "earliest deadline first"})
        self.assertEqual(s, 200)
        self.assertIn("kernel.md", j["reply"])
        s, _, j2 = self.call("POST", "/api/bot/message", {"text": "/status", "conversation": j["conversation"]})
        self.assertIn("rag:", j2["reply"])
        s, _, p = self.call("GET", f"/api/bot/palette?conversation={j['conversation']}")
        self.assertEqual(len(p["entries"]), 4)
        self.assertEqual({e["role"] for e in p["entries"]}, {"user", "bot"})

    def test_render_replicate_morse_and_policy(self):
        s, h, j = self.call("POST", "/api/bot/render", {"config": {"seed": 3, "width": 48, "height": 32}})
        self.assertEqual(s, 200)
        self.assertTrue(base64.b64decode(j["png_b64"]).startswith(b"\x89PNG"))
        self.assertNotIn("Access-Control-Allow-Origin", h)
        s, _, r = self.call("POST", "/api/bot/replicate", {"bytecode_b64": j["bytecode_b64"], "rate": 0.3, "seed": 4, "count": 2})
        self.assertEqual((s, len(r["children"])), (200, 2))
        s, _, m = self.call("POST", "/api/bot/morse", {"bytecode_b64": j["bytecode_b64"]})
        self.assertEqual(m["digest"], j["digest"])
        self.assertEqual(self.call("POST", "/api/bot/render", {"config": {"width": 100000}})[0], 400)
        self.assertEqual(self.call("POST", "/api/bot/replicate", {"bytecode_b64": "AAAA"})[0], 400)
        s, _, j = self.call("POST", "/api/bot/render", {"config": {"compute": {"trigger": "on_pattern", "pattern": "capacity_bound"}}})
        self.assertFalse(j["computed"])                                          # deferred until the pattern exists
        s, _, j = self.call("POST", "/api/bot/render", {"config": {"compute": {"trigger": "on_pattern", "pattern": "capacity_bound"}}, "force": True})
        self.assertTrue(j["computed"])

    def test_donor_gate(self):
        self.assertEqual(self.call("POST", "/api/bot/render", {"config": {"palette": "aurora"}})[0], 403)
        self.assertEqual(self.call("POST", "/api/bot/mesh/cycle", {})[0], 403)
        self.assertIn("Denied", self.call("POST", "/api/bot/message", {"text": "/constellation"})[2]["reply"])
        self.assertEqual(self.call("POST", "/api/bot/donor/activate", {"token": "KD1.bad.bad"})[0], 403)
        tok = DN.issue_token(self.sk, "ada", 30)
        self.assertEqual(self.call("POST", "/api/bot/donor/activate", {"token": tok})[0], 200)
        self.assertEqual(self.call("POST", "/api/bot/render", {"config": {"palette": "aurora"}})[0], 200)
        svc2 = BotService(self.root, use_keras=False)           # restart: token is re-verified from disk
        self.assertTrue(svc2.is_donor())
        svc2.shutdown()
        os.remove(os.path.join(self.root, "config", "donor_pubkeys.json"))
        svc3 = BotService(self.root, use_keras=False)           # issuer key removed -> locked again
        self.assertFalse(svc3.is_donor())
        svc3.shutdown()

    def test_mesh_via_api_for_donor(self):
        self.call("POST", "/api/bot/donor/activate", {"token": DN.issue_token(self.sk, "ada", 30)})
        nodes = [MeshNode(self.svc.mesh_auth, f"n{i}") for i in range(2)]
        for n in nodes:
            n.start()
        try:
            for n in nodes:
                self.assertEqual(self.call("POST", "/api/bot/mesh/register", {"name": n.name, "url": n.url})[0], 200)
            self.assertEqual(self.call("POST", "/api/bot/mesh/register", {"name": "x", "url": "http://8.8.8.8:80"})[0], 400)
            s, _, r = self.call("POST", "/api/bot/mesh/cycle", {})
            self.assertTrue(r["ok"], r)
            self.assertIn("ok=True", self.call("POST", "/api/bot/message", {"text": "/constellation"})[2]["reply"])
        finally:
            for n in nodes:
                n.stop()

    def test_http_hardening(self):
        self.assertEqual(self.call("POST", "/api/bot/message", raw=b"{}", headers={"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.call("POST", "/api/bot/message", raw=b"{bad")[0], 400)
        self.assertEqual(self.call("POST", "/api/bot/message", raw=b" " * (300 * 1024))[0], 413)
        self.assertEqual(self.call("GET", "/api/bot/nope")[0], 404)
        self.assertEqual(self.call("DELETE", "/api/bot/status")[0], 405)
        s, _, j = self.call("POST", "/api/bot/pair/approve", {"code": "123456"})
        self.assertEqual((s, j["error"]["code"]), (403, "approval_disabled"))
        os.environ["KRYSTAL_OPERATOR_KEY"] = "op-secret"
        self.assertEqual(self.call("POST", "/api/bot/pair/approve", {"code": "1"}, {"X-Operator-Key": "wrong"})[0], 401)
        code = self.svc.gateway.pairing.request("node9")
        s, _, j = self.call("POST", "/api/bot/pair/approve", {"code": code}, {"X-Operator-Key": "op-secret"})
        self.assertEqual((s, j["paired"]), (200, "node9"))

    def test_api_quota_returns_429(self):
        self.svc.quotas.define("api_requests", 3, 60.0)
        codes = [self.call("GET", "/api/bot/quota")[0] for _ in range(5)]
        self.assertEqual(codes, [200, 200, 200, 429, 429])

    def test_llm_endpoints_policy_and_no_key_leak(self):
        s, _, j = self.call("GET", "/api/bot/llm/policy")
        self.assertFalse(j["policy"]["allowed"])
        self.assertNotIn("sk-", json.dumps(j))
        s, _, j = self.call("POST", "/api/bot/llm/document", {"kind": "math", "input": "2+2"})
        self.assertEqual((s, j["error"]["code"]), (403, "policy_denied"))
        self.assertEqual(self.call("POST", "/api/bot/llm/document", {"kind": "zzz", "input": "x"})[0], 400)

    def test_risk_train_endpoint_reports_gate(self):
        for e in synth_events(30):
            self.svc.pending.submit(e)
        s, _, j = self.call("POST", "/api/bot/risk/train", {"prefer": "stdlib"})
        self.assertEqual(s, 200)
        self.assertEqual(j["report"]["status"], "insufficient_data")
        s, _, p = self.call("GET", "/api/bot/predict")
        self.assertIn("evaluation", p)
        self.assertFalse(p["trusted"])


if __name__ == "__main__":
    unittest.main()
