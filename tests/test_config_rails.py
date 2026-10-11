"""Tests for configuration rails (krystal_kernel.rails) and the telemetry-theory schemas.

Run: python -m unittest tests.test_config_rails -v
"""

from __future__ import annotations

import copy
import json
import math
import random
import statistics
import unittest
from pathlib import Path

from krystal_kernel.rails import (
    DEFAULT_REGISTRY,
    RAIL_SCHEMA,
    check_rail,
    list_presets,
    load_preset,
    run_rail,
    summarize_result,
    unsupported_keywords,
    validate,
)
from krystal_kernel.rails.library import (
    VERDICT_INCONCLUSIVE,
    VERDICT_NOT_REJECTED,
    VERDICT_REJECTED,
)
from krystal_kernel.rails.rail import RailValidationError

ROOT = Path(__file__).resolve().parents[1]
THEORY_SCHEMAS = ROOT / "krystal_theory" / "schemas"


def _rail(nodes, links, outputs=None):
    doc = {"rail": "t", "version": 1, "nodes": nodes, "links": links}
    if outputs is not None:
        doc["outputs"] = outputs
    return doc


def _src(nid="src", **params):
    base = {"n": 200, "seed": 3}
    base.update(params)
    return {"id": nid, "fn": "source.synthetic_series", "params": base}


def _switches(modes):
    return sum(1 for a, b in zip(modes, modes[1:]) if a != b)


class SchemaValidatorTests(unittest.TestCase):
    def test_types_and_bool_is_not_a_number(self):
        self.assertEqual(validate(3, {"type": "integer"}), [])
        self.assertEqual(validate(3.0, {"type": "integer"}), [])
        self.assertTrue(validate(True, {"type": "integer"}))
        self.assertTrue(validate(True, {"type": "number"}))
        self.assertTrue(validate(3.5, {"type": "integer"}))
        self.assertEqual(validate(None, {"type": ["number", "null"]}), [])

    def test_object_keywords(self):
        s = {"type": "object", "required": ["a"], "additionalProperties": False,
             "properties": {"a": {"type": "string", "pattern": "^x"}}}
        self.assertEqual(validate({"a": "xy"}, s), [])
        errs = validate({"b": 1}, s)
        self.assertTrue(any("missing required property 'a'" in e for e in errs))
        self.assertTrue(any("unexpected property 'b'" in e for e in errs))
        self.assertTrue(validate({"a": "yx"}, s))

    def test_const_enum_do_not_confuse_true_and_one(self):
        self.assertTrue(validate(1, {"const": True}))
        self.assertTrue(validate(True, {"enum": [1, 2]}))
        self.assertEqual(validate(1, {"enum": [1, 2]}), [])

    def test_arrays_and_numbers(self):
        s = {"type": "array", "minItems": 1, "maxItems": 2, "uniqueItems": True,
             "items": {"type": "number", "minimum": 0, "exclusiveMaximum": 1}}
        self.assertEqual(validate([0, 0.5], s), [])
        self.assertTrue(validate([], s))
        self.assertTrue(validate([0.5, 0.5], s))
        self.assertTrue(validate([1], s))

    def test_combinators(self):
        any_s = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
        self.assertEqual(validate(2, any_s), [])
        self.assertTrue(validate(2.5, any_s))
        one_s = {"oneOf": [{"type": "number"}, {"type": "integer"}]}
        self.assertTrue(validate(2, one_s))          # matches both -> not exactly one
        self.assertEqual(validate(2.5, one_s), [])
        self.assertTrue(validate("a", {"not": {"type": "string"}}))

    def test_local_and_external_refs(self):
        s = {"$defs": {"pos": {"type": "number", "minimum": 0}},
             "type": "array", "items": {"$ref": "#/$defs/pos"}}
        self.assertEqual(validate([1, 2], s), [])
        self.assertTrue(validate([-1], s))
        ext = {"$ref": "telemetry_signal.schema.json"}
        errs = validate({}, ext, base_dir=THEORY_SCHEMAS)
        self.assertTrue(any("missing required property" in e for e in errs))

    def test_unsupported_keywords_are_reported(self):
        s = {"type": "object", "properties": {"a": {"type": "string", "format": "date",
                                                    "dependentRequired": {}}},
             "if": {}}
        found = unsupported_keywords(s)
        self.assertIn("/: if", found)
        self.assertIn("/properties/a: dependentRequired", found)


class RegistryTests(unittest.TestCase):
    def test_every_node_is_documented_and_serialisable(self):
        self.assertGreaterEqual(len(DEFAULT_REGISTRY), 18)
        desc = DEFAULT_REGISTRY.describe()
        json.dumps(desc, allow_nan=False)
        for d in desc:
            self.assertTrue(d["doc"], d["name"])
            self.assertIn(d["category"], ("source", "transform", "measure", "diagnose"))

    def test_no_privileged_sources(self):
        """A3 scope: no MSR/RAPL/psutil-backed nodes."""
        import inspect
        import krystal_kernel.rails.library as lib
        src = inspect.getsource(lib)
        for banned in ("import psutil", "rdmsr", "wrmsr", "/dev/cpu", "RAPL_"):
            self.assertNotIn(banned, src)


class RailValidationTests(unittest.TestCase):
    def assertError(self, doc, fragment):
        v, errors = check_rail(doc)
        self.assertIsNone(v)
        self.assertTrue(any(fragment in e for e in errors), errors)

    def test_schema_layer(self):
        self.assertError({"rail": "t", "version": 2, "nodes": [], "links": []}, "/version")
        self.assertError(_rail([{"id": "Bad-Id", "fn": "source.constant"}], []), "pattern")
        self.assertError(_rail([{"id": "a", "fn": "source.constant", "extra": 1}], []),
                         "unexpected property 'extra'")

    def test_unknown_fn_param_and_bounds(self):
        self.assertError(_rail([{"id": "a", "fn": "source.nope"}], []), "unknown node function")
        self.assertError(_rail([{"id": "a", "fn": "source.constant", "params": {"x": 1}}], []),
                         "unknown parameter")
        self.assertError(_rail([_src(n=1)], []), "below minimum")
        self.assertError(_rail([_src(noise="pink")], []), "is not one of")
        self.assertError(_rail([_src(n=True)], []), "must be integer")

    def test_duplicate_ids(self):
        self.assertError(_rail([_src(), _src()], []), "duplicate node id")

    def test_port_errors(self):
        nodes = [_src(), {"id": "m", "fn": "transform.threshold_mode"}]
        self.assertError(_rail(nodes, [{"from": "src.nope", "to": "m.series"}]), "has no output")
        self.assertError(_rail(nodes, [{"from": "src.series", "to": "m.nope"}]), "has no input")
        self.assertError(_rail(nodes, [{"from": "zz.series", "to": "m.series"}]), "unknown node")

    def test_kind_mismatch_and_modes_subtyping(self):
        nodes = [_src(), {"id": "st", "fn": "measure.stats"},
                 {"id": "m", "fn": "transform.threshold_mode"}]
        self.assertError(_rail(nodes, [{"from": "src.series", "to": "st.series"},
                                       {"from": "st.stats", "to": "m.series"}]),
                         "cannot connect record")
        # series -> modes is NOT allowed (only modes -> series).
        nodes2 = [_src(), {"id": "p", "fn": "measure.exceed_fraction"}]
        self.assertError(_rail(nodes2, [{"from": "src.series", "to": "p.modes"}]),
                         "cannot connect series")

    def test_double_driver_and_missing_input(self):
        nodes = [_src("a"), _src("b"), {"id": "m", "fn": "transform.threshold_mode"}]
        self.assertError(_rail(nodes, [{"from": "a.series", "to": "m.series"},
                                       {"from": "b.series", "to": "m.series"}]),
                         "already driven")
        self.assertError(_rail(nodes, []), "required input m.series")

    def test_cycle_and_self_loop(self):
        nodes = [{"id": "a", "fn": "transform.moving_average"},
                 {"id": "b", "fn": "transform.moving_average"}]
        self.assertError(_rail(nodes, [{"from": "a.series", "to": "b.series"},
                                       {"from": "b.series", "to": "a.series"}]),
                         "cycle detected")
        self.assertError(_rail(nodes[:1], [{"from": "a.series", "to": "a.series"}]), "self-loop")

    def test_outputs_must_exist(self):
        self.assertError(_rail([_src()], [], outputs=["src.nope"]), "is not an output port")

    def test_cross_check_and_source_budget(self):
        self.assertError(_rail([{"id": "j", "fn": "source.sleep_jitter",
                                 "params": {"interval_ms": 100, "n": 100}}], []),
                         "n * interval_ms")
        many = [{"id": f"c{i}", "fn": "source.process_cpu_sampler",
                 "params": {"duration_s": 4.0}} for i in range(3)]
        self.assertError(_rail(many, []), "limit is 10 s")

    def test_run_raises_on_invalid(self):
        with self.assertRaises(RailValidationError):
            run_rail(_rail([{"id": "a", "fn": "source.nope"}], []))


class RailExecutionTests(unittest.TestCase):
    def test_deterministic_order_trace_and_provenance(self):
        doc = _rail(
            [{"id": "ref", "fn": "source.constant", "params": {"value": 1.0}},
             {"id": "cmp", "fn": "diagnose.compare_prediction", "params": {"abs_tol": 0.1}},
             {"id": "m", "fn": "transform.threshold_mode", "params": {"threshold": 85}},
             _src(),
             {"id": "p", "fn": "measure.exceed_fraction"}],
            [{"from": "src.series", "to": "m.series"},
             {"from": "m.modes", "to": "p.modes"},
             {"from": "p.p", "to": "cmp.observed"},
             {"from": "ref.value", "to": "cmp.predicted"}])
        r1, r2 = run_rail(doc), run_rail(doc)
        self.assertTrue(r1["ok"])
        self.assertEqual(r1["order"], ["ref", "src", "m", "p", "cmp"])
        self.assertEqual(r1["order"], r2["order"])
        self.assertEqual(r1["values"]["p.p"], r2["values"]["p.p"])  # seeded
        self.assertEqual(len(r1["trace"]), 5)
        self.assertTrue(all(t["wall_ms"] >= 0 for t in r1["trace"]))
        self.assertEqual(r1["provenance"]["src.series"], "synthetic")
        self.assertEqual(r1["provenance"]["p.p"], "synthetic")
        self.assertEqual(r1["provenance"]["ref.value"], "derived")
        self.assertEqual(r1["provenance"]["cmp.verdict"], "synthetic")

    def test_node_failure_is_contained(self):
        doc = _rail([_src(), {"id": "neg", "fn": "source.constant", "params": {"value": -1}},
                     {"id": "h", "fn": "transform.hysteresis_mode"}],
                    [{"from": "src.series", "to": "h.series"}, {"from": "neg.value", "to": "h.h"}])
        r = run_rail(doc)
        self.assertFalse(r["ok"])
        self.assertEqual(r["error"]["node"], "h")
        self.assertIn("hysteresis width", r["error"]["message"])
        self.assertEqual(r["trace"][-1]["status"], "error")

    def test_summary_is_strict_json(self):
        doc = _rail([_src(n=2, amplitude=0.0), {"id": "rho", "fn": "measure.lag1_autocorr"}],
                    [{"from": "src.series", "to": "rho.series"}])
        r = run_rail(doc)
        self.assertTrue(math.isnan(r["values"]["rho.rho"]))
        view = summarize_result(r)
        json.dumps(view, allow_nan=False)
        self.assertIsNone(view["values"]["rho.rho"])

    def test_long_series_are_summarised(self):
        r = run_rail(_rail([_src(n=1000)], []))
        v = summarize_result(r)["values"]["src.series"]
        self.assertEqual(v["n"], 1000)
        self.assertEqual(len(v["head"]), 48)

    def test_nan_scalar_yields_inconclusive(self):
        doc = _rail([_src(n=2, amplitude=0.0), {"id": "rho", "fn": "measure.lag1_autocorr"},
                     {"id": "chk", "fn": "diagnose.bound_check", "params": {"upper": 0.2}}],
                    [{"from": "src.series", "to": "rho.series"},
                     {"from": "rho.rho", "to": "chk.value"}])
        self.assertEqual(run_rail(doc)["values"]["chk.verdict"]["status"], VERDICT_INCONCLUSIVE)


class PresetTests(unittest.TestCase):
    def test_all_presets_validate(self):
        presets = list_presets()
        self.assertGreaterEqual(len(presets), 4)
        for p in presets:
            _, errors = check_rail(p["doc"])
            self.assertEqual(errors, [], p["file"])
            self.assertEqual(p["file"], f"{p['rail']}.rail.json")

    def test_hr1_iid_not_rejected(self):
        r = run_rail(load_preset("hr1_threshold_switching"))
        self.assertEqual(r["values"]["verdict.verdict"]["status"], VERDICT_NOT_REJECTED)

    def test_hr1_autocorrelated_rejected(self):
        r = run_rail(load_preset("hr1_autocorrelated_noise"))
        verdict = r["values"]["verdict.verdict"]
        self.assertEqual(verdict["status"], VERDICT_REJECTED)
        self.assertGreater(abs(r["values"]["rho.rho"]), 0.2)
        self.assertLess(r["values"]["s.rate"], r["values"]["pred.rate"])

    def test_hysteresis_preset(self):
        r = run_rail(load_preset("hysteresis_band_check"))
        self.assertLessEqual(r["values"]["hys_sw.count"], 1)
        self.assertGreater(r["values"]["thr_sw.count"], 1000)
        self.assertEqual(r["values"]["verdict.verdict"]["status"], VERDICT_NOT_REJECTED)

    def test_load_preset_rejects_path_tricks(self):
        with self.assertRaises(ValueError):
            load_preset("../secrets")


class TheoryPropertyTests(unittest.TestCase):
    """Executable checks of statements in krystal_theory (03 and 05)."""

    @staticmethod
    def _iid_modes(n, p, rng):
        return [1 if rng.random() < p else 0 for _ in range(n)]

    def test_theorem_3_1_and_lemma_5_2_monte_carlo(self):
        rng = random.Random(2026)
        # At p = 1/2 Lemma 5.2 and the naive binomial variance coincide; std ratio is
        # sqrt(2(1-3a)/(1-2a)), about 1.33 at p = 0.1, so that p discriminates them.
        p, n, runs = 0.1, 2000, 400
        rates = [_switches(m) / (n - 1) for m in (self._iid_modes(n, p, rng) for _ in range(runs))]
        a = p * (1 - p)
        self.assertAlmostEqual(statistics.fmean(rates), 2 * a, delta=0.005)
        sigma_lemma = math.sqrt(4 * a * (1 - 3 * a) / (n - 1))
        sigma_naive = math.sqrt(2 * a * (1 - 2 * a) / (n - 1))
        sd = statistics.stdev(rates)
        self.assertLess(abs(sd - sigma_lemma) / sigma_lemma, 0.12)
        self.assertGreater(abs(sd - sigma_naive) / sigma_naive, 0.20)

    def test_theorem_3_2_bounded_noise(self):
        for seed in range(25):
            rng = random.Random(seed)
            delta = rng.uniform(0.1, 5.0)
            mean = 85.0 + rng.uniform(-2 * delta, 2 * delta)
            xs = [mean + rng.uniform(-delta, delta) for _ in range(3000)]
            for initial in (0, 1):
                doc = _rail([{"id": "c", "fn": "source.constant", "params": {"value": 0}},
                             {"id": "h", "fn": "transform.hysteresis_mode",
                              "params": {"h": 2 * delta, "initial": initial}}], [])
                spec = DEFAULT_REGISTRY.get("transform.hysteresis_mode")
                out = spec.fn({"series": xs}, spec.resolved_params(doc["nodes"][1]["params"]))
                changes = _switches([initial] + out["modes"])
                self.assertLessEqual(changes, 1, (seed, initial))

    def test_corollary_3_3_h_zero_chatters(self):
        spec = DEFAULT_REGISTRY.get("transform.hysteresis_mode")
        xs = [86.0, 84.0] * 50
        out = spec.fn({"series": xs}, spec.resolved_params({"h": 0.0}))
        self.assertEqual(_switches(out["modes"]), len(xs) - 1)

    def test_lemma_5_1_subsampling_never_adds_switches(self):
        rng = random.Random(5)
        for _ in range(200):
            modes = [rng.randint(0, 1) for _ in range(rng.randint(2, 300))]
            k = rng.randint(1, 10)
            offset = rng.randrange(k)
            self.assertLessEqual(_switches(modes[offset::k]), _switches(modes))

    def test_lemma_5_3_quantisation_error_bound(self):
        spec = DEFAULT_REGISTRY.get("transform.quantize")
        rng = random.Random(9)
        xs = [rng.uniform(-100, 100) for _ in range(5000)]
        for q in (0.1, 0.5, 2.0):
            out = spec.fn({"series": xs}, {"q": q})["series"]
            self.assertLessEqual(max(abs(a - b) for a, b in zip(xs, out)), q / 2 + 1e-9)


class RealSourceTests(unittest.TestCase):
    """Real stdlib telemetry: assert structure only (values are machine-dependent)."""

    def test_clock_quantum(self):
        spec = DEFAULT_REGISTRY.get("source.clock_quantum")
        out = spec.fn({}, spec.resolved_params({"clock": "perf_counter", "samples": 4}))
        self.assertTrue(out["quantum_ms"] > 0)
        self.assertGreater(out["reported_resolution_ms"], 0)

    def test_sleep_jitter_and_cpu_sampler(self):
        r = run_rail(_rail(
            [{"id": "j", "fn": "source.sleep_jitter", "params": {"interval_ms": 1, "n": 5}},
             {"id": "c", "fn": "source.process_cpu_sampler",
              "params": {"duration_s": 0.1, "interval_ms": 20, "duty": 0.5}},
             {"id": "st", "fn": "measure.stats"}],
            [{"from": "c.cpu_pct", "to": "st.series"}]))
        self.assertTrue(r["ok"], r["error"])
        self.assertEqual(len(r["values"]["j.jitter_ms"]), 5)
        self.assertGreaterEqual(len(r["values"]["c.cpu_pct"]), 2)
        self.assertEqual(r["provenance"]["st.stats"], "measured")


class SchemaFileTests(unittest.TestCase):
    def test_rail_schema_file_matches_code(self):
        on_disk = json.loads((ROOT / "schemas" / "krystal_config_rail.schema.json")
                             .read_text(encoding="utf-8"))
        self.assertEqual(on_disk, RAIL_SCHEMA)

    def test_all_schemas_use_supported_keywords_only(self):
        files = [ROOT / "schemas" / "krystal_config_rail.schema.json",
                 *sorted(THEORY_SCHEMAS.glob("*.schema.json"))]
        self.assertGreaterEqual(len(files), 5)
        for f in files:
            schema = json.loads(f.read_text(encoding="utf-8"))
            self.assertEqual(unsupported_keywords(schema), [], f.name)

    def test_theory_templates_and_examples_validate(self):
        theory_schema = json.loads((THEORY_SCHEMAS / "telemetry_theory.schema.json")
                                   .read_text(encoding="utf-8"))
        docs = sorted((ROOT / "krystal_theory" / "priklady").glob("*.theory.json")) + \
            sorted(THEORY_SCHEMAS.glob("*.template.json"))
        self.assertGreaterEqual(len(docs), 2)
        for f in docs:
            doc = json.loads(f.read_text(encoding="utf-8"))
            self.assertEqual(validate(doc, theory_schema, base_dir=THEORY_SCHEMAS), [], f.name)

    def test_example_references_resolve(self):
        """Every rail named in the example exists, and verdicts cite declared hypotheses."""
        ex = json.loads((ROOT / "krystal_theory" / "priklady" /
                         "hr1_prepinanie_pri_prahu.theory.json").read_text(encoding="utf-8"))
        hyp_ids = {h["id"] for h in ex["hypotheses"]}
        sig_ids = {s["id"] for s in ex["signals"]}
        for h in ex["hypotheses"]:
            self.assertTrue(set(h["signals"]) <= sig_ids, h["id"])
            if h["protocol"].get("rail"):
                load_preset(h["protocol"]["rail"])
        for v in ex["verdicts"]:
            self.assertIn(v["hypothesis"], hyp_ids)
            if v.get("rail"):
                load_preset(v["rail"])
        # A hypothesis status may only move by a MEASURED verdict on its own target.
        for h in ex["hypotheses"]:
            if h["status"] in ("NEZAMIETNUTÉ", "ZAMIETNUTÉ"):
                backing = [v for v in ex["verdicts"] if v["hypothesis"] == h["id"]
                           and v["provenance"] == "MEASURED" and v["scope"] == h["target"]
                           and v["status"] == h["status"]]
                self.assertTrue(backing, f"{h['id']} status {h['status']} lacks a measured verdict")

    def test_bad_theory_document_is_rejected(self):
        theory_schema = json.loads((THEORY_SCHEMAS / "telemetry_theory.schema.json")
                                   .read_text(encoding="utf-8"))
        ex = json.loads((ROOT / "krystal_theory" / "priklady" /
                         "hr1_prepinanie_pri_prahu.theory.json").read_text(encoding="utf-8"))
        bad = copy.deepcopy(ex)
        bad["verdicts"][0]["provenance"] = "MEASURED"
        bad["verdicts"][0].pop("measured_on", None)
        errs = validate(bad, theory_schema, base_dir=THEORY_SCHEMAS)
        self.assertTrue(errs, "MEASURED verdict without measured_on must be rejected")
        bad2 = copy.deepcopy(ex)
        bad2["hypotheses"][0]["rejection_criteria"] = []
        self.assertTrue(validate(bad2, theory_schema, base_dir=THEORY_SCHEMAS))


if __name__ == "__main__":
    unittest.main()
