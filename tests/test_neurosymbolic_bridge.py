"""Tests for the neuro-symbolic bridge verifier (no live LLM needed)."""

from __future__ import annotations

from projects.trigonometric_hybrid_platform_framework.neurosymbolic_bridge import build_compendium, verify_answer

FACTS = [{"fact_id": "F_C06_threshold_k", "statement": "NumPy BLAS leaves sequential-FMA order at K = 449",
          "source": "x", "key_path": "x", "evidence": "LOCAL_MEASURED", "value": 449}]
RULES = [{"rule_id": "R_GPU_RECIPE", "statement": "GPU conv kernel computes sequential FMA", "verdict": "SUPPORTED_WITH_SCOPE",
          "derived_from": []}]


def test_verifier_classifies_sentences() -> None:
    text = ("BLAS changes order at K = 449 [F_C06_threshold_k]. "
            "BLAS changes order at K = 512 [F_C06_threshold_k]. "
            "The GPU is always faster. "
            "Compile time is tiny [F_C03_FAKE].")
    res = verify_answer(text, FACTS, RULES)
    assert [s["status"] for s in res["sentences"]] == ["GROUNDED", "NUMBER_MISMATCH", "UNCITED", "FABRICATED_CITATION"]
    assert res["verdict"] == "CONFLICT"


def test_compendium_without_llm_contains_no_generated_text(tmp_path) -> None:
    out = build_compendium({"facts": FACTS, "rules": RULES}, "What is K?", llm={"ok": False, "error": "offline"},
                           path=tmp_path / "kb.md")
    text = (tmp_path / "kb.md").read_text(encoding="utf-8")
    assert out["verification"]["verdict"] == "UNKNOWN"
    assert "LLM unavailable" in text and "R_GPU_RECIPE" in text


def test_compendium_keeps_only_grounded(tmp_path) -> None:
    llm = {"ok": True, "text": "K is 449 [F_C06_threshold_k]. K is 7 [F_C06_threshold_k]."}
    build_compendium({"facts": FACTS, "rules": RULES}, "What is K?", llm=llm, path=tmp_path / "kb.md")
    text = (tmp_path / "kb.md").read_text(encoding="utf-8")
    assert "- K is 449 [F_C06_threshold_k]." in text
    assert "~~K is 7 [F_C06_threshold_k].~~ (NUMBER_MISMATCH)" in text
