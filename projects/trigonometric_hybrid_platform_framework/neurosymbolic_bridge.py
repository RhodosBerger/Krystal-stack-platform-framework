"""Neuro-symbolic bridge: the LLM proposes, the symbolic fact base verifies.

Flow
  1. build_grounded_prompt(): facts + rules -> prompt that demands a [fact_id] citation per sentence
  2. ask_llm(): OpenAI-compatible endpoint (default LM Studio http://127.0.0.1:1234/v1)
  3. verify_answer(): per sentence
       UNCITED              no citation              -> rejected
       FABRICATED_CITATION  cites an unknown ID      -> rejected
       NUMBER_MISMATCH      a number in the sentence does not occur in the cited facts -> rejected
       GROUNDED             otherwise
  4. build_compendium(): knowledge-base page = induced rules + only GROUNDED LLM sentences

PROPOSER = LLM, VERIFIER = fact base. An unavailable LLM yields UNKNOWN, never invented text.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from .fact_synthesis_engine import REPORTS

CITE_RE = re.compile(r"\[((?:F|R)_[A-Za-z0-9_]+)\]")
NUM_RE = re.compile(r"(?<![A-Za-z_])-?\d+(?:\.\d+)?(?:e-?\d+)?")


def build_grounded_prompt(facts: List[Dict[str, Any]], rules: List[Dict[str, Any]], question: str, max_facts: int = 60) -> List[Dict[str, str]]:
    lines = [f"[{f['fact_id']}] {f['statement']} (evidence: {f['evidence']})" for f in facts[:max_facts]]
    lines += [f"[{r['rule_id']}] {r['statement']} (verdict: {r['verdict']})" for r in rules]
    system = ("Answer ONLY from the listed facts. End every sentence with the ID(s) it relies on in square brackets, "
              "e.g. [F_C06_threshold_k]. Copy numbers exactly. If the facts do not answer the question, say so.")
    return [{"role": "system", "content": system},
            {"role": "user", "content": "FACTS:\n" + "\n".join(lines) + f"\n\nQUESTION: {question}"}]


def ask_llm(messages: List[Dict[str, str]], base_url: str = "http://127.0.0.1:1234/v1", model: str = "local-model",
            timeout: float = 60.0) -> Dict[str, Any]:
    body = json.dumps({"model": model, "messages": messages, "temperature": 0.0, "max_tokens": 400}).encode("utf-8")
    req = urllib.request.Request(f"{base_url}/chat/completions", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        return {"ok": True, "text": data["choices"][0]["message"]["content"]}
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _numbers(text: str) -> List[str]:
    return [n for n in NUM_RE.findall(text)]


def verify_answer(text: str, facts: List[Dict[str, Any]], rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    known: Dict[str, str] = {f["fact_id"]: f"{f['statement']} {json.dumps(f['value'], default=str)}" for f in facts}
    known.update({r["rule_id"]: f"{r['statement']} {json.dumps(r, default=str)}" for r in rules})
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    out = []
    for s in sentences:
        cites = CITE_RE.findall(s)
        body = CITE_RE.sub("", s)
        if not cites:
            status = "UNCITED"
        elif any(c not in known for c in cites):
            status = "FABRICATED_CITATION"
        else:
            evidence = " ".join(known[c] for c in cites)
            missing = [n for n in _numbers(body) if n not in evidence]
            status = "NUMBER_MISMATCH" if missing else "GROUNDED"
        out.append({"sentence": s, "citations": cites, "status": status})
    grounded = sum(1 for o in out if o["status"] == "GROUNDED")
    return {"sentences": out, "grounded": grounded, "total": len(out),
            "verdict": "UNKNOWN" if not out else ("SUPPORTED_WITH_SCOPE" if grounded == len(out) else "CONFLICT")}


def build_compendium(synthesis: Dict[str, Any], question: str, llm: Optional[Dict[str, Any]] = None,
                     path: Path = REPORTS / "knowledge_base_compendium.md") -> Dict[str, Any]:
    llm = llm if llm is not None else ask_llm(build_grounded_prompt(synthesis["facts"], synthesis["rules"], question))
    lines = ["# Knowledge base compendium", "", "## Induced rules (symbolic)", ""]
    lines += [f"- **{r['verdict']}** — {r['statement']} `({r['rule_id']})`" for r in synthesis["rules"]]
    lines += ["", f"## LLM answer to: {question}", ""]
    if llm.get("ok"):
        ver = verify_answer(llm["text"], synthesis["facts"], synthesis["rules"])
        lines += [f"- {o['sentence']}" for o in ver["sentences"] if o["status"] == "GROUNDED"]
        rejected = [o for o in ver["sentences"] if o["status"] != "GROUNDED"]
        if rejected:
            lines += ["", "Rejected sentences:", ""] + [f"- ~~{o['sentence']}~~ ({o['status']})" for o in rejected]
    else:
        ver = {"verdict": "UNKNOWN", "reason": llm.get("error")}
        lines.append(f"_LLM unavailable: {llm.get('error')}. No generated text included._")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"path": str(path), "llm_ok": bool(llm.get("ok")), "verification": ver}
