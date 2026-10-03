"""Hybrid retrieval index for the Krystal bot (stdlib only).

Two rankers are fused with reciprocal-rank fusion:

* BM25 over accent-folded word tokens (so Slovak queries match without diacritics)
* cosine over a hashed character n-gram vector (robust to typos / inflection)

Neither is a *semantic* embedding. The index is deliberately model-free so it works on the
default zero-dependency install; an LLM or a learned embedder can be layered on top later.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
import unicodedata
from collections import Counter, OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DIM = 256
_WORD = re.compile(r"[a-z0-9_]+")


def fold(text: str) -> str:
    """Lower-case and strip diacritics ("zámer" -> "zamer")."""
    t = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def tokenize(text: str) -> List[str]:
    return _WORD.findall(fold(text))


def _hvec(text: str, dim: int = DIM) -> List[float]:
    v = [0.0] * dim
    t = f" {fold(text)} "
    for n in (3, 4):
        for i in range(len(t) - n + 1):
            h = int.from_bytes(hashlib.blake2b(t[i:i + n].encode("utf-8"), digest_size=8).digest(), "little")
            v[h % dim] += 1.0 if (h >> 63) & 1 else -1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class Doc:
    __slots__ = ("id", "source", "kind", "text", "meta", "ts", "tokens", "tf", "vec")

    def __init__(self, id: str, source: str, kind: str, text: str, meta: Optional[Dict[str, Any]] = None, ts: Optional[float] = None):
        self.id, self.source, self.kind, self.text = id, source, kind, text
        self.meta = meta or {}
        self.ts = ts if ts is not None else time.time()
        self.tokens = tokenize(text)
        self.tf = Counter(self.tokens)
        self.vec = _hvec(text)

    def to_json(self) -> Dict[str, Any]:
        return {"id": self.id, "source": self.source, "kind": self.kind, "text": self.text, "meta": self.meta, "ts": self.ts}


class RagIndex:
    """Thread-safe in-memory index with optional JSONL persistence and a hard document cap."""

    def __init__(self, path: Optional[str] = None, max_docs: int = 20000, k1: float = 1.4, b: float = 0.75):
        self.path, self.max_docs, self.k1, self.b = path, max_docs, k1, b
        self._docs: "OrderedDict[str, Doc]" = OrderedDict()
        self._df: Counter = Counter()
        self._len_sum = 0
        self._lock = threading.RLock()
        self.evicted = 0
        if path and os.path.exists(path):
            self._load(path)

    # ------------------------------------------------------------------ mutation
    def add(self, source: str, kind: str, text: str, meta: Optional[Dict[str, Any]] = None, doc_id: Optional[str] = None,
            ts: Optional[float] = None) -> str:
        text = text.strip()
        if not text:
            return ""
        did = doc_id or hashlib.blake2b(f"{source}\0{kind}\0{text}".encode("utf-8"), digest_size=10).hexdigest()
        with self._lock:
            if did in self._docs:
                return did  # idempotent: re-ingesting the same text is a no-op
            d = Doc(did, source, kind, text, meta, ts)
            self._docs[did] = d
            self._df.update(d.tf.keys())
            self._len_sum += len(d.tokens)
            while len(self._docs) > self.max_docs:
                self._evict_one()
        return did

    def _evict_one(self) -> None:
        # Evict the oldest *log-like* document first so curated docs outlive telemetry noise.
        victim = next((i for i, d in self._docs.items() if d.kind in ("event", "event_summary", "chat")), None)
        if victim is None:
            victim = next(iter(self._docs))
        self._remove(victim)
        self.evicted += 1

    def _remove(self, did: str) -> None:
        d = self._docs.pop(did, None)
        if d is None:
            return
        for t in d.tf.keys():
            self._df[t] -= 1
            if self._df[t] <= 0:
                del self._df[t]
        self._len_sum -= len(d.tokens)

    def remove_source(self, source: str) -> int:
        with self._lock:
            ids = [i for i, d in self._docs.items() if d.source == source]
            for i in ids:
                self._remove(i)
            return len(ids)

    # ------------------------------------------------------------------ retrieval
    def search(self, query: str, k: int = 5, kinds: Optional[Iterable[str]] = None, min_score: float = 0.0) -> List[Dict[str, Any]]:
        q_tokens = tokenize(query)
        if not q_tokens and not query.strip():
            return []
        kinds = set(kinds) if kinds else None
        with self._lock:
            docs = [d for d in self._docs.values() if kinds is None or d.kind in kinds]
            n = len(self._docs)
            if not docs:
                return []
            avg = (self._len_sum / n) if n else 1.0
            bm: List[Tuple[float, Doc]] = []
            for d in docs:
                s = 0.0
                dl = len(d.tokens) or 1
                for t in set(q_tokens):
                    f = d.tf.get(t)
                    if not f:
                        continue
                    df = self._df.get(t, 0)
                    idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
                    s += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / avg))
                if s > 0:
                    bm.append((s, d))
            qv = _hvec(query)
            cs = sorted(((_cos(qv, d.vec), d) for d in docs), key=lambda x: -x[0])[: max(k * 4, 20)]
        bm.sort(key=lambda x: -x[0])
        fused: Dict[str, float] = {}
        score_bm = {d.id: s for s, d in bm}
        score_cs = {d.id: s for s, d in cs}
        for ranking in (bm[: max(k * 4, 20)], cs):
            for rank, (_, d) in enumerate(ranking):
                fused[d.id] = fused.get(d.id, 0.0) + 1.0 / (60 + rank + 1)
        by_id = {d.id: d for _, d in bm}
        by_id.update({d.id: d for _, d in cs})
        out = []
        for did, sc in sorted(fused.items(), key=lambda x: -x[1])[:k]:
            d = by_id[did]
            # Hits that only matched via weak n-gram overlap are noise; require some lexical OR strong n-gram signal.
            if score_bm.get(did, 0.0) <= 0 and score_cs.get(did, 0.0) < 0.35:
                continue
            if sc < min_score:
                continue
            out.append({"id": did, "score": round(sc, 5), "bm25": round(score_bm.get(did, 0.0), 4), "ngram": round(score_cs.get(did, 0.0), 4),
                        "source": d.source, "kind": d.kind, "text": d.text, "meta": d.meta,
                        "citation": f"{d.source}" + (f"#{d.meta['heading']}" if d.meta.get("heading") else "")})
        return out

    def __len__(self) -> int:
        return len(self._docs)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            kinds = Counter(d.kind for d in self._docs.values())
            return {"documents": len(self._docs), "vocabulary": len(self._df), "by_kind": dict(kinds),
                    "max_docs": self.max_docs, "evicted": self.evicted, "ranker": "bm25+hashed-ngram (not semantic)"}

    # ------------------------------------------------------------------ persistence
    def save(self, path: Optional[str] = None) -> str:
        p = path or self.path
        if not p:
            raise ValueError("no path")
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        tmp = p + ".tmp"
        with self._lock, open(tmp, "w", encoding="utf-8") as f:
            for d in self._docs.values():
                f.write(json.dumps(d.to_json(), ensure_ascii=False) + "\n")
        os.replace(tmp, p)  # atomic: a crash mid-save never leaves a truncated index
        return p

    def _load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    self.add(r["source"], r["kind"], r["text"], r.get("meta"), r.get("id"), r.get("ts"))
                except (ValueError, KeyError):
                    continue


# ---------------------------------------------------------------------- ingestors
_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def chunk_markdown(text: str, max_chars: int = 900) -> List[Tuple[str, str]]:
    """Split markdown into (heading, chunk) pairs. Code fences are never split mid-block."""
    chunks: List[Tuple[str, str]] = []
    heading, buf, in_fence = "", [], False

    def flush() -> None:
        body = "\n".join(buf).strip()
        buf.clear()
        while len(body) > max_chars:
            cut = body.rfind("\n\n", 0, max_chars)
            if cut < max_chars // 3:
                cut = max_chars
            chunks.append((heading, body[:cut].strip()))
            body = body[cut:].strip()
        if body:
            chunks.append((heading, body))

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(line)
        if m:
            flush()
            heading = m.group(2)
            buf.append(line)
        else:
            buf.append(line)
    flush()
    return [(h, c) for h, c in chunks if len(c) > 20]


def ingest_markdown_file(index: RagIndex, path: str, root: Optional[str] = None) -> int:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    rel = os.path.relpath(path, root) if root else os.path.basename(path)
    rel = rel.replace("\\", "/")
    index.remove_source(rel)
    n = 0
    for heading, chunk in chunk_markdown(text):
        if index.add(rel, "doc", chunk, {"heading": heading}):
            n += 1
    return n


def ingest_markdown_tree(index: RagIndex, root: str, max_files: int = 400, skip: Sequence[str] = ("venv", ".venv", ".venv-keras", "node_modules", "target", ".git", "dist")) -> Dict[str, int]:
    files, chunks = 0, 0
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in skip and not d.startswith(".venv")]
        for fn in sorted(fns):
            if fn.lower().endswith(".md") and files < max_files:
                try:
                    chunks += ingest_markdown_file(index, os.path.join(dp, fn), root)
                    files += 1
                except OSError:
                    continue
    return {"files": files, "chunks": chunks}


def event_to_text(ev: Dict[str, Any]) -> str:
    skip = {"seq"}
    parts = [str(ev.get("type", "event"))]
    parts += [f"{k}={v}" for k, v in ev.items() if k not in skip and k not in ("type", "t")]
    return " ".join(parts)


def summarize_events(events: Sequence[Dict[str, Any]]) -> str:
    """Compress a window of kernel events into one retrievable paragraph."""
    if not events:
        return ""
    done = [e for e in events if e.get("type") == "task_done"]
    lat = sorted(e["latency_ms"] for e in done if isinstance(e.get("latency_ms"), (int, float)))
    p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else None
    by_lane = Counter(e.get("lane") for e in done)
    errs = sum(1 for e in done if not e.get("ok", True))
    miss = sum(1 for e in done if e.get("missed"))
    other = Counter(e.get("type") for e in events if e.get("type") != "task_done")
    t0, t1 = events[0].get("t", 0), events[-1].get("t", 0)
    return (f"window seq {events[0].get('seq')}..{events[-1].get('seq')} span {t1 - t0:.1f}s: {len(done)} task_done "
            f"lanes={dict(by_lane)} errors={errs} deadline_missed={miss} p95_latency_ms={p95} other_events={dict(other)}")


def ingest_events(index: RagIndex, events: Iterable[Dict[str, Any]], window: int = 100, source: str = "kernel_events") -> int:
    """Index notable events individually and bulk `task_done` events as windowed summaries."""
    added, buf = 0, []
    for ev in events:
        if ev.get("type") == "task_done":
            buf.append(ev)
            if len(buf) >= window:
                added += bool(index.add(source, "event_summary", summarize_events(buf), {"seq": buf[-1].get("seq")}, ts=buf[-1].get("t")))
                buf = []
        else:
            added += bool(index.add(source, "event", event_to_text(ev), {"seq": ev.get("seq"), "etype": ev.get("type")}, ts=ev.get("t")))
    if buf:
        added += bool(index.add(source, "event_summary", summarize_events(buf), {"seq": buf[-1].get("seq")}, ts=buf[-1].get("t")))
    return added
