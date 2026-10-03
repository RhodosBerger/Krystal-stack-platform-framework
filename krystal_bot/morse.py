"""Morse as the platform's human-legible wire format for bytecode and process streams (stdlib only).

Three things live here:

1. A *lossless* Morse codec for `krystal_bot.bytecode.Programs` (op letter + 3-digit decimal args).
2. A fixed-length integer *tensor form* of any Morse string (dot=1, dash=2, letter gap=3, word gap=4,
   pad=0). This is the numeric input a model, including an OpenVINO-compiled one, can consume.
3. `ProcessPredictor`: an n-gram model over the Morse-coded stream of kernel events that predicts the
   *next* processes. It is evaluated on held-out data against a most-frequent baseline and reports the
   result honestly; it is a statistical predictor, not a neural network, and not run on OpenVINO.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .bytecode import BY_LETTER, OPS, BytecodeError, Program

MORSE = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.", "G": "--.", "H": "....", "I": "..", "J": ".---",
    "K": "-.-", "L": ".-..", "M": "--", "N": "-.", "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.", "S": "...", "T": "-",
    "U": "..-", "V": "...-", "W": ".--", "X": "-..-", "Y": "-.--", "Z": "--..",
    "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-", "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.",
}
UNMORSE = {v: k for k, v in MORSE.items()}
WORD_GAP = " / "


def encode_text(text: str) -> str:
    return " ".join(MORSE[c] for c in text.upper() if c in MORSE)


def decode_text(morse: str) -> str:
    out = []
    for tok in morse.split():
        if tok not in UNMORSE:
            raise ValueError(f"invalid morse token {tok!r}")
        out.append(UNMORSE[tok])
    return "".join(out)


# ---------------------------------------------------------------------- program codec
def encode_program(prog: Program) -> str:
    prog.validate()
    words = []
    for op, args in prog.ops:
        parts = [MORSE[OPS[op][1]]]
        for a in args:
            parts.extend(MORSE[d] for d in f"{a:03d}")
        words.append(" ".join(parts))
    return WORD_GAP.join(words)


def decode_program(morse: str) -> Program:
    ops = []
    for word in morse.split("/"):
        toks = word.split()
        if not toks:
            raise BytecodeError("empty morse word")
        letter = UNMORSE.get(toks[0])
        if letter not in BY_LETTER:
            raise BytecodeError(f"unknown op letter {toks[0]!r}")
        op = BY_LETTER[letter]
        digits = []
        for t in toks[1:]:
            d = UNMORSE.get(t)
            if d is None or not d.isdigit():
                raise BytecodeError(f"invalid digit token {t!r}")
            digits.append(d)
        if len(digits) % 3:
            raise BytecodeError("args must be 3 decimal digits each")
        args = tuple(int("".join(digits[i:i + 3])) for i in range(0, len(digits), 3))
        ops.append((op, args))
    p = Program(ops)
    p.validate()
    return p


# ---------------------------------------------------------------------- tensor form
PAD, DOT, DASH, LGAP, WGAP = 0, 1, 2, 3, 4


def morse_to_tensor(morse: str, length: int = 2048) -> List[int]:
    out: List[int] = []
    for word_i, word in enumerate(morse.split("/")):
        if word_i:
            out.append(WGAP)
        for tok_i, tok in enumerate(word.split()):
            if tok_i:
                out.append(LGAP)
            out.extend(DOT if c == "." else DASH for c in tok)
    if len(out) > length:
        raise ValueError(f"tensor needs {len(out)} elements; limit is {length}")
    return out + [PAD] * (length - len(out))


def tensor_to_morse(t: Sequence[int]) -> str:
    sym = {DOT: ".", DASH: "-", LGAP: " ", WGAP: " / "}
    return "".join(sym[v] for v in t if v != PAD)


# ---------------------------------------------------------------------- event stream
def event_symbol(ev: Dict[str, Any]) -> str:
    """One letter per kernel event. The alphabet is the contract between logs, Morse and the predictor."""
    t = ev.get("type")
    if t == "task_done":
        if not ev.get("ok", True):
            return "X"
        if ev.get("missed"):
            return "M"
        return "P" if ev.get("lane") == "process" else "T"
    return {"task_rejected": "J", "task_expired": "D", "task_retry": "R", "heal": "H", "heal_suppressed": "S", "pattern": "N",
            "health_change": "C", "breaker_opened": "B", "breaker_closed": "K", "backend_error": "E"}.get(t, "O")


SYMBOL_MEANING = {"T": "thread task ok", "P": "process task ok", "M": "deadline missed", "X": "task error", "J": "rejected (back-pressure)",
                  "D": "expired in queue", "R": "retry", "H": "self-heal action", "S": "heal suppressed (budget)", "N": "pattern announced",
                  "C": "health changed", "B": "breaker opened", "K": "breaker closed", "E": "backend error", "O": "other"}


def events_to_symbols(events: Iterable[Dict[str, Any]]) -> str:
    return "".join(event_symbol(e) for e in events)


def symbols_to_morse(symbols: str) -> str:
    return encode_text(symbols)


class ProcessPredictor:
    """Variable-order (1..order) n-gram over the symbol stream with stupid-backoff scoring."""

    def __init__(self, order: int = 4):
        self.order = order
        self.tables: List[Dict[str, Counter]] = [defaultdict(Counter) for _ in range(order + 1)]
        self.unigram: Counter = Counter()
        self.trained_on = 0

    def fit(self, symbols: str) -> None:
        self.unigram.update(symbols)
        for n in range(1, self.order + 1):
            for i in range(n, len(symbols)):
                self.tables[n][symbols[i - n:i]][symbols[i]] += 1
        self.trained_on += len(symbols)

    def predict(self, context: str, k: int = 3) -> List[Tuple[str, float]]:
        scores: Dict[str, float] = defaultdict(float)
        for n in range(min(self.order, len(context)), 0, -1):
            c = self.tables[n].get(context[-n:])
            if c:
                tot = sum(c.values())
                w = 0.4 ** (min(self.order, len(context)) - n)
                for s, v in c.items():
                    scores[s] += w * v / tot
        if not scores and self.unigram:
            tot = sum(self.unigram.values())
            scores = {s: v / tot for s, v in self.unigram.items()}
        z = sum(scores.values()) or 1.0
        return [(s, round(v / z, 4)) for s, v in sorted(scores.items(), key=lambda kv: -kv[1])[:k]]

    def evaluate(self, symbols: str, warmup: Optional[int] = None) -> Dict[str, Any]:
        """Hold out the last 20 %; fit on the first 80 %; report top-1/top-3 against the most-frequent-symbol baseline."""
        n = len(symbols)
        if n < 100:
            return {"status": "insufficient_data", "symbols": n, "note": "need >= 100 symbols to evaluate"}
        cut = int(0.8 * n)
        m = ProcessPredictor(self.order)
        m.fit(symbols[:cut])
        base = m.unigram.most_common(1)[0][0]
        top1 = top3 = base_hit = tot = 0
        for i in range(cut, n):
            pred = [s for s, _ in m.predict(symbols[max(0, i - self.order):i], 3)]
            tot += 1
            top1 += pred[:1] == [symbols[i]]
            top3 += symbols[i] in pred
            base_hit += base == symbols[i]
        res = {"status": "evaluated", "symbols": n, "holdout": tot, "top1": round(top1 / tot, 4), "top3": round(top3 / tot, 4),
               "baseline_most_frequent": round(base_hit / tot, 4), "baseline_symbol": base,
               "beats_baseline": top1 / tot > base_hit / tot + 0.01}
        res["note"] = ("predictor beats the most-frequent baseline on held-out data" if res["beats_baseline"] else
                       "predictor does NOT beat the most-frequent baseline; do not use its output for decisions")
        return res

    def to_json(self) -> Dict[str, Any]:
        return {"order": self.order, "trained_on": self.trained_on, "unigram": dict(self.unigram),
                "tables": [{k: dict(v) for k, v in t.items()} for t in self.tables]}

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "ProcessPredictor":
        m = cls(d["order"])
        m.trained_on, m.unigram = d["trained_on"], Counter(d["unigram"])
        for n, t in enumerate(d["tables"]):
            for k, v in t.items():
                m.tables[n][k] = Counter(v)
        return m
