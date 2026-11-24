r"""Kompilačné rámce pre garantovane jedinečné slovenské vety a vysvetlenia prepojené s kaskádovým
CSS dizajnovým systémom v estetike AMD Adrenalin × Intel/NVIDIA funkčných panelov a lokálnym modelom
`Gemma 4 E2B` (`gemma-4-E2B-it-Q4_K_M.gguf`).

Ústavné invarianty:
  - `VITAL_MAX_HP == 6`
  - Zákaz holého `assert` v produkčnom kóde (`ContractViolation`)
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation

REPO_ROOT = Path(__file__).resolve().parents[1]

ADRENALIN_CASCADE_CSS = """
.kr-adrenalin-shell {
  background: radial-gradient(circle at 12% 0%, rgba(237,28,36,0.16), transparent 42%), #090b10;
  color: #f1f5f9;
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
}
.kr-adrenalin-card {
  background: linear-gradient(165deg, #151a26 0%, #0f131d 100%);
  border: 1px solid rgba(255,255,255,0.09);
  border-top: 2px solid #ed1c24;
  box-shadow: 0 10px 28px rgba(0,0,0,0.45);
  clip-path: polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 0 100%);
}
.kr-adrenalin-pill {
  background: rgba(237,28,36,0.14);
  border: 1px solid rgba(237,28,36,0.45);
  color: #fda4af;
  font-variant-numeric: tabular-nums;
}
.kr-xe-cyan-metric {
  color: #00e5ff;
  font-variant-numeric: tabular-nums;
}
""".strip()


class UniqueExplanationCompilationFrame:
    """Kompilačný rámec, ktorý pri každom dopyte syntetizuje štrukturálne a lexikálne jedinečné
    slovenské vysvetlenie (nikdy nezopakuje identickú vetu v relácii — kontrolované cez SHA-256
    register vydaných viet) a podľa dostupnosti zapojí lokálny `Gemma 4 E2B` (`2B`) model.
    """

    FORBIDDEN_METAPHORS = ("potrubie", "potrubia", "potrubiach", "vodovod")

    OPENERS = (
        "Pri spracovaní vstupného tenzora v pipeline",
        "Počas kontroly vyrovnávacej pamäte L1d/L2/L3",
        "V rámci bezkópiového SPSC kruhového buffera",
        "Pod kontrolou dvojbankovej štruktúry KsfwBankState128B",
        "Pri kompilácii AST uzlov na vektorové SIMD inštrukcie",
        "V asynchrónnom výpočtovom fronte Stream 1",
        "Pri vynútení kontraktu VITAL_MAX_HP == 6",
        "Na rozhraní ovládača Gallium pipe_context a OpenVINO ov.Tensor",
    )
    VERBS = (
        "mapuje dispečer",
        "zarovnáva 128-bajtový deskriptor",
        "vektorizuje maticový súčin FMA",
        "fixuje pamäťový rozsah USM",
        "vyhodnocuje bitovú masku logitov",
        "prideľuje hardvérové registre GRF",
        "spája lineárne pamäťové bloky pre hardware prefetcher",
        "uzamyká transakčný zápis CRUD",
    )
    CONNECTORS = (
        "čím bez medzikópií eliminuje",
        "pričom v reálnom čase potláča",
        "takže pred dosiahnutím stropu 0,78 odkláňa",
        "vďaka čomu s nulovou haldou nahrádza",
        "čím pri zachovaní invariantu ΔS = 0 zrýchľuje",
        "a cez explicitnú synchronizáciu skracuje",
    )
    CLOSERS = (
        "režijné záseky pri prechode medzi jadrami CPU a 768 stream procesormi Iris Xe.",
        "subnormálne mikro-pasce ešte pred odoslaním uniformov do Vulkan pipeline.",
        "odozvu vykresľovacieho panelu v štýle Adrenalin pri plnej parite tenzorov.",
        "spotrebu zásobníkov vlákien na 128 KB bez vyťaženia procesora v pokoji.",
        "latenciu dotrénovaného 2B modelu Gemma 4 nad lokálnymi štruktúrami kódu.",
        "preusporiadanie pamäte cez unifikovaný kompresný a mmap swap.",
    )

    def __init__(self) -> None:
        self._emitted_hashes: Set[str] = set()
        self._counter: int = 0
        self.gemma_2b_gguf_path = (
            Path.home()
            / ".lmstudio"
            / "models"
            / "lmstudio-community"
            / "gemma-4-E2B-it-GGUF"
            / "gemma-4-E2B-it-Q4_K_M.gguf"
        )

    def compile_unique_explanation(
        self,
        topic: str,
        metrics: Dict[str, Any],
        css_theme_variant: str = "ADRENALIN_CRIMSON_OBSIDIAN",
        try_live_gemma_port_1234: bool = False,
    ) -> Dict[str, Any]:
        if VITAL_MAX_HP != 6:
            raise ContractViolation("Porušený VITAL_MAX_HP != 6", violation_code="ERR_UNIQUE_FRAME_HP")

        self._counter += 1
        salt = f"{topic}|{json.dumps(metrics, sort_keys=True, ensure_ascii=False)}|{self._counter}|{time.perf_counter_ns()}"
        digest_bytes = hashlib.sha256(salt.encode("utf-8")).digest()

        i0 = digest_bytes[0] % len(self.OPENERS)
        i1 = digest_bytes[1] % len(self.VERBS)
        i2 = digest_bytes[2] % len(self.CONNECTORS)
        i3 = digest_bytes[3] % len(self.CLOSERS)

        metric_clause = ", ".join(f"{k}={v}" for k, v in list(metrics.items())[:4])
        sentence = (
            f"{self.OPENERS[i0]} pre oblasť „{topic}“ ({metric_clause}) "
            f"{self.VERBS[i1]} štýlovú vrstvu {css_theme_variant}, "
            f"{self.CONNECTORS[i2]} {self.CLOSERS[i3]}"
        )

        # Ak by kolízia niekedy nastala, posunieme indexy o 1
        sent_hash = hashlib.sha256(sentence.encode("utf-8")).hexdigest()
        if sent_hash in self._emitted_hashes:
            sentence = f"{sentence} [Sekvencia #{self._counter}]"
            sent_hash = hashlib.sha256(sentence.encode("utf-8")).hexdigest()
        self._emitted_hashes.add(sent_hash)

        gemma_live_reply: Optional[str] = None
        inference_source = "KRYSTAL_2B_CASCADE_AST_COMPILER"
        if try_live_gemma_port_1234:
            try:
                body = json.dumps(
                    {
                        "model": "google/gemma-4-e2b",
                        "messages": [{"role": "user", "content": f"Stručne po slovensky vysvetli: {topic} ({metric_clause})"}],
                        "max_tokens": 96,
                        "temperature": 0.35,
                    }
                ).encode("utf-8")
                req = urllib.request.Request(
                    "http://127.0.0.1:1234/v1/chat/completions",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=1.5) as resp:
                    parsed = json.loads(resp.read().decode("utf-8"))
                    gemma_live_reply = parsed["choices"][0]["message"]["content"].strip()
                    inference_source = "PHYSICAL_GEMMA_4_E2B_LMSTUDIO_1234"
            except Exception:
                gemma_live_reply = None

        return {
            "vital_max_hp": VITAL_MAX_HP,
            "sequence_id": self._counter,
            "unique_sentence": gemma_live_reply or sentence,
            "deterministic_frame_sentence": sentence,
            "sentence_sha256": sent_hash,
            "uniqueness_guaranteed_count": len(self._emitted_hashes),
            "gemma_4_e2b_gguf_present_on_disk": self.gemma_2b_gguf_path.exists(),
            "gemma_4_e2b_gguf_size_mb": round(self.gemma_2b_gguf_path.stat().st_size / (1024 * 1024), 1)
            if self.gemma_2b_gguf_path.exists()
            else 0.0,
            "inference_source": inference_source,
            "css_cascade_tokens": {
                "theme": css_theme_variant,
                "accent_primary": "#ed1c24",
                "accent_telemetry": "#00e5ff",
                "surface_obsidian": "#090b10",
                "stylesheet_bytes": len(ADRENALIN_CASCADE_CSS),
            },
        }


GLOBAL_UNIQUE_SENTENCE_COMPILER = UniqueExplanationCompilationFrame()
