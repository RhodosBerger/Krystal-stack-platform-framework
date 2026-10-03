"""
Krystal-Stack Project Intelligence
==================================
Measures the repository and turns the findings into ranked priorities,
expertise options and a heavy-compute pattern catalogue.  Standard library only.

Design rule: nothing here is a hand-written opinion about *current state*.
Every priority is backed by detectors that inspect the working tree, the host
toolchain or the git index.  When a gap is fixed, its detector stops firing and
the priority drops out of the open list on the next scan.  Only the *weights*
(strategic impact, compute heat, leverage, effort) are editorial.

Scoring (documented in docs/PROJECT_AUDIT_AND_COMPUTE_PATTERNS.md):

    gap      = min(1, sum(severity of failing detectors))      severities sum to 1
    score    = 100 * (0.35*impact + 0.35*gap + 0.15*heat + 0.15*leverage)
    value    = score / effort_points          (S=1, M=3, L=8)  -> "quick wins"
"""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXCLUDE_DIRS = {
    "venv", ".venv", "target", "dist", ".git", "__pycache__", "node_modules",
    "logs", "test_logs", "compiled_dataset", ".github",
}
CODE_EXT = {".py", ".rs", ".janet", ".js", ".gd", ".gdshader", ".ps1", ".sh", ".bat", ".html", ".css"}
DOC_EXT = {".md"}
EFFORT_POINTS = {"S": 1, "M": 3, "L": 8}
MAX_TEXT_BYTES = 2_000_000


# ─── scan context ────────────────────────────────────────────────────────────

class ScanContext:
    """Cached, read-only view of the working tree and host toolchain."""

    CODE_ROOTS = [
        "src", "krystal_lang", "openworld_engine", "krystal_web_hub", "krystal_janet",
        "mimicry_engine", "holographic_engine", "instances", "wsl_bridge", "godot_project",
        "gamesa_cortex_v2", "krystal-bitboard", "krystal-ob", "krystal-miner", "krystal-vino",
        "openvino_oneapi_system", "antigravity_prompt_engine.py", "active_optic_compositor.py",
        "demo_neural_ascii_engine_win.py",
    ]
    SOURCE_EXT = {".py", ".rs", ".janet", ".js", ".gdshader", ".gd"}

    def __init__(self, root: str = REPO_ROOT):
        self.root = root
        self._text: Dict[str, str] = {}
        self._tracked: Optional[List[str]] = None
        self._code: Optional[List[Tuple[str, str]]] = None

    def path(self, rel: str) -> str:
        return os.path.join(self.root, *rel.split("/"))

    def exists(self, rel: str) -> bool:
        return os.path.exists(self.path(rel))

    def text(self, rel: str) -> str:
        if rel not in self._text:
            p = self.path(rel)
            try:
                if os.path.isfile(p) and os.path.getsize(p) <= MAX_TEXT_BYTES:
                    with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                        self._text[rel] = fh.read()
                else:
                    self._text[rel] = ""
            except OSError:
                self._text[rel] = ""
        return self._text[rel]

    def code_symbols(self, rel: str) -> str:
        """Identifiers + non-docstring string literals of a Python file.

        Used for 'is X actually called' checks so that a docstring or comment merely *mentioning*
        a symbol (e.g. 'no vkCmdDispatch') cannot make it look implemented.
        """
        key = "\0sym:" + rel
        if key not in self._text:
            src = self.text(rel)
            try:
                tree = ast.parse(src)
                doc_nodes = set()
                for n in ast.walk(tree):
                    if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body \
                            and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant):
                        doc_nodes.add(id(n.body[0].value))
                parts = []
                for n in ast.walk(tree):
                    if isinstance(n, ast.Name):
                        parts.append(n.id)
                    elif isinstance(n, ast.Attribute):
                        parts.append(n.attr)
                    elif isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in doc_nodes:
                        parts.append(n.value)
                self._text[key] = "\n".join(parts)
            except SyntaxError:
                self._text[key] = re.sub(r"#.*", "", src)
        return self._text[key]

    def glob(self, rel_dir: str, suffix: str) -> List[str]:
        d = self.path(rel_dir)
        if not os.path.isdir(d):
            return []
        return sorted(f"{rel_dir}/{n}" for n in os.listdir(d) if n.endswith(suffix))

    def tracked(self) -> List[str]:
        if self._tracked is None:
            try:
                out = subprocess.run(
                    ["git", "ls-files", "-z"], cwd=self.root, capture_output=True, timeout=10
                ).stdout.decode("utf-8", "ignore")
                self._tracked = [p for p in out.split("\0") if p]
            except Exception:
                self._tracked = []
        return self._tracked

    def code_files(self) -> List[Tuple[str, str]]:
        if self._code is None:
            found: List[Tuple[str, str]] = []
            for entry in self.CODE_ROOTS:
                full = self.path(entry)
                if os.path.isfile(full):
                    found.append((entry, self.text(entry)))
                    continue
                for dp, dn, fn in os.walk(full):
                    dn[:] = [d for d in dn if d not in EXCLUDE_DIRS]
                    for f in fn:
                        if os.path.splitext(f)[1].lower() in self.SOURCE_EXT:
                            rel = os.path.relpath(os.path.join(dp, f), self.root).replace("\\", "/")
                            if rel == "src/python/project_intelligence.py":
                                continue  # its own signature regexes are not evidence
                            found.append((rel, self.text(rel)))
            self._code = found
        return self._code

    @staticmethod
    def tool(*names: str) -> Optional[str]:
        for n in names:
            p = shutil.which(n)
            if p:
                return p
        return None

    @staticmethod
    def has_module(name: str) -> bool:
        try:
            return importlib.util.find_spec(name) is not None
        except Exception:
            return False


# ─── subsystem catalogue (what exists; metrics are measured) ─────────────────

def _scripts_not_tests(ctx: ScanContext, rel: str) -> int:
    return len(re.findall(r"^\s*def test_", ctx.text(rel), re.M))


SUBSYSTEMS: Dict[str, Dict[str, Any]] = {
    "neural-ascii-core": {"title": "Neural ASCII engine & compositor", "paths": ["demo_neural_ascii_engine_win.py", "active_optic_compositor.py", "ascii_neural_compositor"], "tests": [], "docs": ["docs/ASCII_TECHNICAL_IMPLEMENTATION_GUIDE.md"]},
    "web-hub": {"title": "Localhost Mission Control (HTTP + SSE)", "paths": ["krystal_web_hub"], "tests": [], "docs": ["docs/API_REFERENCE.md"]},
    "vulkan-compute": {"title": "Vulkan compute driver (ctypes)", "paths": ["src/python/vulkan_compute_driver.py"], "tests": ["tests/test_vulkan_driver.py"], "docs": ["docs/research/DEEP_PERFORMANCE_OPTIMIZATION_AND_CORE_ARCHITECTURE.md"], "verified": "vulkan"},
    "cyclic-organism": {"title": "Symplectic Hamiltonian organism", "paths": ["src/python/cyclic_organism_kernel.py", "schemas/cyclic_architecture_schema.json"], "tests": ["tests/test_cyclic_organism.py"], "docs": ["docs/research/CYCLIC_THEORY_AND_ALTERNATIVE_ARCHITECTURAL_SCHEMAS.md"]},
    "krystal-lang": {"title": "Krystal-Lang compiler & topological VM", "paths": ["krystal_lang"], "tests": ["tests/test_krystal_lang.py"], "docs": ["docs/research/KRYSTAL_LANG_GEOMETRIC_BYTECODE_AND_LLM_COMPILATION.md"]},
    "openworld": {"title": "Open-world semantic compiler & renderer", "paths": ["openworld_engine"], "tests": ["tests/test_openworld_engine.py"], "docs": ["docs/research/ADVANCED_OPENWORLD_PROCEDURAL_RENDERING.md"]},
    "mimicry": {"title": "Mimicry compositor (Blender-style modifiers)", "paths": ["mimicry_engine"], "tests": ["tests/test_mimicry_compositor.py"], "docs": ["docs/research/BLENDER_CORE_MODIFIER_MIMICRY_ENGINE.md"]},
    "antigravity-ar": {"title": "Antigravity prompt engine & AR mirror", "paths": ["antigravity_prompt_engine.py", "instances", "templates", "holographic_engine"], "tests": ["tests/test_procedural_bootstrap.py", "tests/verify_advanced_ar_mirror.py"], "docs": ["docs/research/RECURSIVE_AR_MIRROR_AND_ANTIGRAVITY_MANIFOLDS.md"]},
    "janet-port": {"title": "Janet Lisp sub-project", "paths": ["krystal_janet"], "tests": ["tests/test_janet_integration.py"], "docs": ["krystal_janet/README.md"], "verified": "janet"},
    "godot": {"title": "Godot project & hologram shader", "paths": ["godot_project"], "tests": [], "docs": ["godot_project/README.md"], "verified": "godot"},
    "wsl-bridge": {"title": "WSL2 bridge", "paths": ["wsl_bridge"], "tests": [], "docs": []},
    "rust-native": {"title": "Rust crates (bitboard, ob, miner, gamesa)", "paths": ["krystal-bitboard", "krystal-ob", "krystal-miner", "gamesa_cortex_v2"], "tests": [], "docs": [], "verified": "cargo"},
    "npu-openvino": {"title": "NPU / OpenVINO / ONNX", "paths": ["krystal-vino", "openvino_oneapi_system"], "tests": [], "docs": ["legacy_docs/WINDOWS_OPEN_VINO_COGNITION.md"], "verified": "npu"},
    "cnc-copilot": {"title": "Industrial cognitive manufacturing (legacy)", "paths": ["advanced_cnc_copilot", "brain_cortex_v3", "cognitive_forge_foundation"], "tests": [], "docs": ["ARCHITECT_PROFILE.md"]},
}


def _runtime_verified(kind: Optional[str], ctx: ScanContext, tests: int) -> bool:
    if kind == "vulkan":
        return "vkCmdDispatch" in ctx.code_symbols("src/python/vulkan_compute_driver.py")
    if kind == "janet":
        return ctx.tool("janet") is not None
    if kind == "godot":
        return ctx.tool("godot", "godot4", "Godot") is not None
    if kind == "cargo":
        return ctx.tool("cargo") is not None
    if kind == "npu":
        return ctx.has_module("onnxruntime") or ctx.has_module("openvino")
    return tests > 0


def measure_subsystems(ctx: ScanContext) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for sid, meta in SUBSYSTEMS.items():
        files = code_loc = doc_loc = 0
        for rel in meta["paths"]:
            full = ctx.path(rel)
            walker = [(os.path.dirname(full), [], [os.path.basename(full)])] if os.path.isfile(full) else os.walk(full)
            for dp, dn, fn in walker:
                dn[:] = [d for d in dn if d not in EXCLUDE_DIRS]
                for f in fn:
                    ext = os.path.splitext(f)[1].lower()
                    if ext not in CODE_EXT | DOC_EXT | {".json", ".toml", ".yml"}:
                        continue
                    try:
                        with open(os.path.join(dp, f), "rb") as fh:
                            n = fh.read().count(b"\n") + 1
                    except OSError:
                        continue
                    files += 1
                    if ext in DOC_EXT:
                        doc_loc += n
                    elif ext in CODE_EXT:
                        code_loc += n
        tests = sum(_scripts_not_tests(ctx, t) for t in meta["tests"])
        docs_present = sum(1 for d in meta["docs"] if ctx.exists(d))
        density = tests / max(1.0, code_loc / 100.0)
        verified = _runtime_verified(meta.get("verified"), ctx, tests)
        readiness = 0.4 * min(1.0, density) + 0.3 * (1.0 if docs_present else 0.0) + 0.3 * (1.0 if verified else 0.0)
        out[sid] = {
            "id": sid, "title": meta["title"], "files": files, "code_loc": code_loc, "doc_loc": doc_loc,
            "discoverable_tests": tests, "docs_present": docs_present, "runtime_verified": verified,
            "readiness": round(readiness, 2),
        }
    return out


# ─── detectors ───────────────────────────────────────────────────────────────

Detector = Dict[str, Any]


def det(desc: str, sev: float, fn: Callable[[ScanContext], Tuple[bool, str]]) -> Detector:
    return {"desc": desc, "severity": sev, "fn": fn}


def lacks(rel: str, token: str) -> Callable[[ScanContext], Tuple[bool, str]]:
    def _f(ctx: ScanContext) -> Tuple[bool, str]:
        missing = token not in (ctx.code_symbols(rel) if rel.endswith(".py") else ctx.text(rel))
        return missing, f"'{token}' {'absent from' if missing else 'present in'} {rel}"
    return _f


def no_tool(*names: str) -> Callable[[ScanContext], Tuple[bool, str]]:
    def _f(ctx: ScanContext) -> Tuple[bool, str]:
        p = ctx.tool(*names)
        return p is None, f"{'/'.join(names)} {'not found on PATH' if p is None else 'found: ' + p}"
    return _f


def no_module(*names: str) -> Callable[[ScanContext], Tuple[bool, str]]:
    def _f(ctx: ScanContext) -> Tuple[bool, str]:
        have = [n for n in names if ctx.has_module(n)]
        return not have, f"python module(s) {', '.join(names)} {'not importable' if not have else 'importable: ' + ', '.join(have)}"
    return _f


def _pattern_in_tree(pattern: str, ext: str, roots: List[str]) -> Callable[[ScanContext], Tuple[bool, str]]:
    rx = re.compile(pattern)

    def _f(ctx: ScanContext) -> Tuple[bool, str]:
        hits = [rel for rel, txt in ctx.code_files() if rel.split("/")[0] in roots and rel.endswith(ext) and rx.search(txt)]
        return bool(hits), (f"pattern /{pattern}/ in {', '.join(hits[:3])}" if hits else f"pattern /{pattern}/ not found")
    return _f


def _tracked_prefix(prefix: str) -> Callable[[ScanContext], Tuple[bool, str]]:
    def _f(ctx: ScanContext) -> Tuple[bool, str]:
        n = sum(1 for p in ctx.tracked() if p.startswith(prefix))
        return n > 0, f"{n} tracked files under {prefix}"
    return _f


def _big_tracked(limit_mb: int) -> Callable[[ScanContext], Tuple[bool, str]]:
    def _f(ctx: ScanContext) -> Tuple[bool, str]:
        big = []
        for p in ctx.tracked():
            if p.startswith(("venv/", "target/", "dist/")):
                continue
            try:
                s = os.path.getsize(ctx.path(p))
            except OSError:
                continue
            if s > limit_mb * 1_000_000:
                big.append(f"{p} ({s / 1e6:.0f} MB)")
        return bool(big), (f">{limit_mb} MB tracked outside build dirs: {', '.join(big[:3])}" if big else f"no tracked file >{limit_mb} MB outside build dirs")
    return _f


def _gitignore_lacks(*entries: str) -> Callable[[ScanContext], Tuple[bool, str]]:
    def _f(ctx: ScanContext) -> Tuple[bool, str]:
        gi = ctx.text(".gitignore")
        missing = [e for e in entries if not re.search(rf"^/?{re.escape(e)}/?\s*$", gi, re.M)]
        return bool(missing), (f".gitignore missing: {', '.join(missing)}" if missing else ".gitignore covers build/venv dirs")
    return _f


def _ci_missing(token: str) -> Callable[[ScanContext], Tuple[bool, str]]:
    def _f(ctx: ScanContext) -> Tuple[bool, str]:
        text = "".join(ctx.text(p) for p in ctx.glob(".github/workflows", ".yml"))
        return token not in text, f"'{token}' {'absent from' if token not in text else 'present in'} .github/workflows"
    return _f


def _ci_dead_paths(ctx: ScanContext) -> Tuple[bool, str]:
    text = "".join(ctx.text(p) for p in ctx.glob(".github/workflows", ".yml"))
    refs = sorted(set(re.findall(r"(?:src/rust-bot|tests/integration(?:/gpu)?)", text)))
    dead = [r for r in refs if not ctx.exists(r)]
    return bool(dead), (f"CI references non-existent paths: {', '.join(dead)}" if dead else "all CI-referenced paths exist")


def _tests_zero(rel: str) -> Callable[[ScanContext], Tuple[bool, str]]:
    def _f(ctx: ScanContext) -> Tuple[bool, str]:
        n = _scripts_not_tests(ctx, rel)
        return n == 0, f"{n} unittest-discoverable tests in {rel}"
    return _f


def _janet_bad_symbols(ctx: ScanContext) -> Tuple[bool, str]:
    bad = re.compile(r"\((?:math/max|math/min|float)(?=[\s\)])")
    hits = []
    for rel in ctx.glob("krystal_janet", ".janet"):
        src = re.sub(r"#.*", "", ctx.text(rel))
        if bad.search(src):
            hits.append(rel)
    return bool(hits), (f"symbols not in Janet Core API (math/max, math/min, float) in {', '.join(hits)}" if hits else "no known-invalid Janet symbols")


def _janet_chr_multibyte(ctx: ScanContext) -> Tuple[bool, str]:
    rx = re.compile(r'\(chr "[^\x00-\x7f]')
    hits = [rel for rel in ctx.glob("krystal_janet", ".janet") if rx.search(ctx.text(rel))]
    return bool(hits), (f"(chr \"<multi-byte>\") is invalid in {', '.join(hits)}" if hits else "no multi-byte chr literals")


def _wildcard_cors(ctx: ScanContext) -> Tuple[bool, str]:
    n = len(re.findall(r'Access-Control-Allow-Origin["\']\s*,\s*["\']\*', ctx.text("krystal_web_hub/server.py")))
    return n > 0, f"{n} responses send Access-Control-Allow-Origin: * (daemon has POST endpoints that write files)"


def _post_no_content_type(ctx: ScanContext) -> Tuple[bool, str]:
    src = ctx.text("krystal_web_hub/server.py")
    m = re.search(r"def do_POST\(self\):(.*?)(?:\n    def |\Z)", src, re.S)
    body = m.group(1) if m else ""
    missing = "application/json" not in body
    return missing, "do_POST does not enforce Content-Type: application/json (text/plain cross-site POSTs skip CORS preflight)"


def _docs_claim(token: str, absent_pattern: str) -> Callable[[ScanContext], Tuple[bool, str]]:
    rx = re.compile(absent_pattern, re.I)

    def _f(ctx: ScanContext) -> Tuple[bool, str]:
        claiming = [p for p in ctx.glob("docs/research", ".md") if token.lower() in ctx.text(p).lower()]
        implemented = any(rx.search(txt) for rel, txt in ctx.code_files() if not rel.endswith(".md"))
        gap = bool(claiming) and not implemented
        return gap, (f"'{token}' claimed in {', '.join(claiming[:2])} but no source matches /{absent_pattern}/" if gap else f"'{token}' claim backed by source or not made")
    return _f


def _missing_dir(rel: str) -> Callable[[ScanContext], Tuple[bool, str]]:
    def _f(ctx: ScanContext) -> Tuple[bool, str]:
        m = not ctx.exists(rel)
        return m, f"{rel} {'does not exist' if m else 'exists'} (referenced as a roadmap deliverable)"
    return _f


# ─── priority candidates (editorial weights; gaps are measured) ──────────────

CANDIDATES: List[Dict[str, Any]] = [
    {
        "id": "vulkan-real-dispatch", "category": "compute", "subsystems": ["vulkan-compute", "neural-ascii-core"],
        "title": "Wire a real Vulkan compute dispatch (logical device -> SPIR-V -> pipeline -> SSBO -> vkCmdDispatch)",
        "impact": 0.95, "heat": 0.95, "leverage": 0.90, "effort": "L",
        "expertise": ["gpu-compute", "systems-ffi", "perf-engineering"],
        "detectors": [
            det("No logical device creation", 0.4, lacks("src/python/vulkan_compute_driver.py", "vkCreateDevice")),
            det("No compute pipeline / shader module", 0.3, lacks("src/python/vulkan_compute_driver.py", "vkCreateComputePipelines")),
            det("No dispatch call - kernel runs as a pure-Python CPU loop", 0.3, lacks("src/python/vulkan_compute_driver.py", "vkCmdDispatch")),
        ],
        "next_steps": [
            "vkCreateDevice with the compute queue family already discovered (index 0 on Iris Xe).",
            "Create 2 host-visible SSBOs (uint8 glyphs, uint32 RGB) + 1 UBO (camera, time, folds).",
            "Obtain SPIR-V: hand-assembled blob emitted from Python (zero toolchain) or glslc from the Vulkan SDK.",
            "vkCmdDispatch -> fence -> map memory; keep execute_raymarch() as the correctness oracle.",
            "Only then set driver.real_gpu_dispatch = True so UI/API flip to ACTIVE (GPU).",
        ],
        "decision_needed": "SPIR-V source: zero-toolchain hand-assembled module vs. installing the Vulkan SDK (glslc).",
    },
    {
        "id": "python-hotpath-vectorize", "category": "compute", "subsystems": ["neural-ascii-core", "openworld"],
        "title": "Cut the pure-Python raymarch hot path (measured ~46 ms/frame at 96x40 => ~22 FPS ceiling)",
        "impact": 0.85, "heat": 0.90, "leverage": 0.60, "effort": "S",
        "expertise": ["numerical-python", "perf-engineering"],
        "detectors": [
            det("Per-pixel Python loops in the Vulkan-path kernel", 0.5, _pattern_in_tree(r"for x in range\(width\)", ".py", ["src"])),
            det("No vectorized backend available (numpy not installed on this host)", 0.5, no_module("numpy")),
        ],
        "next_steps": [
            "Prototype an opt-in numpy kernel (rays as (N,3) arrays) behind the existing fallback; measure before claiming any speedup.",
            "Keep stdlib-only path as default so 'zero-dependency' remains true.",
            "Add a benchmark that records ms/frame into a JSON artifact the docs can cite.",
        ],
        "decision_needed": "Allow an optional numpy dependency (in a Windows venv - the committed venv/ is Linux-only).",
    },
    {
        "id": "janet-runtime-verify", "category": "language", "subsystems": ["janet-port"],
        "title": "Run the Janet sub-project on a real Janet interpreter",
        "impact": 0.55, "heat": 0.10, "leverage": 0.40, "effort": "S",
        "expertise": ["lisp-functional", "language-runtime"],
        "detectors": [
            det("janet not installed", 0.4, no_tool("janet")),
            det("Calls to functions absent from the Janet Core API", 0.3, _janet_bad_symbols),
            det("(chr \"<multi-byte>\") in entropy code", 0.2, _janet_chr_multibyte),
            det("No test executes janet", 0.1, lambda c: ("subprocess" not in c.text("tests/test_janet_integration.py"), "test_janet_integration.py never spawns janet")),
        ],
        "next_steps": [
            "Install Janet (scoop/winget/release zip) and run: janet krystal_janet/main.janet --test",
            "Make governor.janet consume a luma array from render-frame instead of scanning ANSI bytes.",
            "Add a test that shells out to janet when present and skips otherwise.",
        ],
    },
    {
        "id": "mimicry-ar-test-coverage", "category": "quality", "subsystems": ["mimicry", "antigravity-ar"],
        "title": "Bring mimicry & AR-mirror engines into the unittest suite (currently 0 discoverable tests)",
        "impact": 0.60, "heat": 0.30, "leverage": 0.50, "effort": "S",
        "expertise": ["qa-automation", "procedural-geometry"],
        "detectors": [
            det("test_mimicry_compositor.py is script-style (0 tests discovered)", 0.5, _tests_zero("tests/test_mimicry_compositor.py")),
            det("verify_advanced_ar_mirror.py is script-style (0 tests discovered)", 0.5, _tests_zero("tests/verify_advanced_ar_mirror.py")),
        ],
        "next_steps": [
            "Wrap the existing assertions in unittest.TestCase methods (no new logic needed).",
            "Run via python -m unittest discover tests so CI and local runs agree.",
        ],
    },
    {
        "id": "repo-hygiene-untrack-artifacts", "category": "devops", "subsystems": [],
        "title": "Stop tracking venv/, target/, dist/ and multi-MB logs in git",
        "impact": 0.50, "heat": 0.0, "leverage": 0.70, "effort": "S",
        "expertise": ["devops-release"],
        "requires_user_approval": True,
        "detectors": [
            det("venv/ (Linux site-packages) is tracked", 0.3, _tracked_prefix("venv/")),
            det("Rust target/ build output is tracked", 0.3, _tracked_prefix("target/")),
            det("Large logs tracked outside build dirs", 0.2, _big_tracked(5)),
            det(".gitignore does not cover venv/target/dist", 0.2, _gitignore_lacks("venv", "target", "dist")),
        ],
        "next_steps": [
            "git rm -r --cached venv target dist backend_run_log_stable.txt   (working tree untouched; history is NOT rewritten)",
            "Add venv/, target/, dist/, *.log, backend_run_*.txt to .gitignore.",
            "Optional later: git filter-repo/BFG to shrink existing history (coordinate with collaborators).",
        ],
    },
    {
        "id": "harden-localhost-api", "category": "security", "subsystems": ["web-hub"],
        "title": "Harden the localhost daemon (wildcard CORS + unauthenticated file-writing POST endpoints)",
        "impact": 0.60, "heat": 0.0, "leverage": 0.30, "effort": "S",
        "expertise": ["web-security"],
        "detectors": [
            det("Wildcard CORS on API responses", 0.6, _wildcard_cors),
            det("POST does not enforce JSON content type", 0.4, _post_no_content_type),
        ],
        "next_steps": [
            "Drop Access-Control-Allow-Origin: * (UI is same-origin) or pin to http://127.0.0.1:<port>.",
            "Reject POST unless Content-Type is application/json; check Origin/Host header.",
            "Constrain /api/mimicry/export-godot to a fixed output directory (already scene-id keyed; validate ids).",
        ],
    },
    {
        "id": "ci-reality-check", "category": "devops", "subsystems": [],
        "title": "Make CI test what exists (workflow targets paths that are not in the repo)",
        "impact": 0.60, "heat": 0.0, "leverage": 0.80, "effort": "S",
        "expertise": ["devops-release", "qa-automation"],
        "detectors": [
            det("CI references non-existent paths", 0.6, _ci_dead_paths),
            det("CI never runs the unittest suite used locally", 0.4, _ci_missing("unittest")),
        ],
        "next_steps": [
            "Replace src/rust-bot and tests/integration references with real crates/tests.",
            "Add: python -m unittest discover tests -v",
        ],
    },
    {
        "id": "rust-native-kernels", "category": "compute", "subsystems": ["rust-native"],
        "title": "Build the native Rust tier the docs depend on (crates unbuilt here; krystal-math-simd does not exist)",
        "impact": 0.70, "heat": 0.85, "leverage": 0.60, "effort": "L",
        "expertise": ["rust-native", "perf-engineering"],
        "detectors": [
            det("cargo not on PATH", 0.4, no_tool("cargo")),
            det("krystal-math-simd crate referenced by docs but missing", 0.4, _missing_dir("krystal-math-simd")),
            det("fast_math.py advertised as SIMD but uses no SIMD/vector library", 0.2, _docs_claim("AVX2", r"avx2|import numpy|_mm256|core::arch|std::simd")),
        ],
        "next_steps": [
            "Install rustup in WSL or Windows, cargo build the existing crates.",
            "Create krystal-math-simd with PyO3 and a benchmark vs fast_math.py before quoting any speedup.",
        ],
    },
    {
        "id": "godot-verification", "category": "integration", "subsystems": ["godot"],
        "title": "Verify the Godot project headless (shader + exported scenes never loaded by an engine)",
        "impact": 0.50, "heat": 0.20, "leverage": 0.40, "effort": "M",
        "expertise": ["game-engines"],
        "detectors": [
            det("Godot binary not available", 0.5, no_tool("godot", "godot4", "Godot")),
            det("CI never runs Godot", 0.5, _ci_missing("godot")),
        ],
        "next_steps": [
            "Install Godot 4.x, run: godot --headless --path godot_project --quit",
            "Export a scene via /api/mimicry/export-godot and load it in the headless run.",
        ],
    },
    {
        "id": "npu-path-verification", "category": "compute", "subsystems": ["npu-openvino"],
        "title": "Prove the NPU/ONNX path end-to-end (runtime not installed; phantom_net.onnx never loaded)",
        "impact": 0.60, "heat": 0.70, "leverage": 0.50, "effort": "M",
        "expertise": ["npu-inference"],
        "detectors": [
            det("No onnxruntime/openvino runtime importable", 0.6, no_module("onnxruntime", "openvino")),
            det("No test loads an ONNX model", 0.4, lambda c: ("onnx" not in "".join(c.text(p) for p in c.glob("tests", ".py")).lower(), "no test under tests/ references onnx")),
        ],
        "next_steps": [
            "Create a Windows venv with onnxruntime-directml (or openvino) and load phantom_net.onnx.",
            "Record latency per inference into a JSON artifact; gate NPU claims on it.",
        ],
    },
    {
        "id": "claims-vs-measurements-audit", "category": "quality", "subsystems": [],
        "title": "Reconcile research-doc performance claims with measurements",
        "impact": 0.50, "heat": 0.0, "leverage": 0.50, "effort": "S",
        "expertise": ["perf-engineering", "qa-automation"],
        "detectors": [
            det("'120+ FPS' claimed without a benchmark artifact", 0.5, lambda c: (any("120+ fps" in c.text(p).lower() for p in c.glob("docs/research", ".md")) and not c.exists("tests/benchmark_results.json"), "tests/benchmark_results.json missing while docs quote 120+ FPS")),
            det("'PCIe DMA' readback claimed while no GPU dispatch exists", 0.5, lambda c: (any("pcie dma" in c.text(p).lower() for p in c.glob("docs/research", ".md")) and "vkCmdDispatch" not in c.code_symbols("src/python/vulkan_compute_driver.py"), "docs mention PCIe DMA but driver has no vkCmdDispatch")),
        ],
        "next_steps": [
            "Make tests/benchmark_core_performance.py write tests/benchmark_results.json; cite it from docs.",
            "Mark unmeasured figures as targets, not results.",
        ],
    },
]


# ─── roadmap parsing ─────────────────────────────────────────────────────────

ROADMAP_FILE = "docs/VISION_AND_FUTURE_ROADMAP.md"
PHASE_IMPACT = {1: 0.70, 2: 0.50, 3: 0.40, 4: 0.30}


def parse_roadmap(ctx: ScanContext) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    phase = 0
    counters: Dict[int, int] = {}
    for line in ctx.text(ROADMAP_FILE).splitlines():
        h = re.match(r"^###\s+F[ÁA]ZA\s+(\d+)", line)
        if h:
            phase = int(h.group(1))
            continue
        c = re.match(r"^\s*-\s*\[( |x)\]\s*(.+)$", line)
        if c and phase:
            counters[phase] = counters.get(phase, 0) + 1
            items.append({"phase": phase, "done": c.group(1) == "x", "text": c.group(2).strip(), "idx": counters[phase]})
    return items


def roadmap_candidates(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for it in items:
        if it["done"]:
            continue
        txt = it["text"].lower()
        exp = ["devops-release"] if any(k in txt for k in ("inštal", "install", "msi", "winget")) else ["product-architecture"]
        out.append({
            "id": f"roadmap-phase{it['phase']}-{it['idx']}", "category": "roadmap", "subsystems": [],
            "title": f"[Roadmap F{it['phase']}] {it['text']}",
            "impact": PHASE_IMPACT.get(it["phase"], 0.3), "heat": 0.2, "leverage": 0.3, "effort": "L",
            "expertise": exp, "source": "roadmap",
            "detectors": [det("Unchecked in roadmap", 1.0, lambda c, t=it["text"]: (True, f"unchecked: {t[:80]}"))],
            "next_steps": ["Break down into measurable milestones; tie each to a detector."],
        })
    return out


# ─── compute patterns ────────────────────────────────────────────────────────

PATTERNS: List[Dict[str, Any]] = [
    {"id": "sphere-tracing-sdf", "name": "SDF sphere tracing", "complexity": "O(W*H*K*C_sdf)", "signature": r"def scene_sdf|def execute_raymarch|\(defn raymarch", "heat": "very high",
     "risk": "Dominant cost; pure Python is ~46 ms/frame at 96x40 (measured).", "target_tier": "GPU compute / vectorized"},
    {"id": "symplectic-verlet", "name": "Symplectic velocity-Verlet integrator", "complexity": "O(N) per step, N=4", "signature": r"def symplectic_step|\(defn symplectic-step", "heat": "low",
     "risk": "Cheap; correctness (energy drift) is the concern. Measured max drift 4.3e-4 over 10k steps.", "target_tier": "CPU"},
    {"id": "lockfree-ring-buffer", "name": "Fixed-capacity ring buffer channels", "complexity": "O(1) push/pop", "signature": r"class FastRingBuffer", "heat": "medium",
     "risk": "Python GIL limits true lock-freedom; measured 4.39x vs queue.Queue.", "target_tier": "CPU / Rust"},
    {"id": "queue-partitioned-vm", "name": "Queue-partitioned bytecode VM", "complexity": "O(pipelines*batch)", "signature": r"class TopologicalVM|\(defn step-vm", "heat": "medium",
     "risk": "Throughput metric in Janet port is synthetic (packets*120).", "target_tier": "CPU"},
    {"id": "entropy-backpressure", "name": "Visual-entropy backpressure governor", "complexity": "O(W*H) per frame", "signature": r"backpressure|compute-entropy", "heat": "medium",
     "risk": "Closed loop couples render cost to a scene statistic; needs a stability test under real load.", "target_tier": "CPU"},
    {"id": "fbm-octave-noise", "name": "Multi-octave fBm terrain", "complexity": "O(W*H*octaves)", "signature": r"def fbm|\(defn fbm-terrain", "heat": "high",
     "risk": "Hash-heavy; branch divergence on GPU is mild, trivial to vectorize.", "target_tier": "vectorized / GPU"},
    {"id": "cooperative-chunk-streaming", "name": "Cooperative fiber chunk streaming", "complexity": "O(chunks)", "signature": r"stream-chunks-cooperative", "heat": "low",
     "risk": "Janet-only; unverified until a Janet runtime is installed.", "target_tier": "CPU"},
    {"id": "dual-ssbo-readback", "name": "Dual-SSBO glyph+RGB readback", "complexity": "O(W*H) bytes (5 B/px)", "signature": r"ssbo|SSBO", "heat": "medium",
     "risk": "Only a buffer contract today; becomes real with vkCmdDispatch.", "target_tier": "GPU"},
    {"id": "simd-vector-math", "name": "SIMD / vector math kernels", "complexity": "O(N/lanes)", "signature": r"avx2|import numpy|_mm256|core::arch|std::simd", "heat": "high",
     "risk": "No source implements SIMD yet; docs that say so are aspirational.", "target_tier": "Rust / numpy"},
    {"id": "npu-int8-inference", "name": "INT8 NPU inference offload", "complexity": "O(L*M^2)", "signature": r"onnxruntime|openvino|DmlExecutionProvider|DirectML", "heat": "high",
     "risk": "Runtime not installed on this host; unmeasured.", "target_tier": "NPU"},
]


def detect_patterns(ctx: ScanContext) -> List[Dict[str, Any]]:
    out = []
    driver = ctx.code_symbols("src/python/vulkan_compute_driver.py")
    for p in PATTERNS:
        rx = re.compile(p["signature"], re.I if p["id"] in ("dual-ssbo-readback", "simd-vector-math") else 0)
        evidence = [rel for rel, txt in ctx.code_files() if rx.search(txt)]
        status = "implemented" if evidence else "absent"
        if p["id"] == "dual-ssbo-readback" and evidence and "vkCmdDispatch" not in driver:
            status = "emulated"
        if p["id"] == "sphere-tracing-sdf" and evidence and "vkCmdDispatch" not in driver:
            status = "implemented_cpu_only"
        if p["id"] in ("npu-int8-inference",) and not (ctx.has_module("onnxruntime") or ctx.has_module("openvino")):
            status = "unverified" if evidence else "absent"
        if p["id"] == "cooperative-chunk-streaming" and evidence and not ctx.tool("janet"):
            status = "unverified"
        out.append({
            "id": p["id"], "name": p["name"], "complexity": p["complexity"], "compute_heat": p["heat"],
            "status": status, "target_tier": p["target_tier"], "risk": p["risk"],
            "evidence": evidence[:5], "evidence_count": len(evidence),
        })
    return out


# ─── expertise ───────────────────────────────────────────────────────────────

TRACKS: Dict[str, Dict[str, Any]] = {
    "gpu-compute": {"title": "GPU compute & Vulkan", "skills": ["Vulkan compute pipeline", "SPIR-V", "SSBO/UBO design", "synchronisation"], "subsystems": ["vulkan-compute", "neural-ascii-core"], "learn": ["docs/research/DEEP_PERFORMANCE_OPTIMIZATION_AND_CORE_ARCHITECTURE.md", "docs/research/GAME_ENGINE_TECHNIQUES_AND_PROCEDURAL_RENDERING.md"]},
    "systems-ffi": {"title": "Native FFI & systems (ctypes, DLL ABI)", "skills": ["ctypes struct layout", "ABI correctness", "handle lifetimes"], "subsystems": ["vulkan-compute"], "learn": ["src/python/vulkan_compute_driver.py"]},
    "perf-engineering": {"title": "Performance engineering & benchmarking", "skills": ["profiling", "benchmark design", "complexity modelling"], "subsystems": ["vulkan-compute", "krystal-lang", "openworld"], "learn": ["docs/research/DEEP_PERFORMANCE_OPTIMIZATION_AND_CORE_ARCHITECTURE.md", "tests/benchmark_core_performance.py"]},
    "numerical-python": {"title": "Numerical Python & vectorization", "skills": ["array programming", "memory layout", "numeric stability"], "subsystems": ["openworld", "neural-ascii-core"], "learn": ["openworld_engine/fast_math.py"]},
    "rust-native": {"title": "Rust native kernels & PyO3", "skills": ["Rust", "PyO3 bindings", "SIMD intrinsics"], "subsystems": ["rust-native"], "learn": ["krystal-bitboard", "gamesa_cortex_v2"]},
    "dynamical-systems": {"title": "Hamiltonian / cyclic control theory", "skills": ["symplectic integration", "Lyapunov analysis", "control loops"], "subsystems": ["cyclic-organism"], "learn": ["docs/research/CYCLIC_THEORY_AND_ALTERNATIVE_ARCHITECTURAL_SCHEMAS.md"]},
    "language-runtime": {"title": "Compilers, bytecode & VMs", "skills": ["lexing/parsing", "bytecode design", "VM scheduling"], "subsystems": ["krystal-lang", "janet-port"], "learn": ["docs/research/KRYSTAL_LANG_GEOMETRIC_BYTECODE_AND_LLM_COMPILATION.md"]},
    "lisp-functional": {"title": "Janet / Lisp functional engines", "skills": ["Janet PEG", "fibers", "immutable data", "macro design"], "subsystems": ["janet-port"], "learn": ["docs/research/JANET_TRANSFORMATION_AND_LISP_ARCHITECTURE.md", "krystal_janet/README.md"]},
    "procedural-geometry": {"title": "Procedural generation, SDF & geometry", "skills": ["SDF CSG", "noise/fBm", "symmetry groups", "modifier stacks"], "subsystems": ["openworld", "mimicry", "antigravity-ar"], "learn": ["docs/research/ADVANCED_OPENWORLD_PROCEDURAL_RENDERING.md", "docs/research/BLENDER_CORE_MODIFIER_MIMICRY_ENGINE.md"]},
    "game-engines": {"title": "Godot & game-engine integration", "skills": ["Godot 4 shaders", "scene export", "headless CI"], "subsystems": ["godot"], "learn": ["godot_project/README.md"]},
    "npu-inference": {"title": "NPU / OpenVINO / ONNX inference", "skills": ["ONNX export", "INT8 quantization", "DirectML/OpenVINO EPs"], "subsystems": ["npu-openvino"], "learn": ["legacy_docs/WINDOWS_OPEN_VINO_COGNITION.md"]},
    "web-security": {"title": "Web/API security for local daemons", "skills": ["CORS/CSRF", "origin checks", "input validation"], "subsystems": ["web-hub"], "learn": ["krystal_web_hub/server.py"]},
    "devops-release": {"title": "DevOps, CI & release engineering", "skills": ["GitHub Actions", "WSL2", "packaging (msi/winget)", "repo hygiene"], "subsystems": ["wsl-bridge"], "learn": ["DEVOPS_INTEGRATION_ROADMAP.md", ".github/workflows/ci.yml"]},
    "qa-automation": {"title": "Test automation & verification", "skills": ["unittest/pytest", "property tests", "golden outputs"], "subsystems": ["cyclic-organism", "krystal-lang", "openworld"], "learn": ["tests"]},
    "industrial-cognitive-manufacturing": {"title": "Industrial cognitive manufacturing (CNC)", "skills": ["Shadow Council governance", "Neuro-safety gradients", "Quadratic Mantinel", "Digital-twin entanglement"], "subsystems": ["cnc-copilot"], "learn": ["ARCHITECT_PROFILE.md"], "source": "ARCHITECT_PROFILE.md"},
    "product-architecture": {"title": "Product & platform architecture", "skills": ["roadmapping", "API design", "packaging"], "subsystems": ["web-hub"], "learn": ["docs/VISION_AND_FUTURE_ROADMAP.md"]},
}


def build_expertise(subs: Dict[str, Dict[str, Any]], open_priorities: List[Dict[str, Any]], ctx: ScanContext) -> List[Dict[str, Any]]:
    out = []
    for tid, t in TRACKS.items():
        mapped = [subs[s] for s in t["subsystems"] if s in subs]
        code_loc = sum(s["code_loc"] for s in mapped)
        tests = sum(s["discoverable_tests"] for s in mapped)
        readiness = round(sum(s["readiness"] for s in mapped) / len(mapped), 2) if mapped else 0.0
        if code_loc and readiness >= 0.65 and tests:
            signal = "in_house_strong"
        elif code_loc and readiness >= 0.35:
            signal = "in_house_partial"
        else:
            signal = "gap"
        needed_by = [(p["id"], p["score"]) for p in open_priorities if tid in p["required_expertise"]]
        engagement = {
            "in_house_strong": ["extend in-house", "design review before large changes"],
            "in_house_partial": ["pair-build session", "targeted spike with measurable exit criteria"],
            "gap": ["dedicated spike or external specialist", "self-study path first (see learning_path)"],
        }[signal]
        out.append({
            "id": tid, "title": t["title"], "skills": t["skills"], "staffing_signal": signal,
            "readiness": readiness, "evidence": {"subsystems": t["subsystems"], "code_loc": code_loc, "discoverable_tests": tests},
            "demand_score": round(sum(s for _, s in needed_by), 1), "needed_by": [i for i, _ in needed_by],
            "learning_path": [p for p in t["learn"] if ctx.exists(p)], "engagement_options": engagement,
            "source": t.get("source", "repository measurement"),
        })
    out.sort(key=lambda x: -x["demand_score"])
    return out


# ─── evaluation & snapshot ───────────────────────────────────────────────────

def evaluate(c: Dict[str, Any], ctx: ScanContext) -> Dict[str, Any]:
    findings, gap = [], 0.0
    for d in c["detectors"]:
        try:
            failing, evidence = d["fn"](ctx)
        except Exception as e:  # a broken detector must never take the API down
            failing, evidence = False, f"detector error: {e}"
        findings.append({"description": d["desc"], "severity": d["severity"], "failing": bool(failing), "evidence": evidence})
        if failing:
            gap += d["severity"]
    gap = min(1.0, gap)
    score = 100.0 * (0.35 * c["impact"] + 0.35 * gap + 0.15 * c["heat"] + 0.15 * c["leverage"])
    pts = EFFORT_POINTS[c["effort"]]
    return {
        "id": c["id"], "title": c["title"], "category": c["category"], "source": c.get("source", "measured"),
        "status": "open" if gap > 0 else "healthy", "score": round(score, 1) if gap > 0 else 0.0,
        "gap": round(gap, 2), "impact": c["impact"], "compute_heat": c["heat"], "leverage": c["leverage"],
        "effort": {"size": c["effort"], "points": pts},
        "value_per_effort": round(score / pts, 1) if gap > 0 else 0.0,
        "subsystems": c["subsystems"], "required_expertise": c["expertise"],
        "findings": findings, "next_steps": c.get("next_steps", []),
        "decision_needed": c.get("decision_needed"), "requires_user_approval": c.get("requires_user_approval", False),
    }


class ProjectIntelligence:
    def __init__(self, root: str = REPO_ROOT, ttl: float = 30.0, isolated: bool = False):
        # isolated=True runs the file scan in a child process. Inside the Mission Control
        # server the engine thread is CPU-bound pure Python, so an in-process scan (thousands of
        # file reads, each re-acquiring the GIL) is >10x slower than a standalone one.
        self.root, self.ttl, self.isolated = root, ttl, isolated
        self._lock = threading.Lock()
        self._snap: Optional[Dict[str, Any]] = None
        self._at = 0.0

    def snapshot(self, force: bool = False) -> Dict[str, Any]:
        with self._lock:
            if force or self._snap is None or time.time() - self._at > self.ttl:
                self._snap = self._scan_isolated() if self.isolated else self._scan()
                self._at = time.time()
            return self._snap

    def _scan_isolated(self) -> Dict[str, Any]:
        import json
        try:
            out = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "snapshot", self.root],
                capture_output=True, timeout=60, cwd=self.root,
            )
            if out.returncode == 0:
                return json.loads(out.stdout.decode("utf-8", "ignore"))
        except Exception:
            pass
        return self._scan()  # fall back to in-process scan

    def _scan(self) -> Dict[str, Any]:
        t0 = time.perf_counter()
        ctx = ScanContext(self.root)
        subs = measure_subsystems(ctx)
        roadmap = parse_roadmap(ctx)
        evaluated = [evaluate(c, ctx) for c in CANDIDATES + roadmap_candidates(roadmap)]
        open_p = sorted([e for e in evaluated if e["status"] == "open"], key=lambda e: -e["score"])
        for i, e in enumerate(open_p, 1):
            e["rank"] = i
        healthy = [e for e in evaluated if e["status"] == "healthy"]
        tracked = ctx.tracked()
        toolchain = {n: bool(ctx.tool(*names)) for n, names in {
            "janet": ("janet",), "cargo": ("cargo",), "rustc": ("rustc",), "godot": ("godot", "godot4", "Godot"), "wsl": ("wsl",), "git": ("git",)}.items()}
        toolchain.update({m: ctx.has_module(m) for m in ("numpy", "onnxruntime", "openvino")})
        return {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "scan_ms": round((time.perf_counter() - t0) * 1000, 1),
            "scoring": {"formula": "100*(0.35*impact+0.35*gap+0.15*heat+0.15*leverage)", "gap": "min(1, sum(severity of failing detectors))",
                        "effort_points": EFFORT_POINTS, "value_per_effort": "score / effort_points"},
            "repo": {"tracked_files": len(tracked), "subsystems": list(subs.values()),
                     "roadmap": {"done": sum(1 for r in roadmap if r["done"]), "open": sum(1 for r in roadmap if not r["done"])}},
            "toolchain": toolchain, "priorities": open_p, "healthy": healthy,
            "patterns": detect_patterns(ctx), "expertise": build_expertise(subs, open_p, ctx),
        }

    # ─ query helpers used by the HTTP layer ─

    def overview(self, force: bool = False) -> Dict[str, Any]:
        s = self.snapshot(force)
        return {k: s[k] for k in ("generated_at", "scan_ms", "repo", "toolchain")} | {
            "open_priorities": len(s["priorities"]), "healthy_items": len(s["healthy"]),
            "top3": [{"rank": p["rank"], "id": p["id"], "score": p["score"]} for p in s["priorities"][:3]]}

    def priorities(self, limit: int = 0, sort: str = "score", category: str = "", include_healthy: bool = False, force: bool = False) -> Dict[str, Any]:
        s = self.snapshot(force)
        items = list(s["priorities"])
        if category:
            items = [p for p in items if p["category"] == category]
        if sort == "value":
            items.sort(key=lambda p: -p["value_per_effort"])
        if limit > 0:
            items = items[:limit]
        res: Dict[str, Any] = {"generated_at": s["generated_at"], "sort": sort, "count": len(items), "scoring": s["scoring"], "priorities": items}
        if include_healthy:
            res["healthy"] = s["healthy"]
        return res

    def priority(self, pid: str) -> Optional[Dict[str, Any]]:
        s = self.snapshot()
        return next((p for p in s["priorities"] + s["healthy"] if p["id"] == pid), None)

    def expertise(self, priority_id: str = "", force: bool = False) -> Optional[Dict[str, Any]]:
        s = self.snapshot(force)
        if not priority_id:
            return {"generated_at": s["generated_at"], "count": len(s["expertise"]), "tracks": s["expertise"]}
        p = self.priority(priority_id)
        if p is None:
            return None
        by_id = {t["id"]: t for t in s["expertise"]}
        required = [dict(by_id[t], match_rank=i + 1) for i, t in enumerate(p["required_expertise"]) if t in by_id]
        return {
            "priority": {"id": p["id"], "title": p["title"], "score": p["score"], "effort": p["effort"]},
            "required_tracks": required, "coverage_gaps": [t["id"] for t in required if t["staffing_signal"] == "gap"],
            "decision_needed": p["decision_needed"],
        }

    def patterns(self, force: bool = False) -> Dict[str, Any]:
        s = self.snapshot(force)
        return {"generated_at": s["generated_at"], "count": len(s["patterns"]), "patterns": s["patterns"]}


_default: Optional[ProjectIntelligence] = None


def get_intelligence() -> ProjectIntelligence:
    global _default
    if _default is None:
        _default = ProjectIntelligence(isolated=True)
    return _default


if __name__ == "__main__":
    import json
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "snapshot":  # internal: used by isolated mode
        root = sys.argv[2] if len(sys.argv) > 2 else REPO_ROOT
        sys.stdout.write(json.dumps(ProjectIntelligence(root)._scan(), ensure_ascii=True))
        sys.exit(0)
    pi = ProjectIntelligence()
    what = sys.argv[1] if len(sys.argv) > 1 else "priorities"
    data = {"priorities": pi.priorities, "expertise": pi.expertise, "patterns": pi.patterns, "overview": pi.overview}[what]()
    print(json.dumps(data, indent=2, ensure_ascii=True))
