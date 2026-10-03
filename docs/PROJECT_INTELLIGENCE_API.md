# Project Intelligence API

Read-only JSON endpoints on the Localhost Mission Control daemon (`python -m krystal_web_hub.server 8080`)
that expose the analysis in [PROJECT_AUDIT_AND_COMPUTE_PATTERNS.md](PROJECT_AUDIT_AND_COMPUTE_PATTERNS.md) live.

- Implementation: [`src/python/project_intelligence.py`](../src/python/project_intelligence.py) (stdlib only) and
  `handle_intelligence()` in [`krystal_web_hub/server.py`](../krystal_web_hub/server.py).
- All endpoints are `GET`, return `application/json`, and never modify the repository or run toolchains.
- The scan (~1 s) runs in a **child process** and is cached for **30 s**; add `?refresh=1` to force a rescan.
  (An in-process scan was >10× slower because the engine thread is CPU-bound and starves other threads of the GIL.)
- Every response has `status`: `OK`, `NOT_FOUND` (404), `ERROR` (400/500) or `UNAVAILABLE` (503).

## Endpoints

| Method & path | Query | Purpose |
|---|---|---|
| `GET /api/project/overview` | `refresh` | Repo size, per-subsystem LOC/tests/readiness, toolchain presence, top-3 priorities |
| `GET /api/priorities` | `limit`, `sort=score\|value`, `category`, `include_healthy`, `refresh` | Ranked open priorities with findings and next steps |
| `GET /api/priorities/<id>` | – | One priority (open or healthy) |
| `GET /api/expertise` | `priority=<id>`, `refresh` | Expertise tracks; with `priority`, the tracks that priority needs plus coverage gaps |
| `GET /api/compute-patterns` | `refresh` | Compute-pattern catalogue with measured status |

Categories: `compute`, `quality`, `devops`, `security`, `integration`, `language`, `roadmap`.

## Priority object

```jsonc
{
  "rank": 1,
  "id": "vulkan-real-dispatch",
  "title": "Wire a real Vulkan compute dispatch (...)",
  "category": "compute",
  "source": "measured",            // measured | roadmap
  "status": "open",                // open | healthy
  "score": 96.0,                   // 0-100
  "gap": 1.0, "impact": 0.95, "compute_heat": 0.95, "leverage": 0.9,   // components of score
  "effort": {"size": "L", "points": 8},
  "value_per_effort": 12.0,
  "subsystems": ["vulkan-compute", "neural-ascii-core"],
  "required_expertise": ["gpu-compute", "systems-ffi", "perf-engineering"],
  "findings": [{"description": "...", "severity": 0.4, "failing": true, "evidence": "..."}],
  "next_steps": ["..."],
  "decision_needed": "SPIR-V source: hand-assembled module vs. installing the Vulkan SDK (glslc).",
  "requires_user_approval": false  // true => the suggested action is destructive; never auto-run
}
```

`findings[].evidence` states exactly what was inspected, so any score can be audited.

## Expertise track object

```jsonc
{
  "id": "gpu-compute",
  "title": "GPU compute & Vulkan",
  "skills": ["Vulkan compute pipeline", "SPIR-V", "SSBO/UBO design", "synchronisation"],
  "staffing_signal": "in_house_partial",   // in_house_strong | in_house_partial | gap
  "readiness": 0.49,
  "demand_score": 96.0,
  "needed_by": ["vulkan-real-dispatch"],
  "learning_path": ["docs/research/..."],
  "engagement_options": ["pair-build session", "targeted spike with measurable exit criteria"],
  "evidence": {"subsystems": ["vulkan-compute", "neural-ascii-core"], "code_loc": 1680, "discoverable_tests": 4}
}
```

`GET /api/expertise?priority=<id>` returns:

```jsonc
{
  "status": "OK",
  "priority": {"id": "...", "title": "...", "score": 0, "effort": {}},
  "required_tracks": [{ /* track object */ "match_rank": 1 }],
  "coverage_gaps": ["rust-native"],     // required tracks whose staffing_signal == "gap"
  "decision_needed": "..."
}
```

## Examples

```powershell
curl.exe "http://127.0.0.1:8080/api/priorities?limit=5"
curl.exe "http://127.0.0.1:8080/api/priorities?sort=value&limit=5"        # best return per effort
curl.exe "http://127.0.0.1:8080/api/priorities?category=security"
curl.exe "http://127.0.0.1:8080/api/priorities/vulkan-real-dispatch"
curl.exe "http://127.0.0.1:8080/api/expertise?priority=rust-native-kernels"
curl.exe "http://127.0.0.1:8080/api/compute-patterns"
curl.exe "http://127.0.0.1:8080/api/project/overview?refresh=1"
```

Without the server: `python src/python/project_intelligence.py [priorities|expertise|patterns|overview]`.

## Errors

| Case | HTTP | Body |
|---|---:|---|
| Unknown priority id (path or `?priority=`) | 404 | `{"status":"NOT_FOUND","message":"Unknown priority id."}` |
| `sort` not `score`/`value` | 400 | `{"status":"ERROR","message":"sort must be 'score' or 'value'."}` |
| `limit` not an integer | 400 | `{"status":"ERROR","message":"limit must be an integer."}` |
| Analyzer not importable | 503 | `{"status":"UNAVAILABLE", ...}` |

## Honesty guarantees

- `gap` comes from detectors that inspect real files and the real PATH (e.g. `janet`, `cargo`, `godot` absent ⇒ failing).
- "Is X implemented?" checks read Python **AST identifiers and non-docstring strings**, so a comment or docstring that merely
  mentions `vkCmdDispatch` cannot make Vulkan appear verified.
- `impact`, `compute_heat`, `leverage` and effort sizes are **editorial weights** in the source; change them there.
- Metrics like "46 ms/frame" quoted inside priority text come from earlier timed runs and are labelled *measured*; the
  analyzer does not itself benchmark.

## Caveats

- The new endpoints do not set `Access-Control-Allow-Origin` themselves, but the shared `send_json` helper still
  adds `*` (tracked as `harden-localhost-api`). They are read-only, so the exposure is information only.
- The scan is a static heuristic; it does not run the test suite. Use `python -m unittest discover tests` for that.
