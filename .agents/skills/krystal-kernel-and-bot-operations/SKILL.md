---
name: krystal-kernel-and-bot-operations
description: Cheatsheet for running, testing and troubleshooting the Krystal compute kernel, its OpenAI-compatible API, the krystal_bot package (RAG, quotas, LLM gateway, Keras risk model, pixel-art composer) and the Mission Control hub on Windows.
---

# Krystal Kernel and Bot Operations

## 1. Package map
- `krystal_kernel/`: hardware profiler, EDF scheduler with lanes (thread, process), AIMD concurrency, self-healing supervisor, learning loop, OpenAI-compatible API (`openai_api.py`).
- `krystal_bot/`: `rag.py` (BM25 + hashed n-gram retrieval), `quota.py` (rolling-window quotas), `llm_gateway.py` (policy-gated OpenAI-compatible client), `risk.py` and `pending.py` (log-driven risk model evaluated in quotas), `keras_worker.py` (isolated Keras), `palette.py` (conversation ledger), `composer.py` (pixel-art canvas and scene painters).
- `krystal_web_hub/server.py`: Mission Control. Kernel routes are `/v1/models`, `/v1/embeddings`, `/v1/chat/completions`, `/metrics` and `/api/kernel/*`.

## 2. Commands (PowerShell, repo root)
```powershell
python -m krystal_kernel profile        # measured hardware profile
python -m krystal_kernel calibrate      # thread/process scaling, IPC round trip
python -m krystal_kernel params         # derived parameters with provenance
$env:PYTHONIOENCODING="utf-8"
python -m unittest discover tests       # full suite
python -W error::ResourceWarning -m unittest tests.test_krystal_kernel   # leak check
python -u -m krystal_web_hub.server 8080   # start the hub (run as a daemon)
```

## 3. Keras (isolated)
- Interpreter: `.venv-keras\Scripts\python.exe`. Backend: `torch` (CPU). JAX is blocked on this machine by Windows Application Control, so do not use it and do not try to bypass the block.
- Worker: `.venv-keras\Scripts\python.exe -m krystal_bot.keras_worker` (JSON lines on stdin/stdout). Set `KRYSTAL_KERAS_THREADS=2` to leave cores for the hub.
- The hub never imports Keras. If the worker is missing, the stdlib logistic model and the heuristic are used.

## 4. LLM gateway policy
- Off by default. It needs `enabled: true`, a configured model, and either debug mode (`KRYSTAL_DEBUG=1` or `debug: true`) or a high-performance device (default: at least 16 logical cores and 32 GB RAM).
- A non-private endpoint also needs `allow_remote: true`. Quotas: requests per minute and tokens per day. The API key comes from the environment variable named in `api_key_env`.
- Math and binary-log documentation helpers attach locally computed facts and stamp the output as AI-generated and unverified.

## 5. Verification routine
1. List listeners: `Get-NetTCPConnection -LocalPort 8080 -State Listen`, and `Get-CimInstance Win32_Process -Filter "Name like 'python%'"`.
2. Kill stale hub processes, including the WindowsApps `python.exe` stub and its child.
3. Start one hub and run a Python check script with timeouts. Do not use curl loops.
4. The hub currently serves about 17 requests per second (about 63 ms per request on every route), so rate-limit tests need either a lower configured rate or a unit test.

## 6. Troubleshooting
| Symptom | Cause | Fix |
|---|---|---|
| `ConnectionRefused` under load | Two servers on one port, or listen backlog too small | Kill stale servers; backlog is 128 now |
| `DLL load failed ... Application Control` | JAX or jaxlib blocked | Use the torch backend |
| `UnicodeEncodeError` (cp1252) | Slovak text on the console | `$env:PYTHONIOENCODING="utf-8"` |
| `ResourceWarning` in tests | Leaked pipes or processes | Close workers; run with `-W error::ResourceWarning` |
| Model never "accepted" | Too few positives or no gain over the heuristic | Collect more real logs; do not synthesise data |
