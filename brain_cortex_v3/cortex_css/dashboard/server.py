#!/usr/bin/env python3
"""CortexCSS Dashboard — FastAPI backend + WebSocket live data"""
import sys, os, json, asyncio, time, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

try:
    from fastapi import FastAPI, WebSocket
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from cortex_css.styles.compute_styles import UTILITY_STYLES
from cortex_css.evaluator.cascade_engine import (
    CortexElement, ParallelStyleEvaluator
)

if not HAS_FASTAPI:
    print("Install: pip install fastapi uvicorn")
    sys.exit(1)

app = FastAPI(title="CortexCSS Dashboard")

# Read the HTML from same directory
_HTML_PATH = os.path.join(os.path.dirname(__file__), "index.html")

@app.get("/", response_class=HTMLResponse)
async def root():
    with open(_HTML_PATH) as f:
        return f.read()

@app.get("/api/styles")
async def get_styles():
    return {
        "count": len(UTILITY_STYLES),
        "styles": [
            {"selector": k, "properties": v.properties}
            for k, v in UTILITY_STYLES.items()
        ]
    }

@app.websocket("/ws/race")
async def race_websocket(websocket: WebSocket):
    await websocket.accept()
    evaluator = ParallelStyleEvaluator(racing_mode="balanced")
    try:
        while True:
            # Simuluj HW stav
            cpu_temp = 45 + random.uniform(0, 40)
            gpu_busy  = random.uniform(10, 95)
            on_batt   = random.random() > 0.6

            candidates = {}

            # Dynamicky pridaj triedy podľa HW stavu
            el_gpu = CortexElement("gpu-path")
            el_gpu.add_class(".device-gpu").add_class(".precision-f16")
            el_gpu.add_class(".power-overdrive" if not on_batt else ".power-eco")
            candidates["gpu-path"] = el_gpu

            el_cpu = CortexElement("cpu-path")
            el_cpu.add_class(".device-cpu").add_class(".precision-bf16")
            el_cpu.add_class(".hint-latency").add_class(".streams-2")
            candidates["cpu-path"] = el_cpu

            el_hetero = CortexElement("hetero-path")
            el_hetero.add_class(".device-hetero").add_class(".precision-int8")
            el_hetero.add_class(".hint-throughput").add_class(".streams-4")
            candidates["hetero-path"] = el_hetero

            if on_batt:
                el_eco = CortexElement("eco-path")
                el_eco.add_class(".battery-survivor")
                candidates["eco-path"] = el_eco

            results = evaluator.race(candidates)

            payload = {
                "timestamp": time.time(),
                "hw": {
                    "cpu_temp_c": round(cpu_temp, 1),
                    "gpu_busy_pct": round(gpu_busy, 1),
                    "on_battery": on_batt,
                    "power_mode": "eco" if on_batt else ("overdrive" if cpu_temp < 60 else "balanced")
                },
                "results": [
                    {
                        "name": r.style_name,
                        "winner": r.winner,
                        "device": r.computed.target_device,
                        "precision": r.computed.precision,
                        "hint": r.computed.performance_hint,
                        "streams": r.computed.num_streams,
                        "power_mode": r.computed.power_mode,
                        "latency_ms": r.latency_ms,
                        "throughput_fps": r.throughput_fps,
                        "power_mw": r.power_mw,
                        "score": r.score,
                        "classes": r.computed.applied_classes,
                        "credit_cost": r.computed.credit_cost(),
                    }
                    for r in results
                ]
            }
            await websocket.send_json(payload)
            await asyncio.sleep(1.5)
    except Exception:
        pass
    finally:
        evaluator.shutdown()

if __name__ == "__main__":
    print("\n  CortexCSS Dashboard")
    print(f"  Open: http://localhost:7860\n")
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="warning")
