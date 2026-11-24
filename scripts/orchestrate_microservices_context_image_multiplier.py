#!/usr/bin/env python3
"""Automatizovaný CLI skript pre spustenie DAG mikroslužieb násobiacich kontexty (4×)
a spracovaný obraz (8× pixel-tok) spolu s alokáciou cez Python + `#![no_std]` Rust nadstavbu.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.trigonometric_hybrid_platform_framework.visual_compositor_timeline_damage_and_microservices import (
    execute_compositor_timeline_damage_and_microservices_suite,
)

if __name__ == "__main__":
    print(json.dumps(execute_compositor_timeline_damage_and_microservices_suite(), indent=2, ensure_ascii=False))
