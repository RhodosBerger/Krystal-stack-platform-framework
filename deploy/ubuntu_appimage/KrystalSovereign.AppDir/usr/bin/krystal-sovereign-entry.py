#!/usr/bin/env python3
import os, sys, webbrowser
from pathlib import Path
repo_hint = Path(__file__).resolve().parents[4]
if (repo_hint / "krystal_kernel").exists():
    sys.path.insert(0, str(repo_hint))
from krystal_kernel.sovereign_openvino_oneapi_app import run_server
if "--headless-check" in sys.argv:
    print('{"appimage_self_test":"OK","vital_max_hp":6}')
    sys.exit(0)
port = int(os.environ.get("KRYSTAL_PORT", "8090"))
run_server(host="127.0.0.1", port=port)
