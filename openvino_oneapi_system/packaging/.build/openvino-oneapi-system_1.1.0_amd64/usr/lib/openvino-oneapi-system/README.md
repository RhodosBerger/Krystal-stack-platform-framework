# OpenVINO ONE API Performance System

Separated subproject focused on:
- OpenVINO Runtime inference loop (with safe fallback if OpenVINO is not installed)
- oneAPI-oriented runtime tuning hooks
- evolutionary optimization of runtime parameters
- UUID/timer-based resource scheduling
- advanced telemetry and structured logging
- delegated logging channels (`system`, `telemetry`, `planning`, `policy`, `inference`, `grid_update`)
- 3D grid memory abstraction
- Windows API awareness layer for Windows runtime version

## Run

```bash
cd openvino_oneapi_system
python3 main.py --cycles 10 --interval 0.5
```

## Linux Benchmark Proof

Run baseline vs adaptive algorithmic planning:

```bash
python3 openvino_oneapi_system/benchmark_linux.py --cycles 50
```

Outputs:
- `openvino_oneapi_system/logs/benchmark_latest.txt` (ASCII report)
- `openvino_oneapi_system/logs/benchmark_latest.json` (machine-readable metrics)

## Build DEB

```bash
sh openvino_oneapi_system/packaging/build_deb.sh 1.0.0
```

Package output:
- `openvino_oneapi_system/dist/openvino-oneapi-system_1.0.0_<arch>.deb`

Install:

```bash
sudo dpkg -i openvino_oneapi_system/dist/openvino-oneapi-system_1.0.0_<arch>.deb
```

Commands after install:
- `ovo-runtime`
- `ovo-benchmark`

Optional service mode:

```bash
sudo systemctl enable --now openvino-oneapi-system
sudo systemctl status openvino-oneapi-system
```

Service configuration:
- `/etc/default/openvino-oneapi-system`

## Notes

- This is intentionally isolated from the rest of the repository.
- If `openvino` is available, the runtime uses `openvino.runtime.Core`.
- If not available, a deterministic simulated inference path is used.
- Log files are generated under `openvino_oneapi_system/logs/`.
