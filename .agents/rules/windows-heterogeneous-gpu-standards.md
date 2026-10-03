# Windows Heterogeneous GPU & Terminal Architecture Standards

## Behavioral Rule
When developing, maintaining, or refactoring components within the Krystal-Stack platform on Windows or cross-platform targets:

### 1. Terminal Output Invariant
- **Never rely on `curses` on Windows**: The standard Python `curses` module is Unix-only and fails on Windows without third-party wheels.
- **Always use Win32 Virtual Terminal Processing**:
  - Enable `ENABLE_VIRTUAL_TERMINAL_PROCESSING` (0x0004) and `DISABLE_NEWLINE_AUTO_RETURN` (0x0008) via `ctypes.windll.kernel32.SetConsoleMode`.
  - Use `\x1b[H` (cursor home) instead of clearing the console (`cls` or `\x1b[2J`) to prevent screen tearing and achieve 120+ FPS.
  - Hide the console cursor (`CONSOLE_CURSOR_INFO.bVisible = False`) during live rendering loops to eliminate cursor flicker.
  - Always configure `sys.stdout.reconfigure(encoding='utf-8')` to prevent `UnicodeEncodeError` with Windows code pages (e.g. cp1252).

### 2. IPC Invariant
- **Never hardcode Unix domain sockets** (e.g., `/tmp/*.sock`).
- On Windows, use:
  - Windows Named Pipes (`\\.\pipe\<name>`)
  - Shared Memory (`CreateFileMappingW` / `MapViewOfFile`) for zero-copy high-throughput tensor/pixel transfers.
  - Standard HTTP/SSE/WebSocket loopback (`127.0.0.1:<port>`).

### 3. Hardware Telemetry & Power Management Abstraction
- Abstract Linux `sysfs` (`/sys/class/...`) behind a clean Hardware Abstraction Layer (HAL).
- On Windows, query:
  - **DXGI**: `IDXGIAdapter3::QueryVideoMemoryInfo` for VRAM budgeting and budget change events.
  - **Power & CPU**: `CallNtPowerInformation` and `SetProcessInformation(PROCESS_POWER_THROTTLING_EXECUTION_SPEED)`.

### 4. Zero-Dependency Resilience
- Always provide a zero-external-dependency fallback using Python's standard library (`math`, `ctypes`, `http.server`, `threading`, `json`).
- Ensure mission-critical dashboards, verification scripts, and engines run immediately on bare Python environments without waiting for heavy wheel compilation.
