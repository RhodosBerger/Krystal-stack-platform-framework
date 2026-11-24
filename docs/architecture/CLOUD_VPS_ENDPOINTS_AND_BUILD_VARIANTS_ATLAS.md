# Cloud / VPS Atlas Endpointov, 4 Modulárne Buildy, Docker & Porovnanie s Blender Benchmarkom

- **Ústavný invariant:** `VITAL_MAX_HP == 6`
- **Jednopríkazová inštalácia na VPS:** `bash deploy/vps_cloud_bootstrap/bootstrap_krystal_vps.sh`
- **Docker kontajner hry a benchmarku:** `docker compose -f docker-compose.last_tribe_cyber60s.yml up --build`

## 1. Modulárne Buildy (`BUILD_01..04`)
- **`BUILD_01_VPS_HEADLESS_CLOUD`** (`68 MB`, `SERVER_API_AND_RULE_MODS`): Čisté Linux VPS bez GPU (iba CPU AVX2/AVX-512 + OpenVINO CPU + MCP + SQLite CRUD)
- **`BUILD_02_LAST_TRIBE_CYBER60S_DOCKER`** (`115 MB`, `FULL_CARD_RECIPE_AND_SHADER_MODS`): Kontajnerizovaná webová hra The Last Tribe (60s Cyberpunk Hologram) + In-Game Adrenalin Benchmark
- **`BUILD_03_IRIS_XE_ARC_GAMING_PACK`** (`94 MB`, `VULKAN_ENGINE_PRESET_MODS`): Lokálny PC build: #![no_std] Rust v2.0 + Kisak/Fedora-DLSS + FXAA/TSAA/GTAO/Ray-Trace
- **`BUILD_04_PROJEKTANT_CAD_BLENDER_STUDIO`** (`142 MB`, `CAD_PARAMETRIC_AND_BLENDER_PASS_MODS`): Inžinierska stanica: CAD (DXF/SVG/OBJ) + Blender MCP + 14 GB VRAM Gemma 4 E2B + Nextcloud 2FA

## 2. Kľúčové Cloud / VPS & Prehliadačové Endpointy (`Port 8090`)
- `GET /last_tribe_cyber60s` — 60s Cyberpunk Hologram **The Last Tribe** (`Crystal`, `Druid`, `Toxic`, 4 projekcie mapy, Skill Tree, Crafting, Modding)
- `GET /adrenalin_control_center` — GSAP (`Flip` + `TextPlugin`) morfujúci panel v štýle AMD Adrenalin
- `GET /rust_firmware_studio` — Suverénne `#![no_std]` Rust firmvérové & Kisak-Gallium Vulkan štúdio
- `POST /mcp/v1` — Zjednotený JSON-RPC 2.0 MCP protokol pre LM Studio, Bionic a cloudových agentov
- `GET|POST /api/v1/crud/endpoints` — SQLite WAL CRUD register dynamických endpointov
- `POST /api/v1/dusan_game/accelerate_turn` — 19-hex maticový akcelerátor pre Axiom Engine
- `GET /api/v1/cloud/vps_status_and_blender_bench` — OpenVINO VPS skener, záloha/obnova a porovnanie s Blender Benchmarkom
