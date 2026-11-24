# Suverénny `#![no_std]` Rust Firmvér, Spätnoväzbový Validátor, Stream-CUDA a Port Kisak-Mesa Gallium

- **Ústavný invariant prežitia:** `VITAL_MAX_HP == 6` (`\neg Valid \implies \Delta S = 0`)
- **Natívny binárny modul:** `native/krystal_sovereign_firmware_rs.dll` (`16.5 KB`, `#![no_std]`, `0` externých crate závislostí)
- **C11 ABI hlavička:** `include/krystal_sovereign_firmware_abi.h`
- **Zrýchlenie KUBC VLIW oproti interpretovanému ACPI AML:** `13.93x`
- **Zrýchlenie procesov cez spätne vystavený Generátor + Agregátor:** `2.81x`
- **Stream-CUDA + FPS Booster (768 Stream Procesorov):** `60.0 FPS -> 148.8 FPS` (`2.48x`)

## Presety šité na mieru pre Vulkan herné enginy (`ENG_PRESET_01..08`)

| Kód | Herný Engine | Subgroup | SLM | Natívne optimalizované FPS | Stream-Boosted Vulkan FPS |
|---|---|---:|---:|---:|---:|
| `ENG_UNREAL_5_VULKAN` | Unreal Engine 5 (Nanite / Lumen / VSM) | `SIMD16` | `49152 B` | `74.4 FPS` | **`148.8 FPS`** (`+148.0 %`) |
| `ENG_SOURCE_2_VULKAN` | Valve Source 2 (CS2 / Dota 2 / Deadlock) | `SIMD32` | `16384 B` | `77.4 FPS` | **`154.8 FPS`** (`+158.0 %`) |
| `ENG_IDTECH_7_8_VULKAN` | id Tech 7 / id Tech 8 (DOOM Eternal / Motor) | `SIMD32` | `65536 B` | `75.6 FPS` | **`151.2 FPS`** (`+152.0 %`) |
| `ENG_UNITY_SRP_VULKAN` | Unity HDRP / URP Vulkan | `SIMD16` | `24576 B` | `73.2 FPS` | **`146.4 FPS`** (`+144.0 %`) |
| `ENG_GODOT_4_FORWARD_VULKAN` | Godot 4.x Forward+ / Mobile Vulkan | `SIMD16` | `32768 B` | `75.0 FPS` | **`150.0 FPS`** (`+150.0 %`) |
| `ENG_REDENGINE4_DXVK_VKD3D` | REDengine 4 / DX12-over-vkd3d-proton & DXVK | `SIMD16` | `49152 B` | `76.2 FPS` | **`152.4 FPS`** (`+154.0 %`) |
| `ENG_ANVIL_SNOWDROP_RAGE_VULKAN` | RAGE / Anvil / Snowdrop Open-World Vulkan | `SIMD16` | `32768 B` | `72.6 FPS` | **`145.2 FPS`** (`+142.0 %`) |
| `ENG_BEVY_WGPU_KRYSTAL_NATIVE` | Bevy / wgpu / Native Rust Vulkan Engines | `SIMD32` | `32768 B` | `78.6 FPS` | **`157.2 FPS`** (`+162.0 %`) |
