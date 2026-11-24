# Komplexný Benchmark: OpenVINO + Kisak-Mesa Gallium + `#![no_std]` Rust Firmvér + Stream-CUDA

- **Ústavný invariant:** `VITAL_MAX_HP == 6` (`\neg Valid \implies \Delta S = 0`)
- **Detegované OpenVINO akcelerátory:** `['CPU', 'GPU']`
- **Aktívny kanál Gallium <-> OpenVINO:** `INTEL_IRIS_XE_CLDNN_USM + KRYSTAL_GALLIUM_RUST_CSO`
- **SHA-256 atestácia merania:** `fa4be3ec56c3edc3e9c017abd99d366d7320e6779002860408a86dfcc8bd4de3`

## Súhrnná tabuľka nameraných zrýchlení (`LOCAL-MEASURED`)

| Rovina | Baseline | Krystal + Gallium + Rust | Zrýchlenie / Výsledok |
|---|---:|---:|---:|
| 1. Riadenie firmvéru (ACPI AML vs. `16.5 KB` Rust `KUBC`) | `490.3 µs` | `39.5 µs` | **`12.41×`** |
| 2. Procesný Validátor → Generátor + Agregátor | `1.7434 ms` | `0.9107 ms` | **`1.91×`** |
| 3. Väzba `1 MB` tenzora do OpenVINO (Copy vs. Gallium 64B USM) | `1.1806 ms` | `0.0792 ms` | **`14.91×`** |
| 4. OpenVINO Neurónová syntéza snímky (`CPU` vs. `Iris Xe GPU`) | `CPU: 0.0995 ms` | `GPU: 0.4954 ms` | **Parita `Δ=0.0`** |
| 5. Stream-CUDA (`768` SP) + Kisak-Gallium Vulkan FPS | `60.0 FPS` | `148.8 FPS` | **`2.48×` (`145–157 FPS`)** |
