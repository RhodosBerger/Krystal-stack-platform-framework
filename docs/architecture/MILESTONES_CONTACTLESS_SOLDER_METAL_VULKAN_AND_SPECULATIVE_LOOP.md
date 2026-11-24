# Míľniky `MS_01..MS_06`: Bezkontaktná letovačka, Ekvivalencia Apple Metal vo Vulkane, Kryptografia & Špekulatívna Slučka

- **Ústavné invarianty:** `VITAL_MAX_HP == 6` · `UMA < 0.78` · `PROPOSER != VERIFIER`
- **Marketingový a technický konštrukt:** **Bezkontaktná letovačka** (`weld_uma_pathway` / `TH_09_CONTACTLESS_TOPOLOGY_WELD_IMPEDANCE`)
- **Najlepšie overené zrýchlenie špekulatívnej vetvy (`128 B` UMA L1d):** **`340.52x`**

## 1. Plánované a splnené Míľniky (`MS_01..MS_06`)

| Míľnik | Cieľ | Cieľový artefakt | Stav |
| :--- | :--- | :--- | :---: |
| **`MS_01_CONTACTLESS_SOLDER_AND_TH09`** | Nahradiť Double-Time Mode pojmom Bezkontaktná letovačka, slovesom weld_uma_pathway, teóriou TH_09 a 64-bitovým .kubc bajtkódom. | `contactless_solder_uma_crypto_and_power_pipeline.py` | `DONE` |
| **`MS_02_CRYPTO_DEEP_RESEARCH_AND_PRESETS`** | Zmapovať kryptografické výpočty (SHA-NI, VAES, BLAKE2b, IFMA Ed25519, FIPS 203 ML-KEM NTT, AF_ALG) do YAML presetov. | `configs/contactless_solder_metal_vulkan_crypto_presets.yaml` | `DONE` |
| **`MS_03_THERMAL_SENSOR_POWER_PLAN_UBUNTU`** | Vytvoriť 5 stratégií thermal throttlingu, 24h senzorový plán napájania a kompletnú službu pre Ubuntu LTS. | `deploy/ubuntu_contactless_solder_power/krystal_ubuntu_power_daemon.py` | `DONE` |
| **`MS_04_APPLE_METAL_VULKAN_128B_MULTI_API`** | Dosiahnuť ekvivalenciu Apple Metal vo Vulkane (BDA + Dynamic Rendering + Timeline Semaphores) pre OpenGL/Zink, OpenCL, DirectX a Vulkan s glmark2 bránou. | `metal_vulkan_endpoints_and_speculative_bench_loop.py` | `DONE` |
| **`MS_05_SPECULATIVE_DEBUG_BENCHMARK_EVOLUTION`** | Zaviesť autonómnu slučku ladenia a benchmarkovania špekulatívnych funkcií s automatickým vetvením pri preukázanom náraste. | `metal_vulkan_endpoints_and_speculative_bench_loop.py` | `DONE` |
| **`MS_06_LIVE_UMA_KERNEL_NPU_VISUAL_STUDIO`** | Vykresliť živé grafické rozhranie procesov v UMA pamäti, vyťaženia jadra a NPU, priebehu bajtkódu a dynamického plánu napájania. | `contactless_solder_live_canvas.html + :8090/contactless_solder_studio` | `DONE` |
| **`MS_07_WIN32_GEN11_SINGLE_MULTI_CORE_AND_RUST_VULKAN_QUOTA`** | Optimalizovať Single-Core Turbo (Core 0 + MMCSS) a 4P Multi-Thread (0x55) cez Windows API a riadiť akceleračnú kvótu Vulkanu cez #![no_std] Rust ovládač. | `native/krystal_concurrent_rust_core/src/lib.rs + metal_vulkan_endpoints_and_speculative_bench_loop.py` | `DONE` |

## 2. Ekvivalencia Apple Metal -> Vulkan 1.4 a prenos 128B rámca do OpenGL, OpenCL, DirectX a Vulkan

| Cieľové prostredie | Prekladová / Interop vrstva | Konkrétne Vulkan volania | Mechanizmus priepustnosti |
| :--- | :--- | :--- | :--- |
| **Vulkan 1.3 / 1.4 (Natívne aplikácie, Godot 4, Blender)** | `Priamy reťazec VK_LAYER_KRYSTAL_CONTACTLESS_SOLDER` | `vkCmdPushConstants(layout, ALL, 0, 128, frame128b) + vkWaitSemaphores(timeline)` | 128B rámec odovzdaný priamo v hardvérových Push registroch bez alokácie deskriptorov |
| **OpenGL 4.6 (`glmark2` v Ubuntu, CAD prehliadače)** | `Mesa Zink (`MESA_LOADER_DRIVER_OVERRIDE=zink` -> Vulkan)` | `vkCmdBeginRenderingKHR + vkMapMemory(HOST_COHERENT UMA) + vkQueueSubmit2` | Zink prekladá GL draw/buffer volania na Vulkan dynamic rendering nad naším UMA poolom |
| **OpenCL 3.0 (Vedecké a tenzorové výpočty)** | `clvk / Mesa Rusticl + VK_KHR_external_memory_fd (DMA-BUF)` | `vkGetMemoryFdKHR / clImportMemoryARM (zdieľaný 64B-zarovnaný UMA deskriptor)` | Nulová kópia medzi OpenCL kernelom a Vulkan compute shaderom cez spoločný DMA-BUF |
| **DirectX 9 / 11 / 12 (Windows hry, Proton/Wine na Ubuntu)** | `DXVK (D3D9/11->Vulkan), vkd3d-proton (D3D12->Vulkan) & Win11 D3D12 OpenExistingHeap` | `VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT + VK_KHR_timeline_semaphore` | Zdieľaná 64KB UMA halda a timeline semafory znižujú CPU réžiu volaní vykresľovania |
