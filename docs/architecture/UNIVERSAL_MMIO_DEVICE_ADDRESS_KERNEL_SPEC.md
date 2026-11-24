# Univerzálne Pamäťovo-Adresné Jadro (`MMIO Aperture`), Zber Hierarchie do Pamäte a Absorpcia Všetkých Celkov

> **Ústavný invariant:** `VITAL_MAX_HP == 6` (`\neg Valid \implies \Delta S = 0`) | **Pamäťová apertúra:** `16 KB` (`0x0000..0x3FFF`)

## 1. Fáza 1 — Získanie hardvérovej hierarchie priamo do pamäte (`0x1000..0x1FFF`, `64 B / uzol`)

| Uzol | Rodič | Offset v `mmap` | PCI ID | Bázová adresa | Šírka pásma | Značka uzla |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0` | `None` | `0x1000` | `0x80869a14` | `0x00000000` | `34.13 GB/s` | `ROOT_SOC_HOST_BRIDGE` |
| `1` | `0` | `0x1040` | `0x80869a14` | `0x00010000` | `88.0 GB/s` | `CPU_4C_8T_AVX512` |
| `2` | `0` | `0x1080` | `0x80869a14` | `0x00100000` | `55.1 GB/s` | `L2_CACHE_512KB_SRAM` |
| `3` | `0` | `0x10c0` | `0x80869a14` | `0x01000000` | `17.81 GB/s` | `UMA_DRAM_CONTROLLER` |
| `4` | `0` | `0x1100` | `0x80869a49` | `0x10000000` | `34.13 GB/s` | `IGPU_IRIS_XE_96EU_BAR` |
| `5` | `1` | `0x1140` | `0x80869a14` | `0x20000000` | `55.1 GB/s` | `ONEDNN_ONETBB_4T_TILE` |
| `6` | `2` | `0x1180` | `0x80869a49` | `0x30000000` | `34.13 GB/s` | `OPENVINO_LEVELZERO_USM` |
| `7` | `3` | `0x11c0` | `0x10de2684` | `0x40000000` | `31.5 GB/s` | `PCIE_REBAR_DGPU_SLOT` |
| `8` | `4` | `0x1200` | `0x1af41050` | `0x50000000` | `12.0 GB/s` | `WSL2_WNTC_VPP_HIL_BRIDGE` |

## 2. Fáza 2 — Možnosti absorpcie všetkých dostupných celkov (`ABS_OPT_01..04`)

| Možnosť | Rozsah mutácie | Verdikt | Určenie |
| :--- | :---: | :---: | :--- |
| `ABS_OPT_01_READONLY_HIERARCHY_MIRROR` | `READ_ONLY (Delta_S == 0)` | **`SUPPORTED_FOR_DISCOVERY`** | Fáza L0 auditu, diagnostika, bezpečné UI projekcie (`UAP_01`). |
| `ABS_OPT_02_ZERO_TOUCH_DECORATOR_BRIDGE` | `BOUNDARY_ADAPTER_ONLY (SourceCodeDelta == 0)` | **`SUPPORTED_WITH_SCOPE`** | Rýchla integrácia externých Python/OpenVINO/C++ knižníc bez zásahu do ich kódu. |
| `ABS_OPT_03_UNIFIED_MMIO_BAR0_AND_128B_BUS` | `CAPABILITY_GATED_ZERO_COPY` | **`RECOMMENDED_PRIMARY_ARCHITECTURE`** | Hlavná produkčná architektúra Krystal Stacku (Windows + Linux/Unix + Rust + Vulkan + OpenVINO). |
| `ABS_OPT_04_MONOLITHIC_RING0_KERNEL_FUSION` | `UNBOUNDED_RING0 (Vysoké riziko BSOD / Kernel Panic)` | **`REJECTED_BY_SAFETY_GATE`** | Zamietnuté pre štandardnú prevádzku — porušuje suverénny princíp P1 a zvyšuje pádové riziko. |

## 3. Absorbované rodiny celkov v `128 B` pamäťovej zbernici (`9` celkov, `402` súborov)

| Slot | Celok | Offset na zbernici | Priradený HW uzol | Súbory | SHA-256 skratka |
| :---: | :--- | :---: | :--- | :---: | :---: |
| `0` | `UNIT_01_KRYSTAL_KERNEL_CORE` | `0x2000` | `CPU_4C_8T_AVX512` | `250` | `397d353614839f98` |
| `1` | `UNIT_02_TRIGONOMETRIC_HYBRID_PKG` | `0x2080` | `OPENVINO_LEVELZERO_USM` | `68` | `6d9247b6b3a0cb98` |
| `2` | `UNIT_03_SOVEREIGN_OPENVINO_V2` | `0x2100` | `IGPU_IRIS_XE_96EU_BAR` | `13` | `547e2d1099baaa4a` |
| `3` | `UNIT_04_C11_ABI_MMIO_HEADERS` | `0x2180` | `ROOT_SOC_HOST_BRIDGE` | `13` | `805ea0fa049a28a1` |
| `4` | `UNIT_05_C_SIMD_SPSC_RUNTIME` | `0x2200` | `L2_CACHE_512KB_SRAM` | `7` | `4a9de1fcb0b5958a` |
| `5` | `UNIT_06_RUST_GUARDIAN_ALLOCATOR` | `0x2280` | `UMA_DRAM_CONTROLLER` | `12` | `26fb564862d04a9e` |
| `6` | `UNIT_07_RUST_BOT_VULKAN_MMIO` | `0x2300` | `IGPU_IRIS_XE_96EU_BAR` | `24` | `5a077a783e1d4217` |
| `7` | `UNIT_08_JSON_AUTHORITY_SCHEMAS` | `0x2380` | `ROOT_SOC_HOST_BRIDGE` | `11` | `5b10e943c2e9454e` |
| `8` | `UNIT_09_ONEAPI_CONDA_CONFIGS` | `0x2400` | `ONEDNN_ONETBB_4T_TILE` | `4` | `b90543a60f567877` |

## 4. Namerané zrýchlenie adresného prístupu k hierarchii (`LOCAL_MEASURED`)
- **Štandardný JSON/IOCTL dotaz:** `16582.37 ns/dotaz`
- **Priamy adresný `mmap` `u32` slot (`0x1000 + idx*64`):** **`379.1 ns/dotaz`** (**`43.74x` rýchlejšie**, rozdiel `-16203.27 ns`).
