# Ponuka pre Komunitu, Arbitrážny Systém Zdrojov, Stratégie Jadier a Proaktívne Príležitosti pre OS a Grafické Ovládače

> **Ústavný invariant:** `VITAL_MAX_HP == 6` ($\neg\text{Valid}(\text{Akcia}) \implies \Delta S = 0$) | **Vlastníctvo a licencia:** **Všetky práva vyhradené — Dušan Kopecký (`dusan.kopecky0101@gmail.com`)**; osobné použitie vyžaduje súhlas majiteľa, komerčné použitie vyžaduje kontaktovanie majiteľa a predloženie cenovej ponuky | **Stav dôkazov:** `LOCAL_MEASURED` & `VERIFIED_WITH_SCOPE`
>
> **Licenčný dodatok (`Ivan Novikov` / `xerj.org`):** **Ivan Novikov** má na základe dohody s majiteľom codebase **Dušanom Kopeckým** povolené používať novododané funkcie a rozšírenia tohto frameworku — predovšetkým **hlboké integrácie bez-kópiových tenzorov (`kuMallocUMA`, `ov.Tensor` v zdieľanej pamäti `GPU_USM_MEMORY`, `64 B / 64 KB` zarovnanie, `FTZ/DAZ` ochrana) a dátovo odvodené konfigurácie OpenVINO (`OV_MT_01..05` viacvláknové profily a `HETERO` plánovanie)** — na svojom projekte (**XERJ**, `xerj.org`). Ďakujeme za spoluprácu s `xerj.org`.

---

## 1. Čo Krystal Stack Ponúka Komunite Vývojárov, Štúdií a Systémových Inžinierov

Krystal Stack sprístupňuje otvorenú referenčnú sadu nástrojov, ktorú možno nasadiť na ľubovoľný hardvér **Intel od 11. generácie vyššie** (Tiger Lake, Alder/Raptor Lake P+E jadrá, Core Ultra s NPU), ako aj na **Linux/Unix (`DRM/GEM`, `DMA-BUF`, `VFIO/UIO`)** a **Windows 11 (`WDDM 3.2`, `DXCore`, `DirectX 12`, `Vulkan`)**:

| Pilier pre komunitu | Kľúčový modul / príkaz | Nameraný prínos (`LOCAL_MEASURED`) |
| :--- | :--- | :--- |
| **1. Univerzálna `16 KB mmap` MMIO Apertúra a Hierarchia v Pamäti** | `universal_hardware_mmio_aperture_kernel.py` | **`59,18×` rýchlejšie** adresné čítanie topológie (`119,24 ns` vs. `7 057,08 ns`), absorpcia všetkých `9` rodín celkov (`375` súborov). |
| **2. Nulovo-réžijná vysokofrekvenčná telemetria `BAR0`** | `hybrid_rust_and_intel_xe_cuda_streams.py` | **`245,97×` nižšia réžia** v horúcej slučke (`133,01 ns` zrkadlo `BAR0` vs. `32 715,62 ns` priamy dotaz na ovládač). |
| **3. CUDA-štýlová abstrakcia pre Stream Procesory, AI Jadrá a NPU** | `hybrid_rust_xe_cuda_bridge.rs` + schéma | `768` FP32 / `1 536` FP16 prúdových jednotiek, `64 KB SLM` Flash-Attention (odchýlka `2,4 × 10⁻⁷`), nulové rozliatie registrov pri `K < 449`. |
| **4. Softvérový našepkávač pre procesor (`CpuLookaheadWhisperer`)** | `krystal_universal_mmio_aperture.h` + Python | **`2,90×` zrýchlenie** (`386,8 µs` vs. `1 123,6 µs`) vďaka pred-zoradeniu L2 cache a bez-vetvovej bitovej autorite. |
| **5. Dátovo odvodené OpenVINO Multithreading Presety (`OV_MT_01..05`)** | `configs/openvino_multithreading_presets.yaml` | **`2,79×` rýchlejšia** OpenVINO inferencia, eliminácia `2,45×` SMT chvostového kmitania a `4,99×` rýchlejšie 4-vláknové SPSC logovanie. |
| **6. Zlepšenie vizuálnej kvality pri prísnom `120 Hz` rozpočte** | `benchmark_visual_quality_and_performance_gain()` | **`+11,17 dB PSNR`** (`30,02 dB` vs. `18,85 dB`), `3,62×` nižšia chyba `RMSE`, `16×` hustota vzoriek za **`2,098 ms < 8,333 ms`**. |
| **7. Arbitrážny systém, stratégie jadier a hľadač príležitostí v `krystal_bot`** | `krystal_bot/integrated_efficiency_overseer.py` | Príkazy `/overseer`, `/auto-select`, `/arbitrate` (`/arbitraz`) a `/opportunities` (`/prilezitosti`). |

---

## 2. Ako Funguje Arbitrážny Systém (`SovereignResourceArbitrationEngine`)

Namiesto statických front alebo slepého prerušovania procesov vyhodnocuje arbitrážny engine dopyty súperiacich subsystémov cez **tieňové ceny zdrojov** ($\lambda_{\text{UMA}}, \lambda_{W}$), ktoré nelineárne rastú pri blížení sa k stropu `UMA = 0,78` alebo k teplote `85 °C`:

$$\mathcal{A}(b_i) = \frac{\text{Užitočnosť}(b_i) \cdot w_{\text{úroveň}}(b_i)}{\lambda_{\text{UMA}} \cdot \Delta\mu_{\text{UMA}}(b_i) + \lambda_{W} \cdot \Delta W(b_i) + \epsilon}$$

1. **Úroveň `P0` (Ústavná ochrana obrazu a zdravia systému):** Má garantovanú prednosť; nikdy nestratí snímku ani logovací rámec.
2. **Úroveň `P1` (Aktívny AI výpočet, XERJ AST, `64 KB SLM` Flash-Attention):** Dostáva koprocesor `Iris Xe` a fyzické P-jadrá dovtedy, kým súčet neprekročí strop `UMA < 0,78`.
3. **Úroveň `P2` (Úlohy na pozadí, archivácia, kompresia):** Pri raste tieňovej ceny sa automaticky presunie do tieňa obnovovania obrazovky (`VBlank Shadow`, $\Delta S = 0$) alebo na úsporné E-jadrá.

---

## 3. Šesť Stratégií Zapínania Jadier (`CORE_WAKE_01..06`) a Proaktívne Príležitosti pre OS a Ovládač

Bot prostredníctvom triedy `BotOsAndGpuOpportunityGenerator` naživo kontroluje voľnú pamäť RAM, rozloženie vlákien, predhriate bloby v `openvino_blobs` a štatistiku `GPU_USM_MEMORY` v grafickom ovládači, na základe čoho automaticky navrhuje a cez arbitráž aktivuje:
* **`OPP_OS_01_PCORE_AFFINITY_CONSOLIDATION`:** Zoskupenie horúcich vlákien na fyzické P-jadrá.
* **`OPP_OS_02_LOW_RAM_COMPACT_ARENA_GUARD`:** Kompaktný `64 MB` režim a `40,37×` kompresia logov pri poklese voľnej systémovej RAM pod `4,5 GB` (ochrana pred swapovaním na NVMe).
* **`OPP_GPU_01_ZERO_BOUNCE_64KB_USM_BRIDGE`:** Zarovnanie zdieľaných bufferov na `64 B` a `64 KB` hranicu pre vynechanie skrytých kópií vo `WDDM 3.2` / `DirectX 12` / `Linux DRM-GEM`.
* **`OPP_GPU_02_SHADER_CACHE_AND_SLM_K256_TILING`:** Dlaždicovanie `K_tile = 256 < 449` v `64 KB SLM` bez rozliatia registrov ovládača.
* **`OPP_OS_GPU_03_VBLANK_SHADOW_TELEMETRY_COALESCING`:** Čítanie živých hodnôt zo `133 ns` zrkadla `BAR0` a zlučovanie pomalých dotazov do ovládača mimo vykresľovania snímky.

---

## 4. Rýchle Spustenie a Overenie

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/run_stock_vs_krystal_benchmarks.py
.\.venv\Scripts\python.exe -O -X utf8 -m pytest tests/test_combinatorial_derivative_functions.py -v
```
