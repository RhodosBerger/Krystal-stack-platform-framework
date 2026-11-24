# Komunitná Znalostná Báza — Často Kladené Otázky k Codebase Krystal Stack

## FAQ_01_WHY_VITAL_MAX_HP_6: Prečo sa v celej codebase (Rust, C11 ABI, Python, SQLite CRUD aj Vulkan vrstve) vyžaduje konštanta VITAL_MAX_HP == 6?
VITAL_MAX_HP == 6 je ústavný invariant prežitia a typovej integrity stavu (pôvodne ukotvený v Axiom Engine projektu dusan-game a rozšírený na celú architektúru KSPP P10). Slúži ako deterministická strážna kotva: ak poškodený pamäťový rámec alebo neplatné volanie zmení túto hodnotu, transakčná brána okamžite zamietne zápis a zachová pôvodný stav (ΔS = 0).
- **Súvisiace súbory:** `krystal_kernel/constants.py`, `include/krystal_sovereign_firmware_abi.h`, `native/krystal_sovereign_firmware_rs/src/lib.rs`

## FAQ_02_WHY_NO_BARE_ASSERT: Prečo je v produkčných moduloch zakázaný príkaz `assert` a všade sa vyvoláva `ContractViolation`?
Pri spustení interpretera s optimalizáciou `python -O` Python úplne odstráni všetky holé `assert` príkazy z bajtkódu. Explicitná výnimka `ContractViolation` garantuje, že kontroly hraníc UMA, 128B zarovnania a bezpečných frekvencií čipsetu ostanú aktívne aj v produkčnom `-O` režime.
- **Súvisiace súbory:** `krystal_kernel/constants.py`, `tests/test_combinatorial_derivative_functions.py`

## FAQ_03_HOW_NOSTD_RUST_FITS_L1I: Ako sa natívne Rust moduly (`10,0 KB`, `15,0 KB` a `16,5 KB`) zmestia celé do 32 KB L1i inštrukčnej cache?
Kompilujeme ich s `#![no_std]`, `panic = 'abort'`, `lto = 'fat'`, `strip = 'symbols'` a nulovými externými závislosťami cez `rustc` + `Zig` linker. Tým odpadá ~1 MB réžia štandardnej knižnice a Tokio runtimeu (62,5× menšia binárka) a jadro vykonáva `LOCK CMPXCHG`, `TZCNT` a `RDTSC` priamo z L1i cache.
- **Súvisiace súbory:** `native/krystal_sovereign_firmware_rs/src/lib.rs`, `scripts/build_concurrent_rust_modules.py`

## FAQ_04_WHAT_IS_BEZKONTAKTNA_LETOVACKA: Čo presne v kóde znamená pojem „Bezkontaktná letovačka“ (`weld_uma_pathway`) a teória `TH_09`?
Ide o bezkópiové premostenie pamäťových rozhraní v zdieľanej UMA pamäti (`64 B` / `128 B` zarovnanie, `ov.Tensor(..., shared_memory=True)`, `VK_KHR_buffer_device_address`) bez medzikópií CPU↔GPU, riadené 64-bitovým VLIW bajtkódom `.kubc` a obmedzovačom saturácie `UMA < 0,78`.
- **Súvisiace súbory:** `projects/trigonometric_hybrid_platform_framework/contactless_solder_uma_crypto_and_power_pipeline.py`

## FAQ_05_HOW_OPENVINO_MEETS_GALLIUM_AND_DLSS: Ako spolupracujú Kisak/Fedora Mesa Gallium ovládače, Stream-CUDA (`768` SP) a OpenVINO?
Bez úpravy zdrojového kódu Mesa (`SourceCodeDelta == 0`) sa `64 B`-zarovnané UMA buffery zdieľajú priamo s `ov.Tensor(shared_memory=True)` na `CPU` aj `Intel Iris Xe GPU`. Stream-CUDA na `768` FP32 prúdových procesoroch počíta v chronometrickom tieni (`Stream 1`) pohybové vektory a syntézu medzisnímok, čím zvyšuje snímkovú frekvenciu hier vo Vulkane.
- **Súvisiace súbory:** `krystal_kernel/openvino_gallium_driver_bridge.py`, `projects/trigonometric_hybrid_platform_framework/sovereign_rust_firmware_cuda_gallium_suite.py`

## FAQ_06_WHAT_IS_DIFFERENTIAL_GEARBOX: Ako fungujú prevody s diferenciálom pri prepínaní threadov (vlákien) a frekvencií?
Trieda `DifferentialThreadGearbox` volí bezpečný frekvenčný stupeň `G1..G5` (`400–4200 MHz` CPU, `400–1300 MHz` Iris Xe) a samosvorne delí inštrukčný tok medzi CPU thready (Poloos A) a stream procesory iGPU (Poloos B) podľa okamžitého odporu L2/L3 cache a UMA zbernice.
- **Súvisiace súbory:** `projects/trigonometric_hybrid_platform_framework/differential_gearbox_ux_flows_and_community_faq.py`

## FAQ_07_HOW_2B_GEMMA_LEARNS_EDITS: Ako sa dvojbiliónová Gemma (`Gemma 4 E2B`) učí z procesov v pipeline a prečo je rýchlejšia?
Predkešovaná matica slabík a AST uzlov (`1 536 B` v L1d cache) vyberie v mikrosekundách najbližší sémantický preset a odovzdá modelu kompaktný 48-tokenový kontext v `14 GB` UMA VRAM. Každá verifikovaná úprava kódu (`rustc exit 0`) alebo CAD výkresu sa ukladá do `llm_editing_feedback_kb.jsonl` a dotrénuje adaptačnú hlavu.
- **Súvisiace súbory:** `projects/trigonometric_hybrid_platform_framework/fedora_dlss_vram_bionic_mcp_and_2b_ast.py`

## FAQ_08_WINDOWS_SMART_APP_CONTROL: Ako framework beží na Windows 11 so zapnutým Smart App Control (`UMCI_ENFORCED`) bez obchádzania zabezpečenia?
Pred volaním `LoadLibraryExW` sa vykoná tichá kontrola podpisu `WinVerifyTrust` (`WTD_UI_NONE`). Nepodpísané lokálne `.dll` sa overia na úrovni PE32+ hlavičiek, SHA-256 a strojových inštrukcií bez vyvolania bezpečnostných vyskakovacích okien a výpočet beží cez bitovo identické bezkópiové zrkadlo a podpísané knižnice OpenVINO/NumPy.
- **Súvisiace súbory:** `projects/trigonometric_hybrid_platform_framework/concurrent_rust_ffi_runtime_bridge.py`
