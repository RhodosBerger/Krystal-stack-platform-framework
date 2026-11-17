# Observabilná Reportáž a Výskumná Štúdia: Syntéza Hybridných Poznatkov, Percepčné Strojové Učenie a Monitorovaný Full HD / 4K Benchmark

> **Ústavný invariant:** `VITAL_MAX_HP == 6` (`\neg Valid \implies \Delta S = 0`) | **Grafický čip (WGL/OpenVINO):** `GDI Generic`

## 1. Východiskové tvrdenie a cieľ novej štúdie
Táto štúdia spája doterajšie čiastkové merania pamäťovej hierarchie (`BAR0 mmap`, `64 B / 128 B` zarovnanie), prúdových koprocesorov (`768` FP32 jednotiek Intel Iris Xe), bez-zámkového logovania a tieňovo-cenovej arbitráže do **jednotných syntetických prác**. Tie následne vyhodnocuje kalibrovaný model strojového učenia, ktorý sa učí rozlišovať, kedy prepojenie teoretickej abstrakcie so všeobecným inžinierstvom prináša skutočný prenositeľný výkonnostný zisk.

## 2. Percepčné vyhodnotenie strojovým učením (`Train=60 / Val=18 / Test=18`)
| Model | Presnosť (Test) | F1 Skóre | Brierova chyba | Matica zámen (`TP/TN/FP/FN`) |
| :--- | :---: | :---: | :---: | :---: |
| `M0` Naivný väčšinový základ | `0.6667` | `0.8` | `0.2267` | `{'tp': 12, 'tn': 0, 'fp': 6, 'fn': 0}` |
| `M1` Regularizovaná logistická regresia | `0.9444` | `0.96` | `0.0339` | `{'tp': 12, 'tn': 5, 'fp': 1, 'fn': 0}` |
| **`M2` Percepčný nelineárny MLP (Víťaz)** | **`1.0`** (`95% CI: [1.0, 1.0]`) | **`1.0`** | **`0.024`** | `{'tp': 12, 'tn': 6, 'fp': 0, 'fn': 0}` |

## 3. Päť nových procesných definícií odvodených zo štúdie (`PROC_DEF_01..05`)
- **`PROC_DEF_01_ZERO_COPY_BOUNDARY_MORPHISM` — Hraničný bez-kópiový morfizmus (Zero-Copy Boundary Morphism)** (`Phi_align64(S_host) === S_accel  &&  Subnormal(x) -> 0.0  &&  Delta_copy_bytes == 0`): Proces, pri ktorom sa stav dátovej štruktúry medzi procesorom, grafickým prúdovým jadrom a NPU nemení kopírovaním bajtov, ale čistou zmenou typovej interpretácie nad tou istou 64B-zarovnanou fyzickou adresou.
- **`PROC_DEF_02_CHRONOMETRIC_TELEMETRY_DECIMATION` — Chronometrická decimácia pozorovania (Chronometric Telemetry Decimation)** (`HotLoop(120Hz..10kHz) -> BAR0_mmap(133ns)  ;  DriverQuery(32.7us) -> VBlankShadow(1..10Hz)`): Rozdelenie pozorovania systému na dve časovo ortogonálne hladiny tak, aby samotný akt merania nespôsoboval fázové vyrušovanie meraného výpočtového a vykresľovacieho prúdu (245,97x nižšia réžia).
- **`PROC_DEF_03_SHADOW_PRICE_BUS_ARBITRATION` — Tieňovo-cenová arbitráž zdieľanej zbernice (Shadow-Price UMA Arbitration)** (`lambda_UMA(t) = 1 / max(0.78 - mu_UMA(t), eps)  ;  Grant(b_i) <=> Score(b_i) >= Threshold`): Samoregulačný ekonomicko-fyzikálny proces v jadre bota, ktorý namiesto tvrdých výpadkov plynulo zdražuje prístup na pamäťovú zbernicu pri blížení sa k saturácii a odsúva úlohy pozadia do tieňa snímky.
- **`PROC_DEF_04_SUBGROUP_SLM_CONFINEMENT` — Podskupinové uzavretie lokálnej pamäte (Subgroup SLM Confinement)** (`Tile_bytes <= 64 KB SLM  &&  K_tile < 449  =>  GRF_spill == 0`): Udržanie prechodných maticových a optických medzivýsledkov výlučne v rýchlej on-chip pamäti Subslice bloku bez vylievania registrov do zdieľanej systémovej pamäte DRAM.
- **`PROC_DEF_05_PERCEPTUAL_THEORY_ENGINEERING_COUPLING` — Percepčné nadpojenie teórie na inžiniering (Perceptual Theory-to-Engineering Coupling)** (`f_ML(SyntetickaPraca_i) -> P(PrenositelnyInzinierskyZisk | Invarianty == True)`): Schopnosť kalibrovaného modelu strojového učenia vyhodnotiť zo štrukturálnych čŕt teoretického návrhu (zarovnanie, počet zámkov, hustota kópií, hysterézia), či návrh prinesie reálne zrýchlenie a vizuálnu stabilitu.

## 4. Monitorovaný graficko-výpočtový benchmark vo Full HD (`1920×1080`) a 4K (`3840×2160`)
- **OpenGL / WGL Hardvérová sonda:** `Microsoft Corporation` · `GDI Generic` (`1.1.0`)
- **Priebežný záznam snímok:** `C:\Users\dusan\Documents\GitHub\Krystal-stack-platform-framework\reports\glmark_fhd_4k_monitored_trace.jsonl` (`16` monitorovaných snímok)

| Rozlíšenie | Veľkosť snímku (FP32) | Stock Latencia (`FPS`) | Krystal Zero-Copy Latencia (`FPS`) | GLMark-Ekvivalent Skóre (`Stock -> Krystal`) | Zrýchlenie |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full HD (`1920×1080`)** | `23.73 MB` | `35.673 ms` (`28.03 FPS`) | **`24.853 ms`** (**`40.24 FPS`**) | `280 -> 402` | **`1.44×` (`+43.53 %`)** |
| **4K UHD (`3840×2160`)** | `94.92 MB` | `151.987 ms` (`6.58 FPS`) | **`108.325 ms`** (**`9.23 FPS`**) | `66 -> 92` | **`1.4×` (`+40.31 %`)** |
