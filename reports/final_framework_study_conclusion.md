# Formálne zakončenie štúdie: Krystal Trigonometric & Neuro-Symbolic Platform Framework

- **Konvolučný index genézy**: `4.793 / 6.000` (`VITAL_MAX_HP == 6`)
- **Vizuálna projekcia architektúry**: `reports/final_framework_projection.svg`

## 1. Citované a empiricky overené výsledky

- [X01_CACHE_HIERARCHY_CLIFF] Nameraný pomer priepustnosti L1d voči UMA DRAM = 1.4x (n=15).
- [X03_SUBNORMAL_ASSIST_VS_FTZ_DAZ] Potvrdené rozdielne správanie subnormálov: NumPy spomalenie 14.004x vs. OpenVINO oneDNN FTZ/DAZ nulovanie=False.
- [X05_LITTLE_ENDIAN_128B_VS_JSON] Binárny 128 B rámec so SHA-256 je 5.038x rýchlejší než UTF-8 JSON.
- [F_C06_threshold_k] Zúžený zákon OpenBLAS sekvenčného FMA pre K < 449 platí bez protipríkladu na všetkých otestovaných profiloch.
- [PHASE_POOL_ZERO_REALLOC] Predalokovaný 4-fázový pool vykonal 5-uzlový graf textúr s reallocated_bytes == 0 (SHA-256=54d2dc286dbe).
- [SYNTHETIC_NET_2B_TRAIN] Sieť syntetického poznania (22 trojíc) znížila krížovú entropiu modelu z 3.6619 na 0.2752.

## 2. Vzorce rozkladu integrácií do presností

- **`F_PREC_01_CROSS_DEVICE_DRIFT`**: `eps_rel(CPU, GPU) = ||y_CPU - y_GPU||_2 / max(||y_ref||_2, 1e-12) < 1e-3` — Oddeľuje neškodnú neasociatívnosť FMA (~2.5e-6) od skutočného porušenia výpočtu.
- **`F_PREC_02_FREIVALDS_SOUNDNESS`**: `P(accept | A*B != C) <= 2^(-rounds)  (pri rounds=6 je horná hranica 0.015625)` — Subkubická O(rounds * n^2) garancia správnosti maticového vetvenia a operátorov.
- **`F_PREC_03_AMORTIZED_CROSSOVER`**: `n* = (T_compile(GPU) - T_compile(CPU)) / (T_median(CPU) - T_median(GPU))` — Presná hranica počtu inferencií, kde perzistentná CACHE_DIR preklápa výhodu na Iris Xe.
- **`F_PREC_04_CITATIONAL_GROUNDING`**: `Grounding(S) = |{s in S : CitesExist(s) and NumbersMatch(s)}| / |S|` — Kvantifikuje presnosť neuro-symbolického výstupu bez tichej tolerancie halucinácií.

## 3. Zachované otvorené rezíduá a hranice (`UNKNOWN / CONFLICT`)

- [C08_NGINX_PORT_MISMATCH] V deploy/nginx_krystal_subdomain_ssl.conf ostáva otvorený nesúlad portu 8089 vs. 8090 (zámerne neprepísané ticho).
- [UNKNOWN_RAPL_MSR] Priame hardvérové čítanie joulov (RAPL) a frekvencií jadier pod AVX-512 vyžaduje jadrový ovládač mimo používateľského priestoru.
