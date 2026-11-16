# Epistemická kremíkovo-neurosymbolická informatika (`HENC-2026`)

> **Ústredná téza:** Štatistický neurónový model je len navrhovateľ hypotéz a sekvencií (PROPOSER), ktorého výpočtová stopa na kremíku (cache linajky, FMA poradie, UMA priepustnosť) aj pravdivostná hodnota (citácie faktov, Freivalds, invarianty stavu) musia byť nezávisle overené pred akýmkoľvek zápisom do stavu (VERIFIER).

## 1. Sedem pilierov odboru (Teoretická genéza učenia)

- P1 Fyzika reprezentácie čísel: IEEE-754 neasociatívnosť, FMA vs. bloková redukcia (K=449) a FTZ/DAZ pri subnormáloch.
- P2 Operátorová dualita a hranica invertovateľnosti: adjungovaná konvolúcia K^T, Krylov CGLS a strata hodnosti za ReLU.
- P3 Pravdepodobnostná kontrola v subkubickom čase: Freivalds O(k*n^2) s hornou hranicou chyby 2^-k.
- P4 Amortizovaná teória heterogénneho prepínania: C(d,n) = T_compile + n*T_median, CACHE_DIR a strop zbernice UMA < 0.78.
- P5 Hysteretické riadenie s hornou semivarianciou: okamžitá ochrana pri preťažení, 12-kroková zotrvačnosť pri návrate.
- P6 Pamäťová sémantika 64 B / 128 B rámcov: little-endian binárne zmluvy s 32 B SHA-256 odtlačkom.
- P7 Verifikovaná generatívna vrstva: kauzálny transformátor viazaný na formálnu bázu faktov a pravidiel.

## 2. Natrénovaný a do OpenVINO skompilovaný `NeuroSymbolicMicroLLM`

> [!NOTE]
> Ide o kompaktný kauzálny mikro-transformátor trénovaný analytickými gradientmi priamo na overených faktoch repozitára (nie o miliardový predtrénovaný základný model). Každá vygenerovaná veta prechádza symbolickým verifikátorom `verify_answer()`.

- **Parametre**: `7104` (`d_model=32`, `vocab=35`, `seq_len=24`)
- **Pokles stratovej funkcie (Cross-Entropy)**: `3.6619 -> 0.2752` (`60` epoch, `LOCAL_MEASURED`)
- **Cieľ kompilácie `opset13`**: `CPU`

### Ukážky generovania pod dohľadom symbolického verifikátora

| Prefix | Vygenerovaná veta (OpenVINO `opset13`) | Verdikt `verify_answer` |
|---|---|---|
| `NumPy BLAS` | `NumPy BLAS threshold K equals 449 [F_C06_threshold_k].` | `SUPPORTED_WITH_SCOPE` |
| `CPU median` | `CPU median latency is 3.519 ms and GPU median is 3.24 ms [F_C03_CPU_median_ms] [F_C03_GPU_median_ms].` | `SUPPORTED_WITH_SCOPE` |
| `Sync graph` | `Sync graph check verdict is VERIFIED_WITH_SCOPE [F_C01_verdict].` | `SUPPORTED_WITH_SCOPE` |
