# Technická Špecifikácia pre Programátora: Pipeline Uzly, L1/L2/L3 Cache, CRUD API, Rubikova Grupa, UMA Matice a `dusan-game`

- **Ústavný kontrakt:** `VITAL_MAX_HP == 6` (`ContractViolation` pri porušení, `0` holých `assert`)
- **Rád permutačnej grupy členenia výstupu (3×3×3 Rubik):** `43252003274489856000` stavov (aktuálny index stavu: `34573442916612898624`)

## Presný tok vstupov a výstupov jazykového modelu `Gemma 4 E2B` (`2B`)
1. **Vstup z UI a kódu (`Raw Input`):** `dict(ui_control_text: str, code_snippet: str)`.
2. **Predkešovaný enkóder (`SyllableAstPrecachePipeline`):** Konvertuje znakové 3-gramy (sloty `0..47`) a počty syntaktických uzlov Python `ast` (sloty `48..63`) na normalizovaný vektor `float32[64]` a vynásobí ho L1d-rezidentnou maticou `float32[6, 64]` (`1 536 B`) za **`269.8 µs`** -> vybraný preset `PRESET_SEM_06_DUSAN_GAME_AXIOM`.
3. **Kompaktný prefill vstup do `Gemma 4 E2B`:** Namiesto opakovaného prepočítavania tisícok tokenov dostáva model predviazaný tenzor `input_ids: int32[1, 48]` + bezkópiovú `int8` KV-cache v `14 GB` UMA VRAM.
4. **Výstup (`Output`):** `logits: float32[1, 48, 256000]` filtrované symbolickou maskou + kompilovateľný `#![no_std]` Rust modul + štruktúrovaný `.md` výstup zoradený podľa permutácie stien `D_FALSIFIER_GATE, U_SIGNATURE_ABI, B_CACHE_PREFETCH, R_PIPELINE_STAGE, L_OUTPUT_LOGITS, F_INPUT_TENSORS`.

## Namerané technické metriky subsystémov
- **L1d / L2 / L3 priepustnosť:** `L1d (32 KB) = 2.7 GB/s`, `L2 (512 KB) = 10.6 GB/s`, `L3 (4 MB) = 2.97 GB/s` -> zvolená politika `POLICY_L1D_ZERO_GC_128B_RING`.
- **UMA maticový násobič s hardvérovým prefetcher-om (`64B` krok vs. nepriamy scatter):** **`15.68×`** (`2.4443 ms -> 0.1558 ms`).
- **Štatistický ML model (`Test R² / RMSE`):** Naivný priemer `RMSE=1.13875 ms` -> Polynomiálny interakčný model **`RMSE=0.04017 ms`** (`R²=0.99875`).
- **Akceleračné API pre `dusan-game` (`Axiom Engine / Poslední Kmen`):** `2483.0 stavov/ms` na 19-hexovom poli (`103.1 µs` pre dávku `256` stavov).
- **SQLite WAL CRUD databáza (`krystal_crud_endpoints.sqlite3`):** `2` aktívnych API záznamov.
