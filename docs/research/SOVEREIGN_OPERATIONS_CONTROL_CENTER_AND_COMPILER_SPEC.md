# SOVEREIGN OPERATIONS CONTROL CENTER & COMPILER AGGREGATOR

**Špecifikácia a Používateľská Príručka:** `SOVEREIGN-OPERATIONS-COMPILER-V1`  
**Skript Bash Agregátora:** [`sovereign_operations_compiler.sh`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/sovereign_operations_compiler.sh)  
**Kompilačný a Agregačný Engine:** [`krystal_kernel/sovereign_operations_compiler.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/sovereign_operations_compiler.py)  
**Ústavný Invariant:** `VITAL_MAX_HP == 6` ($\neg\text{Valid}(A) \implies \Delta S \equiv 0$)  
**Theory DAG Uzol:** `THEORY-SOVEREIGN-OPERATIONS-COMPILER` (`DOKAZANE`)  
**Výsledky Testov:** 103/103 testov úspešných za 0.029 s

---

## 1. Účel a Exekutívny Prehľad

Pre potreby centralizovaného riadenia celého ekosystému `Krystal-stack-platform-framework` sme vytvorili **Bash Agregátor pre Riadiace Centrum Prevádzky a Kompilátor**.
Skript [`sovereign_operations_compiler.sh`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/sovereign_operations_compiler.sh) spája kompiláciu formálnych teórií, 128B zero-copy kontraktov a kontrolu fázových tieňových rozpočtov do jedného vysokooptimalizovaného, hermetického príkazu s farebným ANSI výstupom a automatickou detekciou prostredia (WSL2 / Linux / Windows).

```mermaid
graph TD
    A[Spustenie: ./sovereign_operations_compiler.sh] --> B[KROK 1: 6-Fázový Kompilátor a Agregátor]
    B --> S1[1. Audit Theory DAG - 28 uzlov, 0 cyklov]
    B --> S2[2. Kompilácia 128B Zero-Copy Kontraktov]
    B --> S3[3. Validácia 4 Adaptovaných Interpolátorov]
    B --> S4[4. Rekonciliácia 2-Fázového Tieňového Cyklu - 3.5ms]
    B --> S5[5. Prieskum Koordinácie Subsystémov - WSL/Godot/Terrain/SoC]
    B --> S6[6. Syntéza 5-Šošovkovej Redundantnej Inferencie]
    B --> C[KROK 2: Plná Hermetická Regresia - 103 Unit Testov]
    C --> D[KROK 3: Emisia a Overenie Certifikátu HMAC-SHA256]
    D --> E[artifacts/SOVEREIGN_OPERATIONS_CONTROL_RECEIPT.json]
```

---

## 2. Priebeh 6 Kompilačných a Operačných Fáz

Kompilačný engine [`SovereignOperationsCompiler`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/sovereign_operations_compiler.py) vykonáva v KROKU 1 sekvenciu 6 deterministických kontrol:

| Fáza | Názov Fázy | Overovaný Invariant / Podmienka | Nameraný Čas | Výsledok |
| :---: | :--- | :--- | :---: | :---: |
| **1** | `THEORY_DAG_AUDIT` | Acyklickosť grafu teórií, $\ge 25$ uzlov, platnosť rodičovských väzieb | **0.03 ms** | `PASS` |
| **2** | `CONTRACT_FRAME_COMPILATION` | 128-bajtové zarovnanie, HMAC-SHA256 podpis, `BinaryContractType.EVIDENCE` | **0.12 ms** | `PASS` |
| **3** | `INTERPOLATOR_VALIDATION` | Hex-HSL sektory (`0x10`..`0x60`), $\det(\mathbf{J}) \equiv 1.0$, $Z_0 = 50.0\ \Omega$ | **0.13 ms** | `PASS` |
| **4** | `SHADOW_CYCLE_RECONCILIATION`| Výpočtový rozpočet $\le 3.5\text{ ms}$, tlak zbernice $\mu_{\text{UMA}} < 0.78$ | **0.02 ms** | `PASS` |
| **5** | `SUBSYSTEM_COORDINATION_POLL`| 4/4 modernizovaných subsystémov (WSL, Godot, Terrain, SoC) verifikovaných | **0.16 ms** | `PASS` |
| **6** | `REDUNDANT_INFERENCE_SYNTHESIS`| 15 aspektov $\times$ 5 šošoviek (75 claimov), konsenzus $\ge 95\%$ | **0.23 ms** | `PASS` |
| **CELKOVO** | **Agregácia Riadiaceho Centra** | **Všetkých 6 fáz autorizovaných, `vital_max_hp == 6` zachované** | **0.72 ms** | **100% OPERATIONAL** |

---

## 3. Hermetická Regresia a Testovacie Pokrytie

V KROKU 2 spúšťa agregátor kompletnú sadu unit testov pokrývajúcu celý stack:
- [`tests/test_sovereign_operations_compiler.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/tests/test_sovereign_operations_compiler.py) (9 testov)
- [`tests/test_sovereign_runtime.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/tests/test_sovereign_runtime.py) (8 testov)
- [`tests/test_redundant_epistemic_inference_engine.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/tests/test_redundant_epistemic_inference_engine.py) (8 testov)
- [`tests/test_future_theory_integrations_orchestrator.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/tests/test_future_theory_integrations_orchestrator.py) (8 testov)
- [`tests/test_chrono_symplectic_processor_architecture.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/tests/test_chrono_symplectic_processor_architecture.py) (6 testov)
- [`tests/test_pattern_rhythm_cadence_calculator.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/tests/test_pattern_rhythm_cadence_calculator.py) (12 testov)
- [`tests/test_cartesian_hex_hsl_grid.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/tests/test_cartesian_hex_hsl_grid.py) (8 testov)
- [`tests/test_latent_offset_safety_thresholds.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/tests/test_latent_offset_safety_thresholds.py) (10 testov)
- [`tests/test_matrix_algorithms_two_phase_bridge.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/tests/test_matrix_algorithms_two_phase_bridge.py) (8 testov)
- [`tests/test_two_phase_reasoning_computing_alternator.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/tests/test_two_phase_reasoning_computing_alternator.py) (8 testov)
- [`tests/test_uma_inductive_voltage_governor.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/tests/test_uma_inductive_voltage_governor.py) (8 testov)
- [`tests/test_silicon_telemetry_lexicon.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/tests/test_silicon_telemetry_lexicon.py) (5 testov)
- [`tests/test_hw_acceleration_bridges.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/tests/test_hw_acceleration_bridges.py) (5 testov)

**Výsledok:** **103/103 testov zelených za 0.029 s** s nulovým zlyhaním.

---

## 4. Auditný Certifikát Riadiaceho Centra (SOVEREIGN_OPERATIONS_CONTROL_RECEIPT.json)

V KROKU 3 agregátor vygeneruje a overí nemenný JSON certifikát v priečinku `artifacts/`:
- **Cesta k súboru:** [`artifacts/SOVEREIGN_OPERATIONS_CONTROL_RECEIPT.json`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/artifacts/SOVEREIGN_OPERATIONS_CONTROL_RECEIPT.json)
- **Štruktúra certifikátu:**
  * `receipt_id`: Unikátny časový a kryptografický identifikátor.
  * `timestamp_ns`: Monotónny hardvérový čas.
  * `stages_passed` / `total_stages_run`: Pomer úspešných etáp (6/6).
  * `is_fully_operational`: Globálny booleovský stav (striktné `True`).
  * `system_metrics`: Počet kanálov (6), rozpočet VBlank tieňa ($3.5\text{ ms}$), impedancia zbernice ($50\ \Omega$), počet uzlov Theory DAG (28).
  * `hmac_digest`: SHA-256 kryptografický odtlačok garantujúci nepopierateľnosť.
  * `vital_max_hp`: Vždy rovné ústavnému číslu `6`.

---

## 5. Príručka Použitia a Dostupné Prepínače

```bash
# 1. Spustenie plného kompilačného a operačného cyklu (Kroky 1 až 3)
bash sovereign_operations_compiler.sh

# 2. Rýchla kompilácia a overenie teórií a kontraktov (Krok 1)
bash sovereign_operations_compiler.sh --compile-only

# 3. Spustenie cez Python priamo v prostredí Windows alebo Linux
python -m krystal_kernel.sovereign_operations_compiler
```
