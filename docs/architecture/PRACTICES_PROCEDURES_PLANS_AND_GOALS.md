# Praktiky, postupy, celodenný plán, merateľné góly a riadiace mikroprompty

- **Ústavný invariant:** `VITAL_MAX_HP == 6` · ochrana zbernice `UMA < 0.78`
- **Overený 120 Hz rozpočet dejov:** `3.315 ms / 8.333 ms`
- **Kontrolný bod autonómneho exekútora:** `3` krokov dokončených, `1` bezpečne odložený pri 15 % tokenovej rezerve (`SHA-256: 3b842e287b25837c...`)

---

## 1. Overené inžinierske praktiky (`PRAC_01..PRAC_04`)

| ID | Praktika | Záväzné pravidlo v projekte |
| :--- | :--- | :--- |
| `PRAC_01_ZERO_SUBNORMAL_INGEST` | **Vstupná sanitizácia float32 pred každým SIMD/OpenVINO uzlom** | Každá externá súradnica a váha prechádza cez canonicalize_float32, čím sa predchádza 14,0x–37,4x mikroinštrukčnému spomaleniu. |
| `PRAC_02_CACHELINE_128B_CONTRACTS` | **Binárne 128 B rámce (2 x 64 B cache riadky) namiesto JSONu** | Medzimodulová výmena v kritickej slučke používa výhradne KPKT/ZeroCopyContractFrame so SHA-256 (5,04x zrýchlenie). |
| `PRAC_03_CHRONOMETRIC_UMA_GUARD` | **Ochrana zdieľanej zbernice UMA < 0.78 s hysteréziou 12 vzoriek** | Okamžitý zoskok OP_01 na CPU_4T pri špičke (0 rámcov meškania) a opätovné zapnutie GPU až po 12 čistých vzorkách. |
| `PRAC_04_SEPARATED_PROPOSER_VERIFIER` | **Prísne oddelenie návrhu od certifikácie (PROPOSER != VERIFIER)** | Žiadny LLM ani simulátor necertifikuje vlastný výstup; vyžaduje sa Freivalds O(n²), pytest ReceiptContract alebo citácia [F_...]. |

---

## 2. Štandardizovaný pracovný postup (`PROC_01..PROC_04`)

| Krok | Postup |
| :--- | :--- |
| `PROC_01_OBSERVE_AND_ANCHOR` | Zamkni invarianty (VITAL_MAX_HP == 6, UMA < 0.78) a načítaj lokálnu telemetriu bez predpokladov z pamäte. |
| `PROC_02_CHEAPEST_DISCRIMINATING_TEST` | Formuluj hypotézu s 5-ticou (Mechanizmus, Efekt, Baseline, Predikcia, Falzifikátor) a spusti najlacnejší CDT test. |
| `PROC_03_SURGICAL_IMPLEMENTATION` | Vykonaj minimálny reverzibilný zásah so zárukou ~Valid(Action) => Delta_S == 0. |
| `PROC_04_VERIFY_AND_CHECKPOINT` | Over pozitívne aj hraničné testy (pytest), zapíš SHA-256 doklad do reports/ a ulož git commit pred ďalším blokom. |

---

## 3. Merateľné góly a falzifikátory (`G1..G6`)

| Gól | Cieľová metrika | Aktuálne zmeraný stav | Podmienka okamžitého vyvrátenia (`Falzifikátor`) |
| :--- | :--- | :--- | :--- |
| `G1_120HZ_DETERMINISM` | Súčet všetkých 13 real-time dejov v tiku <= 8.33 ms (120 Hz) | **2.835 ms (SPLNENÉ, rezerva 5.498 ms)** | Akýkoľvek real-time tik presiahne 8.33 ms alebo spôsobí výpadok scanoutu pri UMA >= 0.78. |
| `G2_ZERO_UNBOUND_NOMENCLATURE` | 0 neukotvených pojmov a 100 % AST overenie príkazov CALL v engine | **63/63 pojmov ukotvených, 0 neviazaných, AST skenuje všetky moduly (SPLNENÉ)** | Prítomnosť čo i len jedného pojmu bez viazanej funkcie/testu v nomenclature_registry.py. |
| `G3_MULTI_ARENA_THROUGHPUT` | >= 5 súbežných 60 Hz arén (4 hráči, 37 hexov) na jedinom mobilnom CPU | **7 súbežných 60 Hz arén (13 362,7 potvrdených ťahov/s, SPLNENÉ)** | Pokles priepustnosti ClashGame.apply pod 9 750 rámcov/s. |
| `G4_GODOT_VR_LAYER_DENSITY` | 10/20 aktívnych vrstiev v Godote 4 (0x3ff) pri UMA < 0.78 a 5/20 (0x1f) v tieni | **10/20 a 5/20 overené v .tscn exporte aj DSL parseri (SPLNENÉ)** | Prekročenie masky 0x1f počas stavu UMA >= 0.78. |
| `G5_SOVEREIGN_ZERO_OPENAI` | 100 % lokálna prevádzka mosta Antigravity <-> Carbon + Gemma/OpenVINO bez OpenAI | **ContractViolation pri pokuse o provider='openai' + 4 disjunktné sady Gemma (SPLNENÉ)** | Akékoľvek volanie externého OpenAI endpointu alebo priesak medzi train a eval splitom. |
| `G6_TOKEN_EXHAUSTION_CONTINUITY` | Plynulé vykonávanie pracovných blokov až do tokenového stropu s nulovou stratou stavu | **Riadené cez EpistemicMicroPromptRouter (MP_01..08) + SHA-256 checkpointy (AKTÍVNE)** | Ukončenie bloku pri vyčerpaní tokenov bez uloženého JSON checkpointu a čistého stavu. |

---

## 4. Celodenný plán vykonávania (`BLK_01..BLK_06`)

| Blok | Časové okno | Fáza | Stav | Výstup bloku |
| :--- | :---: | :---: | :---: | :--- |
| `BLK_01_MORNING_SILICON` | `09:45 – 10:15` | `R1` | **`COMPLETED`** | OpenVINO konvolučný reťazec, FMA sčítacie poradie, prah OpenBLAS K=449 a Freivalds O(n²). |
| `BLK_02_MIDDAY_ARENA_AND_SHELL` | `10:15 – 12:00` | `R2` | **`COMPLETED`** | Clash Battle Royale (37 hexov, 128 B potvrdenia), heterogénny CLI shell, DxDiag a 126,4x AST zrýchlenie. |
| `BLK_03_AFTERNOON_VR_GONIO_PARSER` | `12:00 – 14:50` | `R2 + R3` | **`COMPLETED`** | Merania X01..X05, goniometrické jadro DESC_01..03, Godot 10/20 vrstiev, .kopt, 4 živé štúdiá a DSL/AST parser. |
| `BLK_04_CROSS_LAYER_AND_EXECUTOR` | `14:50 – 16:30` | `R4` | **`COMPLETED_NOW`** | Medzivrstvové kopiloty XLP_01..04 + TokenBudgetedAutonomousExecutor a router mikropromptov MP_01..08. |
| `BLK_05_GEMMA_ADAPTER_AND_LOD_SWEEP` | `16:30 – 19:00` | `R3 + R5` | **`QUEUED_READY`** | Dávkový lokálny tréning adaptérov nad gemma_neurosymbolic_finetune_sets.jsonl a viac-dlaždicový MultiMesh stress-test. |
| `BLK_06_LONG_HORIZON_GOAL_STRESS` | `19:00 – vyčerpanie tokenov (/goal)` | `R4 + R5` | **`QUEUED_READY`** | Dlhodobý záťažový beh 7 súbežných 60 Hz arén, regresný audit všetkých modulov a finálny SHA-256 checkpoint. |

---

## 5. Sada riadiacich mikropromptov (`MP_01..MP_08`) až do vyčerpania tokenov
> Podľa `AGENTS.md` §10 nepoužívame jeden monolitický gigaprompt, ale špecializované mikroprompty prepínané routerom `EpistemicMicroPromptRouter`. Pre dlhý autonómny beh v chate odporúčame príkaz **`/goal`**, a pri priblížení k limitu kontextového okna príkaz **`/compact`**.

| ID | Typ úlohy | Fáza | Riadiaci mikroprompt | Stop podmienka & Checkpoint |
| :--- | :--- | :---: | :--- | :--- |
| `MP_01_SILICON_CDT_PROBE` | `silicon_benchmark` | `R1` | Zamkni VITAL_MAX_HP==6. Zmeraj n>=10 vzoriek (medián, p95, stdev) na i5-1135G7 + Iris Xe, over canonicalize_float32 a zapíš výsledok s triedou LOCAL_MEASURED do reports/. | Ulož JSON stopu ak p95 <= rozpočet a rel_err < 1e-3. |
| `MP_02_DSL_AND_AST_GATE` | `parser_and_contracts` | `R2` | Rozober vstupné body HEX/PT3/RAY/LAYER_MASK/CALL cez deterministický parser, over každý CALL voči AST katalógu a pri neplatnom vstupe vyhoď ContractViolation so zárukou Delta_S == 0. | Spusti pozitívny aj hraničný pytest test pred zápisom. |
| `MP_03_GONIOMETRIC_GODOT_SYNC` | `spatial_and_vr_render` | `R3` | Prepočítaj DESC_01..03 z uhla theta (|sin²+cos²-1| < 1e-12), nastav masku vrstiev Godot na 0x3ff (UMA < 0.78) alebo 0x1f (UMA >= 0.78) a zosynchronizuj .kopt Fresnel záznamy. | Over platnosť .tscn a .kopt SHA-256 hashu. |
| `MP_04_UMA_HYPERVISOR_BACKBRANCH` | `hypervisor_and_qos` | `R4` | Pri UMA >= 0.78 aktivuj OP_01_BACKBRANCH_UMA_FALLBACK: odpoj GPU_ALLOCATE, skráť rámce 16->8, zníž kvótu WSL2 RAM o 2 GB a odošli 128 B KPKT paket s 12-vzorkovou hysteréziou návratu. | Potvrď sha256_verified == True na všetkých 128 B paketoch. |
| `MP_05_CROSS_LAYER_COPILOT_AUDIT` | `cross_layer_integration` | `R4` | Každé premostenie do DCC, Robotiky, WSL2 stanice a IDE označ najprv ako PATTERN_CANDIDATE so 6-ticou, spusti izolovaný diskriminačný beh a povýš status iba pri preukázanom výsledku. | Zapíš cross_layer_pattern_demo.json a over všetky 4 vrstvy. |
| `MP_06_GEMMA_SOVEREIGN_FINETUNE` | `local_ml_training` | `R5` | Over nulový priesak medzi train a eval v gemma_neurosymbolic_finetune_sets.jsonl, zakáž provider='openai', trénuj lokálne váhy/adaptéry a vyžaduj citáciu [F_...] pri každom faktickom výstupe. | Zaznamenaj počiatočnú a koncovú stratu (loss) + eval presnosť. |
| `MP_07_MULTI_ARENA_STRESS_LOOP` | `endurance_stress` | `R4 + R5` | Spusti záťažovú slučku 7 súbežných 60 Hz arén (ClashGame + tenzorová difúzia D⁻¹A), sleduj p95 latenciu a pri odchýlke Freivalds rel_gap > 1e-4 okamžite aktivuj OP_03_FREIVALDS_RESIDUAL_TRAP. | Ulož priebežnú štatistiku každých N kôl do reports/. |
| `MP_08_TOKEN_CAP_HANDOFF_GUARD` | `token_budget_guard` | `ALL` | Keď zostávajúci rozpočet tokenov klesne pod bezpečnú rezervu (<= 15 %), nespúšťaj novú neuzavretú vetvu: dokonči bežiaci test, zapíš SHA-256 checkpoint s presným ďalším krokom a odporuč /compact alebo /goal. | Zapíš autonomous_day_checkpoint.json a ukonči krok čisto. |
