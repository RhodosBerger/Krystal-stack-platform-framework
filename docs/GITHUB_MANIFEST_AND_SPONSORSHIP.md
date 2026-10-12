# KRYSTAL-STACK: GITHUB REPOSITORY MANIFEST, COMMERCIAL EVIDENCE & SPONSORSHIP (2026/2027)

Dokumentácia pripravená na okamžité použitie pre GitHub profil, popisy repozitára, Release Notes, sociálne siete a komunitnú i firemnú výzvu na licencovanie a sponzorstvo.

---

## 1. Popisky na GitHub (Repository Descriptions & Metadata)

### A. Krátky Popis (GitHub "About" Box – max 350 znakov)
```text
Krystal-Stack 2026: Heterogeneous compute & cyber-physical framework governed by Commercial Evidence Levels. Features CrystalSec 3.0 (UMCI-trusted native SIMD/Vulkan break-even), KATG-006 CNC adaptive thermal governor (scrap-risk reduction & SSV), local Edge AI (Granite/OpenVINO), Janet Lisp DSL, Godot 4.x procedural engine & VorpX VR (Ncon 130°).
```

### B. Zoznam Štítkov (GitHub Topics / Tags)
Skopírujte do poľa **Topics** v nastaveniach repozitára:
```text
crystalsec, vulkan, simd, avx2, vnni, edge-ai, cnc-governor, cyber-physical, openvino, directml, janet-lang, godot4, vorpx-vr, ncon-vr, procedural-generation, high-rpm-machining, scrap-risk-reduction, hardware-integrity
```

### C. Webstránka Repozitára (Website URL)
```text
https://github.com/RhodosBerger/Krystal-stack-platform-framework
```

---

## 2. Architektonické Piliere & Úrovne Komerčnej Evidencie

Všetky subsystémy Krystal-Stacku sú klasifikované podľa **Commercial Evidence Frameworku**:
- `PRODUCTION_AVAILABLE`: Plne nasaditeľné, otestované v test suite, empiricky verifikované.
- `PILOT_AVAILABLE`: Pripravené na zákaznícke pilotné programy, integračné inžinierstvo a zmluvné vyhodnotenie.
- `SIMULATION_VALIDATED`: Overené v dynamických modeloch; hardvérová validácia prebieha.

```
                                      KRYSTAL-STACK ARCHITECTURE
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                                   MISSION CONTROL HUB & NOC                                     │
 │        (Interaktívne Web Štúdiá: Fleet Telemetria, CNC Monitor, Kampaň, Ncon VR, Trhy)          │
 └────────────────────────────┬──────────────────────────────────────┬─────────────────────────────┘
                              │                                      │
 ┌────────────────────────────▼───────────────────────┐ ┌────────────▼─────────────────────────────┐
 │            JANET LISP PROCEDURAL DSL               │ │    GODOT 4.x PROCEDURAL ENGINE            │
 │  20 validovaných skriptov: balistika, variety,     │ │  Generovanie .tscn scénických stromov,   │
 │  rezonancia totémov, 256k stavov, 6 Max HP         │ │  PBR materiály, SDF raymarching terrain   │
 │  (Maturity: PRODUCTION_AVAILABLE)                  │ │  (Maturity: PRODUCTION_AVAILABLE)         │
 └────────────────────────────┬───────────────────────┘ └────────────┬─────────────────────────────┘
                              │                                      │
 ┌────────────────────────────▼──────────────────────────────────────▼─────────────────────────────┐
 │                         HETEROGENEOUS COMPUTE & HARDWARE INTEGRITY                              │
 │ • CrystalSec 3.0: Windows Device Guard / UMCI Trusted AVX2/VNNI + Vulkan (PRODUCTION_AVAILABLE)│
 │ • KATG-006 CNC: Adaptive Thermal Governor (SSV chatter dampening, scrap reduction) (PILOT)      │
 │ • Local Edge AI: IBM Granite 4B SLM · DirectML NPU · OpenVINO · 0 Cloud API Fees (PRODUCTION)   │
 └─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Kľúčové Komponenty v Repozitári:
1. **CrystalSec 3.0 Enterprise Hardware Security & Crossover Engine (`PRODUCTION_AVAILABLE`)**:
   - `src/c/taxonomy_simd.c` (AVX2 widening bez saturácie, native VNNI `vpdpbusd`).
   - `src/python/native_binary_trust_inspector.py` (overenie Device Guard / UMCI a WinVerifyTrust).
   - `scripts/benchmark_native_cpu_vs_vulkan.py` (fyzická QPC mikroarchitektonická izolácia a kalibrácia $N^*_{\text{e2e}}$).
   - `artifacts/CRYSTALSEC_V3_TRUE_CPU_GPU_BREAK_EVEN.json`.
2. **KATG-006 Cyber-Physical CNC Suite (`PILOT_AVAILABLE`)**:
   - `krystal_kernel/cnc_adaptive_thermal_governor.py` (spindle speed variation pri $n \ge 12\,000\,\text{RPM}$ na tlmenie rezonancií).
   - Workflow na zníženie rizika nepodarkov (**Scrap-Risk Reduction**) cez diferenciálnu opravu G-kódu pri výpadku chladenia alebo teplotných varovaniach.
   - Diskrétny model vitality nástrojov **`ToolHealthState ∈ [0,6]`**.
3. **Autonómna Edge AI & Heterogénna Orchestrácia (`PRODUCTION_AVAILABLE`)**:
   - Lokálna inferencia SLM (IBM Granite 4B, Llama 3.2, DeepSeek) bez nutnosti platených cloudových tokenových API.
   - MultiLock Matrix Autocompleter s RAG kvótami a nulovou závislosťou od externých serverov.
4. **Vzdušný Bojový Systém & Kinetika (Godot 4.x / Janet DSL / VorpX VR)**:
   - 20 validovaných Janet DSL skriptov v `krystal_janet/` (`PRODUCTION_AVAILABLE`).
   - Vznášajúce sa ostrovy s 240mm mínometom, rogala s kĺzavosťou 7.5:1 a padáky.
   - VorpX VR stereoskopia a podpora ultra-širokouhlých headsetov Ncon by Korrado (130° FOV, `PILOT_AVAILABLE`).

---

## 3. Komerčné Licenčné Úrovne & Sponzorstvo (2026/2027)

Pre detailné informácie si prečítajte:
- 📄 **[MONETIZATION_PLAN.md](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/MONETIZATION_PLAN.md)**: Komplexná biznis stratégia a rozpis ROI.
- 🛡️ **[COMMERCIAL_SUPPORT.md](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/COMMERCIAL_SUPPORT.md)**: Úrovne zmluvnej podpory (SLA) a konzultačné služby.

### Prehľad Licenčných Úrovní:

| Úroveň | Cena | Cieľová Skupina | Kľúčové Výhody & Dodávky |
| :--- | :--- | :--- | :--- |
| **Community** | **$0** (Free) | Študenti, výskumníci, open-source | Plný Apache 2.0 kód, 20 Janet skriptov, komunitná GitHub podpora |
| **Pro Dev / Indie Studio** | **$49 / mesiac**<br>*(alebo $399 / rok)* | Nezávislí vývojári, herné štúdiá | Komerčná licencia (royalty-free do $250k obratu), nightly buildy, prioritný kanál |
| **CNC Pilot Partner (KATG-006)** | **$249 / mesiac**<br>*(na 1 vreteno)* | CNC obrobne, nástrojárne, aerospace | Pilotný runtime adaptívneho regulátora (SSV), scrap-risk reduction workflow, telemetrické audity |
| **Enterprise Edge / Security** | **$499 / mesiac**<br>*(alebo $4,990 / rok)* | Dátové centrá, obrana, Edge AI | CrystalSec 3.0 UMCI overené binárky, lokálny SLM runtime (Granite), NOC Admin konto, 4h SLA (Severity 1) |
| **Corporate Advisory Partner** | **$20,000 / rok** | Hardvéroví výrobcovia, priemysel | Kreslo v Architecture Advisory Council, bimesačné poradenské hovory, prioritné ladenie jadier |
| **Technical Steering Partner** | **$50,000 / rok** | Polovodičové a globálne korporácie | Hlasovacie právo v Technical Steering Committee, 24/7 eskalácia, zákazkový vývoj jadier na mieru |

---

## 4. Platobné Možnosti & Podniková Fakturácia

1. **Priamy Bankový Prevod / Fakturácia (SEPA / SWIFT)**:
   - Oficiálna komerčná faktúra s DPH pre firemných zákazníkov (Net 30 pre zmluvných partnerov).
   - Kontakt: **`dusan.kopecky0101@gmail.com`**
2. **GitHub Sponsors (Okamžitá Aktivácia)**:
   - [github.com/sponsors/RhodosBerger](https://github.com/sponsors/RhodosBerger)
   - Automatické pridelenie rolí, repozitárov a licenčných kľúčov.
3. **Alternatívne Web3 / Krypto Zúčtovanie (Voliteľné)**:
   - **USDT (TRC20)** / **Bitcoin (BTC)** / **Ethereum (ETH)**: Dostupné na požiadanie pre medzinárodných technologických partnerov.

---

## 5. Kontakt na Hlavného Architekta

- **Hlavný Architekt & Vývojár**: Dušan Kopecký ([@RhodosBerger](https://github.com/RhodosBerger))
- **E-mail**: `dusan.kopecky0101@gmail.com`
- **GitHub**: [github.com/RhodosBerger](https://github.com/RhodosBerger)
