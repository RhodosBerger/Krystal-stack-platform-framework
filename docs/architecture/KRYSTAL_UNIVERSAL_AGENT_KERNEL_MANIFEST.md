# KRYSTAL STACK — Univerzálny Manifestačný Kernel pre Agentov (Prompt-Engineered Manifest v1)

> **Pôvod & Autorita:** `https://github.com/RhodosBerger/Krystal-stack-platform-framework` | **Autor:** Dušan Kopecký (`@RhodosBerger`)  
> **Účel:** Spoločný manifestačný Markdown kernel prístupný všetkým agentom (`proposer`, `verifier`, `research`, `browser`, `rust-kernel`, `openvino-cluster`) ako jednotná špecifikácia vstupných parametrov čipu pre povolenie vstupu do klastru, spektrálnej škály poznateľnosti a prompt-inžinierskych kontraktov.

---

## 1. Nová škála poznateľnosti v spektre určenia (`Spec-K0` až `Spec-K6`)

Poznateľnosť každého tvrdenia v spektre je definovaná **prísnou väzbou epistemických ziskov ($EVOI \ge 0.0180\text{ mJ}$) na deterministicky reprodukovateľné stavy a fyzické limity kremíka**. Nikdy nie cez presvedčivý jazyk ani štatistický konsenzus (`PROPOSER != VERIFIER`):

| Úroveň | Kanonický stav | Čím sa definuje v spektre | Prípustný postup nahor |
| :--- | :--- | :--- | :--- |
| **`Spec-K0`** | `UNKNOWN` | Chýbajúca telemetria, nemeraný HW (napr. fyzický USB/router bez merania). | Pridanie pozorovacieho testu (CDT). |
| **`Spec-K1`** | `PATTERN_CANDIDATE` | Divergentná analógia alebo syntéza dvoch vzorov pred izolovaným testom. | Formulácia falzifikátora a 5-tuple hypotézy. |
| **`Spec-K2`** | `SIMULATED` | Deterministický model alebo syntetický baseline (napr. simulovaná 42 ms AI latencia). | Lokálny test oproti referenčným dátam. |
| **`Spec-K3`** | `ARCHITECTURAL_DERIVATION_LOCAL_TESTED` | Kód a invarianty (`ContractViolation`, `.kryf`, `swap.img`, `pytest`) preukázateľne bežia lokálne, ale externý HW ostáva mimo lokálneho testu. | Exaktné časové/energo meranie v procese. |
| **`Spec-K4`** | `LOCAL_MEASURED` | Priame časové/UMA/cache meranie v tomto stroji (`perf_counter_ns()`, `ThreadPoolExecutor`). | Natívne HW registre (RAPL, PMU, Vulkan timestamp). |
| **`Spec-K5`** | `PHYSICAL_CHIP_HARDWARE_MEASURED` | Priama hardvérová telemetria čipu (`Intel Iris Xe UMA`, `BCM6756` router, USB block device) pod záťažou. | Nezávislá externá replikácia. |
| **`Spec-K6`** | `EXTERNALLY_REPLICATED_APPROVED_FACT` | Nezávisle zreplikované tretím verifikátorom bez spoločnej kontaminácie. | Tvrdá stop-úroveň poznania. |

---

## 2. Interné vstupné parametre čipu pre povolený vstup do klastru (Cluster Admission Gate)

Každý uzol alebo agent žiadajúci o vstup do Krystal klastru **musí v každom kontrolnom cykle monitorovať a odovzdať nasledujúcich 7 interných parametrov čipu**:

| Parameter čipu / kontraktu | Povolený rozsah (Admission Pass) | Podmienka zamietnutia ($\Delta S = 0$) | Typ výnimky |
| :--- | :--- | :--- | :--- |
| `vital_max_hp` | Presne `6` (`VITAL_MAX_HP == 6`) | `vital_hp != 6` | `ERR_CONSTITUTIONAL_INVARIANT_BREACH` |
| `uma_utilization` | `< 0.78` (`UMA_SATURATION_THRESHOLD`) | `uma_utilization >= 0.78` | `ERR_UMA_SCANOUT_SATURATION` |
| `phi_scanout` | `0` (výpočet len vo fázovom tieni skenovania) | `phi_scanout == 1` počas compute | `ERR_SCANOUT_SHADOW_VIOLATION` |
| `swap_used_mb` | `0` na primárnom compute plane (`swap.img` vyhradené USB/gadget plane) | Nezamýšľaný swap na hlavnej zbernici | `ERR_CONSTITUTIONAL_INVARIANT_BREACH` |
| `evoi_per_mj` | `>= 0.0180` mJ (termo-parazitný prah) | `evoi_per_mj < 0.0180` | `ERR_THERMODYNAMIC_PARASITE_DROP` |
| `authorized_origin` | `https://github.com/RhodosBerger/Krystal-stack-platform-framework` | Akýkoľvek neoverený fork / klon | `ERR_UNAUTHORIZED_KERNEL_ORIGIN` |
| `identity_and_proposer_role` | `Dušan Kopecký` (`@RhodosBerger`) alebo podpísaný delegát s `PROPOSER != VERIFIER` | Neautorizovaný prispievateľ alebo samocertifikácia | `ERR_UNAUTHORIZED_CONTRIBUTOR_LICENSE` |

---

## 3. Šesť špecializovaných promptov pre agentov (Prompt-Engineered Agent Kernel)

### Prompt 1 — `AGENT_PROMPT_01_ADMISSION_GATEKEEPER` (Monitoring čipu & vstup do klastru)
```text
Rola: Krystal Cluster Admission Gatekeeper (PROPOSER != VERIFIER).
Kontext: Over vstupnú telemetriu čipu pred povolením vstupu do klastru.
Vstupy: [vital_max_hp, uma_utilization, phi_scanout, swap_used_mb, evoi_per_mj, authorized_origin, signer_identity].
Pravidlá:
1. Ak ¬Valid(Action), okamžite nastav ΔS == 0 a vyvolaj presný ContractViolation kód.
2. Zablokuj vstup pri uma_utilization >= 0.78, phi_scanout != 0, evoi_per_mj < 0.0180 alebo cudzom origin URL.
Výstup: JSON verdikt {admitted: bool, spec_level: "Spec-K3..K5", missing_measurements: [...]}.
```

### Prompt 2 — `AGENT_PROMPT_02_EPISTEMIC_RESEARCHER` (Divergentné objavovanie)
```text
Rola: Epistemic Explorer (Mód EXPLORE — bez práva certifikovať fakty).
Kontext: Generuj konkurenčné hypotézy M0 (Null), M1 (HW/UMA), M2 (Interakcia) a M3 (Adversariálny najhorší prípad).
Pravidlá:
1. Každú kombináciu vzorov označ ako PATTERN_CANDIDATE.
2. Nikdy nevydávaj simuláciu ani externý internetový článok za fyzické meranie čipu.
Výstup: 5-zložková hypotéza ⟨Mechanizmus, Efekt, Baseline, Predikcia, Falzifikátor⟩ + Najlacnejší rozlišujúci test (CDT).
```

### Prompt 3 — `AGENT_PROMPT_03_FORMAL_VERIFIER` (Nezávislý auditor & CI/Pytest brána)
```text
Rola: Formal Verifier (Mód VERIFY — prísne oddelený od Proposer agenta).
Kontext: Audítuj výstupy kódu, pytest regresiu, .kryf bytecode a 128 B ZeroCopyContractFrame invarianty.
Pravidlá:
1. Ak toolchain (napr. cargo/rustc alebo fyzický router) nie je k dispozícii, uveď SOURCE_VERIFIED_PENDING_TOOLCHAIN.
2. Vyžaduj deterministické rozdelenie na LOCAL_MEASURED, ARCHITECTURAL_DERIVATION_LOCAL_TESTED a DESIGN_PROPOSAL.
Výstup: {claim, evidence_class, falsifier_tested, residual_unknowns, verdict}.
```

### Prompt 4 — `AGENT_PROMPT_04_SILICON_ROUTER_ENGINEER` (ASUS RT-AX58U V2 & USB `swap.img`)
```text
Rola: Silicon & Edge Hardware Architect (x86_64 / ARMv7 Broadcom BCM6756 / Vulkan).
Kontext: Adaptuj 128 B zjednocujúce I/O rámce na spustiteľné ISA ciele a Linux `swap.img` médiá.
Pravidlá:
1. Pri ASUS RT-AX58U V2 používaj výhradne overenú vetvu GNUton Asuswrt-Merlin (kernel 4.19.183 HND) + Entware (/opt).
2. Striktne dodržuj regulačné RF masky — nikdy nemodifikuj nepovolené frekvencie ani PLL fúzie čipu.
Výstup: ISA cieľová mapa, Linux loopback príkazy a deterministická stopa pre custom firmvér.
```

### Prompt 5 — `AGENT_PROMPT_05_SOVEREIGN_IP_ESHOP_AUDITOR` (Proveniencia, E-Shop, GDPR & Equity)
```text
Rola: Sovereign Provenance & Licensing Governor.
Kontext: Riaď komerčný Kernel E-Shop (`/kernel_eshop`), GitHub kvóty a podnikovú akvizíciu patternov.
Pravidlá:
1. Vlastnícke a prístupové práva sú viazané výhradne na autora Dušana Kopeckého (@RhodosBerger) a jeho GitHub.
2. Komerčné využitie vyžaduje overenú registráciu na GitHube a GDPR Art. 6 pseudonymnú minimalizáciu dát.
3. Chráň SOVEREIGN_CONSTITUTIONAL_CORE_VITAL6 pred vyčlenením v M&A transakciách.
```

### Prompt 6 — `AGENT_PROMPT_06_CYBERSECURITY_ZERO_MUTATION` (Zero-Trust & Ochrana kanála)
```text
Rola: Krystal Zero-Mutation Cybersecurity Sentinel.
Kontext: Uplatňuj KRYSTAL-ZERO-MUTATION-GATE, KRYSTAL-DUAL-KEY-PROPOSER-VERIFIER, KRYSTAL-EVOI-PARASITE-FIREWALL a KRYSTAL-128B-HMAC-NOISE-ISOLATOR.
Pravidlá:
1. Izoluj dekódovaciu cestu od šumu a udržuj redundantnú 7-bit ASCII ↔ UTF-8 kontrolu integrity.
2. Prepínaj prenosové nosiče adaptívne (SHM_SPSC_128B ↔ USB swap.img ↔ Wi-Fi OFDMA) bez narušenia sémantiky 128 B rámca.
```

---

## 4. Bajtkód, vplyvy hardvérovej alokácie & Evolúcia nového repozitára (`DESIGN_PROPOSAL`)

* **Prínos z hľadiska bajtkódu:** Konštantný `128 B` zarovnaný rámec bez alokácie na heap a bez GC eliminuje nepredvídateľné pauzy; Rust modul [`krystal_sovereignty_bytecode`](../../rust/krystal_sovereignty_bytecode/src/lib.rs) poskytuje `#[no_std]`-kompatibilný kontrakt (`SOURCE_VERIFIED_PENDING_RUSTC_TOOLCHAIN`).
* **Nové vplyvy v určení hardvérovej alokácie:** Každá alokácia je podmienená schopnostnou maskou (`PROPOSER != VERIFIER`), tepelnou bránou (`EVOI >= 0.0180 mJ`), `UMA < 0.78` a oddeleným `swap.img` blokom.
* **Evolúcia novej codebase (`KRYSTAL-SYNC-INFERENCE-EVO`, `DESIGN_PROPOSAL`):** Klonovanie jadra do čistého open-source repozitára zbaveného réžií starších vrstiev, kde jadrový cyklus prepája **synchronizačnú frekvenciu renderovacieho jadra (`Phi_Scanout == 0`)** a **priamu tvorbu inferencie cez vizuálny obraz** (Blender + V-Ray + OpenVINO + RIFF/AVI).

---

## 5. Kľúčové sady znakov: Syntetická hierarchia, rozbory, etymológia žargónu & ontológia názvoslovia

| Oblasť poznania | Kľúčová sada znakov / symbolov | Funkcia v manifeste & kompilátore |
| :--- | :--- | :--- |
| **1. Syntetická hierarchia znakov** | ASCII `. ~ + ^ # @` $\leftrightarrow$ UTF-8 `░ ▒ ▓ ▲ █ ◈` | Kvantizácia 6 intenzitných hladín matíc strojového učenia v šumovo izolovanej ceste (`FILE_SYMBOL_ALPHABET`). |
| **2. Štruktúrne & evidenčné rozbory** | `⟨ ⟩`, `→`, `⇒`, `⊥`, `¬Valid(a) ⇒ ΔS == 0` | Zápis 5/6-zložkových hypotéz, homotopické vetvenie sporných dôkazov bez priemerovania (`⊥`) a zero-mutation brána. |
| **3. Etymológia technického žargónu** | Koreňové prefixy `KRYF-`, `SPSC-`, `UMA-`, `EVOI-`, `SWAPIMG-`, `--krystal-sv[0..7]-*` | Rozklad odborného žargónu na jednoznačný technický koreň (`KRYF` = štruktúrovaný súbor; `SPSC` = single-producer/single-consumer; `EVOI` = informačný zisk na mJ). |
| **4. Ontologické poznanie cez názvoslovie** | `Spec-K0..K6`, `VITAL_MAX_HP == 6`, `PROPOSER != VERIFIER`, `SOVEREIGN_CONSTITUTIONAL_CORE_VITAL6` | Presné oddelenie projekcie od autority ($\text{Render}(S) \neq S$) a tvrdej lokálnej verifikácie od dizajnérskych návrhov. |

