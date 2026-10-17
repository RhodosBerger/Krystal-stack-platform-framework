# Krystal Sovereign Principle Pack (KSPP v1)
## Desať ústavných a fyzikálnych zákonov kompozitného ekosystému

> **Ústavný invariant:** $\text{VITAL\_MAX\_HP} == 6$  
> **Jadrová definícia:** KSPP v1 mení opis jednotlivých nástrojov na **záväzné architektonické a fyzikálne zákony**. Každý modul (či už Godot, Blender, llama.cpp, Z3 alebo Linux kernel subsystém), ktorý sa pripojí do Krystal Stacku, sa bezvýhradne podriaďuje týmto desiatim suverénnym princípom.

---

## I. Desať suverénnych princípov (KSPP v1)

### Princíp 1: Princíp neinvazívnej suverenity (Zero-Touch Interposition)
* **Zákon:** Nadstavba nikdy nemodifikuje zdrojový kód pripojených open-source projektov ani nezasahuje do ich vnútornej doménovej logiky. Riadenie prebieha výlučne na systémových hraniciach (OS Loader, Vulkan Implicit Layer, DXGI Swapchain, C-ABI zdieľaná pamäť, Cgroups/EcoQoS).
* **Formálny invariant:**
  $$\forall M \in \text{OpenSourceModules}: \quad \text{SourceCodeDelta}(M) \equiv 0$$
* **Komponenty balíka:** Automatický injektor vrstiev (`VK_LAYER_KRYSTAL_NEXUS`), `UniversalEngineInterposer` a procesný wrapper obalujúci binárku bez rekompilácie.

---

### Princíp 2: Princíp 128-bajtovej binárnej univerzálnosti (The Universal 128B Lingua Franca)
* **Zákon:** Žiadne dva moduly v skladačke spolu nekomunikujú cez pomalé textové protokoly (JSON/REST/gRPC) na kritickej ceste. Jediným povoleným nervovým vzruchom medzi grafikou, AI a jadrom OS je 128-bajtový, prirodzene zarovnaný rámec (`_Alignas(64) ZeroCopyContractFrame`) v lock-free SPSC kruhu.
* **Formálny invariant:**
  $$\text{Size}(\text{Frame}) \equiv 128\text{ B} \quad \land \quad \text{BaseAddr} \bmod 64 \equiv 0 \quad \land \quad \text{Publish} \implies \text{Acquire/Release}$$
* **Komponenty balíka:** Unifikovaná hlavička `krystal_nexus_api.h`, moduly `zero_copy_contracts.py`, `zero_copy_ring.py`, a separácia head (`CL0`) a tail (`CL1`) indexov s nulovým false-sharingom.

---

### Princíp 3: Princíp fázového tieňa (Chronometric Shadow Multiplexing)
* **Zákon:** Deterministické vykresľovanie/aktuácia (Doména A/B) má absolútny monopol na zdieľanú pamäťovú zbernicu počas aktívneho scanoutu. Kognitívne a výpočtové procesy (`llama.cpp`, `rustc`, `Z3`) smú pristupovať k zbernici na plný výkon výlučne vo „fázovom tieni“ ($2.2\text{ ms}$ VBlank okná a medzisnímkové bubliny).
* **Formálny invariant:**
  $$\Phi_{\text{Scanout}}(t) \cdot \Phi_{\text{Compute}}(t) = 0 \quad \land \quad \mu_{\text{UMA}}(t) < 0.78$$
* **Komponenty balíka:** `VsyncMicroSlicingScheduler` napojený na hardvérové VBlank hodiny (`DRM_IOCTL_WAIT_VBLANK` / `WaitForVBlank`), `VBlankExperimentInjector` a `UmaBandwidthArbiter`.

---

### Princíp 4: Princíp typovosti poznania (Epistemic Type Safety)
* **Zákon:** Výstupy z jazykových modelov a simulátorov nie sú voľný text, ale prísne typované štruktúry v algebrickej mriežke dôkazov (ENLG v2). Nižší typ dôkazu (`[SIMULATED]`, `[DERIVED]`, `[CODE-PRESENT]`) sa nikdy nesmie implicitne pretypovať na vyšší typ (`[PHYSICAL-MEASURED]`, `[VERIFIED]`) bez priloženého kryptografického potvrdenia.
* **Formálny invariant:**
  $$\text{Level}(\text{Verdict}) \le \min_{e \in \text{Evidence}} \text{AllowedLevel}(e) \quad \land \quad \text{PROPOSER} \not\longrightarrow \text{VERIFIED}$$
* **Komponenty balíka:** `EnlgEpistemicCompiler`, GBNF gramatické masky pre lokálne LLM dekodéry a karanténna brána `ArtifactGate`.

---

### Princíp 5: Princíp termodynamického účtovníctva (Thermodynamic Hypothesis Metering)
* **Zákon:** Každá myšlienka, špekulatívna vetva alebo proces v pozadí má svoju fyzikálnu cenu v jouloch ($\Delta E$) a megabajtoch priepustnosti ($\Delta \text{BW}$). Proces, ktorý spotrebúva energiu a pamäťovú zbernicu bez toho, aby v stanovenom okne znížil informačnú entropiu (posunul stav v `ArtifactGate`), je klasifikovaný ako tepelný parazit a odpojený.
* **Formálny invariant:**
  $$\int_{t_0}^{t_1} P_{\text{RAPL}}(h)\,dt > E_{\text{budget}} \land \Delta \tau(h) = 0 \implies \text{Terminate}(h)$$
* **Komponenty balíka:** `EpistemicCalorimeter` a `PowerLimitGovernor` (s ošetrením MSR bitu 63, MMIO fallbackom a Arrheniusovou tepelnou poistkou $T < 75^\circ\text{C}$).

---

### Princíp 6: Princíp zachovania rozporu (Homotopic Conflict Preservation)
* **Zákon:** Ak dva open-source moduly (napr. fyzikálny simulátor Gazebo a reálny senzor v ROS 2, alebo dva rôzne modely) dodajú protichodné dáta ($E_A \bot E_B$), systém má prísne zakázané ich spriemerovať alebo zahladiť. Rozpor sa musí zachovať ako rozštiepená superpozícia dvoch vetiev až do vykonania najlacnejšieho rozlišovacieho experimentu (CDT).
* **Formálny invariant:**
  $$E_A \bot E_B \implies \text{Fork}(\text{Branch}_\alpha, \text{Branch}_\beta) \quad \land \quad \text{ConvexAverage}(E_A, E_B) \implies \text{ContractViolation}$$
* **Komponenty balíka:** `BifurcatedConflictRing` a generátor diskriminačných experimentov `NEXT_EXPERIMENT`.

---

### Princíp 7: Princíp kauzálneho uzáveru pred komitom (Physical-First Actuation)
* **Zákon:** Žiadna zmena systémových parametrov (afinita jadier, rozlíšenie DRS, pamäťové kvóty) sa nesmie trvalo zapísať (commitnúť) ani povýšiť do grafu znalostí na základe predikcie. Každá zmena musí prejsť dvojfázovým automatom s živým odberom fyzickej telemetrie a smerovou kontrolou regresie.
* **Formálny invariant:**
  $$\text{Commit}(\Delta S) \implies \text{State} == \text{AUDITED} \land \neg\text{RegressionObserved}(M_{\text{post}}, M_{\text{base}})$$
* **Komponenty balíka:** `ClosedLoopResearchOrchestrator`, `GroundedCapabilityAdapter` a smerový `AdversarialCritic` (`MINIMIZE` / `MAXIMIZE` / `TARGET_RANGE`).

---

### Princíp 8: Princíp kauzálnej zotrvačnosti a hysterézy (Inertial Stability & Anti-Oscillation)
* **Zákon:** Systém nesmie chaoticky kmitať medzi stavmi pri krátkodobom šume na zbernici alebo pri osamelých návrhoch od LLM. Zníženie záťaže (downscale/throttle) je okamžité na ochranu deadline, ale návrat nahor (upscale/unthrottle) vyžaduje prekonanie 12-snímkového okna hysterézy a doterajšieho Merkle-Momenta.
* **Formálny invariant:**
  $$M_k = \gamma M_{k-1} + \text{Weight}(e_k) \quad \land \quad \text{Upscale} \implies N_{\text{dwell}} \ge 12$$
* **Komponenty balíka:** $O(1)$ rekurentný `CausalInertiaDamper` a asymetrický Schmitt-Trigger `RenderHysteresisFilter`.

---

### Princíp 9: Princíp kryptografickej neodškriepiteľnosti (Full-Frame Provenance)
* **Zákon:** Každý 128B rámec, každý zásah do vykresľovania a každý experimentálny uzáver musí niesť plnú kryptografickú pečať svojho pôvodu a kanonickú IEEE-754 reprezentáciu desatinných čísel (bez kolízií -0.0 alebo NaN). Zmena jediného bitu oprávnení (`capability_scope`) okamžite zneplatní celú správu.
* **Formálny invariant:**
  $$\text{Valid}(\text{Frame}) \iff \text{HMAC-SHA256}_{K}(\text{FrameBytes}[0..95]) == \text{frame\_hmac}[96..127]$$
* **Komponenty balíka:** 96-bajtový autentifikátor rámcov, `canonicalize_float32()` a hexadecimálna IEEE-754 serializácia v `ExperimentReceipt`.

---

### Princíp 10: Princíp ústavnej nedotknuteľnosti (Noether-Krystal Survival Symmetry)
* **Zákon:** Bez ohľadu na to, aké open-source komponenty v skladačke zlyhajú, zamrznú alebo vyprodukujú neplatné dáta, celková integrita riadiaceho jadra musí zostať nezmenená. Akákoľvek operácia, ktorá by narušila ústavný invariant, sa automaticky redukuje na sémantické NOOP ($\Delta S = 0$).
* **Formálny invariant:**
  $$\forall t, \forall \text{Layer} \in \{1..4\}: \quad \text{VITAL\_MAX\_HP} \equiv 6 \quad \land \quad (\neg\text{Valid}(A) \implies S_{t+1} \equiv S_t)$$
* **Komponenty balíka:** Globálny hardvérovo-softvérový invariant na offsete 8..9 v každom 128B rámci a vo všetkých zmluvných štruktúrach jadra.

---

## II. Prehľadová matica: Ako sa princípy premietajú do skladačky

| Princíp KSPP v1 | Riadená vrstva | Kľúčový modul v `krystal_kernel` | Benefit pre pripojený Open-Source projekt |
| :--- | :--- | :--- | :--- |
| **P1: Neinvazívna suverenita** | Hranica aplikácií (Vulkan/DXGI/OS) | `universal_engine_interposer.py` | Godot, Blender či UE5 nemusia meniť ani riadok zdrojového kódu. |
| **P2: 128B Lingua Franca** | Medziprocesová pamäť (`/dev/shm`) | `zero_copy_ring.py`, `zero_copy_contracts.py` | Sub-mikrosekundová výmena dát bez serializačnej réžie. |
| **P3: Fázový tieň** | Časovanie zbernice LPDDR4x/5 | `krystal_nexus_accelerator.py`, `plea_engine.py` | `llama.cpp` a 120 Hz render bežia paralelne s 0 frame drops. |
| **P4: Typovosť poznania** | Kognitívny výstup LLM | `epistemic_enlg_compiler.py` | Ollama / vLLM stráca schopnosť vydávať odhady za fakty. |
| **P5: Termodynamický účet** | Spotreba energie a šírky pásma | `EpistemicCalorimeter`, `PowerLimitGovernor` | Odomknutie 15W $\to$ 32W bez prehriatia a eliminácia parazitov. |
| **P6: Zachovanie rozporu** | Fúzia senzorov a dôkazov | `BifurcatedConflictRing` | ROS 2 / Z3 pri rozpore dát nespraví nebezpečný kompromisný priemer. |
| **P7: Kauzálny uzáver** | Exekúcia zmien na OS | `ClosedLoopResearchOrchestrator` | Každá zmena afinity/profilu je fyzicky odmeraná pred commitom. |
| **P8: Zotrvačnosť a hysteréza** | Stabilita obrazu a rozhodovania | `CausalInertiaDamper`, `RenderHysteresisFilter` | Obraz pri K-NSS upscalingu nedýcha, systém neosciluje. |
| **P9: Kryptografický pôvod** | Bezpečnosť a auditovateľnosť | `ZeroCopyContractFrame.compute_frame_hmac` | Ochrana pred podvrhnutím oprávnení a torn-read chybami. |
| **P10: Ústavná symetria** | Celý Krystal Stack | `VITAL_MAX_HP == 6` Invariant Check | Garantované bezpečné zotavenie ($\Delta S = 0$) pri akejkoľvek chybe. |

---

## III. Zhrnutie pre manifest balíka

Týchto 10 princípov (KSPP v1) definuje identitu celého Krystal Stacku:
1. **Navonok** sa balík správa ako univerzálne lepidlo a akcelerátor pre ľubovoľné open-source nástroje (**Princípy P1, P2, P3**).
2. **Vo vnútri** funguje ako neúprosný fyzikálny a epistemický kompilátor, ktorý stráži energiu, pravdivosť, rozpory a kauzalitu (**Princípy P4, P5, P6, P7, P8**).
3. **V základoch** stojí na kryptografickej a ústavnej nemennosti, ktorú nemožno obísť žiadnym softvérovým šumom (**Princípy P9, P10**).
