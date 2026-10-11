# KRYSTAL-STACK // METODIKA PRECHODU DO ĎALŠEJ FÁZY A NOČNÝ STABILNÝ BUILD
=============================================================================
**Verzia dokumentu:** 2.0-STABLE  
**Dátum a čas revízie:** 11. Október 2026, 19:15 CET  
**Cieľový míľnik:** Nočný stabilný integračný build (Deadline: 00:00 Polnoc)  
**Kódové označenie:** *Operation Midnight Polaris*  
**Autorita:** Senior Principal Systems & Cyber-Physical Architect  
**Záväzný systémový invariant:** `VITAL_MAX_HP == 6` (Striktne nemenný)

---

## 1. Strategický Kontext a Účel Metodík

Cieľom tohto dokumentu je formalizovať exaktné systémové postupy a metodiky, ktoré posúvajú projekt Krystal-Stack z fázy špecifikácií a prototypov do **stabilného, verifikovaného a demonštrovateľného buildu**. Do polnoci musí byť systém schopný na reálnych hardvérových zostavách predviesť konkrétne merateľné ukazy (exhibits) demonštrujúce kauzálnu integritu nášho pripravovaného frameworku.

V súlade s kľúčovým epistemickým pravidlom:
$$\text{Observe} \longrightarrow \text{Model} \longrightarrow \text{Explain} \longrightarrow \text{Predict} \longrightarrow \text{Falsify} \longrightarrow \text{Validate} \longrightarrow \text{Intervene}$$
je akýkoľvek zásah alebo optimalizácia podmienená kauzálnou merateľnosťou a elimináciou *Observer Effectu*.

---

## 2. Definované Hardvérové Zostavy (Target Setups)

Framework Krystal-Stack je navrhnutý pre multi-platformovú kooperáciu na štyroch overiteľných zostavách:

### Zostava A: Local Bare-metal (Windows 11 + Intel Iris Xe Graphics)
- **Host:** Windows 11 Enterprise (NT Kernel), CPU Intel Core i7 / i5 (4 P-cores / 8 threads).
- **Akcelerátor:** Integrované GPU Intel Iris Xe Graphics cez Vulkan Direct Compute a Level Zero.
- **Rola v demonštrácii:** Hlavný uzol vývojárskeho prostredia, vizuálny Mission Control Hub (`localhost:8080`), real-time rasterizácia a Vulkan compute pipeline.

### Zostava B: WSL2 / Linux Execution Node (Ubuntu 22.04 LTS + OpenVINO GPU Passthrough)
- **Host:** WSL2 Kernel 5.15+ s priamym prístupom k `/dev/dri/card0` a `/dev/dri/renderD128`.
- **Stack:** i915 kernel driver, OpenCL NEO (`intel-opencl-icd`), Level Zero, OpenVINO Runtime 2024+.
- **Rola v demonštrácii:** CrystalSec 2.0 Causal LLM Execution Plane, reálna inferencia neurónových modelov, meranie TTFT, ITL a KV-cache alokácie.

### Zostava C: Sovereign Container / MicroVM (Arch Linux + cgroups v2)
- **Host:** Headless Arch Linux kontajner (`docker/Dockerfile.archlinux`, `docker-compose.gov-integrity.yml`).
- **Stack:** Strict cgroups v2 resource isolation, CFS quota limits, systemd telemetry, Communication Integrity Plane.
- **Rola v demonštrácii:** Deterministická validácia, nemenný audit ledger, zachovanie zmlúv o zdrojoch a izolácia služieb bez tichej degradácie.

### Zostava D: Cyber-Physical CNC Edge Simulator
- **Host:** Real-time deterministický cyklus (1 kHz / 1 ms delta).
- **Stack:** Model tepelného gradientu vretena ($\nabla T$), dynamický diferenciál RPM ($\Delta \text{RPM}$), algoritmus núdzového chladenia a bezpečnostná rezerva kolapsu.
- **Rola v demonštrácii:** Dôkaz, že matematická a radarová metodológia funguje identicky v softvérovom aj fyzikálnom svete.

---

## 3. Metodika M1: Verifikácia Stacku a Stanovenie „Golden Baseline“

Pred spustením akéhokoľvek experimentu musí systém prejsť deterministickou kontrolou prítomnosti hardvérových rozhraní:

1. **Pre-flight inšpekcia:**
   - Overenie prítomnosti `/dev/dri/renderD128` (resp. Vulkan Device na Windows).
   - Kontrola práv v skupinách `video` a `render`.
   - Zistenie dostupnosti ACPI teplotných zón (`/sys/class/thermal/` alebo Windows WMI).
2. **Meranie pokoja (Quiescent State):**
   - Spustenie `AdaptiveTelemetryGovernor` v režime `QUIESCENT_BASELINE` (2 Hz) počas $t = 5$ sekúnd.
   - Nameranie bazálnej latencie, teploty jadier a zaťaženia procesora.
   - Uloženie referenčného stavového vektora $H_0$.

---

## 4. Metodika M2: Eliminácia „Observer Effectu“ a Dynamické Vzorkovanie

Telemetrický subsystém nesmie svojou vlastnou činnosťou zmeniť výsledky merania:

1. **Výpočet réžie pozorovateľa:**
   - Pre každý odber telemetrie sa meria čas vykonania funkcie $t_{\text{tick}}$.
   - Guvernér počíta projekciu záťaže CPU:
     $$\text{Overhead}_{\text{pct}} = (f_{\text{sampling}} \cdot t_{\text{tick}}) \times 100\%$$
2. **Kritériá zásahu (Observer Effect Guard):**
   - **Limit:** Maximálna povolená réžia je **$2.5\%$ CPU slice**.
   - Ak $\text{Overhead}_{\text{pct}} > 2.5\%$, guvernér okamžite zastropuje frekvenciu:
     $$f_{\text{safe}} = \frac{0.025}{t_{\text{tick}}}$$
   - Záznam o zastropovaní (`is_observer_throttled = True`) sa povinne zapíše do telemetrie.
3. **Prechod medzi režimami:**
   - $|dH/dt| \le \theta \implies \text{QUIESCENT\_BASELINE}$ ($2\text{ Hz}$).
   - $|dH/dt| > \theta \implies \text{BURST\_TRANSIENT\_CAPTURE}$ ($50 - 100\text{ Hz}$, resp. $f_{\text{safe}}$).
   - Stabilizácia $N=15$ tikov $\implies \text{COOLDOWN\_RECOVERY} \to \text{QUIESCENT\_BASELINE}$.

---

## 5. Metodika M3: Exekúcia 8-fázového Vertikálneho Slice-u

Preverenie celého reťazca jedného príkazu na zvolenej zostave:

$$\text{Prompt} \xrightarrow{1} \text{Context} \xrightarrow{2} \text{Routing} \xrightarrow{3} \text{Inference} \xrightarrow{4} \text{Tool Proposal} \xrightarrow{5} \text{Validation} \xrightarrow{6} \text{Execution} \xrightarrow{7} \text{Observation} \xrightarrow{8}$$

### Akceptačné kritériá fázy:
- **TTFT (Time-To-First-Token):** $p95 \le 45\text{ ms}$ (na GPU/OpenVINO) resp. $p95 \le 120\text{ ms}$ (na CPU fallbacku).
- **ITL (Inter-Token Latency):** $p95 \le 12\text{ ms}$.
- **Throttling:** Striktne $nr\_throttled == 0$. Akékoľvek throttlingové prepadnutie je klasifikované ako anomália.
- **Invariant:** `VITAL_MAX_HP == 6` pred, počas aj po exekúcii.

---

## 6. Metodika M4: Bounded Configuration Search & Reprodukovateľnosť

Skúmanie priestoru nastavení je prísne ohraničené a nesmie vyústiť do ad-hoc "tweakovania":

1. **Povolené dimenzie zmien:**
   - Iba premenné z `ALLOWED_CONTROLLED_VARIABLES` (`cpu.threads`, `openvino.streams`, `cpu.affinity`, `queue_max_depth`).
2. **Pravidlo izolácie:**
   - Mení sa striktne **1 premenná** (alebo 1 kontrolovaný pár, napr. `threads` $\times$ `streams`), pričom všetky ostatné ostávajú fixované na bazálnych hodnotách.
3. **Empirická replikácia ($N = 5$ opakovaní):**
   - Každá konfigurácia sa meria 5-krát.
   - Počíta sa výberový priemer $\mu$ a smerodajná odchýlka $\sigma$.
   - Koeficient variácie $CV = \frac{\sigma}{\mu}$.
   - Reprodukovateľnosť:
     $$\text{Repeatability}_{\text{pct}} = \max(0.0, (1.0 - CV) \times 100\%)$$
4. **Povýšenie na `SUPPORTED`:**
   - Iba konfigurácie s $\text{Repeatability} \ge 85\%$ sú zaradené do finálnej Pareto mapy.

---

## 7. Metodika M5: Multi-disciplinárna Syntéza do Radaru a Váh

Výsledky merania zo všetkých štyroch domén sa transformujú do jednotného vizuálneho a matematického jazyka:

1. **Normalizácia na 5-bodovú škálu ($0.0 - 5.0$):**
   - Každá disciplína mapuje svoje fyzikálne alebo výpočtové parametre na 4 hlavné osi radaru.
2. **Podmienka váh:**
   - Každý aspekt $i$ má priradenú relatívnu váhu $w_i$, kde $\sum_{i=1}^4 w_i = 1.0$.
3. **Výpočet kompozitného indexu $U$:**
   $$U = \sum_{i=1}^4 w_i \cdot s_i \in [0.0 - 5.0]$$
4. **Generovanie 2D SVG polygónu:**
   - Prepojenie bodov na osiach do vizuálneho diamantu, ktorý je okamžite premietaný do [multi_disciplinary_radar_studio.html](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/static/multi_disciplinary_radar_studio.html).

---

## 8. Metodika M6: Nočný Integračný Protokol (Midnight Release Protocol)

Presný sled krokov pre zostavenie a overenie finálneho stabilného buildu pred polnocou:

1. **Krok 1 (19:30 - 20:30): Automatizovaný Test Suite Run**
   - Spustenie unit testov: `test_crystalsec_v2_llm_execution_plane.py`, `test_crystalsec_v2_bounded_search_and_observer_effect.py`, `test_multi_disciplinary_measurement_summarizer.py`.
   - Podmienka: **100% testov green**, žiadne preskočené kritické testy.
2. **Krok 2 (20:30 - 21:30): Spustenie Multi-Target Demonštrátora**
   - Spustenie `scripts/midnight_stable_build_demonstrator.py`.
   - Vytvorenie reálneho demonštračného záznamu a overenie všetkých 4 ukážok (exhibits).
3. **Krok 3 (21:30 - 22:30): Verifikácia Web Hubu a Radar Studia**
   - Kontrola HTTP servera na porte 8080.
   - Overenie funkčnosti endpointov `/api/measurement-disciplines/summary` a `/measurement-radar`.
4. **Krok 4 (22:30 - 23:30): Vystavenie Validačného Certifikátu**
   - Vygenerovanie podpísaného certifikátu `artifacts/MIDNIGHT_STABLE_BUILD_CERTIFICATE.json`.
5. **Krok 5 (23:30 - 23:59): Git Tag a Release Seal**
   - Vytvorenie tagu `v2.0-midnight-stable`.
   - Potvrdenie čistého working tree.
