# Kognitívne Štruktúry: OpenVINO & Windows API
*Architektúra neurónového riadenia operačného systému*

**Dátum:** 14. Január 2026
**Vrstva:** Cognition Layer (Vrstva Vedomia)

---

## 1. Definícia Kognície v OS
Kognícia v našom systéme nie je len "myslenie", je to **cyklus spätnej väzby** medzi hardvérom a softvérom.
Definujeme tri kognitívne stavy (Brainwaves), ktoré systém nadobúda:

### A. Stav ALPHA (Idle / Synthesis)
*   **Symbol:** `≈` (Flow)
*   **Podmienka:** Systém je v kľude, užívateľ číta alebo pozerá video.
*   **OpenVINO Úloha:** Behajú "Background Daemons" na kontrolu integrity Hex Gridu.
*   **Windows API Akcia:** `SetProcessWorkingSetSize` (Trimming RAM), `SetThreadPriority(LOW)`.

### B. Stav BETA (Active / Calculation)
*   **Symbol:** `::` (Grid Active)
*   **Podmienka:** Bežná práca, písanie kódu, rendering UI.
*   **OpenVINO Úloha:** Predikcia pohybu myši (Kalman), pre-fetching textúr.
*   **Windows API Akcia:** Udržiavanie "Memory Coherence", štandardná priorita.

### C. Stav GAMMA (Hyper-Focus / Combat)
*   **Symbol:** `⚡` (Force)
*   **Podmienka:** Hranie hry, kompilácia, tréning AI.
*   **OpenVINO Úloha:** "Tunnel Vision". Ignorovanie všetkého okrem hlavného vlákna.
*   **Windows API Akcia:** `SetPriorityClass(REALTIME_PRIORITY_CLASS)`, Zamykanie stránok v RAM (`VirtualLock`).

---

## 2. Dátový Most (The Bridge)

### Vstup (Sensory Input - Windows API)
Používame `kernel32.dll` na získanie "surových vnemov":
1.  **GlobalMemoryStatusEx:** Fyzický tlak na pamäť.
2.  **GetThreadTimes:** Presný čas CPU strávený v jadre vs. užívateľskom móde.
3.  **Performance Counters:** Nízkoúrovňové "tiky" procesora.

### Procesor (The Brain - OpenVINO)
Dáta normalizujeme do tenzora (vektor čísel) a pošleme do modelu.
*   **Input Vector:** `[RAM_Load, CPU_Kernel_Time, User_Input_Rate, Thermal_Delta]`
*   **Model:** `latency_predictor.xml` (Predtrénovaná sieť).
*   **Output:** Pravdepodobnosť prechodu do stavu Gamma.

### Výstup (Motor Output - Windows API)
Na základe rozhodnutia OpenVINO systém vykoná fyzickú zmenu v OS:
*   **Thread Injection:** Úprava priority bežiaceho vlákna.
*   **Process Affinity:** "Pribitie" procesu na konkrétne (najchladnejšie) jadro.

---

## 3. Symbolika v Neurónovej Sieti
Neurónová sieť sa neučí len čísla. Učí sa "Symbolické Vzory".
*   Ak sieť vidí sekvenciu `Ψ -> ⚡` (Predikcia vedie k Sile), posilní váhy (Weights) pre túto cestu.
*   Ak vidí `⚡ -> Ω` (Sila vedie k Odporu/Prehriatiu), zníži váhy (Inhibícia).

Toto vytvára **"Reflexívny OS"**, ktorý sa učí vyhýbať bolesti (Prehriatie/Lag) a vyhľadávať odmenu (Plynulosť/FPS).
