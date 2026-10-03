# Edukačný sprievodca: Čo predstavuje Krystal-Stack Platform Framework?
**Komplexný manuál pre študentov, vývojárov, grafických inžinierov a architektov**  
*Verzia: 1.0 (2026) | Autori: Dušan Kopecký & Krystal-Stack Open Architecture Team*

---

## 1. Filozofická a Technologická Myšlienka: "Hardvér ako Živý Organizmus"

Väčšina súčasných operačných systémov (Windows, Linux) a herných enginov pristupuje k počítaču mechanicky:
- Procesor (CPU) staticky vykonáva kód riadok po riadku.
- Grafická karta (GPU) pasívne renderuje trojuholníky do framebufferu.
- Pamäť (RAM/VRAM) je len pasívny zoznam bajtov.

**Krystal-Stack prináša radikálnu zmenu paradigmy (Biomimetický / Kybernetický model):**
Počítač nie je mŕtvy stroj, ale **dynamický biologický organizmus**:
- Každý watt elektrickej energie, každý cyklus GPU a každý megabajt VRAM je **obmedzený prírodný zdroj**.
- Systém má svoju **nervovú sústavu** (telemetria, termálne senzory, zbernica [HyperStateBus](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/active_optic_compositor.py)).
- Systém má **reflexy** (miechové reflexy do 1 milisekundy chránia pred tepelným zrútením – Thermal Runaway).
- Systém má **kognitívnu kôru (Brain Cortex / Shadow Council)**, kde tri autonómne entity (Tvorca, Audítor, Účtovník) neustále vyjednávajú optimálnu alokáciu zdrojov.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       TRADIČNÝ VS. ORGANICKÝ COMPUTING                     │
├───────────────────────────────┬─────────────────────────────────────────────┤
│ TRADIČNÝ PRÍSTUP              │ KRYSTAL-STACK (ORGANICKÝ MODEL)             │
├───────────────────────────────┼─────────────────────────────────────────────┤
│ Statický OS scheduler (CFS)   │ Trhový mechanizmus (Economic Governor)     │
│ Pasívny výstup na monitor     │ Aktívna optika (Active Optic Compositor)    │
│ Pád aplikácie pri preťažení   │ Vizuálny backpressure (Automatické škrtenie)│
│ Všetko beží na GPU            │ Heterogénna triáda (CPU + GPU + NPU + WSL)  │
│ ASCII je len textový filter   │ ASCII je neurálny jazyk kompresie a telemetrie│
└───────────────────────────────┴─────────────────────────────────────────────┘
```

---

## 2. Kľúčové Piliere Architektúry

### Pilier I: Heterogénna Triáda (CPU + GPU + NPU)
V moderných čipoch (Intel Core Ultra, AMD Ryzen AI, Apple Silicon M-series, Qualcomm Snapdragon X) už nemáme len CPU a grafiku. Máme **tri výpočtové svety**:
1. **CPU (Generalista)**: Rýchly v sekvenčnej logike a obsluhe Windows/Linux volaní.
2. **GPU (Paralelista)**: Určený na rendering herného sveta, milióny polygónov a zložité svetelné shadery.
3. **NPU (Špecialista)**: Maticový tenzorový akcelerátor s príkonom iba **1–2 Watty**, ktorý dokáže počítať neurálne siete 10-krát úspornejšie ako grafická karta.

> **Prečo je to revolučné pre hry a ASCII?**  
> V bežných hrách grafická karta beží na 99 % výkonu. Krystal-Stack presúva analýzu scény a neurálnu kompozíciu na NPU. Hra nestratí ani jedno jediné FPS a počítač šetrí desiatky wattov energie.

---

### Pilier II: Vizuálna Entropia & Vizuálny Backpressure
Jedným z najoriginálnejších objavov Krystal-Stacku je koncept: **Obraz na obrazovke je najlepšia telemetria stavu počítača.**

Matematicky meriame tri zložky chaosu:
$$E_{total} = 0.4 \cdot E_{spatial} + 0.3 \cdot E_{temporal} + 0.3 \cdot E_{frequency}$$

1. **Priestorová entropia ($E_{spatial}$)**: Udáva, nakoľko je scéna rozbitá na mikroskopické detaily a šum.
2. **Časová entropia ($E_{temporal}$)**: Udáva, ako prudko sa obraz mení medzi dvoma po sebe idúcimi snímkami.
3. **Frekvenčná entropia ($E_{frequency}$)**: Rýchla Fourierova transformácia (FFT) zisťujúca výskyt vysokofrekvenčného kmitania.

```
NORMÁLNY STAV (Nízka entropia E = 0.25)        KRITICKÝ CHAOS (Vysoká entropia E = 0.88)
┌──────────────────────────────────────┐       ┌──────────────────────────────────────┐
│ [STABLE STATUS] CPU: 45% | GPU: 62%  │       │ █▓▒░▓█▒░▓█▒░▓█▒░ [CRITICAL CRASH]    │
│ ████████████████░░░░░░░░░░░░░░░░░░░░ │       │ ░▒▓█░▒▓█░▒▓█░▒▓█░ ERROR: TDR STALL   │
└──────────────────────────────────────┘       └──────────────────────────────────────┘
                   │                                              │
                   ▼                                              ▼
          Udržuj plný výkon                             AKTIVUJ BACKPRESSURE!
          Všetky systémy v norme                        Zníž záťaž CPU o 30%
                                                        Prealokuj pamäť
```

Keď entropia prekročí $0.70$, **Visual Backpressure** okamžite priškrtí procesy na pozadí a zjednoduší štýl zloženia, čím zabráni pádu GPU (TDR eventu) skôr, než zareaguje samotný operačný systém Windows.

---

### Pilier III: Smerové Sobel Vektory & Typografický Matching
Prečo Krystal-Stack ASCII nevyzerá ako obyčajný textový filter?
Bežné programy vezmú jas pixelu (0–255) a priradia mu znak z reťazca: ` .:-=+*#%@`. Výsledok je rozmazaný a bez hrán.

Krystal-Stack používa **konvolučné Sobelove jadrá**:
- Vypočíta gradient $G_x$ a $G_y$.
- Určí presný uhol hrany: $\theta = \arctan(G_y / G_x)$.
- Podľa uhla priradí skutočný smerový znak:
  - $90^\circ \rightarrow$ `|` (zvislá stena)
  - $0^\circ \rightarrow$ `-` (podlaha / horizont)
  - $45^\circ \rightarrow$ `/` (stúpajúca hrana)
  - $135^\circ \rightarrow$ `\` (klesajúca hrana)

Výsledkom je čistý technický nákres (blueprint), v ktorom ľudské oko okamžite rozozná tvary a siluety.

---

## 3. Pre koho je Krystal-Stack určený?

| Cieľová skupina | Ako projekt využijú | Kľúčový modul |
| :--- | :--- | :--- |
| **Herní vývojári (Godot/Vulkan)** | ASCII a kyberpunkové shadery s nulovým dopadom na GPU; unikátny umelecký štýl. | [`godot_project/`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/godot_project/) |
| **Študenti Computer Science** | Štúdium heterogénneho computingu, NPU inferencie, raymarchingu a Linux/WSL mostov. | [`docs/research/`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/docs/research/) |
| **DevOps & Cloud Inžinieri** | Vizuálny monitoring serverov s automatickým potláčaním špičiek (Visual Backpressure). | [`krystal_web_hub/`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/) |
| **Priemyselná automatizácia (CNC)** | Digitálne dvojčatá strojov a robotických ramien bežiace na embedded PC bez GPU. | [`advanced_cnc_copilot/`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/advanced_cnc_copilot/) |
