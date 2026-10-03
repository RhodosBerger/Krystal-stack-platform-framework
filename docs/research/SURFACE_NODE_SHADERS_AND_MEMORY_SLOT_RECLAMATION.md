# Krystal-Stack Research: Simulácia Povrchov v Node Editore & Spätný Zber Prázdnych Slotov Pamäte

**Dátum:** 3. Október 2026  
**Autor:** Krystal-Stack Shader Architecture & Memory Reclamation Group  
**Kľúčový Invariant:** `VITAL_MAX_HP = 6` (Pravidlo maximálnej celistvosti systému)  
**Zlatý Pomer:** $\phi = 1.61803398875$, $\phi^{-1} \approx 0.61803398875$

---

## 1. Úvod a Problémová Formulácia

Procedurálne generovanie komplexných povrchov (kryštály, alchymistické minerály, toxické kyselinové polia, zvetraný kameň a kovová patina) v reálnom čase vyžaduje viacvrstvové úpravy materiálu (adjustment layers, triplanar blending, Fresnel rim glow, normálová perturbácia a kontaktné samotienenie). 

Tradičné prístupy čelia dvom zásadným limitom:
1. **Pevná kombinačná variabilita:** Ak má šéder statické parametre, povrchy pôsobia repetitívne a predvídateľne.
2. **Pamäťová réžia:** Generovanie miliónov unikátnych variantov textúr alebo materiálových presetov bežne vyžaduje masívne alokácie VRAM / Host RAM, čo vedie k fragmentácii pamäte a latencii zberníc.

Architektúra **Krystal-Stack** rieši tento problém spojením **Vizuálneho Node Editora** so **Spätným Zberačom Prázdnych Slotov Pamäte (Reverse Memory Slot Scavenger)**.

---

## 2. Viacvrstvová Simulácia Povrchov v Node Editore

Vizuálny Node Editor prepája jednotlivé komponenty materiálu cez uzly a prepojenia:

### 2.1 Vrstvy a Triky Úprav (Adjustment Nodes)
- **Základná Vrstva (Base Layer):** Definuje Albedo farbu, základnú drsnosť (roughness) a primárny kovový lesk (metallic).
- **Šumová Perturbácia (Noise Perturbation):** Fraktálny Voronoi a Perlin šum modulujúci reliéf a bump mapu.
- **Kontaktné Samotienenie (Horizon-Based Parallax Occlusion):** Počíta vnútorné kontaktné tiene podľa výškovej mapy reliéfu. Zabraňuje tomu, aby hlboké ryhy svietili rovnako ako vyvýšené hrebene.
- **Harmonický Fresnel Lem (Golden Mean Rim Glow):** Využíva $\phi^{-1} \approx 0.618$ na výpočet uhla okrajového svetla, simulujúceho vnútornú kryštalickú rezonanciu.

---

## 3. Spätné Vychytávanie Prázdnych Slotov v Blokoch Pamäte

### 3.1 Mechanizmus Pamäťového Slabu (Memory Block Slab)
Pamäťový blok je rozdelený na $N = 64$ slotov s pevnou veľkosťou (napr. 4 KB per slot = 256 KB slab). Počas behu aplikácie procesy alokujú a uvoľňujú sloty (textúry, compute buffery, geometriu), čím vznikajú **voľné diery (fragmentácia pamäte)**.

### 3.2 Spätný Zber (Reverse LIFO Scavenging)
Namiesto náhodnej alokácie alebo čakania na periodický Garbage Collector, **spätný zberač** skenuje blok od najvyššieho indexu ($63 \to 0$):
1. Deteguje neobsadené sloty (`is_occupied == False`).
2. Označí ich ako **recyklované (`reclaimed = True`)**.
3. Z každej pozície $i$ a entropického posunu generuje permutačný seed:
   $$S_i = |\sin((i \cdot 13.37 + 1.618) \cdot \pi)|$$
4. Mapuje index slotu $i \pmod 8$ na jednu z 8 samostatných funkcionalít šédra:
   - `Slot % 8 == 0`: Mikro-fazetový index lomu (IOR 1.54).
   - `Slot % 8 == 1`: Hĺbka kontaktného tieňa.
   - `Slot % 8 == 2`: Podpovrchový chromatický rozptyl (Subsurface scattering).
   - `Slot % 8 == 3`: Kaustická toxická viskozita.
   - `Slot % 8 == 4`: Druidská oklúzia dutín a mikro-mach.
   - `Slot % 8 == 5`: Harmonický Fresnel lem.
   - `Slot % 8 == 6`: Anizotropná metalická patina.
   - `Slot % 8 == 7`: Volumetrická absorpcia podľa Beer-Lambertovho zákona.

---

## 4. Kombinatorický Nárast Scenárov (Scenario Permutation Explosion)

Využitie prítomnosti a poradia spätne zachytených voľných slotov $k$ vytvára **kombinatorický nárast vizuálnych scenárov bez spotreby jediného bajtu dodatočnej pamäte**:

$$\text{Scenáre}(k) = P(8, \min(8, k)) \times 2^{\frac{\min(16, k)}{4}} = \frac{8!}{(8 - \min(8, k))!} \times 2^{\frac{\min(16, k)}{4}}$$

### Príklad rastu:
- Pri $k = 4$ uvoľnených slotoch: $P(8, 4) \times 2^1 = 1,680 \times 2 = 3,360$ scenárov.
- Pri $k = 8$ uvoľnených slotoch: $P(8, 8) \times 2^2 = 40,320 \times 4 = 161,280$ scenárov.
- Pri $k = 13$ uvoľnených slotoch: $40,320 \times 2^{3.25} \approx 383,618$ unikátnych povrchov!

Systém tak premieňa fragmentáciu pamäte (bežne vnímanú ako chybu alebo neefektivitu) na **procedurálne bohatstvo herného sveta a šédrov**.

---

## 5. Implementácia v Architektúre Krystal-Stack

1. **Python Core Engine:** [surface_node_shader_and_memory_reclaimer.py](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/economic_engine/surface_node_shader_and_memory_reclaimer.py)
   - Dataclass `MemorySlot`, trieda `MemoryBlockSlab`, `SurfaceNodeShaderEngine`, singleton `GLOBAL_SURFACE_NODE_ENGINE`.
2. **Janet DSL Subproject:** [surface_node_shader_and_memory_reclaimer.janet](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_janet/surface_node_shader_and_memory_reclaimer.janet)
   - Funkcie `create-memory-slab`, `reverse-scavenge-empty-slots`, `calculate-combinatorial-scenarios`.
3. **REST API na porte 8089:** [krystal_engine_core.py](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/krystal_engine_core.py)
   - `GET /api/surface-nodes/catalog`
   - `GET /api/surface-nodes/default-graph`
   - `POST /api/surface-nodes/evaluate`
   - `POST /api/surface-nodes/scavenge-memory`
   - `GET /surface-nodes` (webové štúdio)
4. **Interaktívne Webové Štúdio:** [surface_node_editor_studio.html](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/static/surface_node_editor_studio.html)
   - Vizuálny editor s ťahateľnými uzlami a Bézierovými drôtmi.
   - Živé 3D plátno s interaktívnou kryštálovou guľou a rotáciou.
   - 64-slotová matica pamäte s pulzujúcimi spätne zachytenými slotmi.
   - Počítadlo kombinatorických scenárov.
5. **Krystal WebOS Desktop Integrácia:** [krystal_webos_desktop.html](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/static/krystal_webos_desktop.html)
   - Ikona `🌐 Node Editor` na ploche, v Štart menu a plávajúce okno.
6. **Platformový Invariant:**
   - Striktné dodržanie `VITAL_MAX_HP = 6`.
