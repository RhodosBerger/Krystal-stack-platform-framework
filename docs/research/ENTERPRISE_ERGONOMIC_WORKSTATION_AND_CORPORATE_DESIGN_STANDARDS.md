# KRYSTAL ENTERPRISE WORKSTATION // CORPORATE DESIGN SYSTEM & ERGONOMIC WORKSTATION ARCHITECTURE
**Standardy AAA Titulov, Bootstrap 5.3 Responzívny Grid a Korporátny Vzhľad**
*Autor: Dušan Kopecký & Krystal-Stack Research Council (2026)*

---

## 1. Exekutívne Zhrnutie & Benchmark Konkurenčných Titulov

V súlade s požiadavkou na vytvorenie rozhrania spĺňajúceho štandardy popredných titulov konkurencie (DCC editory **Unreal Engine 5**, **Unity 6**, **Blender 4**, telemetrické systémy **Bloomberg Terminal / Datadog APM** a generatívne svety **Google Project Genie**) bol navrhnutý a implementovaný **Krystal Enterprise Workstation**.

Tento systém prekonáva bežné webové prototypy a poskytuje:
1. **Denzitne optimalizovanú ergonómiu (Information Density):** Žiadne prázdne plytvanie priestorom; panely sú usporiadané podľa logiky profesionálnych pracovných staníc (inšpektor vľavo, duálny viewport v strede, ekonomika a telemetria vpravo, lore kronika v dokovateľnom spodnom paneli).
2. **Plnú responzivitu cez Bootstrap 5.3:** Plynulé prispôsobenie od mobilných zariadení cez tablety až po ultraširoké monitory (4K/Ultrawide) s využitím moderných komponentov (`accordion`, `nav-pills`, `offcanvas`, `card`, `modal`).
3. **Sofistikovaný korporátny dizajn (Dark Obsidian Corporate Theme):** Vyladená paleta tmavých tónov, tenké 1px deliace linky, jemné sklenené rozmazanie (`backdrop-filter: blur(16px)`), pulzujúce LED indikátory stavu hardvéru a čistá typografia (Inter pre UI, Cinzel pre exekutívne titulky, JetBrains Mono pre telemetriu).

---

## 2. Architektúra Trojstĺpcového Ergonomického Rozvrhnutia

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  KRYSTAL ENTERPRISE NAV: Brand Badge | Intel NPU: 5.4W | 0.038 J/tok | 💎 2,450 KC | ⛶  │
├─────────────────────────┬───────────────────────────────────┬──────────────────────────┤
│  ĽAVÝ INŠPEKTOR (col-3) │  STREDNÝ DUÁLNY VIEWPORT (col-6)  │  PRAVÁ TELEMETRIA (col-3)│
│                         │                                   │                          │
│  [▼] GOOGLE MAPS 3D     │  ┌─────────────────────────────┐  │  ┌────────────────────┐  │
│      • Praha / Ba / Ty  │  │ Fotorealistický Bayer ASCII │  │  │ Korporátna Pokladňa│  │
│      • Polomer & GPS    │  │ [░▒▓█ Sobel & Iskry]        │  │  │ 💎 2,450 KC        │  │
│      • Export Blender   │  ├─────────────────────────────┤  │  └────────────────────┘  │
│                         │  │ Natívna 3D Polygónová Sieť  │  │  ┌────────────────────┐  │
│  [▼] OHRANIČENÝ 3D KLIET│  │ [Three.js WebGL & Klietka]  │  │  │ Matica Achievementov│  │
│      • X/Y/Z posuvníky  │  └─────────────────────────────┘  │  │ 6 míľnikov (Bronz-  │  │
│      • ±3.5m / 215 m³   │                                   │  │ Mythic) + Mint tlač │  │
│                         │  HUD: GENE Akcie [0..5] (WASD)    │  └────────────────────┘  │
│  [▼] RASTER MIXER       │  [0]IDLE [1]ORBIT [2]MORPH ...    │  ┌────────────────────┐  │
│      • Bayer 4x4 / 8x8  │                                   │  │ Intel RAPL Telemetria│  │
│      • Sub-pixely       │                                   │  │ 5.4W vs 24.5W Llama │  │
└─────────────────────────┴───────────────────────────────────┴──────────────────────────┘
│  DOKOVATEĽNÝ SPODNÝ DRAWER (OFFCANVAS): KRONIKA CODEX & SLM GENERÁTOR NARRATÍVY       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Ľavý Panel: Priestorový Inšpektor a Mestský Extractor
- **Google Maps 3D Extractor:** Výber reálnych miest (Praha Staré Město, Bratislavský Hrad a Dunaj, Tokyo Shibuya Crossing) alebo zadanie ľubovoľných GPS súradníc. Generuje presné pôdorysy, výšky podlaží a strešné vežičky.
- **Export do Blender 4.x Modifikátorov:** Automaticky generuje Python skript (`blender_import_google_maps.py`) a `.obj` súbor pre import s automatickým nasadením `Bevel`, `Solidify` a `Displace` modifikátorov.
- **Ohraničený 3D Priestor (Bounded Cage):** Ergonomické posuvníky pre rozmery $X, Y, Z$, výpočet uzavretého objemu v $\text{m}^3$ a prepínače pre drôtenú klietku a 1.0m metrickú podlahovú mriežku.
- **Procedural Raster Mixer:** Prepínanie Bayerových matíc (4x4 a 8x8), sub-pixelových kvadrantov (`▖▗▘▙▚▛▜▝▞▟`), smerových Sobelových hrán a zrkadlových trblietok.

### 2.2 Stredný Panel: Špičkový Duálny Viewport & Latentný Kontrolér
- **Split Viewport:** Naľavo beží fotorealistický ASCII compositor s voliteľnými CRT scanlines; napravo beží interaktívne 3D WebGL plátno (Three.js) s reálnou mestskou geometriou, osvetlením a kamerovým orbitom.
- **Project GENE Akčný Kontrolér:** Tlačidlá latentných akcií `[0..5]` s ergonomickým ovládaním cez klávesové skratky (`WASD` pre priestorovú navigáciu, `Medzerník` pre expanziu kryštálu, `Q/E` pre rotáciu kamery).

### 2.3 Pravý Panel: Ekonomická Pokladnica & Intel Telemetria
- **Korporátny Ledger:** Zostatok vyťažených Krystal Kreditov (KC), prehľad dividend a ukazovateľ odomknutých úspechov.
- **Matica Achievementov:** Prehľadné karty míľnikov rozdelené podľa kategórií (Taktický boj, Mestská architektúra, Optická telemetria, Lore) s odznakmi tried (Bronze, Silver, Gold, Mythic). Každá karta umožňuje jedným klikom odomknúť míľnik, získať kredity a vygenerovať novú kapitolu v kronike.
- **Hardvérový Guvernér (Intel OpenVINO vs llama.cpp):** Reálne telemetrické snímanie spotreby cez RAPL registre, energetická efektivita na token, latencia prvého tokenu (TTFT) a dynamická spätná väzba (backpressure) pri prehriatí.

### 2.4 Spodný Panel: Kronika Codex & SLM Storyteller Terminal
- Dokovateľný a responzívny Bootstrap `offcanvas` spodný drawer.
- Zobrazuje generované fantasy kapitoly kroniky, ktoré sú pri odomknutí achievementu vytvárané lokálnym modelom ($\le 2\text{B}$ parametrov) s minimálnou energetickou stopou na Intel NPU.
- Umožňuje interaktívne zadanie ľubovoľnej témy a okamžité vygenerovanie kapitoly s priamou telemetrickou pečiatkou.

---

## 3. Dizajnové Tokeny Korporátneho Štýlu

| Kategória | Token / Trieda | Špecifikácia / Hex | Účel a Použitie |
| :--- | :--- | :--- | :--- |
| **Podklad** | `--corp-bg-base` | `#090C12` | Absolútne tmavé pozadie aplikácie |
| **Povrch panelu** | `--corp-bg-surface` | `#111622` | Základné karty a panely workstationu |
| **Zvýšená karta** | `--corp-bg-card` | `#151C2C` | Aktívne prvky a vnútorné kontajnery |
| **Ohraničenie** | `--corp-border-subtle` | `rgba(255, 255, 255, 0.08)` | 1px jemné deliace linky |
| **Primárny akcent** | `--corp-cyan` | `#00F0FF` | Kľúčové ovládacie prvky, aktívne lúče |
| **Menný akcent** | `--corp-amber` | `#FFB800` | Krystal Kredity, zlaté míľniky |
| **Hardvérový status**| `--corp-emerald` | `#00E676` | Intel NPU aktívny, nominálny chod |
| **Kritický status** | `--corp-ruby` | `#FF3366` | Tepelné škrtenie, preťaženie pamäte |
| **Typografia UI** | `font-corporate` | `'Inter', sans-serif` | Maximálna čitateľnosť pri 11–13px |
| **Typografia Kód** | `font-mono` | `'JetBrains Mono', monospace` | Telemetria, súradnice, ASCII matica |

---

## 4. REST API Endpointy Servera

Všetky endpointy sú plne integrované a overené v [`krystal_web_hub/server.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/server.py):

| Metóda | Cesta | Popis |
| :--- | :--- | :--- |
| `GET` | `/code-gene` | Servuje nový Enterprise Workstation (`code_gene_studio.html`) |
| `GET` | `/api/achievements` | Vráti zoznam achievementov, kategórie, odmeny a stav odomknutia |
| `POST`| `/api/achievements/unlock` | Odomkne míľnik, pripíše kredity a spustí SLM generovanie kapitoly |
| `GET` | `/api/achievements/codex` | Vráti zoznam všetkých zaznamenaných kapitol kroniky |
| `GET` | `/api/telemetry/hardware` | Poskytuje porovnávací audit (Intel NPU vs llama.cpp) a živé RAPL watty |
| `POST`| `/api/narrative/synthesize`| Vygeneruje na požiadanie fantasy kapitolu cez SLM prompt šablónu |
| `GET` | `/api/google_maps/cities` | Zoznam reálnych miest k dispozícii pre 3D extrakciu |
| `POST`| `/api/google_maps/extract` | Extrakcia 3D geometrie a WGS84 projekcia do metrickej klietky |
| `POST`| `/api/google_maps/export_blender` | Generovanie Blender 4.x Python skriptu a `.obj` geometrie |
