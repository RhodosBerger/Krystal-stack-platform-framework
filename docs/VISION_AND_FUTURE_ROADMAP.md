# Vízia Budúcich Funkcionalít a Strategická Roadmapa (2026 – 2030)
**Krystal-Stack Platform Framework — Od Terminálu k Priestorovému Neurálnemu Computingu**  
*Dokument: STRATEGIC-VISION-V3 | Schválená vízia | Dátum: 2026-10-01*  
*Autor: Dušan Kopecký & Krystal-Stack Research Council*

---

## 🌌 1. Veľká Vízia: Nový Jazyk Medzi Strojom a Vedomím

Svet sa ocitol v grafickej pasci: videohry a simulácie vyžadujú stále drahšie 600-wattové grafické karty, masívnu šírku pásma a obrovské dátové centrá, pričom 95 % prenášaných pixelov predstavuje len vizuálny balast.

**Krystal-Stack predstavuje tretiu cestu:**
Namiesto surových megabajtov RGB pixelov transformujeme realitu na **sémantický holografický neurálny prúd (Neural Holographic Token Stream)**.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                EVOLUČNÁ PARADIGMA KRYSTAL-STACKU                       │
├───────────────────────┬────────────────────────┬───────────────────────────────────────┤
│ ÉRA                   │ TECHNOLÓGIA            │ CHARAKTERISTIKA                       │
├───────────────────────┼────────────────────────┼───────────────────────────────────────┤
│ 1. Textový filter     │ 1D Jasové ASCII        │ Statické, pomalé, CPU-bound           │
│ 2. Vizuálna telemetria│ Active Optic Compositor│ Sobel hrany, vizuálna entropia        │
│ 3. Súčasnosť (2026)   │ Heterogénna triáda     │ NPU akcelerácia, WSL2, Godot, Web Hub │
│ 4. Priestorové AR     │ Volumetrický hologram  │ AR okuliare, mikrodáta 20 KB/s        │
│ 5. Bio-Kybernetika    │ BCI Neuro-Feedback     │ Adaptácia podľa mozgových vĺn         │
└───────────────────────┴────────────────────────┴───────────────────────────────────────┘
```

---

## 🗺️ 2. Strategická Roadmapa (2026 – 2030)

### FÁZA 1: Dokončenie Hybridnej Platformy (Q4 2026)
- **Cieľ**: Pevné prepojenie Windows hosta s WSL2 a hernými enginmi.
- **Kľúčové míľniky**:
  - [x] Localhost Mission Control Hub s live SSE streamom na porte 8080.
  - [x] Obojsmerný most WSL2 (`wsl_bridge/bridge_daemon.py`).
  - [x] Integrácia Godot 4.x Screen-Space Hologram shadera (`godot_project/`).
  - [x] Prototyp procedurálneho 3D SDF a volumetrického projektora.
  - [ ] Balíčkovanie samostatného Windows inštalátora (`.msi` a Chocolatey/winget balík).

---

### FÁZA 2: Priestorové AR Okuliare a Taktické Kompresné Prenosy (2027)
- **Cieľ**: Premietanie holografického ASCII priamo do zorného poľa cez AR okuliare (Meta Quest 3/Pro, Apple Vision Pro, XREAL Air 2, HoloLens 2).
- **Inovácia: Taktický nízko-dátový prenos (Ultra-Low Bandwidth Stream)**:
  - Prenášať 4K video vojakom v teréne, do ponorky alebo na vesmírnu stanicu vyžaduje 25–50 Mbps.
  - Holografický prúd Krystal-Stacku prenáša kompletnú 3D scénu v ASCII/SDF vektoroch pri dátovom toku iba **15 až 30 Kilobajtov za sekundu (KB/s)**!
  - 1000-násobná úspora prenosového pásma umožňuje live 3D taktický prenos aj cez pomalé rádiové alebo satelitné spojenia.

```
[Dron / Hra / 3D Senzor] ──▶ NPU Kompresia ──▶ 20 KB/s Rádio ──▶ AR Okuliare (Hologram)
```

---

### FÁZA 3: Bio-Kybernetická Spätná Väzba (BCI Neuro-Adaptation) (2028)
- **Cieľ**: Napojenie na spotrebiteľské a priemyselné EEG čítače (Muse, Emotiv, OpenBCI).
- **Mechanizmus**:
  - Systém v reálnom čase monitoruje mozgové vlny operátora:
    - **Alpha vlny (8–12 Hz)**: Pokoj, bdelosť, flow-state.
    - **Theta / High-Beta (> 20 Hz)**: Kognitívne preťaženie, stres, panika.
  - **Zatvorená riadiaca slučka (Closed-Loop Bio-Backpressure)**:
    - Keď systém zistí prudký nárast operátorovho stresu, kompozitor okamžite zníži vizuálnu entropiu obrazovky, potlačí rušivé textúry a ponechá len najdôležitejšie navigačné vektory a varovný HUD.
    - Chráni pilotov, chirurgov a operátorov ťažkých CNC strojov pred kognitívnym kolapsom.

---

### FÁZA 4: Autonómne Seba-Vyvíjajúce sa Herné Svety (2029+)
- **Cieľ**: Hry, ktoré nepotrebujú gigabajty statických assetov, ale generujú sa samé.
- **Koncept "The Endless Terminal Universe"**:
  - Malý lokálny jazykový model (SLM) funguje ako Dungeon Master a Game Director.
  - Na základe pohybu hráča a telemetrie SLM v zlomku sekundy generuje:
    - Nové 3D matematické rovnice povrchov (Signed Distance Fields).
    - Pravidlá fyziky a nepriateľských jednotiek.
    - Dynamické Vulkan compute shadery kompilované za behu cez SPIR-V.
  - Nekonečný herný vesmír s veľkosťou inštalácie **pod 50 MB**, schopný bežať na akomkoľvek hardvéri.

---

### FÁZA 5: Swarm Consensus a Decentralizovaná Výpočtová Sieť (2030)
- Nadväzuje na modul [krystal-bitboard](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal-bitboard).
- P2P Gossip protokol umožňujúci miliónom domácich a firemných počítačov spojiť svoje NPU a GPU jednotky do globálneho superpočítača, ktorý autonómne rieši klimatické simulácie, rendering a biomedicínske výpočty.

---

## 🏁 Záver
Krystal-Stack sa z pôvodného experimentu s ASCII a GPU schedulingom stáva ucelenou **víziou novej generácie softvérovej a hardvérovej symbiózy**. Spojenie nízkej energetickej náročnosti (NPU), okamžitej čitateľnosti (ASCII geometria) a umelej inteligencie (SLM režisér) definuje technológiu, ktorá pretrvá aj v ére priestorových počítačov.
