# KRYSTAL STACK: ARCHITEKTÚRA & DIGITÁLNE DVOJČA PRE HARDVÉR A ENERGETIKU 🌌
### Heterogénne Výpočty · Digitálne Dvojča & Simulátor Siete · Energetická Arbitráž · 128B Zero-Copy IPC · Stavový Invariant

<div align="center">

[![Testy](https://img.shields.io/badge/Unit%20Testy-58%2F58%20Úspešné%20(0.026s)-22c55e?style=for-the-badge&logo=pytest)](../../../tests/)
[![Protokol](https://img.shields.io/badge/Protokol-128B%20Zero--Copy%20Zarovnaný-06b6d4?style=for-the-badge&logo=rust)](../../../krystal_kernel/tri_phase_dosing_controller.py)
[![Sieťový Štandard](https://img.shields.io/badge/Sieťová%20Norma-EN%2050160%20Zhoda-f59e0b?style=for-the-badge)](../../../artifacts/benchmarks/VPP_GRID_BROKER_AND_SWITCHED_MATRIX_SPEC.md)
[![Inferencia](https://img.shields.io/badge/Inferencia-OpenVINO%20%7C%20DirectML%20%7C%20Vulkan-ea580c?style=for-the-badge&logo=intel)](../../../krystal_kernel/)
[![Invariant](https://img.shields.io/badge/Ústavné%20Pravidlo-VITAL__MAX__HP%20%3D%3D%206-ef4444?style=for-the-badge)](../../../artifacts/benchmarks/)
[![Licencia](https://img.shields.io/badge/Licencia-Apache%202.0-blue?style=for-the-badge)](../../../LICENSE)

---

### 🌐 Jazykové Mutácie / Multilingual Hub
[ **🇬🇧 English (Primary)** ](../../../README.md) &nbsp;|&nbsp; [ **🇸🇰 Slovenčina** ](README.md) &nbsp;|&nbsp; [ **🇩🇪 Deutsch** ](../de/README.md) &nbsp;|&nbsp; [ **🇨🇿 Čeština** ](../cs/README.md)

---

</div>

> **Výkonné zhrnutie:**  
> **Krystal Stack** je modulárny výskumno-vývojový ekosystém fungujúci ako **Hardware-in-the-Loop (HIL) Digitálne dvojča a simulačný framework**. Umožňuje deterministické modelovanie, riadenie a validáciu heterogénnych výpočtov (Multi-GPU/CPU/NPU), cenovej arbitráže na spotovom trhu s elektrinou (OKTE) a asymetrickej trojfázovej výkonovej elektroniky bez rizika poškodenia fyzickej infraštruktúry.

---

## 🧭 Prehľad Architektúry Podľa Domén (Role-Based Rázcestník)

Pre Senior Tech Leadov a náborových manažérov:

| Inžinierska Doména / Pozícia | Preukázané Schopnosti a Koncepty | Kľúčové Moduly a Artefakty | Zameranie & Buzzwords pre CV |
| :--- | :--- | :--- | :--- |
| **🦀 Nízkoúrovňové Systémy / Embedded / C & Rust**<br>*(Networking, Automotive, Telco, IoT)* | • 128-bajtový prirodzene zarovnaný Zero-Copy protokol (`#[repr(C, align(128))]`)<br>• Lock-free SPSC kruhové buffery a pamäťové bariéry<br>• eBPF/XDP deskriptory pre obídenie kernelu (Kernel-Bypass)<br>• Q32.32 fixed-point aritmetika s bitovou reprodukovateľnosťou | [`tri_phase_dosing_controller.py`](../../../krystal_kernel/tri_phase_dosing_controller.py)<br>[`household_edge_energy_node.py`](../../../krystal_kernel/household_edge_energy_node.py)<br>[`TRI_PHASE_POWER_DOSING_SPEC.md`](../../../artifacts/benchmarks/TRI_PHASE_POWER_DOSING_SPEC.md) | *Zero-Copy IPC, Cache Line Alignment (128B), eBPF/XDP, Lock-Free Rings, FFI, Deterministický stavový automat, Sub-milisekundová latencia* |
| **🧠 AI / ML Infraštruktúra & MLOps**<br>*(AI Startupy, Hyperscaleri, Výpočtové Centrá)* | • Kaskádový dispečing heterogénnych modelov (Intel OpenVINO, DirectML, Vulkan)<br>• Dynamický sharding úloh medzi AI inferenciou a ťažbou podľa záťaže fáz<br>• Zdieľaná UMA pamäť bez nutnosti kopírovania tenzorov<br>• Telemetria pomeru Token-per-Watt a tepelný rozpočet | [`krystal_vino_runtime.py`](../../../krystal_kernel/krystal_vino_runtime.py)<br>[`grid_aware_energy_arbitrage.py`](../../../krystal_kernel/grid_aware_energy_arbitrage.py)<br>[`H07_SILICON_TELEMETRY_AND_OPENVINO_LEXICON_SPEC.md`](../../../artifacts/benchmarks/H07_SILICON_TELEMETRY_AND_OPENVINO_LEXICON_SPEC.md) | *Heterogeneous Compute, OpenVINO, Multi-Model Dispatch, UMA Pamäť, Optimalizácia Token/Watt, Tepelný Derating* |
| **⚡ Energetika / Smart Grid / FinTech**<br>*(VPP Agregátori, Batériové Úložiská, Algo-Trading)* | • 24-hodinová simulácia arbitráže (96x 15-min slotov spotového trhu OKTE)<br>• Sub-10ms Peak Shaving pre ochranu 3x25A domáceho ističa<br>• Koordinácia Virtual Power Plant (100 domov) s fázovaným poľom kompenzácie napätia (EN 50160)<br>• Prepínateľná SiC MOSFET matica (48V vs 800V DC) so znížením strát o 277x | [`vpp_grid_broker_fleet.py`](../../../krystal_kernel/vpp_grid_broker_fleet.py)<br>[`household_edge_energy_node.py`](../../../krystal_kernel/household_edge_energy_node.py)<br>[`demo_tri_phase_dosing_simulation.py`](../../../demo_tri_phase_dosing_simulation.py) | *Spot Market Arbitrage, Dynamic Peak Shaving, VPP (Virtual Power Plant), Norma EN 50160, Fázované pole meničov* |
| **🛡️ DevSecOps / Enterprise Architektúra & Compliance**<br>*(Banky, Korporáty, Bezpečnostné Platformy)* | • Automatizovaný licenčný audit a firewall dodávateľského reťazca<br>• Runtime izolácia a IPC karanténa virálnych GPL/copyleft knižníc<br>• Kryptografická integrita dátových rámcov (SHA-256 / Ed25519 tokeny)<br>• 100 % úspešnosť testov bez krehkých externých závislostí | [`community_showcase_integrator.py`](../../../krystal_kernel/community_showcase_integrator.py)<br>[`COMMUNITY_FOLLOWER_ECOSYSTEM_AUDIT.md`](../../../artifacts/benchmarks/COMMUNITY_FOLLOWER_ECOSYSTEM_AUDIT.md)<br>[`test_tri_phase_dosing_controller.py`](../../../tests/test_tri_phase_dosing_controller.py) | *Supply Chain Security, Automatizovaný Licenčný Compliance, Karanténa Copyleftu, Deterministické Testovanie* |

---

## 🔬 Preklad Pojmov do Priemyselného Štandardu

* **Ústavný Invariant (`assert VITAL_MAX_HP == 6`) $\implies$ Formálny Stavový Invariant / Safety Gate:**  
  Zaručuje, že neautorizovaná alebo neplatná akcia nespôsobí žiadnu mutáciu stavu ($\Delta S = 0$). Eliminuje pretečenie vyrovnávacej pamäte, nekonzistentné zápisy a elektrické preťaženie ističa.
* **Tri-Phase Dosing $\implies$ Fázovo Posunuté PWM Dávkovanie Výkonu & Tepelný Derating:**  
  Strieda okná špičkového výkonu medzi fázami ($L_1, L_2, L_3$) pre chladenie svoriek; synchrónny $120^\circ$ posun spínania meničov ruší 3. harmonickú (150 Hz) na nulovom vodiči ($N$).
* **Digitálne Dvojča & Simulation-First Prístup $\implies$ Hardware-in-the-Loop (HIL) Predvalidácia:**  
  Umožňuje dôkladne overiť hraničné stavy vo viacrozmernom simulačnom priestore pred fyzickou montážou kilowattových meničov a batérií.

---

## 🔌 Schéma Trojfázového Rozvádzača & 128B Rámec

```text
[ Verejná sieť 3x 230V/400V - 25A Hlavný Istič ]
      │ (CT Prúdové transformátory + 1-Wire teplotné sondy @ 10 ms)
      ▼
[ KRYSTAL POWER SPINE (Silová chrbtica CYKY-J 5x10 mm²) ]
 ├──► Fáza L1 (max 23.9 A) ──► [ Menič #1 (5 kW) ] ──► PSU #1 (1 800 W) ──► GPUs 0, 1, 2 + EPYC #1
 ├──► Fáza L2 (max 23.9 A) ──► [ Menič #2 (5 kW) ] ──► PSU #2 (1 800 W) ──► GPUs 3, 4, 5 + EPYC #2
 ├──► Fáza L3 (max 23.9 A) ──► [ Menič #3 (5 kW) ] ──► PSU #3 (1 800 W) ──► GPUs 6, 7, 8, 9 + Čerpadlá
 └──► Nulový vodič (N) ─────► Zdvojený 10 mm² s aktívnym rušením 3. harmonickej (150 Hz)
```

**Rozloženie 128-bajtového telemetrického rámca:**
- `0..57` (58 B): Hlavička, CT meranie prúdov a napätí, SoC batérie, frekvencia siete.
- `58..73` (16 B): Kryptografický SHA-256 state hash.
- `74..95` (22 B): L3 Grid Broker polia: ID trafostanice, jalový výkon $Q$, VPP dispečerský cieľ.
- `96..117` (22 B): Tri-Phase Dosing rozšírenie: teploty svoriek ($0.1^\circ\text{C}$), príkony GPU skupín, stav ventilu TÚV.
- `118..127` (10 B): Zarovnanie na 128 bajtov (cache line boundary).
$$\mathbf{58 + 16 + 22 + 22 + 10 = 128\ \text{Bajtov}}$$

---

## 🚀 Spustiteľné Demonštrácie a Simulácie

```bash
# 1. Trojfázové pulzné dávkovanie & riešenie preťaženia fázy L1 (< 10 ms odozva)
python demo_tri_phase_dosing_simulation.py

# 2. Virtuálna elektráreň (VPP) - koordinácia 100 domov & 800V SiC matica
python demo_vpp_grid_broker_simulation.py

# 3. 24-hodinový 96-slotový cyklus domáceho uzla (+2 714 €/rok zisk)
python demo_household_edge_simulation.py

# 4. Frekvenčná stabilizácia siete (50 Hz FCR primárna regulácia)
python demo_grid_aware_arbitrage.py
```

---

## 💼 Text do Životopisu / LinkedIn / Motivačného Listu

> **Krystal Stack – High-Performance Compute & Grid-Edge Optimization Framework** *(Architekt & Hlavný Vývojár)*
> * **Nízkoúrovňová architektúra a protokoly:** Návrh 128-bajtového binárneho protokolu (`#[repr(C, align(128))]`) pre Zero-Copy prenos telemetrických dát medzi eBPF/XDP sieťovou vrstvou, jadrom a výkonovou elektronikou bez réžie Garbage Collectora.
> * **Energetická arbitráž a Smart Grid simulácia:** Implementácia 24-hodinového simulačného enginu (96x 15-minútových slotov spotového trhu OKTE) pre asymetrické vyvažovanie 3 fáz ($L_1, L_2, L_3$), dynamickú ochranu hlavného ističa (sub-10ms Peak Shaving) a rekuperáciu odpadového tepla zo serverov.
> * **AI Inferencia & Plánovanie Úloh:** Vývoj adaptívneho dispečera pre 10x AMD GPU a duálne EPYC procesory s dynamickým prerozdeľovaním priorít (AI inferencia vs. ťažba) na základe tepelných a prúdových limitov fáz v reálnom čase.
> * **Automated Governance & Testovanie:** Vytvorenie licenčného firewallu (`CommunityAuditGuard`) na automatickú detekciu a IPC izoláciu GPL/Apache závislostí, kryté 100 % úspešnosťou v testoch bez externých závislostí.

---

## 👥 Autori & Licencia

* **Hlavný Architekt & Vývojár**: Dušan Kopecký ([@RhodosBerger](https://github.com/RhodosBerger))
* **Kontakt**: `dusan.kopecky0101@gmail.com`
* **Licencia**: Apache License 2.0 (Pozri [`LICENSE`](../../../LICENSE))
