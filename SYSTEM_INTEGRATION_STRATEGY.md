# Stratégia Integrácie: Stromové Vedomie (Yggdrasil)
*Metodológia vzájomného informovania komponentov a riadenia Kernelu*

**Dátum:** 14. Január 2026
**Vrstva:** Integračná (Holistická)

---

## 1. Stromová Štruktúra Dát (The State Tree)
Namiesto plochých logov budujeme **Hierarchický Strom Stavov**. Každý komponent je vetva, ktorá nesie svoje "ovocie" (dáta).

**Štruktúra `Yggdrasil_State.json`:**
```json
{
  "ROOT": {
    "HARDWARE_LAYER": {
      "Kernel": { "ram_pressure": 45, "page_faults": 12 },
      "Sensors": { "temp_cpu": 65, "vrm_voltage": 1.2 }
    },
    "COGNITIVE_LAYER": {
      "OpenVINO": { "state": "GAMMA", "focus": "COMPILER_THREAD" },
      "Axioms": { "active_rules": ["LAW_OF_HIGH_LOAD"] }
    },
    "TOPOLOGY_LAYER": {
      "HexGrid": { "fragmentation": 0.1, "active_cells": 4096 },
      "Cache": { "hit_rate": 0.98 }
    }
  }
}
```

## 2. Tok Informácií (Mutual Awareness)
Komponenty nie sú izolované.
*   **Guardian** vidí do **Kernelu**: "Vidím, že máš veľa Page Faults."
*   **Grid** vidí do **OpenVINO**: "Vidím, že si v stave GAMMA, takže pripravujem rýchle buňky."

## 3. Stratégie Riadenia Windows Kernelu
Na základe analýzy Stromu (Tree Walker) systém aplikuje pokročilé stratégie:

### A. Stratégia "Thermal Affinity"
*   **Podmienka:** Senzory hlásia lokálne prehriatie Jadra 0.
*   **Akcia:** Strategos pošle príkaz Kernelu: `SetProcessAffinityMask`.
*   **Výsledok:** Presunie výpočet na chladnejšie Jadro 4.

### B. Stratégia "Entangled Threads" (L3 Cache Locality)
*   **Podmienka:** Krystal-Lang detekuje silný vzťah `[Thread_A] <-> [Thread_B]`.
*   **Akcia:** Strategos prikáže Kernelu držať tieto vlákna na rovnakom CCX (Core Complex).
*   **Výsledok:** Zdieľanie L3 Cache, nulová latencia pri komunikácii.

### C. Stratégia "Gamma Locking"
*   **Podmienka:** OpenVINO prejde do stavu GAMMA (Hyper-Focus).
*   **Akcia:** Strategos volá `VirtualLock` (Windows API).
*   **Výsledok:** Zakáže swapovanie pamäte tohto procesu na disk. Musí ostať v RAM.

---

*Tento dokument definuje pravidlá pre `system_tree_orchestrator.py`.*
