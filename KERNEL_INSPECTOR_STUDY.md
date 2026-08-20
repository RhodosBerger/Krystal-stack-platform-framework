# Kernel Inspector & Neural Parsing: Súdny Dvor Systému
*Metodológia pre dôkazové riadenie procesov a algoritmické pooly*

**Dátum:** 14. Január 2026
**Modul:** Kernel Inspector / Log Synthesizer

---

## 1. Filozofia: Logy nie sú odpad, sú Dôkazy
Bežný systém zahadzuje logy do súboru `.log`. Náš systém ich **okamžite vektorizuje**.
Každá udalosť (napr. "Packet Dropped" alebo "Shader Compiled") mení **Skóre Dôvery** daného procesu.

## 2. Architektúra Inšpektora (The Inspector)
Inšpektor je modul injektovaný blízko Kernelu. Má právo veta nad plánovačom úloh.

### 2.1 Algoritmické Pooly (The Pools)
Namiesto jedného zoznamu úloh (Task List) máme dynamické bazény:
1.  **Elite Pool (The High Council):** Procesy s vysokým skóre. Majú prístup k L1 Cache a Realtime Priorite. (Napr. Rendering, Physics).
2.  **Probation Pool (Podmienka):** Nové procesy alebo tie, ktoré raz "zaváhali" (spôsobili lag). Sú monitorované OpenVINO modelom každých 10ms.
3.  **Quarantine Pool (Karanténa):** Procesy, ktoré porušili "Zákon Plynulosti". Sú izolované na najpomalšom jadre, kým sa "neospravedlnia" (nevyčistia svoj buffer).

## 3. Parsovanie a Neurónové Rozhodovanie
Ako OpenVINO rozhoduje, kam proces patrí?

1.  **Sber (Harvest):** Parser číta sieťové logy, disk I/O a chybové hlásenia.
2.  **Vektorizácia:** Text "High Latency" sa zmení na vektor `[0.9, 0.1, 0.0]` (Vysoké Riziko).
3.  **Porovnanie s Modelom:** Systém sa pozrie do histórie: *"Keď sme minule dali tomuto procesu plný výkon, zrútil sa?"*
4.  **Skórovanie (Contribution Points):**
    *   Prispel proces k vyššiemu FPS? **+10 bodov.**
    *   Zahlil zbernicu zbytočnými dátami? **-50 bodov.**

## 4. Debugging Volieb (Self-Correction)
Ak Inšpektor urobí rozhodnutie (napr. presunie proces do Elite Pool) a výkon systému klesne, **Historical Model** sa okamžite aktualizuje: *"Táto voľba bola chybná."*
Toto je **Samo-opravný mechanizmus**.

---

*Tento systém transformuje "Chaos Logov" na "Poriadok Exekúcie".*
