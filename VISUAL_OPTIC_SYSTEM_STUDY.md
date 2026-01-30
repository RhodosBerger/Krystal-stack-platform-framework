# Visual Optic System: Displej ako Riadiaca Entita
*Štúdia o vizuálnom jazyku, sémantickej reflexii a optickej spätnej väzbe*

**Dátum:** 14. Január 2026
**Modul:** Active Optic / Visual Cortex

---

## 1. Definícia: Displej nie je Výstup, je to Stav
V našej architektúre prestávame vnímať monitor ako "Zobrazovač".
Definujeme ho ako **"Event Horizon" (Horizont Udalostí)** systému.
Je to miesto, kde sa **Abstraktná Matematika** (kód) mení na **Vizuálnu Realitu** (obraz).

### 1.1 Optika ako Inšpektor
Tento modul funguje ako "Digitálne Oko", ktoré sa pozerá *späť* do systému cez obrazovku.
*   **Vstup:** Renderované objekty (Okná, Hry, Terminály).
*   **Analýza:** Sú tieto objekty harmonické? Sú v "Grids"? Je tam vizuálny šum (Glitch)?
*   **Výstup:** Príkaz pre Kernel.

## 2. Tematická Závislosť a Organizácia
Objekty na obrazovke nie sú náhodné. Sú to **Tematické Celky**.
*   **Príklad:** Okno prehliadača a Okno textového editora sú vizuálne oddelené, ale v pamäti (Hex Grid) môžu byť "prepletené", ak užívateľ kopíruje text z webu do editora.
*   **Optická Reakcia:** Ak Optika vidí, že tieto dve okná sú vedľa seba, pošle signál do `HyperStateBus`, aby Kernel **zlúčil ich pamäťové stránky** (Memory Deduplication).

## 3. Ovplyvňovanie Procesu (Visual Backpressure)
Toto je kľúčová mechanika. Displej riadi výkon.

### Scenár: Vizuálna Entropia
*   **Situácia:** Na obrazovke je 50 otvorených okien, časticové efekty a notifikácie.
*   **Detekcia:** Optika nameria vysokú **Vizuálnu Entropiu** (Chaos).
*   **Reakcia:** Optika povie Guardianu: *"Užívateľ to nestíha vnímať. Zníž framerate pre neaktívne okná na 1 FPS a uvoľni GPU pre hlavný fokus."*

Týmto sa Displej stáva **Regulátorom**. Ak je obraz príliš zložitý pre človeka, systém automaticky spomalí/zjednoduší backend, aby šetril energiu.

## 4. Vizuálny Jazyk ako Debugger
Chyba v systéme sa neprejaví ako "Blue Screen", ale ako **Vizuálna Metafora**.
*   Namiesto pádu aplikácie sa okno jemne "rozostrí" (Blur).
*   To signalizuje užívateľovi (aj systému): *"Tento proces stráca koherenciu. Potrebuje reštart alebo viac RAM."*

---

*Tento dokument slúži ako základ pre implementáciu `active_optic_compositor.py`.*
