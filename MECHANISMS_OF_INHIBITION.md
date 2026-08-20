# Mechanizmy Inhibície: Napätie vs. Teoretický Rámec
*Pokročilá dokumentácia pre GAMESA/KrystalStack Architektúru*

**Dátum:** 14. Január 2026
**Typ:** Architektonická Štúdia
**Status:** Schválené pre implementáciu

---

## 1. Úvod do Konceptu Inhibície
V našom systéme "inhibícia" neznamená len zastavenie procesu. Je to aktívne, vypočítané rozhodnutie "nevykonať" akciu, aby sa zachovala integrita celku. Rozlišujeme dve hlavné vetvy:

## 2. Inhibícia Napätia (Voltage Inhibition - Hardware Layer)
Tento mechanizmus funguje na najnižšej úrovni (Rust/C vrstva) a reaguje na fyzické limity.

### 2.1 Princíp Fungovania
*   **Vstup:** Telemetria z VRM (Voltage Regulator Module) a teplotných senzorov.
*   **Logika:** Ak `T(core) > T(limit)` alebo `V(droop) > V(threshold)`, Guardian okamžite aplikuje "Hardware Brake".
*   **Reakcia:** Namiesto podtaktovania (čo je pomalé), systém "zmrazí" prísun inštrukcií do pipeline.
*   **Analógia:** Záchranná brzda vo vlaku.

### 2.2 Matematický Model
$$ I_{volt} = \begin{cases} 1 & \text{ak } V_{current} > V_{max\_safe} \\ 0 & \text{inak} \end{cases} $$

## 3. Inhibícia Teoretického Rámca (Theoretical Framework Inhibition - Logic Layer)
Toto je "Prefrontálny Cortex" nášho systému. Inhibuje procesy nie preto, že *nemôže*, ale preto, že *nechce* (na základe predikcie).

### 3.1 Princíp Fungovania
*   **Vstup:** Historické logy, OpenVINO predikcie, "Intuícia" systému.
*   **Logika:** "Aj keď máme voľnú RAM, nespustíme tento proces, pretože vieme, že o 5ms bude potrebná pre renderovanie prioritnejšej textúry."
*   **Reakcia:** Proces je odložený do "Virtual Waiting Room" (vo WSL kontajneri).

### 3.2 Metodológia Rozhodovania
Vychádza z našich prúdov poznania:
1.  **Introspekcia:** Systém analyzuje vlastné minulé zlyhania.
2.  **Abstrakcia:** Vytvára sa model "ideálneho stavu".
3.  **Aplikácia:** Ak sa realita odchyľuje od ideálu, aplikuje sa inhibícia.

---

## 4. Prepojenie s Hexadecimálnou Sustavou
Inhibícia sa aplikuje na konkrétne adresy v mriežke.
*   **Príklad:** Blok `0x7FFF0000` až `0x7FFF00FF` je označený flagom `INHIBITED_THEORETICAL`.
*   Zápis do tohto bloku je presmerovaný do swapu, čím sa chráni kritická pamäť.

---

*Tento dokument slúži ako základ pre vývoj "Guardian Hero" logiky vo verzii 4.0.*
