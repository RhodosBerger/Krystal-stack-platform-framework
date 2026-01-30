# Coding Scheme Paradigma: Jazyk Vzťahov (Krystal-Lang)
*Nový spôsob uchopenia informácií pre telemetriu a riadenie*

**Dátum:** 14. Január 2026
**Vrstva:** Interpretačná Vrstva (Shell)

---

## 1. Filozofia: Od Príkazov k Vzťahom
Bežný kód je imperatívny ("Urob toto").
Krystal-Lang je **relačný** ("Toto súvisí s týmto takto").

Namiesto písania skriptov definujeme **Topológiu Problému**.
*   Nie: `if temp > 90 then shutdown()`
*   Ale: `[TEMP_SENSOR] --(CORRELATES)--> [SHUTDOWN_PROTOCOL] @ {threshold: 90}`

## 2. Syntax Jazyka
Jazyk je navrhnutý pre rýchle písanie v Shell/Bash prostredí.

`[ZDROJ] [OPERÁTOR] [CIEĽ] @ {META_DATA}`

### Operátory:
*   `->` **Direct Flow:** Priamy tok dát alebo príkazu.
*   `=>` **Inductive Link:** Logická väzba (ak sa stane A, stane sa B).
*   `::` **Static Relation:** Definícia štruktúry (A je súčasťou B).
*   `<>` **Sync:** Obojstranná synchronizácia (Entangled Memory).

## 3. Telemetria ako Sieť Vzťahov
Logovanie v tomto jazyku nevytvára riadky textu, ale **Graf**.
Keď nastane chyba, systém nevyplľuje "Error at line 5", ale ukáže **pretrhnuté spojenie** v grafe.

### Príklad Debuggingu:
**Vstup:** `kry status --graph`
**Výstup:**
```
[CPU_0] --(OK)--> [RAM]
[CPU_1] --(LATENCY 500ms)--> [SSD]  <-- !!! PROBLEM NODE
```

## 4. Bash Integrácia
Interpreter je navrhnutý ako "Pipe-friendly" nástroj.
Môžeme reťaziť príkazy systému s Linux nástrojmi:
`kry telemetry | grep "LATENCY" | kry optimize`

Týmto sa otvára možnosť pre "Shell-based AI Management".
