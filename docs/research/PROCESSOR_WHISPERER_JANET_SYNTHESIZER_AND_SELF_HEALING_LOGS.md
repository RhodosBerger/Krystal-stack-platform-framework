# PROCESSOR WHISPERER, JANET SCRIPT SYNTHESIZER & SELF-HEALING GF(2) LOGS
**Architektúra Skriptovacieho Syntetizátora, Hexadecimálnych Bytecode Streamov, Samoopravovacích Binárnych Matíc a Našepkávača pre Procesor**
*Autor: Dušan Kopecký & Krystal-Stack Hardware Systems Council (2026)*

---

## 1. Prehľad Systému & Motivácia

V nadväznosti na požiadavku vytvorenia vlastného skriptovacieho syntetizátora a hardvérového našepkávača bol implementovaný modul:
📁 [`krystal_kernel/processor_whisperer.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/processor_whisperer.py)

Tento systém prepája 4 kľúčové vrstvy:
1. **Janet Script Engine Synthesizer:** Generuje S-expression skripty priamo odvodené z interných funkcií enginu a kompiluje ich do 64-bitovo zarovnaného binárneho bytecode s hexadecimálnym logovaním.
2. **Binary Matrix Self-Healing Log (GF(2) Hamming/SECDED):** Využíva binárne paritné matice nad poľom $\mathbb{F}_2$ na určenie pôvodu prepnutia prúdu (C/P-states) a zápisov do L1 vs L3 pamäte, pričom automaticky opravuje porušené záznamy.
3. **Instruction Set Meta Governor (PGO):** Na základe meta-informácií z logov dynamicky ovplyvňuje inštrukčné sady procesora (AVX-512 FMA, AVX2 unrolling, `PREFETCHT0`, `MOVNTDQ` non-temporal stores).
4. **Processor Instruction Whisperer ("Našepkávač pre procesor"):** Operuje bleskovo v operačnej pamäti (RAM ring buffer) pre nanosekundové predikcie a logy zapisuje asynchrónne na SSD bez blokovania výpočtového vlákna.

---

## 2. Janet Script Synthesizer & Hexadecimálny Bytecode

Syntetizátor generuje S-výrazy (podobné jazyku Janet / Lisp), ktoré mapujú reálne funkcie enginu:
```clojure
# Krystal-Stack Janet Engine Script: NeoPraha_Alchemical_Matrix
(module krystal/engine-core :seed 42)

  # 1. Non-negotiable Architectural Invariant
  (vital/assert-hp :expected 6)

  # 2. Multioctave Procedural Terrain Elevation
  (terrain/multioctave :seed 42 :octaves 6 :height-scale 4.5)

  # 3. Skeuomorphic Bohemian Alchemical Geometries
  (sdf/chalice :rot-y 1.25 :stem-h 0.55 :bowl-r 0.78)
  (sdf/athame :crossguard-w 0.65 :blade-l 1.45 :tilt-x 0.25)

  # 4. Urban Morphic Google Maps Footprint Extrusion
  (urban/extrude-spire :city "praha_old_town" :height-m 48.0 :gables 2)

  # 5. Artillery Plunging Mortar Dispersion
  (ballistics/mortar-ellipse :range-m 180.0 :apex-h 42.0 :dispersion-sigma 4.2)

  # 6. Spacetime Bullet-Time Dilation & Coxeter Mirror
  (bullet-time/dilate :factor 0.35 :duration-s 1.2)
  (coxeter/dihedral-fold :folds 6 :amplitude 1.8)

  # 7. Procedural Halftone Raster & Economic Minting
  (dither/bayer-sample :matrix :bayer-4x4 :luma 0.72)
  (metabolic/pulse :phase :beta :vital-hp 6)
  (ledger/mint-credits :amount 250 :faction "Kryštálový Kmeň")
```

### Binárna Reprezentácia & Hexadecimálny Dump
Kompilátor prevádza tieto inštrukcie do 8-bajtových inštrukčných slov zarovnaných na 64 bitov s hlavičkou `0x4B 0x53 0x59 0x4E` (`KSYN`):
```
0000  4B 53 59 4E 01 00 00 2A  01 00 00 06 00 00 00 00  |KSYN...*........|
0010  02 00 00 06 40 90 00 00  03 00 00 37 00 00 00 4E  |....@......7...N|
0020  04 00 00 91 00 00 00 19  05 00 00 30 00 00 00 02  |...........0....|
0030  06 00 00 B4 00 00 00 2A  07 00 00 23 00 00 00 78  |.......*...#...x|
0040  09 00 00 06 00 00 00 12  08 00 00 04 00 00 00 48  |...............H|
0050  0A 00 00 02 00 00 00 06  0B 00 00 FA 00 00 00 01  |................|
0060  00 00 00 00 00 00 00 00                           |........|
```

---

## 3. Samoopravovací Log pomocou Binárnych Matíc v $\mathbb{F}_2$

Pre presné rozlíšenie hardvérových prechodov kódujeme stav do 4 dátových bitov $(d_1, d_2, d_3, d_4)$:
- $d_1$: `CURRENT_SWITCH` (zmena frekvencie P-state, prepnutie C-state, RAPL prúdový limit)
- $d_2$: `L1_CACHE_WRITE` (zápis do L1 dátovej cache, alokácia stack frame, $\sim 1\text{ ns}$)
- $d_3$: `L3_CACHE_WRITE` (spätný zápis do L3 LLC, dirty line eviction, $\sim 12\text{ ns}$)
- $d_4$: `DRAM_BUS_SPILL` (vytečenie do hlavnej pamäte, memory wall)

### Paritná Matica $H$ nad $\mathbb{F}_2$ (Hamming [7, 4]):
Generujú sa 3 paritné bity:
$$p_1 = d_1 \oplus d_2 \oplus d_4$$
$$p_2 = d_1 \oplus d_3 \oplus d_4$$
$$p_3 = d_2 \oplus d_3 \oplus d_4$$

Kódové slovo $c = [p_1, p_2, d_1, p_3, d_2, d_3, d_4]$.

Kontrolná matica $H$:
$$H = \begin{pmatrix} 1 & 0 & 1 & 0 & 1 & 0 & 1 \\ 0 & 1 & 1 & 0 & 0 & 1 & 1 \\ 0 & 0 & 0 & 1 & 1 & 1 & 1 \end{pmatrix}$$

### Syndrómové Dekódovanie & Samooprava:
Pre prijatý vektor $r$ sa spočíta syndróm $s = H \cdot r^T \pmod 2$:
- Ak $s = 0$, záznam je neporušený.
- Ak $s \ne 0$, hodnota syndrómu udáva presný index bitu (1..7), ktorý bol poškodený.
- Invertovaním chybného bitu log vykoná **okamžitú samoopravu v pamäti** a spoľahlivo určí, či udalosť pramenila z prepnutia prúdu, L1 cache alebo L3 cache.

---

## 4. Ovplyvňovanie Inštrukčných Sád (ISA Meta Governor)

Meta-informácie zbierané cez log dynamicky riadia výber vektorových a pamäťových inštrukcií:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   DYNAMICKÝ VÝBER INŠTRUKČNEJ SADY (PGO ISA SELECTOR)                  │
├──────────────────────┬────────────────────────┬────────────────────────────────────────┤
│ Stav Pamäte / Prúdu  │ Odporúčaná ISA         │ Optimalizácie a Prefetch               │
├──────────────────────┼────────────────────────┼────────────────────────────────────────┤
│ L1 Hit Ratio >= 85%  │ AVX512_FMA_UNROLLED    │ 8x rozbalenie slučiek, FMA inštrukcie  │
│ L3 Aktivita 15 - 45% │ AVX2_PREFETCH_AHEAD    │ 4x unroll, PREFETCHT0 64B dopredu      │
│ L3 Spilling > 45%    │ STREAMING_NT_STORES    │ MOVNTDQ non-temporal bypass zbernice   │
│ Prúdové prepnutie    │ SCALAR_COMPACT_SSE4    │ Zúženie šírky mikro-operácií, úspora W │
└──────────────────────┴────────────────────────┴────────────────────────────────────────┤
```

---

## 5. Našepkávač pre Procesor: In-Memory + Asynchrónne SSD Logovanie

```
                    ┌───────────────────────────────────────────────┐
                    │               VÝPOČTOVÝ KERNEL                │
                    └───────┬───────────────────────────────▲───────┘
                            │                               │
                1. Hardware │                   4. Okamžitý │ In-Memory
                   Udalosť  ▼                      ISA Hint │ (20-50 µs)
                    ┌───────────────────────────────┴───────┐
                    │     PROCESSOR INSTRUCTION WHISPERER   │
                    │   (RAM Ring Buffer: 4096 záznamov)    │
                    └───────┬───────────────────────────────┘
                            │
                 2. Zápis do│ Asynchrónny Non-blocking
                    Fronty  ▼ Flush (100 ms tick)
                    ┌───────────────────────────────────────┐
                    │      SSD PERSISTENCE WORKER THREAD    │
                    │   (logs/whisperer_ssd_audit.hexlog)   │
                    └───────────────────────────────────────┘
```

1. **In-Memory Vrstva (RAM Ring Buffer):**
   - Rýchly kruhový buffer v operačnej pamäti.
   - Poskytuje inštrukčné našepkanie (`whisper_instruction_hint`) v časoch $20\text{--}50\text{ }\mu\text{s}$.
2. **Asynchrónna SSD Vrstva:**
   - Samostatné vlákno na pozadí dávkovo zapisuje hexadecimálne záznamy na disk bez spomaľovania hlavného CPU vlákna.

---

## 6. REST API Endpointy na Serveri

Integrované v [`krystal_web_hub/server.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/server.py):
- `GET /api/whisperer/synthesize`: Syntetizuje Janet S-expression skript a vráti hexadecimálny dump.
- `GET /api/whisperer/status`: Vráti stav RAM ring buffera, samoopravovacieho logu a ISA odporúčanie.
- `POST /api/whisperer/pulse`: Simuluje príchod hardvérovej udalosti, vykoná syndrómové dekódovanie a vráti našepkanú inštrukciu.

---

## 7. Overovacia Testovacia Sada

Vytvorená testovacia sada [`verify_processor_whisperer_and_synthesizer.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/verify_processor_whisperer_and_synthesizer.py) dosiahla **100% úspešnosť**:
```
=================================================================
KRYSTAL-STACK: VERIFYING PROCESSOR WHISPERER & JANET SYNTHESIZER
=================================================================
[TEST 1] Testing Janet Script Synthesizer & Hexadecimal Bytecode...
  -> Synthesizer OK: 12 S-expressions, 104 B bytecode, valid hex dump.
[TEST 2] Testing GF(2) Binary Matrix Self-Healing & Syndrome Attribution...
  -> Binary matrix OK: 7/7 single-bit flip anomalies repaired, total healed=7.
[TEST 3] Testing Instruction Set Meta Governor...
  -> ISA Meta Governor OK: Dynamic tier selection (AVX-512 FMA, Streaming NT Stores, Scalar SSE4) verified.
[TEST 4] Testing Processor Whisperer & Asynchronous SSD Flush...
  -> Processor Whisperer OK: In-memory query took 20.6µs, SSD log flushed (2944 bytes).
=================================================================
ALL TESTS PASSED WITH 100% INTEGRITY!
=================================================================
```
