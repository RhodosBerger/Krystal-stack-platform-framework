# Register tvrdení

Každé tvrdenie, ktoré sa objavilo v kóde, testoch alebo dokumentácii k triáži, s jedným stavom:

`DOKÁZANÉ` · `NAMERANÉ` · `MODEL` (parameter alebo odvodenina z parametrov) · `NEOVERENÉ` ·
`OPRAVENÉ` (pôvodné znenie bolo nepravdivé, nahradené) · `STIAHNUTÉ` (bez náhrady) · `ZAMIETNUTÉ` (hypotéza vyvrátená meraním)

*Posledná aktualizácia:* 2026-10-09.

## A. Triáž a plánovanie (`psychological_coping_cortex_mcp_bridge.py`)

| # | Tvrdenie | Stav | Doklad |
|---|---|---|---|
| A1 | Triáž je totálna a deterministická | DOKÁZANÉ | [Veta 1.1](01_triazny_operator.md#veta-11-totálnosť-a-determinizmus) |
| A2 | Triáž vyberá zo všetkých 7 mechanizmov | **OPRAVENÉ** → vyberá len 4 | [Veta 1.2](01_triazny_operator.md#veta-12-dosiahnuteľná-množina), `test_h2_*` |
| A3 | Neznámy stav (NaN/inf) vedie k bezpečnej politike | DOKÁZANÉ (po oprave H3) | [Veta 1.3](01_triazny_operator.md#veta-13-fail-safe), `test_h3_*` |
| A4 | Núdzová úloha je vždy naplánovaná pred nominálnou | DOKÁZANÉ (po oprave H1); pôvodne nepravdivé | [Vety 2.1, 2.2](02_planovanie_a_poradie.md), `test_h1_*` |
| A5 | Vyšší stres ⇒ vyššia urgentnosť | **NEPLATÍ** pre $c$ a $e$; platí pre $\theta$ | [Vety 2.3, 2.4](02_planovanie_a_poradie.md#22-nemonotónnosť-priority) |
| A6 | Poradie minimalizuje vážené čakanie | MODEL — len pri váhach z vety 2.6 | [Veta 2.6](02_planovanie_a_poradie.md#veta-26-implikované-váhy-aktuálnej-politiky), H-S1 |
| A7 | Časové rozpočty 50/120/350/500 µs | MODEL | bez merania |
| A8 | UI plánovač zohľadňuje teplotu | OPRAVENÉ (H4) | `test_h4_*` |
| A9 | Hint špekulácie zodpovedá hĺbke špekulácie | DOKÁZANÉ (po oprave H9) | [Veta 1.5](01_triazny_operator.md#veta-15-konzistencia-špekulácie), `test_h9_*` |
| A10 | Prahové prepínanie je stabilné | **NEPLATÍ** pri $h=0$ a šume | [Dôsledok 3.3](03_riadenie_a_stabilita.md#dôsledok-33-h--0-nič-negarantuje) |

## B. Klasifikácia synoným

| # | Tvrdenie | Stav | Doklad |
|---|---|---|---|
| B1 | „Úspora 78,5 % pásma embeddingov“ (konštanta) | **STIAHNUTÉ** — konštanta nezávislá od vstupu, odporovala vlastným počtom (99,6 %) | repro H6, `test_h6_*` |
| B2 | Úspora odvodená z `MODEL_VOCAB_SIZE`/`MODEL_CLUSTER_SIZE` | MODEL, `figure_status=MODEL_UNVERIFIED` | kód |
| B3 | Vyhľadanie embeddingu stojí $O(|V|)$ na token | **OPRAVENÉ** — gather je $O(d)$; $O(|V|)$ platí pre výstupnú projekciu | H-L1 |
| B4 | Klasifikácia je presná | OPRAVENÉ — podreťazce dávali falošné zhody („start“→„art“); voľné prefixy („mutex“→„mute“) nahradené pravidelnými príponami (-s, -es, -d, -ed, -ing) | `test_h5_*` |

## C. Kremík a výkon (`SiliconDieUnlocker`)

| # | Tvrdenie | Stav | Doklad |
|---|---|---|---|
| C1 | „i5-1135G7 == i7-1165G7, parita potvrdená“ | **OPRAVENÉ** — rovnaký návrh čipu, ale i5 má fúzami vypnuté EU (80 vs 96) a L3 (8 vs 12 MB); fúzy nie sú vratné cez MSR. Nič sa nedetegovalo. | `die_parity_note`, test |
| C2 | „+193,8 % priepustnosti“ | **STIAHNUTÉ** — vzorec násobil takt jadra taktom ringu (GHz²), nemá význam priepustnosti | repro H8b |
| C3 | +88,9 % ako horná hranica | MODEL — pomer taktov 3,4/1,8 pri výpočtovo viazanej záťaži a konštantnom IPC | kód, test |
| C4 | Zápisy MSR 0x610 / 0x620 / 0x618 | **NEOVERENÉ, neprebehli** — `msr_writes_performed=False` | kód |
| C5 | DRAM RAPL (0x618) je na tomto čipe dostupný a „odblokovaný“ | **OPRAVENÉ** — neoverené; report teraz vracia `stock_dram_rapl_limited=None` (neznáme) a `unlocked_dram_rapl_active=False` | test silicon report |
| C6 | Udržateľný takt 3,4 GHz pri 32 W, AF 0,9118, −40 mV | MODEL | bez merania |

## D. Nadväzujúce merania z auditu (prevzaté, nie overené týmto podprojektom)

| # | Tvrdenie | Stav | Zdroj |
|---|---|---|---|
| D1 | Drift energie symplektického integrátora 4,34·10⁻⁴ / 10 000 krokov | NAMERANÉ (podľa auditu; počet behov neuvedený) | [audit §2](../docs/PROJECT_AUDIT_AND_COMPUTE_PATTERNS.md) |
| D2 | `FastRingBuffer` 4,39× rýchlejší než `queue.Queue` | NAMERANÉ (mikrobenchmark podľa auditu; stroj, dátum a počet behov neuvedené) | audit §2 |
| D3 | `FastRingBuffer` je lock-free | OPRAVENÉ — kvôli GIL nie | audit §2 |
| D4 | SDF raymarch 37–46 ms/snímka pri 96×40 | NAMERANÉ (podľa auditu; počet behov neuvedený) | audit §2 |
