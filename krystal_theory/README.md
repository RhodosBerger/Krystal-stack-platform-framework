# Krystal Theory — teoretický podprojekt

> **Iba teória.** Tento adresár neobsahuje spustiteľný kód, benchmarky ani marketingové čísla.
> Obsahuje definície, axiómy, vety s dôkazmi, falzifikovateľné hypotézy s protokolmi testovania
> a register tvrdení. Implementácia žije v [`krystal_kernel/`](../krystal_kernel/), merania v
> [`docs/PROJECT_AUDIT_AND_COMPUTE_PATTERNS.md`](../docs/PROJECT_AUDIT_AND_COMPUTE_PATTERNS.md).

## Účel

1. **Oddeliť to, čo je dokázané, od toho, čo je namerané, a od toho, čo je len domnienka.**
2. Dať každému rozhodovaciemu pravidlu v kóde presnú matematickú podobu, aby sa dalo
   overiť, *čo* pravidlo garantuje a *čo* nie.
3. Vyrábať ciele pre inžiniersku prácu: každá veta, ktorej predpoklady kód nespĺňa,
   je konkrétna úloha; každá hypotéza je konkrétne meranie.

## Rozsah (verzia 0.1)

| Oblasť | Stav |
|---|---|
| Teória plánovania a riadenia — triážny operátor z [`psychological_coping_cortex_mcp_bridge.py`](../krystal_kernel/psychological_coping_cortex_mcp_bridge.py) | **rozpracované (v0.1)** |
| Hamiltonovský / cyklický organizmus | mimo rozsahu v0.1 (pozri [CYCLIC_THEORY…](../docs/research/CYCLIC_THEORY_AND_ALTERNATIVE_ARCHITECTURAL_SCHEMAS.md)) |
| Teória informácie, termodynamika výpočtu, SDF geometria | mimo rozsahu v0.1 |

## Štruktúra

| Súbor | Obsah |
|---|---|
| [00_notacia_a_konvencie.md](00_notacia_a_konvencie.md) | Značenie, stavové štítky, pravidlá dokazovania |
| [01_triazny_operator.md](01_triazny_operator.md) | Formálna definícia triáže, totálnosť, rozklad stavového priestoru, fail-safe |
| [02_planovanie_a_poradie.md](02_planovanie_a_poradie.md) | Striktné poradie, (ne)monotónnosť, Smithovo pravidlo a implikované váhy |
| [03_riadenie_a_stabilita.md](03_riadenie_a_stabilita.md) | Prepínanie pri prahu (chattering), hysterézia, termálny reléový cyklus |
| [04_hypotezy_a_protokoly.md](04_hypotezy_a_protokoly.md) | Falzifikovateľné hypotézy s presnými protokolmi a kritériami zamietnutia |
| [REGISTER_TVRDENI.md](REGISTER_TVRDENI.md) | Každé číselné alebo kvalitatívne tvrdenie a jeho stav |
| [OTVORENE_PROBLEMY.md](OTVORENE_PROBLEMY.md) | Otvorené problémy a z nich odvodené ciele |

## Pravidlá podprojektu

1. **Žiadny kód.** Výnimkou sú krátke výpisy v blokoch ```` ```text ```` pri citovaní pravidiel z kódu.
2. **Každé tvrdenie má štítok** (pozri [00](00_notacia_a_konvencie.md#stavové-štítky)).
3. **Väzba na kód je explicitná.** Keď veta hovorí o funkcii, uvádza súbor a symbol.
   Keď sa kód zmení, veta sa musí znovu overiť; v hlavičke súboru je dátum poslednej kontroly.
4. **Čísla len s pôvodom.** Číslo z merania musí mať stroj, dátum a počet behov
   (pravidlo [engineering-verification-and-honesty](../.agents/rules/engineering-verification-and-honesty.md)).
   Inak je to *parameter modelu* alebo *cieľ*.
5. **Zamietnutá hypotéza sa nemaže.** Presúva sa do registra so stavom `ZAMIETNUTÉ` a dôvodom.

*Posledná kontrola voči kódu: 2026-10-09.*
