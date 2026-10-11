# 04 — Hypotézy a protokoly

Každá hypotéza má: **tvrdenie**, **predpoklady**, **protokol**, **kritérium zamietnutia** a **stav**.
Stav sa mení len meraním; namerané údaje sa ukladajú ako JSON artefakt s menom stroja,
dátumom a počtom behov.

Cieľový stroj, ak nie je uvedené inak: Intel Core i5-1135G7, Iris Xe (bez NPU/TPU), Windows.

---

## H-R1 — Nezávislosť vzoriek teploty pri prahu
- **Tvrdenie.** Pri ustálenej záťaži s priemerom $\theta$ blízko $\theta^\*$ je empirická frekvencia
  prepínania bezpamäťového režimu v 95 % intervale spoľahlivosti okolo $2\hat p(1-\hat p)$ (veta 3.1).
- **Predpoklady.** A3.1.
- **Protokol.** 1) Ustálená záťaž s $\theta$ blízko 85 % $T_{j,\max}$. 2) Vzorkovanie $\theta$ pri 10 Hz
  počas 600 s, 3 behy. 3) $\hat p$ = podiel vzoriek $\ge \theta^\*$; $\hat s$ = podiel susedných párov so zmenou.
  4) Autokorelácia $\rho_1$ postupnosti $\mathbf{1}[\theta_k \ge \theta^\*]$.
- **Zamietnutie.** $|\hat s - 2\hat p(1-\hat p)|$ mimo 95 % IS v ≥ 2 z 3 behov, alebo $|\rho_1| > 0{,}2$.
  Pri zamietnutí platí veta 3.1 len ako horná hranica pre nezávislý šum a pre 3.2 treba
  odhad $\delta$ z rozsahu šumu, nie z rozptylu.
- **Stav.** NEOVERENÉ (chýba zdroj teploty s dostatočnou frekvenciou).

## H-R2 — Politika REG znižuje príkon
- **Tvrdenie.** Prepnutie záťaže do politiky REG zníži priemerný príkon balíka o aspoň 10 %.
- **Predpoklady.** Existuje záťaž, ktorá politiky skutočne rešpektuje (dnes ich kód len *vracia*).
- **Protokol.** Striedanie 60 s REG / 60 s PROJ, 10 cyklov, príkon z RAPL čítačov
  (iba čítanie, bez zápisu MSR), porovnanie priemerov párovým testom.
- **Nástroj (vyžaduje schválenie).** Čítanie MSR/RAPL na Windows potrebuje ovládač v režime jadra
  a administrátorské práva. Konkrétny nástroj musí používateľ schváliť pred meraním (pravidlo 2 v
  `engineering-verification-and-honesty`); bez schválenia je H-R2 blokovaná. Blokovanie zo strany
  Application Control / antivírusu sa neobchádza, len sa zaznamená.
- **Zamietnutie.** Rozdiel < 10 % alebo $p > 0{,}05$. Pri zamietnutí sú vety 3.5–3.6 pre tento systém
  bezpredmetné (príkon nezávisí od režimu).
- **Stav.** NEOVERENÉ; blokované cieľom G4 (spotrebiteľ politík).

## H-R3 — Tepelný model 1. rádu
- **Tvrdenie.** Skok príkonu $P_L \to P_H$ dá priebeh $T(t)$ s $R^2 \ge 0{,}95$ pri prekladaní exponenciálou (lema 3.4).
- **Protokol.** 5 skokov, preloženie $T_\infty$ a $\tau$ metódou najmenších štvorcov, uložiť $\hat R_{\mathrm{th}}, \hat\tau$.
- **Zamietnutie.** Medián $R^2 < 0{,}95$ → model 1. rádu nestačí, vety 3.5–3.6 treba rozšíriť na 2. rád.
- **Predikcia na overenie.** S $\hat\tau, \hat T_L, \hat T_H$ predpovedať periódu cyklu z vety 3.6;
  zamietnuť, ak sa nameraná perióda líši o > 20 %.
- **Stav.** NEOVERENÉ.

## H-S1 — Zamýšľané váhy úloh
- **Tvrdenie (dizajnové).** Autor považuje úlohu pod kontenciou za ≤ 24 % tak dôležitú ako
  nominálnu úlohu (veta 2.6).
- **Protokol.** Nie je to meranie: autor buď potvrdí váhy, alebo zadá vlastné $w_m$; rank sa potom
  prepočíta podľa WSPT a porovná s `TRIAGE_POLICY`.
- **Zamietnutie.** Autor zadá $w_{\mathrm{COMP}} > 0{,}24$ → aktuálny rank nie je WSPT-optimálny a treba ho zmeniť.
- **Stav.** OTVORENÉ (čaká na rozhodnutie).

## H-L1 — „Úspora pásma“ klasifikácie synoným
- **Tvrdenie.** Klasifikácia tokenov do skupín mechanizmov nezmení počet bajtov prečítaných
  pri vyhľadaní embeddingu (gather je $O(d)$ na token bez ohľadu na veľkosť slovníka).
- **Protokol.** Na referenčnom modeli zmerať bajty načítané vrstvou embeddingu s klasifikáciou
  a bez nej, 1 000 dotazov.
- **Zamietnutie.** Rozdiel > 1 %.
- **Dôsledok.** Ak sa *nezamietne*, pole `embedding_bandwidth_saved_pct` treba premenovať alebo odstrániť,
  lebo modeluje náklad, ktorý pri vyhľadaní embeddingu nevzniká (vzniká vo výstupnej projekcii).
- **Stav.** NEOVERENÉ.
