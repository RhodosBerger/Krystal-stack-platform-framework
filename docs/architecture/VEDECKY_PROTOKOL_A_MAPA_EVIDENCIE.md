# Vedecký protokol a mapa evidencie (V-SCIENCE-EVIDENCE-01)
**Document ID:** `SPEC-SCIENCE-EVIDENCE-01`  
**Milestone:** `V-SCIENCE-EVIDENCE-01`  
**Classification:** Vedecký epistemický protokol, audit štúdií & Mapa evidencie  
**Constitutional Invariant:** `VITAL_MAX_HP == 6`  

---

## 1. Účel a epistemické východiská

Tento dokument formalizuje **Vedecký protokol a tvorbu Mapy evidencie** pre systém Antigravity a rámec Krystal Stack.

Cieľom vedeckého skúmania **nie je umelé sploštenie reality do jediného dominantného naratívu**, ale **štruktúrovaná, auditovateľná mapa evidencie (Evidence Map)** so zachovaním neistoty, protichodných signálov a chýbajúcich premenných.

> ### Základný axióm
> **„Cieľ je mapa evidencie, nie jedna správna realita.“**  
> Pravdivostná hodnota sa nikdy nepriraďuje autorite, popularite, počtu článkov ani spoločenskému konsenzu.

---

## 2. Architektúra Vedeckého protokolu

```
                                  [ Vedecký Claim ]
                                          |
                                          v
      +-----------------------------------------------------------------------+
      |                  5-ZLOŽKOVÝ VEDECKÝ AUDIT CLAIMU                      |
      |                                                                       |
      |   ⟨ Pôvod, Metóda, Typ štúdie, Financovanie, Konflikt záujmov (COI) ⟩ |
      +-----------------------------------------------------------------------+
                                          |
                        +-----------------+-----------------+
                        |                                   |
                        v                                   v
      +-----------------------------------+   +-------------------------------+
      |    METODOLOGICKÁ PREVIERKA        |   |   HODNOTENIE BIASU A COI      |
      |                                   |   |                               |
      |  - Vnútorná & vonkajšia validita  |   |  - COI zvyšuje riziko biasu   |
      |  - Vzorka (N) a reprezentatívnosť |   |  - COI != dôkaz manipulácie   |
      |  - Štatistická sila / power       |   |  - Skríning publikačného      |
      |  - Predregistrácia vs. HARKing    |   |    biasu (file-drawer)        |
      +-----------------------------------+   +-------------------------------+
                                          |
                                          v
      +-----------------------------------------------------------------------+
      |                       MAPA EVIDENCIE (EVIDENCE MAP)                   |
      |                                                                       |
      |    Modely: [M0: NULL] <---> [M1: Direct] <---> [M2: Confounder]       |
      |    Ku každému modelu: Čo ho oslabuje? Čo podporuje? Čo falzifikuje?   |
      |                                                                       |
      |    ROZPOR ŠTÚDIÍ ---> ZÁKAZ PRIEMEROVANIA ---> ScientificConflictNode |
      +-----------------------------------------------------------------------+
                                          |
                                          v
      +-----------------------------------------------------------------------+
      |                     DETERMINISTICKÁ BRÁNA POVÝŠENIA                   |
      |                                                                       |
      |  - Internet == Discovery (nie verdikt)                                |
      |  - Lokálna auditovaná evidencia neprepísaná externým konsenzom        |
      |  - Povýšenie do lokálnej pravdy čaká na nezávislý externý audit       |
      +-----------------------------------------------------------------------+
```

---

## 3. Normatívne pravidlá vedeckého auditu

### 3.1. Pravda nepatrí autorite, popularite ani počtu článkov
* **Zákaz odvolávania sa na prestíž:** Titul autora, impakt faktor časopisu alebo inštitucionálna prestíž nie sú dôkazom pravdivosti.
* **Konsenzus = Prior, nie dôkaz:** Vedecký alebo spoločenský konsenzus funguje výhradne ako Bayesovský prior (určuje prioritu, čo skúmať), nikdy nie ako verdikt o empirickej pravdivosti.
* **Zákaz zamieňania recenzie za dôkaz:** Naratívny prehľadový článok (review), abstrakt alebo meta-recenzia bez prístupu k primárnym mikro-dátam nie je priamy dôkaz.

### 3.2. 5-zložkový audit každého tvrdenia (Scientific 5-Tuple)
Pre každé skúmané tvrdenie $C$ systém vedie záznam:
$$\text{ClaimAudit} = \langle P, M, T, F, \text{COI} \rangle$$
1. **$P$ (Pôvod / Provenance):** Primárny archív, surové merania, autori, dátum zberu.
2. **$M$ (Metóda / Methodology):** Experimentálny dizajn, orakulum, kontrolné skupiny, slepý/dvojito slepý protokol.
3. **$T$ (Typ štúdie / Study Type):** Randomizovaná kontrolovaná štúdia (RCT), longitudinálna kohorta, observačná štúdia, in-silico simulácia, formálny dôkaz.
4. **$F$ (Financovanie / Funding):** Grantové agentúry, komerčné subjekty, štátne dotácie.
5. **$\text{COI}$ (Konflikty záujmov):** Deklarované aj skryté finančné a inštitucionálne väzby.

### 3.3. Konflikt záujmov (COI): Riziko biasu $\neq$ Dôkaz manipulácie
* **Epistemické rozlíšenie:** Prítomnosť konfliktu záujmov ($\text{COI} = \text{True}$) zvyšuje apriórne riziko metodologického skreslenia (bias risk).
* **Zákaz ad-hominem:** $\text{COI}$ **nie je dôkazom podvodu, falšovania dát ani manipulácie**. Tvrdenie s $\text{COI}$ sa nezamieta a priori; vyžaduje však prísnejší metodologický audit, nezávislú replikáciu a kontrolu publikačného biasu.

### 3.4. Metodologické previerky validity
Každá štúdia vstupujúca do evidencie sa podrobuje previerke:
* **Vnútorná validita (Internal Validity):** Boli eliminované mätúce premenné (confounders)? Bola kontrolná skupina izolovaná?
* **Vonkajšia validita (External Validity):** Sú výsledky prenositeľné mimo laboratórne podmienky a špecifickú hardvérovú/demografickú vzorku?
* **Vzorka ($N$) a reprezentatívnosť:** Je veľkosť vzorky dostatočná na prekonanie štatistického šumu? Bola vzorka náhodná alebo samovýberová?
* **Publikačný bias & File-Drawer Problem:** Neboli negatívne výsledky potlačené (skríning funnel plot asymetrie)?
* **Predregistrácia vs. HARKing:** Bola hypotéza predregistrovaná vopred, alebo išlo o *Hypothesizing After the Results are Known* (p-hacking)?
* **Nezávislá replikácia $\neq$ Opakovanie:** Citovanie toho istého pôvodného článku v 50 ďalších prácach je iba echo (opakovanie), nie 50 nezávislých replikácií.

### 3.5. Zákaz spriemerovania pri rozpore štúdií
* Ak Štúdia $A$ namerala efekt $E = +1.8$ a Štúdia $B$ namerala $E = -0.4$, je **prísne zakázané spriemerovať výsledok na $E = +0.7$**.
* Pri nezhode systém vytvorí explicitný uzol sporu (`ScientificConflictNode`) a identifikuje **chýbajúcu informáciu** (rozdiel v metodike, iný hardvérový stepping, odlišné meracie prístroje, nezachytený confounder).

### 3.6. Konkurenčné modely a $M_0$ (NULL Model)
Pre každú otázku systém povinne udržiava portfólio modelov, kde **$M_0$ je vždy nulová hypotéza** (Null Hypothesis):
* **$M_0$ (NULL):** Pozorovaný jav je výsledkom náhody, systémového šumu, artefaktu meracieho prístroja alebo placebo efektu.
* Pre každý model $M_k$ musí mapa evidencie explicitne špecifikovať:
  1. **Čo model oslabuje (Weakening factors).**
  2. **Čo model podporuje (Supporting evidence).**
  3. **Čo model definitívne falzifikuje (Decisive falsifier).**

### 3.7. Internet je Discovery, nie Verdict
* Web slúži výhradne na objavovanie kandidátov na hypotézy a externých dátových bodov.
* Lokálnu auditovanú telemetriu a experimentálne merania v projekte **nie je dovolené prepísať externým konsenzom**.
* Každé povýšenie do lokálneho faktického poznania (`APPROVED_FACT`) podlieha nezávislému externému auditu autora (`proposer != evaluator`).

---

## 4. Konštitučné a hardvérové invarianty

1. **`VITAL_MAX_HP == 6`:** Konštitučný invariant je nemenný naprieč celou mapou evidencie a všetkými stavovými prechodmi.
2. **Auditovateľná stopa:** Žiadny claim nesmie vstúpiť do `EvidenceMap` bez priradeného 5-zložkového auditu a podmienky falzifikácie.
