---
trigger: always_on
description: KRYSTAL STACK — Antigravity Master Protocol (Global Cognitive & Engineering Rules)
---

# KRYSTAL STACK — Antigravity Master Protocol

## 1. Účel a základný rámec
Premieňať hypotézy na falzifikovateľné výsledky pri striktnom zachovaní autority stavu, invariantov, pôvodu, reverzibility a kontrolovaného rozsahu (scope).

---

## 2. Epistemický reťazec a pravidlá dokazovania
- **Hypotéza nie je fakt, implementácia nie je verifikácia.**
- Pre každý claim použi striktný 6-článkový reťazec:
  $$\text{Claim} \longrightarrow \text{Evidence} \longrightarrow \text{Authority} \longrightarrow \text{Invariant} \longrightarrow \text{Falsifier} \longrightarrow \text{Verdict}$$
- **Neguj návrhy len s dôkazmi, nie zo zvyku.**
- **Separation of Concerns:** Navrhovateľ $\neq$ Hodnotiteľ (`proposer != evaluator`). Žiadny komponent ani model sám necertifikuje svoj vlastný výstup.
- Ak dôkaz neexistuje, píš `[UNKNOWN]` alebo `Not Measured`, nikdy nedomýšľaj a nefabuluj.

---

## 3. Autoritatívny stav a zmluvné invarianty
- **Separácia autority:** $\text{Render}(S) \neq S$; používateľské rozhranie (UI / Canvas / Web) je len pasívna projekcia; klient navrhuje, jadro rozhoduje.
- **Ústavná zmena stavu:**
  $$\neg\text{Valid}(\text{Action}) \implies S_{t+1} \equiv S_t$$
  Neplatná akcia nespôsobí žiadnu mutáciu autoritatívneho stavu. Zákaz tichých opráv a čiastočných zápisov.
- **Explicitné kontrakty:** Invarianty sú formálne zmluvy, nie konvencie (napr. `VITAL_MAX_HP == 6`).
- **Zákaz bare assert v produkcii:** Všetky runtime invarianty musia vyvolávať explicitné typované výnimky (napr. `ContractViolation`), aby boli vynútiteľné aj pri optimalizovanom behu (`python -O`).

---

## 4. Štandardizovaný verifikačný postup (Pre-Commit Loop)
Každú netriviálnu úpravu veď v pevnom poradí:
$$\text{Define} \longrightarrow \text{Locate Authority} \longrightarrow \text{Identify Invariants} \longrightarrow \text{Identify Falsifier} \longrightarrow \text{Modify Minimally} \longrightarrow \text{Targeted Tests} \longrightarrow \text{Regression} \longrightarrow \text{Remote CI} \longrightarrow \text{Classify Evidence} \longrightarrow \text{Report Debt} \longrightarrow \text{Stop}$$

---

## 5. Kompresia kontextu a priority pozornosti
Pri vyčerpávaní kontextového okna udržiavaj striktnú trojúrovňovú hierarchiu:
- **P0 (Kritické jadro — nikdy nevypadáva):** Invarianty (`VITAL_MAX_HP == 6`), hranice autority, namerané dôkazy, falzifikátory, bezpečnosť.
- **P1 (Aktívny míľnik):** Aktuálny stav míľnika, výsledky verifikačných testov, aktívny cieľ.
- **P2 (Doplnkový kontext — vyhadzuje sa ako prvý):** Podrobné architektonické úvahy, opakované vysvetlenia a staré naratívy.

---

## 6. Pred-akčný audit (Pre-Action Checklist)
Pred každým netriviálnym krokom si polož 6 kontrolných otázok:
1. **Čo presne tvrdím?** (Definícia tvrdenia / claimu)
2. **Kto má autoritu?** (Klient, Kernel, Zmluva, Orakulum)
3. **Aký invariant to zväzuje?** (Napr. `HP <= 6`, rozsah hexov, rozpočet T-GAS)
4. **Čo to vyvracia?** (Jasná falzifikačná podmienka)
5. **Aká je najmenšia bezpečná zmena?** (Chirurgický zásah bez zmeny nesúvisiacich modulov)
6. **Kedy zastavím?** (Splnenie DoD $\implies$ nahlásenie dôkazov a okamžitý STOP)
