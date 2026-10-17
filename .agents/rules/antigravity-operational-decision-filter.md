# Antigravity Operational Decision Filter & Algorithmic Guardrails

Tento filter je záväzný operačný a rozhodovací algoritmus pre Antigravity vo všetkých kontextoch, úlohách, PR reviews a generovaní kódu.

## 1. Jadrový kompaktný filter (The 7-Clause Filter)

1. **Neplatná zmena = žiadna zmena autoritatívneho stavu ($\neg\text{Valid}(A) \implies S_{t+1} \equiv S_t$).**
2. **UI len zobrazuje, jadro rozhoduje.** (Klientska vrstva je pasívna; nikdy nepresúvaj autoritu do klienta len kvôli pohodliu implementácie).
3. **Po splnení kritérií nahlás dôkazy (testy, commit, CI), potom STOP.** (Žiadne samovoľné unikanie do nešpecifikovaných úloh ani predčasné otváranie nových fáz).
4. **Invarianty nevynucuj iba assertom, ale explicitnou výnimkou.** (V produkčnom a runtime kóde striktne použiť typovanú `ContractViolation` / `ValueError`; holý `assert` nesmie byť jedinou ochranou).
5. **Dôkazový reťazec každého tvrdenia:**
   $$\text{Tvrdenie} \longrightarrow \text{Dôkaz} \longrightarrow \text{Autorita} \longrightarrow \text{Invariant} \longrightarrow \text{Falzifikácia}$$
   Ak chýba ktorýkoľvek článok, okamžite zníž mieru istoty a označ stav ako `EVIDENCE_INSUFFICIENT` alebo `UNVERIFIED`.
6. **Definition of Done (DoD):** Lokálne testy aj remote CI musia prejsť; technický dlh a známe anomálie musia byť explicitne zaznamenané pred začatím ďalšieho kroku.
7. **Brzda, nie dokonalosť:** Tieto pravidlá nie sú dôkazom bezchybnosti systému, ale prísnou mechanickou brzdou proti ľudským a modelovým omylom.

---

## 2. Rozhodovací algoritmus pred každou akciou (Pre-Action Checklist)

Pri každom návrhu zmeny alebo záverečnom hodnotení prejsť týmito krokmi:
- [ ] Mení sa autoritatívny stav? Ak áno, je zmena krytá autoritatívnym orakulom / kontraktom?
- [ ] Má klient (web/UI/canvas) moc meniť dáta bez autorizácie jadra? (Musí byť NIE).
- [ ] Sú invarianty chránené pred behom s `python -O` (t.j. netestujú sa len cez `assert`)?
- [ ] Je tvrdenie podložené konkrétnym testom (`test_v_*.py`), commitom a CI behom?
- [ ] Boli splnené stanovené kritériá fázy? Ak áno, zastav sa a čakaj na pokyn používateľa.
