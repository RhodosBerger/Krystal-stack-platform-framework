# SOVEREIGN TACTICAL & INVARIANT GAMEPLAY MECHANICS SPECIFICATION
## Formálna Špecifikácia Nových Herných a Taktických Mechaník pre Poslední Kmen & Krystal Stack

> **Mantra:** *Drž cieľ stabilný, plán môžeš meniť, kombinuj len označené charakteristiky, nevyrob istotu, otváraj rámec pri rozpore, neoverené ostáva neznáme.*  
> **Ústavný Invariant:** `VITAL_MAX_HP == 6` | $\neg\text{Valid}(A) \implies \Delta S \equiv 0$ (Zero Mutation on Breach)  
> **Normatívny Rámec:** KSPP v1 (P1-P10) | Separation of Concerns (Proposer $\neq$ Evaluator)  
> **Katalógová Teória:** `THEORY-SOVEREIGN-INVARIANT-TACTICAL-MECHANICS`

---

## 1. Výkonný Súhrn a Filozofia

Tento dokument kodifikuje **novú generáciu taktických a herných mechaník**, ktoré nevychádzajú z ľubovoľných "game designových náhod", ale sú **priamym odrazom fyzikálnych, matematických a ústavných zákonov 6 vrstiev Krystal Stacku ($L_0$ až $L_5$)**.

Každá herná akcia, pohyb, kúzlo alebo balistický výstrel je prepojený s príslušnou vrstvou architektúry, čím sa premosťuje taktický súboj na hexovej sieti hry *Poslední Kmen* s nízkoúrovňovým riadením kremíka, časovania a symplektickej geometrie.

---

## 2. Šesť Nových Invariantných Mechaník

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        6 NOVÝCH INVARIANTNÝCH MECHANÍK HRY                             │
├────┬────────────────────────────┬──────────────────────────────────────────────────────┤
│ M1 │ VBLANK PHASE-SHIFT         │ L1: Fázový tieňový únik pred priamou paľbou          │
├────┼────────────────────────────┼──────────────────────────────────────────────────────┤
│ M2 │ SYMPLECTIC LENS CURVING    │ L2: Goniometrické zakrivenie striel za roh bez LOS   │
├────┼────────────────────────────┼──────────────────────────────────────────────────────┤
│ M3 │ 12-FRAME DWELL RELEASE     │ L3: Merkle-Momentum absorpcia a seizmický protiúder  │
├────┼────────────────────────────┼──────────────────────────────────────────────────────┤
│ M4 │ VERTIGO PERISCOPE LOCK     │ L2/L4: Rušenie zákopového krytia cez Dolly Zoom      │
├────┼────────────────────────────┼──────────────────────────────────────────────────────┤
│ M5 │ CONSTITUTIONAL QUARANTINE  │ L5: Karanténa neplatných ťahov s pravidlom ΔS ≡ 0    │
├────┼────────────────────────────┼──────────────────────────────────────────────────────┤
│ M6 │ AUTONOMIC STANCE SWITCH    │ L0..L5: Prepínanie bojových postojov podľa záťaže    │
└────┴────────────────────────────┴──────────────────────────────────────────────────────┘
```

---

### Mechanika 1: VBlank Shadow Phase-Shift (Fázový Tieňový Skok)
- **Architektonická Väzba:** Vrstva $L_1$ (Chronometric Shadow Multiplexing, VBlank tieňové okno $\tau_{\text{shadow}} \ge 3.50\,\text{ms}$).
- **Mechanizmus:** Hrdina aktivuje fázový posun za 2 Aetherové kryštály. Počas aktívneho raster scanoutu ($T_{\text{active}} = 12.80\,\text{ms}$) prechádza jednotka do nehmotného stavu.
- **Rovnica fázového oddelenia:**
  $$\Phi_{\text{Scanout}}(t) \cdot \Phi_{\text{Vulnerable}}(t) = 0$$
- **Herný Efekt:**
  - Všetky priame útoky (šípy, guľky, kúzla s priamou líniou) minú cieľ (100% Phase Evasion).
  - Plánovanie odvetného úderu sa musí dokončiť v tieňovom rozpočte $\le 3.50\,\text{ms}$.
- **Riziko (Back-EMF Feedback):** Ak celková saturácia pamäte UMA presiahne strop $\mu_{\text{UMA}} \ge 0.78$, fázový plášť sa zrúti a jednotka utrpí $-1\,\text{HP}$ elektrický spätný ráz.

---

### Mechanika 2: Symplectic Lens Curving (Goniometrické Zakrivenie Trajektórie)
- **Architektonická Väzba:** Vrstva $L_2$ (Spatial & Symplectic Geometry, $SO(2)$ rotácie, polárna krivosť $\kappa$).
- **Mechanizmus:** Umiestnenie kryštalickej gravitačnej šošovky na hexové pole. Každý projektil prelietajúci cez sektor $D_6$ mení svoj vektor hybnosti podľa Liouvilleovho invariantu:
  $$\det(\mathbf{J}) \equiv \frac{\partial(q', p')}{\partial(q, p)} \equiv 1.0$$
- **Polárna krivosť trajektórie:**
  $$\kappa(r, \theta) = \frac{|r^2 + 2(r')^2 - r r''|}{(r^2 + (r')^2)^{3/2}}$$
- **Herný Efekt:**
  - Strely nekopírujú priamku, ale obtiekajú prekážky a citadelárne múry.
  - Zásah nepriateľa za rohom alebo v zákope bez nutnosti priamej priamky viditeľnosti (Line-of-Sight).
  - Energia zásahu neklesá, pretože symplektický integrátor prísne zachováva fázový objem.

---

### Mechanika 3: 12-Frame Dwell Hysteresis & Seismic Release (Hysterézny Pancier a Seizmický Ráz)
- **Architektonická Väzba:** Vrstva $L_3$ (Cognitive Cadence, KSPP P8 — 12-rámcová hysterézia DVFS, Merkle-Momentum $\gamma = 0.85$).
- **Mechanizmus:** Jednotka s pasívnym pancierom *Dwell Armor* pri zásahu neodpovedá unáhlene. Zadržiava kinetickú energiu zásahov v akumulátore po dobu presne $N_{\text{dwell}} = 12$ snímok ($200\,\text{ms}$ pri $60\,\text{Hz}$).
- **Rekurentná hybnosť Merkle-Momentum:**
  $$M_k = 0.85 M_{k-1} + 0.15 \cdot \text{Damage}_k$$
- **Herný Efekt:**
  - Ak súper v priebehu 12 snímok zaútočí znova (flurry útoky, sekvencie rán), každá ďalšia rana akumuluje dodatočnú energiu.
  - Na 13. snímke nastane *Seizmické uvoľnenie (Seismic Release)*: kruhová tlaková vlna s polomerom 1 hex, ktorá udelí útočníkovi $150\%$ nahromadenej energie a status `stunned` (omráčenie na 1 kolo).

---

### Mechanika 4: Vertigo Periscope Lock (Goniometrické Zameriavanie Sapérskych Zákopov)
- **Architektonická Väzba:** Vrstva $L_2/L_4$ (Goniometrický Vertigo Dolly Zoom invariant a Sapper trench dispersion engine).
- **Mechanizmus:**
  - Sapéri vykopávajú zákopové línie na hexoch. Jednotky v zákope majú $75\%$ redukciu plošného poškodenia z mínometov.
  - Pozorovateľská jednotka (Spotter / Prieskumný balón) aktivuje periskopický mód *Vertigo Lock*.
- **Invariant Vertigo Dolly Zoom:**
  $$H(t) \cdot \tan\left(\frac{\theta_{\text{FOV}}(t)}{2}\right) \equiv C_{\text{const}}$$
- **Herný Efekt:**
  - Optický zoom zúži $\theta_{\text{FOV}}$, zatiaľ čo relatívna veľkosť terča ostáva bitovo fixovaná.
  - Odhalí skryté jednotky v zákopoch, eliminuje ich $75\%$ redukciu a zníži rozptyl mínometu $\sigma_{\text{mortar}}$ z $3.5$ hexov na $0.0$ (chirurgický priamy zásah s kritickým poškodením).

---

### Mechanika 5: Constitutional Quarantine & Zero-Mutation Tether (Ústavná Karanténna Pečať)
- **Architektonická Väzba:** Vrstva $L_5$ (Constitutional Gatekeeper, $\text{VITAL\_MAX\_HP} \equiv 6$, $\neg\text{Valid}(A) \implies \Delta S \equiv 0$).
- **Mechanizmus:** Hráč uvalí na cieľového nepriateľa *Ústavnú pečať*. Pečať monitoruje každú deklarovanú akciu cieľa voči prísnym zmluvným pravidlám.
- **Formálny kontrakt karantény:**
  $$\text{Action} \notin \text{ValidEnvelope} \implies \text{Quarantine}(\text{Action}) \land \Delta S \equiv 0$$
- **Herný Efekt:**
  - Ak sa cieľ pokúsi o akciu mimo povolené mantinely (napr. pokus o presun cez prekážku, pokus o liečenie nad 6 HP, alebo útok bez dostatku many), akcia je okamžite odmietnutá (`QUARANTINED_SAFETY_BREACH`).
  - Cieľ stratí ťah ($\Delta S = 0$, žiadna zmena polohy ani stavu), zatiaľ čo spotrebovaná mana cieľa prepadne a presunie sa do *Studne duší* hráča ako čistý Aether.

---

### Mechanika 6: Autonomic Hypervisor Stance Switching (Bojové Postoje Hypervízora)
- **Architektonická Väzba:** Celý stack $L_0..L_5$ cez autonómne ladiace profily [`TuningProfile`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/bot_diagnostic_and_tuning_toolkit.py#L56-L63).
- **Mechanizmus:** Hrdina môže v reálnom čase prepínať bojové postoje, ktoré modifikujú prevádzkové parametre bota a bojovníka:
  1. **Postoj ULTRA_PERFORMANCE (Bojový Berserk):**
     - $PL_1 = 32\,\text{W}$, $+25\%$ rýchlosť útoku a pohybový dosah.
     - Vyššia spotreba many ($+1$ Mana/útok) a tepelná daň.
  2. **Postoj LOW_POWER_EFFICIENCY (Kamenná Meditácia):**
     - $PL_1 = 15\,\text{W}$, spotreba znížená o $45\%$.
     - Pasívna regenerácia $+1\,\text{HP}$ za kolo (striktne po strop 6 HP) a maskovanie pred radarom.
  3. **Postoj LOW_LATENCY_GAMING (Bleskový Reflex):**
     - IPC lehota znížená na $0.80\,\text{ms}$.
     - $100\%$ úspešnosť smerových úskokov (`duck_down`, `dodge_left`, `dodge_right`) v duelovej matici The West.
  4. **Postoj BALANCED (Kmeňová Rovnováha):**
     - Štandardný harmonický stav s optimálnym pomerom energie a obrany.

---

## 3. Matematická a Algoritmická Validácia

Každá z mechaník je plne deterministická, reprodukovateľná zo stavu $S_0$ a overená testovacou procedúrou:
- Zachovanie HP v intervale $[0, 6]$.
- Nulový pamäťový únik pri odmietnutí akcie ($\Delta S \equiv 0$).
- Zachovanie symplektického fázového priestoru $\|\det(\mathbf{J}) - 1.0\| < 10^{-12}$.
