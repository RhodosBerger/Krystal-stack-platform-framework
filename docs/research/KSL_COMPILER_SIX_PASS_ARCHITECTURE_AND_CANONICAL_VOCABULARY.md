# KSL COMPILER 6-PASS ARCHITECTURE & CANONICAL VOCABULARY REVISION
## Od Fyzikálnych Metafor k Merateľným Typovým Invariantom a Binárnemu Prekladu

---

### Epistemický status dokumentu
- **Klasifikácia:** `[FORMAL ENGINEERING SPECIFICATION / COMPILER ARCHITECTURE]`
- **Stav:** `[IMPLEMENTED & TEST-VERIFIED]` ([`krystal_kernel/ksl/compiler_passes.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/ksl/compiler_passes.py), [`tests/test_ksl_6pass_compiler.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/tests/test_ksl_6pass_compiler.py))
- **Ústavné invarianty:** `THEORY_NODES == 56` (0 cyklov, žiadny Node 57) | `VITAL_MAX_HP == 6` | `Proposal != Authority`
- **Účel:** Odstrániť metaforické formulácie („physics cosplay“) a transformovať šesť paradigiem do 6 sekvenčných kompilátorových fáz jazyka KSL vedúcich k deterministickému generovaniu KIR a 64-bitových inštrukcií KµISA pre FPGA mailbox.

---

## 1. Kanonická Revízia Tvrdení (Substitučná Matica)

Pre potreby patentových prihlášok, technického due diligence a rigorózneho auditu sa všetky výstupy riadia touto záväznou substitučnou maticou:

| Pôvodná Formulácia (Zrušená / Metafora) | Nová Kanonická Formulácia (Implementačný Štandard) | Klasifikácia Dôkazu |
|---|---|:---:|
| $\det(J)=1 \iff \text{VITAL\_MAX\_HP}=6 \iff \Delta S=0$ | Oddelené invarianty: $\mathcal{I}_{\text{resource}}: C(A) \le Q$ (Constrained Resource Accounting) a $\mathcal{I}_{\text{state}}: \text{Invalid}(A) \implies \Delta S_{\text{authoritative}} = 0$. | `[FORMAL-INVARIANT]` |
| $\Delta \tau = 0 \implies \text{Thermal Parasite}$ | $\text{CUD} = \frac{\Delta \text{UsefulWork}}{\Delta E}$; ak $\text{CUD} < h_{\min} \land E > E_{\text{budget}} \implies \text{Throttle/Terminate}$ (s výnimkou pre GC, verifikáciu a kalibráciu). | `[MEASURABLE-METRIC]` |
| $\Phi_{\text{scanout}} \cdot \Phi_{\text{compute}} = 0$ | $\text{TemporalOverlap}(A,B) \implies \text{ResourceSets}(A) \cap \text{ResourceSets}(B) = \emptyset$ (alebo explicitná arbitráž cez TurnToken). | `[SCHEDULER-RULE]` |
| Elektrická impedancia $Z_0, \Gamma$, $\text{RL} \ge 18\,\text{dB}$ | Bezrozmerná výpočtová analógia: $Z_c = \frac{\text{QueueDepth} \cdot \text{Latency}}{\text{EffectiveBandwidth}}$ a koeficient odrazu záťaže $\Gamma_c = \frac{Z_{\text{load}} - Z_{\text{base}}}{Z_{\text{load}} + Z_{\text{base}}}$. | `[COMPUTATIONAL-ANALOGY]` |
| $\Delta S = 0$ (všeobecne pri zlyhaní) | Rozlíšenie $\Delta S_{\text{logical}}$ vs. $\Delta S_{\text{external}}$. Pre fyzický svet (CNC/GPIO) platí: $\text{Irreversible}(A) \implies \text{PreCommitValidation}(A)$. | `[TRANSACTIONAL-LAW]` |
| „100 % eliminácia halucinácií LLM“ | $\text{UnverifiedLLMOutput} \not\implies \text{AuthoritativeCommit}$ (zabránenie neoverenému návrhu získať exekučnú autoritu). | `[TYPE-SYSTEM-GUARANTEE]` |
| $E_{\text{total}} > 0{,}70 \implies \text{imminent TDR}$ | Heuristický signál pre CDT hypotézu: $H_1: P(\text{TDR} \mid E_{\text{total}})$ predikuje pád skôr než konvenčná telemetria vs. $H_0$. | `[CDT-HYPOTHESIS]` |
| Faraday-Lenz $\Delta V_{\text{droop}} \le 50\,\text{mV}$ | Návrhový cieľ napájacej siete čipu/PCB závislý od $L_{\text{eff}}, \text{ESR}, \text{ESL}, C_{\text{decoupling}}$. | `[HARDWARE-DESIGN-HYPOTHESIS]` |

---

## 2. KSL Epistemický Typový Systém a Kontrakty

V bežných jazykoch je skalárny `float` zo senzora typovo nerozlíšiteľný od `float`, ktorý vygeneroval neurónový model alebo heuristika. KSL zavádza striktné obálky pôvodu dát (Provenance Wrappers):

```text
       Proposed<T>          (Stochastický výstup z LLM / neoverená heuristika)
            │
     [constrain / predict]
            ▼
       Predicted<T>         (Výstup z interného modelu pred exekúciou)
            │
  [execute + observe(L3)]
            ▼
       Measured<T>          (Surová telemetria zo senzora / počítadla so šumom)
            │
 [validate(tolerance, oracle)]
            ▼
       Validated<T>         (Dáta spárované s predikciou a podpísané v AepReceipt)
```

### Zákon o Neprelievaní Typov (Non-Promotion Law)
Priradenie:
```ksl
let x: Measured<f32> = p;  // kde p je Predicted<f32> alebo Proposed<f32>
```
vyvolá okamžitú chybu pri kompilácii (`EpistemicTypeMismatchError`). Jediný spôsob, ako premeniť `Predicted<T>` a `Measured<T>` na `Validated<T>`, je prejsť blokom `validate ... against ... tolerance ...`, ktorý emituje kryptografický receipt.

### Klasifikácia Reverzibility Operácií
Každá operácia v KSL deklaruje svoj vzťah k vonkajšiemu svetu ($\Delta S_{\text{external}}$):
1. **`reversible`:** Čisto pamäťová alebo tieňová operácia v kandidátskom priestore; pri zlyhaní stačí zahodiť pointer ($\Delta S_{\text{logical}} = 0$).
2. **`compensatable`:** Operácia, ktorá zapísala do zdieľaného buffera alebo zmenila takt, ale vyžaduje explicitný blok `on_abort { ... }` (napr. vrátenie napätia alebo CNC odskok).
3. **`irreversible`:** Fyzická akcia bez možnosti vrátenia v čase (napr. zapnutie lasera, zápis do eFuse). Kompilátor vyžaduje, aby všetky vstupy boli `Validated<T>` a vynucuje dvojstupňovú bránu `PreCommitValidation`.

---

## 3. Šesť Kompilátorových Prechodov KSL (kslc)

Namiesto pridávania uzlov do teoretického grafu je 6 paradigiem implementovaných ako 6 sekvenčných kompilátorových fáz:

```text
KSL Source (.ksl)
  │
  ├─► Pass 1: Epistemic Type Checker      (Mindset V: Proposed ↛ Measured ↛ Validated)
  ├─► Pass 2: 3D Resource Budgeter        (Mindset I: Q = (t, m, e), CUD >= h_min check)
  ├─► Pass 3: Authority & Reversibility   (Mindset IV: Proposal != Authority, PreCommit pre Irreversible)
  ├─► Pass 4: Temporal & Conflict Graph   (Mindset II & III: PhaseWindow overlap, Z_c, TurnTokens)
  ├─► Pass 5: Transition Verifier Gen     (AEP L7/L8: Generovanie porovnania Predicted vs Measured)
  └─► Pass 6: Machine Lowering            (Mindset VI: KIR -> KµISA 64-bit -> KSAC / FPGA Mailbox)
```

### Prechod 1: Epistemic Type Checker
Verifikuje nemennosť proveniančného zväzu. Blokuje tiché pretypovania medzi neovereným návrhom (`Proposed`), predikciou (`Predicted`) a meraním (`Measured`).

### Prechod 2: 3D Resource Budgeter
Kvantifikuje trojzložkový rozpočet $Q = (t, m, e)$ (čas v $\text{ns}$, tlak UMA zbernice v intervale $[0, 1]$ a energiu v $\text{nJ}$). Vypočítava skóre Compute-Utility Density (CUD) a overuje, že odhadované náklady neprekročia deklarovaný limit operácie.

### Prechod 3: Authority & Reversibility Checker
Vynucuje ústavné pravidlo $\text{Proposal} \neq \text{Authority}$. Pre operácie `compensatable` vyžaduje prítomnosť kompenzačného bloku; pre `irreversible` vyžaduje predchádzajúcu validáciu. Odmieta akýkoľvek commit bez validačného receiptu.

### Prechod 4: Temporal & Computational Impedance Scheduler
Modeluje časové okná a bezrozmernú analógiu výpočtovej impedancie:
$$Z_c = \frac{\text{QueueDepth} \cdot \text{Latency}}{\text{EffectiveBandwidth}}, \qquad \Gamma_c = \frac{Z_{\text{load}} - Z_{\text{base}}}{Z_{\text{load}} + Z_{\text{base}}}$$
Zabezpečuje, že operácie s prekryvom časových okien majú disjunktné množiny hardvérových prostriedkov alebo používajú arbitráž cez `TurnToken`.

### Prechod 5: Transition Verification Synthesis
Syntetizuje uzavretú reťaz KIR inštrukcií (Krystal Intermediate Representation):
`ACQUIRE_TURN_TOKEN` $\to$ `SAMPLE_TELEMETRY_L3` $\to$ `EVAL_PREDICTOR_L4` $\to$ `CHECK_CONSTRAINTS_L2` $\to$ `COMMIT_CANDIDATE_L6` $\to$ `VERIFY_RESIDUAL_L8` $\to$ `EMIT_RECEIPT_RELEASE`.

### Prechod 6: Machine Lowering
Generuje deterministické 64-bitové inštrukčné slová KµISA pre FPGA L0 Mailbox.

---

## 4. KµISA 64-Bitové Binárne Slovo

Každá inštrukcia KIR sa deterministicky kóduje do 64-bitového slova:

```text
 63      56 55    52 51             40 39                  20 19                   0
+----------+--------+-----------------+----------------------+----------------------+
|  Opcode  |  Lane  |  Target / UMA   |      Time (ns)       |     Energy (nJ)      |
|  (8 bit) | (4 bit)|    (12 bit)     |       (20 bit)       |       (20 bit)       |
+----------+--------+-----------------+----------------------+----------------------+
```

- **Bity [63:56]:** Opcode (`0x10` TurnToken, `0x20` Telemetry, `0x30` Predictor, `0x40` Constraint, `0x50` Commit, `0x60` Verify, `0x70` Receipt)
- **Bity [55:52]:** Exekučný pruh (`0x0` L0, `0x1` L1, `0x2` L2, `0x3` L3)
- **Bity [51:40]:** Adresa UMA bunky / cieľového registra
- **Bity [39:20]:** Časový rozpočet v nanosekundách (rozsah až $1.04\text{ ms}$)
- **Bity [19:0]:** Energetický rozpočet v nanojouloch (rozsah až $1.04\text{ mJ}$)

Príklad vygenerovaného slova pre `COMMIT_CANDIDATE_L6` (L1, bunka `0x04A`, $600\text{ ns}$, $260\text{ nJ}$):
$$\mathbf{0x50104A0025800104}$$

---

## 5. Záver: Zachovanie Teoretického Grafu na 56 Uzloch

Táto architektúra rieši kľúčový formálny cieľ: **žiadna nová teória, žiadny Node 57, žiadny T13**. Šesť paradigiem sa transformovalo z neformálnych metafor na rigoróznu kompilátorovú pipeline, ktorú možno formálne verifikovať a bitovo overovať voči kremíkovému hardvéru.
