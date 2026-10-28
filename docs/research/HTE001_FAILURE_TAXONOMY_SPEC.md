# HTE-001 — RUN-0001 FAILURE TAXONOMY & AUDIT SPECIFICATION
## Formálna Klasifikácia Tried a Štádií Zlyhania Fyzického Prechodu

---

### Epistemický status dokumentu
- **Klasifikácia:** `[FORMAL ENGINEERING SPECIFICATION / RESEARCH CONTRACT]`
- **Účel:** Odstrániť neinformatívny binárny status `FAIL` a priradiť každému neúspešnému behu presnú príčinnú triedu a štádium zlyhania.
- **Ústavné invarianty:** `Proposal != Authority` | `VITAL_MAX_HP == 6` | `Prediction != Evidence`
- **Pravidlo evidencie:** Každý neúspešný beh zostáva v triede `FPGA_BOARD_MEASURED` a povinne sa zaznamenáva do akvizičného denníka (prevencia survivorship bias).

---

## 1. Dvojrozmerná Taxonómia Zlyhania: $(C_{\text{failure}}, S_{\text{stage}})$

Každé zlyhanie fyzického behu je klasifikované ako usporiadaná dvojica:
$$\mathbf{Failure} = \langle \text{FailureClass}, \text{FailureStage} \rangle$$

---

## 2. Triedy Zlyhania ($\text{FailureClass}$)

Definovaných je 7 navzájom disjunktných tried zlyhania:

### 1. `PROVENANCE_FAILURE`
- **Podmienka:** Nezhoda kryptografických hashov medzi bežiacim kódom a preregistrovaným manifestom ($H_{\text{software,runtime}} \neq H_{\text{software,manifest}}$).
- **Kauzálny mechanizmus:** Spustenie nemapovaného kódu, neuložené zmeny v skripte, rozchod commitu alebo orakula.
- **Dopad:** Akvizičný beh je neplatný ($\text{AcquisitionValid} == \text{False}$).

### 2. `ACQUISITION_FAILURE`
- **Podmienka:** Neúplná alebo poškodená fyzická akvizícia:
  $$N_{\text{active conversions}} < 700 \quad \lor \quad N_{\text{independent conversions}} < 1000 \quad \lor \quad \Delta t_{\text{burst}} < 200\text{ ms} \quad \lor \quad \text{dropped\_samples} > 0$$
- **Kauzálny mechanizmus:** Zlyhanie I2C zbernice, timeout čítania, pretečenie vyrovnávacej pamäte, predčasné ukončenie burstu.
- **Dopad:** Akvizičný beh je neplatný ($\text{AcquisitionValid} == \text{False}$).

### 3. `BITSTREAM_MISMATCH`
- **Podmienka:** Hardvérový podpis FPGA bitstreamu alebo Device DNA na doske KCU116 nezodpovedá manifestu:
  $$\text{SHA256}(\text{ActiveBitstream}) \neq \text{Manifest.bitstream\_hash}$$
- **Kauzálny mechanizmus:** Nahratie nesprávneho bitstreamu do pamäte FPGA.
- **Dopad:** Akvizičný beh je neplatný ($\text{AcquisitionValid} == \text{False}$).

### 4. `SENSOR_FAILURE`
- **Podmienka:** Výpadok komunikácie s monitorom INA226, strata ACK na I2C adrese `0x40`/`0x41`, neplatné registračné čítanie (napr. `0xFFFF` pri zbernici v tri-state), alebo pretečenie vnútorného registra prúdu.
- **Kauzálny mechanizmus:** Hardvérová chyba senzora, strata napájania senzora, chybná pull-up rezistorová sieť.
- **Dopad:** Akvizičný beh je neplatný ($\text{AcquisitionValid} == \text{False}$).

### 5. `INVARIANT_FAILURE`
- **Podmienka:** Porušenie ktoréhokoľvek exaktného invariantu množiny $I_0 = \{\text{memory}\}$ alebo ústavného limitu HP:
  $$R_{\text{memory}} \neq 0 \quad (|R_{\text{memory}}| > 10^{-9}) \quad \lor \quad \text{vital\_max\_hp} \neq 6$$
- **Kauzálny mechanizmus:** Poškodenie pamäťového rámca, nezarovnaný zápis, zlyhanie logiky commitu.
- **Dopad:** Akvizícia je platná ($\text{AcquisitionValid} == \text{True}$), ale sémantický prechod zlyhal ($\text{SemanticVerdict} == \text{FAIL}$, $\Delta S_{\text{authoritative}} \equiv 0$).

### 6. `ENVELOPE_FAILURE`
- **Podmienka:** Prekročenie preregistrovanej tolerančnej obálky pre ktorékoľvek spojité pole $j \in I_+$:
  $$\max_{j \in I_+} |r_j| = \max_{j \in I_+} \left|\frac{R_j}{\epsilon_j}\right| > 1.0$$
- **Kauzálny mechanizmus:** Vyššia latencia, než predikoval model ($T > 208\text{ ns}$), alebo vyššia dynamická spotreba energie ($E_{\text{op}} > 0.45\ \mu\text{J}$).
- **Dopad:** Akvizícia je platná ($\text{AcquisitionValid} == \text{True}$), ale fyzikálna hypotéza prechodu zlyhala ($\text{TransitionPass} == \text{False}$, $\text{SemanticVerdict} == \text{FAIL}$).

### 7. `BIAS_FAILURE`
- **Podmienka:** Zlyhanie štatistického testu presnosti naprieč minimálnou sadou $N \ge 30$ behov:
  $$\exists j \in I_+: \quad \text{CI}_{95\%}(\bar{r}_j) \not\subset [-1.0, 1.0]$$
- **Kauzálny mechanizmus:** Systematické vychýlenie modelu (systematic drift/bias), hoci jednotlivé behy ležali na hrane tolerancie.
- **Dopad:** Celý experiment HTE-001 je neúspešný ($\text{PASS}_{\text{HTE001}} == \text{False}$).

---

## 3. Štádiá Zlyhania ($\text{FailureStage}$)

Definovaných je 6 sekvenčných fáz exekúcie:
1. `PREFLIGHT`: Predletová kontrola identity súborov, hashov, prístupových práv a stavu FPGA dosky.
2. `IDLE_BEFORE`: Prvá fáza merania kľudového baseline ($50\text{ ms}$).
3. `ACTIVE_BURST`: Fáza aktívneho vykonávania $10^6$ operácií koprocesora KSAC-1 ($\ge 200\text{ ms}$).
4. `IDLE_AFTER`: Druhá fáza merania kľudového baseline ($50\text{ ms}$).
5. `TRANSITION_EVALUATION`: Vyhodnotenie exaktných invariantov $I_0$ a tolerančnej obálky $I_+$.
6. `EXPERIMENT_STATISTICS`: Vyhodnotenie globálnej obálky a neparametrického Bootstrap 95% CI naprieč celou sadou behov.

---

## 4. Matica Odozvy Riadiaceho Systému

| Trieda Zlyhania | Platnosť Akvizície | Epistemická Trieda | Verdikt | Reakcia Riadiaceho Systému |
|---|:---:|:---:|:---:|---|
| **`PROVENANCE_FAILURE`** | `INVALID` | `PENDING` | `FAIL` | Zastaviť beh; odmietnuť zápis do receiptu; generovať `ProvenanceAuditAlert`. |
| **`ACQUISITION_FAILURE`** | `INVALID` | `PENDING` | `FAIL` | Zahodiť beh zo štatistického súboru; opakovať akvizíciu po resete I2C. |
| **`BITSTREAM_MISMATCH`** | `INVALID` | `PENDING` | `FAIL` | Zastaviť; reprogramovať FPGA bitstream z overeného archívu. |
| **`SENSOR_FAILURE`** | `INVALID` | `PENDING` | `FAIL` | Hardvérový reset zbernice I2C; kontrola napájacích napätí. |
| **`INVARIANT_FAILURE`** | `VALID` | `FPGA_BOARD_MEASURED` | `FAIL` | Zapečatiť `QuarantineReceipt`; $\Delta S \equiv 0$; izolovať riadiaci subsystém. |
| **`ENVELOPE_FAILURE`** | `VALID` | `FPGA_BOARD_MEASURED` | `FAIL` | Zapečatiť receipt; $\Delta S \equiv 0$; zachovať beh v dátovom súbore. |
| **`BIAS_FAILURE`** | `VALID` | `FPGA_BOARD_MEASURED` | `FAIL` | Zapečatiť kompletný experimentálny denník; falzifikovať hypotézu $H_1$. |

---

## 5. Kauzálna Diagnostika a Pravidlá Opakovaného Behu (Rerun Matrix)

Z hľadiska vedeckej metodológie nie je každý `FAIL` ekvivalentný:
- **Technické/Transportné zlyhanie (Transport Artifact):** Neplatná akvizícia dovoľuje opakovať meranie bez falzifikácie hypotézy.
- **Fyzikálne/Sémantické zlyhanie (Empirical Falsification):** Platná akvizícia s prekročením obálky predstavuje nezvratný dôkaz a vyžaduje trvalé zachovanie v dátovom súbore.

| Trieda Zlyhania | Kauzálna Doména | Platnosť Evidencie | Povolené Opakovanie (Rerun) | Vyžaduje Revíziu Manifestu? | Vedecký Dopad na $H_1$ |
|---|---|:---:|:---:|:---:|---|
| **`PROVENANCE_FAILURE`** | Riadenie verzie kódu | `NULL` | ÁNO (po zosúladení commitu) | NIE | Žiadny (beh nebol autorizovaný) |
| **`ACQUISITION_FAILURE`** | I2C zbernica / Buffer | `INVALID` | ÁNO (po resete I2C linky) | NIE | Žiadny (technický artefakt transportu) |
| **`BITSTREAM_MISMATCH`** | Konfigurácia čipu | `INVALID` | ÁNO (po flashnutí správneho bitstreamu) | NIE | Žiadny (nesprávny testovaný objekt) |
| **`SENSOR_FAILURE`** | Merací obvod | `INVALID` | ÁNO (po oprave hardvéru senzora) | NIE | Žiadny (zlyhanie meracieho prístroja) |
| **`INVARIANT_FAILURE`** | Architektonický kontrakt | `VALID` | **NIE** (ako rovnaký beh) | ÁNO (odhalená architektonická chyba) | Okamžitá diskvalifikácia lowering pipeline |
| **`ENVELOPE_FAILURE`** | Fyzikálna dynamika | `VALID` | **NIE** (ako rovnaký beh) | **ZÁKAZ** (beh zostáva v súbore) | Dôkaz pre falzifikáciu hypotézy $H_1$ |
| **`BIAS_FAILURE`** | Štatistické vychýlenie | `VALID` | **NIE** | **ZÁKAZ** | Nezvratná empirická falzifikácia $H_1$ |

