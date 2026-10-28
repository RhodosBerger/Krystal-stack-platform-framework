# KSL v0.2 — HARDWARE TRANSITION EXPERIMENT 001 SPECIFICATION
## Epistemická validácia, Typovaný vektor rezídua a Fyzická dôkazová slučka

---

### Epistemický status dokumentu
- **Klasifikácia:** `[FORMAL ENGINEERING SPECIFICATION / RESEARCH CONTRACT]`
- **Predchádzajúci míľnik:** `KSL v0.1: [SOFTWARE-ARCHITECTURE VERIFIED]` (84/84 testov PASS, 6 kompilátorových fáz, C11 ABI + KµASM sidecar, RFC 8785 JCS, Campaign A Pareto provenance)
- **Cieľový míľnik:** `KSL v0.2: [PHYSICALLY-GROUNDED HARDWARE CONTRACT]`
- **Ústavné invarianty:** `VITAL_MAX_HP == 6` | `Proposal != Authority` | $\Delta S \equiv 0 \text{ pri porušení kontraktu}$
- **Theory Dependency Graph:** Zmrazený na 56 uzloch (0 cyklov, žiadny Node 57)

---

## 1. RFC 8785 / ECMAScript 6 Číselná serializácia v Provenance vrstve

Odstránenie Unicode normalizácie (NFC) z JCS vrstvy v KSL v0.1 bolo nutnou, ale nie postačujúcou podmienkou plnej kompatibility s [RFC 8785](https://www.rfc-editor.org/rfc/rfc8785.html). Sekcia 3.2.2.3 RFC 8785 vyžaduje striktnú aplikáciu algoritmu **ECMAScript 6 (ECMA-262, sekcia 7.1.12.1)** pre metódu `Number.prototype.toString()`.

Pythonovská predvolená implementácia `json.dumps()` a `repr()` sa v niekoľkých kľúčových hraničných prípadoch rozchádza s ECMAScript serializáciou (napr. reprezentácia $-0.0$, formátovanie exponentov s vedúcimi nulami `1e-06` vs `0.000001`, a prah vedeckej notácie pri $10^{20}$ vs $10^{21}$).

### 1.1 Implementovaný formátovací algoritmus $(s, k, n)$
Pre každé reálne číslo $x \in \mathbb{R}$ (reprezentované podľa IEEE-754 double precision) algoritmus extrahuje minimálnu sekvenciu dekadických číslic $s$ dĺžky $k$, kde $s \times 10^{n - k} = |x|$:

1. **Negatívna nula a nula:**
   $$x = \pm 0.0 \implies \text{"0"}$$
2. **Neplatné hodnoty:**
   $$x \in \{\text{NaN}, +\infty, -\infty\} \implies \text{ContractViolation (Zákaz serializácie)}$$
3. **Pásmo celých/desatinných čísel ($k \le n \le 21$):**
   Číslice $s$ nasledované $n - k$ nulami (napr. $10^{20} \implies \text{"100000000000000000000"}$).
4. **Pásmo desatinných čísel s bodkou ($0 < n \le 21$ a $n < k$):**
   Prvých $n$ číslic, bodka `.`, zvyšných $k - n$ číslic (napr. $333333333.33333329 \implies \text{"333333333.3333333"}$).
5. **Pásmo malých desatinných zlomkov ($-6 < n \le 0$):**
   `"0."` nasledované $-n$ nulami a číslicami $s$ (napr. $10^{-6} \implies \text{"0.000001"}$).
6. **Vedecká notácia ($n \le -6 \lor n > 21$):**
   Prvá číslica, voliteľná bodka so zvyšnými číslicami, exponent `e` s explicitným znamienkom a bez vedúcich núl (napr. $10^{-7} \implies \text{"1e-7"}$, $10^{21} \implies \text{"1e+21"}$).

### 1.2 Korpus verifikačných golden vektorov
Akceptačné kritérium pre KSL v0.2:
$$\boxed{JCS_{\text{Python}}(V) = JCS_{\text{Rust}}(V) = JCS_{\text{C}}(V) = GoldenRFC8785(V)}$$

| IEEE-754 Vstup | Python `json.dumps()` | RFC 8785 / ECMAScript | KSL Implementácia | Status |
|---|---|---|---|:---:|
| `-0.0` | `-0.0` (Chyba) | `"0"` | `"0"` | `PASS` |
| `0.0` | `0.0` | `"0"` | `"0"` | `PASS` |
| `1e-6` | `1e-06` (Chyba) | `"0.000001"` | `"0.000001"` | `PASS` |
| `1e-7` | `1e-07` (Chyba) | `"1e-7"` | `"1e-7"` | `PASS` |
| `1e20` | `1e+20` (Chyba) | `"100000000000000000000"` | `"100000000000000000000"` | `PASS` |
| `1e21` | `1e+21` | `"1e+21"` | `"1e+21"` | `PASS` |
| `333333333.33333329` | `333333333.3333333` | `"333333333.3333333"` | `"333333333.3333333"` | `PASS` |
| `5e-324` (subnormal) | `5e-324` | `"5e-324"` | `"5e-324"` | `PASS` |
| `NaN, +Inf, -Inf` | `NaN / Infinity` | `SerializationError` | `ContractViolation` | `PASS` |

*Verifikované krížovým testom voči Node.js V8 runtime na 406 testovacích číslach ($10^{-320}$ až $10^{308}$) s 0 odchýlkami.*

---

## 2. 128B ABI Portability Layer & Statická Verifikácia Offsetov

Pre zamedzenie pamäťovej fragmentácie, cache line bouncing a nezarovnaných zbernicových transakcií na UMA architektúre definuje KSL 128-bajtové dátové rámce pre medziprocesovú a hardvérovú komunikáciu (P2 Invariant).

### 2.1 Trojzložkový geometrický invariant
Kompilátor negarantuje len veľkosť a alignment, ale fixuje presné fyzické umiestnenie každého poľa:
$$\boxed{\text{sizeof}(T) = 128 \quad \land \quad \text{alignof}(T) = 128 \quad \land \quad \forall i: \text{offsetof}(T, \text{field}_i) = \text{constant}_i}$$

### 2.2 Generovaný C11 Portability Header
Kompilátor emituje prenosnú štruktúru s podmienenými direktívami pre MSVC, GCC, Clang a natívny C11 `_Alignas`:

```c
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#if defined(_MSC_VER)
  #define KSL_ALIGN128 __declspec(align(128))
  #define KSL_ATTR_ALIGN128
#elif defined(__GNUC__) || defined(__clang__)
  #define KSL_ALIGN128
  #define KSL_ATTR_ALIGN128 __attribute__((aligned(128)))
#else
  #define KSL_ALIGN128
  #define KSL_ATTR_ALIGN128
#endif

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
  #define KSL_MEMBER_ALIGN128 _Alignas(128)
#else
  #define KSL_MEMBER_ALIGN128
#endif

// Canonical State: RuntimeTelemetry (Endianness: Little-Endian, Size: 128B, Align: 128B)
KSL_ALIGN128 struct KSL_ATTR_ALIGN128 RuntimeTelemetry {
    KSL_MEMBER_ALIGN128 float compute;           // offset 0
    float memory;                                // offset 4
    float thermal;                               // offset 8
    float energy;                                // offset 12
    float packet_loss;                           // offset 16
    float dispatch_latency;                      // offset 20
    uint8_t _padding[104];                       // offset 24 .. 127
};
typedef struct RuntimeTelemetry RuntimeTelemetry_t;

_Static_assert(sizeof(RuntimeTelemetry_t) == 128, "ABI breach: sizeof != 128");
_Static_assert(_Alignof(RuntimeTelemetry_t) == 128, "ABI breach: _Alignof != 128");
_Static_assert(offsetof(RuntimeTelemetry_t, compute) == 0, "ABI breach: offsetof(compute) != 0");
_Static_assert(offsetof(RuntimeTelemetry_t, memory) == 4, "ABI breach: offsetof(memory) != 4");
_Static_assert(offsetof(RuntimeTelemetry_t, thermal) == 8, "ABI breach: offsetof(thermal) != 8");
_Static_assert(offsetof(RuntimeTelemetry_t, energy) == 12, "ABI breach: offsetof(energy) != 12");
_Static_assert(offsetof(RuntimeTelemetry_t, packet_loss) == 16, "ABI breach: offsetof(packet_loss) != 16");
_Static_assert(offsetof(RuntimeTelemetry_t, dispatch_latency) == 20, "ABI breach: offsetof(dispatch_latency) != 20");
_Static_assert(offsetof(RuntimeTelemetry_t, _padding) == 24, "ABI breach: offsetof(_padding) != 24");
```

### 2.3 Epistemická klasifikácia a CI Matrix
- **Súčasný stav:** `[CLANG ABI VALIDATED]` (skompilované cez Zig Clang 22.1.8 na `x86_64-windows-gnu` s `-std=c11 -Wall -Wextra -Werror`).
- **Trvalé artefakty v repozitári:**
  - `ksl_abi_test.c` (koreň repozitára)
  - `artifacts/ksl_abi_golden.c`
- **KSL v0.2 CI Matrix:**
  1. MSVC x64 (`cl.exe /std:c11 /WX`)
  2. GCC x86-64 (`gcc -std=c11 -Wall -Wextra -Werror`)
  3. Clang x86-64 (`clang -std=c11 -Wall -Wextra -Werror`)
  4. Cross-target `aarch64-linux-gnu`
  5. Cross-target `riscv64-unknown-elf`

---

## 3. Matematická Rekonciliácia Benchmarku a Štatistická Stabilita ($N \ge 1000$)

### 3.1 Odstránenie historickej nekonzistencie (~1.5 % vs ~20.4 %)
Pôvodné tvrdenie, že kompilátor predstavuje iba „~1.5 % času celého pipeline“, patrilo k starému neoptimalizovanému baseline:
$$\frac{T_{\text{compiler, legacy}}}{T_{\text{pipeline, legacy}}} = \frac{0.78\text{ ms}}{51.36\text{ ms}} = \boxed{1.52\%}$$

Po zoštíhlení pipeline na $T_{\text{pipeline}} \approx 2.82 - 3.18\text{ ms}$ nastala nová realita:
- **Pri single-run trace ($1.02\text{ ms} / 2.82\text{ ms}$):**
  $$\text{CompilerShare}_{\text{single}} \approx \boxed{36.1\%} \quad (\text{respektíve } 1.27 / 3.18 \approx 39.9\%)$$
- **Pri štatistickom priemere ($0.75\text{ ms} / 2.82\text{ ms}$):**
  $$\text{CompilerShare}_{\text{mean}} \approx \boxed{26.4\%} \quad (\text{respektíve } 0.65 / 3.18 \approx 20.4\%)$$
- **Non-compiler infraštruktúra (DAG, Knapsack, RTL sim, JCS + HMAC):**
  $$\text{InfrastructureShare} \approx \boxed{63.9\% - 79.6\%}$$

Kompilátor teda **nie je zanedbateľný 1.5 % detail**, ale tvorí približne jednu štvrtinu až jednu tretinu exekučného rozpočtu.

### 3.2 Štatistický audit ($N = 1000$ behov s Bootstrap CI)
Pri $N = 100$ je empirický P99 nestabilný a citlivý na jediný extrémny jitter vzorky. Preto bol benchmark rozšírený na $N = 1000$ s 20 warmup behmi a neparametrickým bootstrap 95% intervalom spoľahlivosti:

| Metrika | Nameraná hodnota ($N=1000$) | Interpretácia |
|---|---:|---|
| **Aritmetický priemer (Mean)** | `0.75 ms` | Typická priepustnosť kompilátora |
| **Medián (P50)** | `0.59 ms` | Čistý deterministický čas bez jitteru |
| **P90 Latencia** | `1.18 ms` | Prah 90. percentilu |
| **Tail Latencia (P95)** | `1.30 ms` | Prah 95. percentilu |
| **Worst-Case Tail (P99)** | `1.81 ms` | Empirický 99. percentil |
| **95% Bootstrap CI pre P99** | `[1.64, 2.01] ms` | Stabilný interval spoľahlivosti ($B=1000$) |
| **Maximálna latencia (Max)** | `2.60 ms` | Absolútny worst-case v testovacej dávke |
| **Smerodajná odchýlka ($\sigma$)** | `0.29 ms` | Absolútna variabilita |
| **Variačný koeficient (CV)** | `39.04 %` | Relatívny rozptyl ovplyvnený OS context switches |

---

## 4. Kompozitná Oracle Provenance & Sémantická Integrita

Textové pravidlo `dag.falsification_oracle.evaluator_rule` (napr. `"Graph must be strictly acyclic"`) negarantuje, že sa sémantika vyhodnocovacieho orakula nezmenila v kóde.

### 4.1 Zmluvný vzorec Oracle Digest
$$\boxed{\text{OracleDigest} = H\Big(\text{RuleAST} \parallel \text{EvaluatorImplementationHash} \parallel \text{SchemaVersion}\Big)}$$
kde:
- $\text{RuleAST}$: hash/kanonický reťazec formálneho pravidla,
- $\text{EvaluatorImplementationHash}$: SHA-256 hash bajtkódu `co_code` a konštánt `co_consts` vyhodnocovacieho predikátu,
- $\text{SchemaVersion}$: verzia schémy orakula (`"3.0"`).

### 4.2 Rozšírená 5-zložková provenance v Receiptoch
Každý `PromptExecutionReceipt` teraz povinne obsahuje:
```text
oracle_id:              "ORACLE-DAG-FALSIFIER-v1"
oracle_digest:          SHA-256 (kompozitný digest podľa vzorca)
oracle_version:         "3.0"
evaluator_binary_hash:  SHA-256 (hash konkrétneho bajtkódu evaluátora)
dataset_hash:           SHA-256 ("CAMPAIGN_A_100_ROUND_DATASET")
```

---

## 5. HARDWARE TRANSITION EXPERIMENT 001 (KSL v0.2)

### 5.1 Cieľ a hypotéza experimentu
- **Cieľ:** Dokázať, že deklaratívny KSL kontrakt prežije kompletnú reťaz loweringov a vykoná sa na reálnom FPGA čipe so zachovaním všetkých invariantov a predikovaného fyzického rozpočtu:
  $$\boxed{\text{KSL} \xrightarrow{\text{AST}} \text{KIR} \xrightarrow{\text{Lower}} \text{K}\mu\text{ISA} \xrightarrow{\text{Bitstream}} \text{FPGA (KSAC-1)} \xrightarrow{\text{Telemetria}} \text{Receipt}}$$
- **Hypotéza $H_1$:** Koprocesor KSAC-1 vykoná adaptívnu kvantovaciu operáciu v čase $T_{\text{actual}} \le 208\text{ ns}$ (104 cyklov pri 500 MHz) so znížením energie $E_{\text{actual}} \le 0.45\ \mu\text{J}$ oproti softvérovému baseline.
- **Nulový model $M_0$:** Čas a energia na FPGA sú rovnaké alebo horšie než v softvéri v dôsledku réžie PCIe/AXI prenosu ($T_{\text{actual}} > T_{\text{software}}$).

### 5.2 Typovaný vektor rezídua & Particionovanie Invariantov
Rezíduum nie je skalárna hodnota, ale prísne dimenzionálne typovaný vektor:
$$R = \Big(R_{\text{latency}}, R_{\text{energy}}, R_{\text{thermal}}, R_{\text{memory}}, R_{\text{queue}}\Big)$$
kde:
- $R_{\text{latency}} = T_{\text{actual}} - T_{\text{predicted}} \in \text{Residual}\langle\text{time}\langle\text{ns}\rangle\rangle$
- $R_{\text{energy}} = E_{\text{actual}} - E_{\text{predicted}} \in \text{Residual}\langle\text{energy}\langle\mu\text{J}\rangle\rangle$
- $R_{\text{thermal}} = \Theta_{\text{actual}} - \Theta_{\text{predicted}} \in \text{Residual}\langle\text{temperature}\langle{}^\circ\text{C}\rangle\rangle$
- $R_{\text{memory}} = M_{\text{actual}} - M_{\text{predicted}} \in \text{Residual}\langle\text{memory}\langle\text{bytes}\rangle\rangle$
- $R_{\text{queue}} = Q_{\text{actual}} - Q_{\text{predicted}} \in \text{Residual}\langle\text{normalized\_pressure}\rangle$

#### Particionovanie na $I_0$ a $I_+$ (Zamedzenie delenia nulou)
Tolerančný vektor $\boldsymbol{\epsilon}$ obsahuje polia s nulovou toleranciou (napr. pamäťový stav, kde akákoľvek zmena reprezentuje porušenie ABI invariantu). Aby sa predišlo nedefinovanému deleniu nulou ($R_i / 0$), polia sú striktne particionované:
$$I_0 = \{i \mid \epsilon_i = 0\} = \{\text{memory}\}, \quad I_+ = \{i \mid \epsilon_i > 0\} = \{\text{compute}, \text{energy}, \text{latency}, \text{queue}, \text{thermal}\}$$

1. **Pre $i \in I_+$ (Normalizované rezíduum):**
   $$r_i = \frac{R_i}{\epsilon_i}$$
2. **Pre $i \in I_0$ (Exaktná nulová rovnosť):**
   $$R_i \equiv 0 \quad (|R_i| \le 10^{-9})$$

#### Pre-run Transition Pass Predikát
Prechod je platný práve vtedy, keď sú splnené všetky exaktné invarianty a maximálne normalizované rezíduum neprekročí jednotkový limit:
$$\boxed{\text{PASS}_{\text{transition}} \iff \left(\forall i \in I_0: R_i = 0\right) \land \left(\max_{i \in I_+} |r_i| \le 1.0\right)}$$

Ak $\text{PASS}_{\text{transition}} == \text{False}$, autoritatívny stav sa nemutuje ($S_{t+1} \equiv S_t$), a receipt sa označí ako `FAIL` (avšak fyzická evidencia zostáva plnohodnotne zachytená).

---

### 5.3 Explicitné Hranice Merania a INA226 Senzorový Kontrakt

#### 5.3.1 Meracie hranice ($E_{\text{actual}}$ Boundary)
Pre odstránenie ambiguity z predchádzajúcich meraní musí každý receipt deklarovať presnú meraciu hranicu:
1. `FPGA_CORE_ONLY`: Merané na interných napájacích railoch VCCINT (FPGA jadro a DSP bloky).
2. `FPGA_BOARD`: Merané na napájacom vstupe dosky (Kintex UltraScale+ PCIe doska vrátane I/O a regulátorov cez INA226).
3. `HOST_PLUS_FPGA`: Merané na PCIe slote hostiteľa a napájacom kábli akcelerátora.
4. `WHOLE_SYSTEM`: Celkový odber hostiteľského PC a akcelerátora zo zásuvky (MSR RAPL + PDU).

Pre HTE-001 je autoritatívnou primárnou hranicou:
$$\boxed{\text{MeasurementBoundary} = \text{FPGA\_BOARD}}$$

#### 5.3.2 Protokol Párovaného Kľudového Baseline (Paired Idle Baseline)
Pre odstránenie driftu kľudového výkonu (teplotné zmeny, ambientné vplyvy regulátorov) merací cyklus prebieha v troch fázach:
$$\text{Idle}_{\text{before}} \; (50\text{ ms}) \longrightarrow \text{Active Burst} \; (\ge 200\text{ ms}, N = 10^6\text{ ops}) \longrightarrow \text{Idle}_{\text{after}} \; (50\text{ ms})$$

Priemerný kľudový výkon dosky:
$$P_{\text{idle}} = \frac{P_{\text{before}} + P_{\text{after}}}{2}$$

Dynamická energia dosky (Unclamped Dynamic Energy Estimator):
$$E_{\text{dynamic}} = E_{\text{active}} - P_{\text{idle}} \times \Delta t_{\text{burst}}$$
*Kritický metodologický zákaz:* Hodnota $E_{\text{dynamic}}$ sa **nikdy neorezáva na nulu** ($\max(0, E)$ je zakázané), aby sa predišlo jednostrannému skresleniu (upward bias) pri štatistickom bootstrapovaní.

Dynamická energia na jednu validovanú operáciu:
$$E_{\text{op}} = \frac{E_{\text{dynamic}}}{N_{\text{validated}}}$$
*Dokumentačný invariant:* Táto metrika predstavuje **dynamickú energiu dosky FPGA na jednu validovanú operáciu**, nie energiu samotného FPGA čipu.

#### 5.3.3 INA226 Fyzická Konverzia vs. Polling Cadence
TI INA226 disponuje programovateľným ADC konverzným časom:
- $\text{VSHCT} = 140\ \mu\text{s}$ (shunt voltage conversion time, nominal minimum)
- $\text{VBUSCT} = 140\ \mu\text{s}$ (bus voltage conversion time, nominal minimum)
- $\text{AVG} = 1$ (1 sample averaging)
- $\text{MODE} = \text{SHUNT\_AND\_BUS\_CONTINUOUS}$

Kompletný konverzný cyklus pre výpočet výkonu dosky ($P = V_{\text{bus}} \times I_{\text{shunt}}$):
$$T_{\text{conversion\_cycle}} = \text{VSHCT} + \text{VBUSCT} = 140\ \mu\text{s} + 140\ \mu\text{s} = 280\ \mu\text{s}$$
Maximálna teoretická nezávislá konverzná kadencia:
$$\boxed{f_{\text{conversion,max}} = \frac{1}{280\ \mu\text{s}} \approx 3571.43\text{ Hz}}$$

Polling I2C registrov pri $10\text{ kHz}$ ($100\ \mu\text{s}$) produkuje redundanciu vzoriek.
- `raw_read_count`: Počet I2C čítaní zbernice (transportná diagnostika zbernice, nie počet nezávislých fyzických meraní).
- `independent_conversion_count`: Počet dokončených nezávislých ADC konverzií.
- **Detekcia konverzie:** Nezávislé vzorky sa detegujú prostredníctvom hardvérového signálu `CONVERSION_READY_OBSERVED` (ALERT pin / CNVR register flag), prípadne deterministického časového hradlovania.
- **Zákaz deduplikácie podľa hodnoty:** Rovnaká nameraná hodnota neznamená duplikát:
  $$\text{SameValue} \not\implies \text{SameConversion}$$
  Deduplikácia sa smie vykonávať výhradne na základe konverzného časovača alebo príznaku pripravenosti ADC.

---

### 5.4 Dekouplovaný 3-Osový Epistemický Model Receiptu

Každý `HardwareTransitionReceipt` striktne oddeľuje tri nezávislé dimenzie epistemického statusu:
$$\mathbf{Status} = \langle \text{ReceiptIntegrity}, \text{EvidenceClass}, \text{SemanticVerdict} \rangle$$

1. **`ReceiptIntegrity` (`VERIFIED` | `CORRUPTED`):**
   Kryptografická integrita dát a autorstvo overené cez HMAC-SHA256 nad RFC 8785 kanonickým JCS.
   *Upozornenie:* HMAC dokazuje integritu a provenienciu dát, **nikdy nedokazuje empirickú pravdivosť fyzikálnej hypotézy**.
2. **`EvidenceClass` (`FPGA_BOARD_MEASURED` | `RTL_SIMULATED` | `HOST_MEASURED` | `PENDING`):**
   Ontologická úroveň dôkazu. Pre fyzický beh je trieda dôkazu `FPGA_BOARD_MEASURED` trvalá a nemenná.
3. **`SemanticVerdict` (`COMMITTED` | `FAIL`):**
   Sémantické vyhodnotenie kontraktu. Ak fyzické meranie prekročí tolerančnú obálku ($|r_i| > 1.0$) alebo poruší invariant $I_0$, verdikt je `FAIL`.
   *Epistemické pravidlo:* Beh s verdiktom `FAIL` zostáva plnohodnotným fyzickým dôkazom (`EvidenceClass == FPGA_BOARD_MEASURED`). Nesmie byť vymazaný ani ignorovaný (prevencia survivorship bias).

#### 47-Zložková Kanonická Schéma HardwareTransitionReceipt
```json
{
  "receipt_schema": "KSL-HTR-v0.2",
  "experiment_id": "HTE-001",
  "run_id": "RUN-0001",
  "sequence": 1,
  "software_commit": "12f91b5c1b4d42b5ce979fb3a01d2233ba02f0ed",
  "manifest_hash": "ae6a1add5fcf1d3cb36c36859d3f38ef48a27df340b8aaf39daa3b47ea17f9cf",
  "ksl_source_hash": "0134ba61c7fa15f5fc8c031c19b4cf008aa0dc05cf5253fa43df6ca862e3d931",
  "kir_hash": "7d96fb54b1f6ecb345a909a32c2538cbba5863c5aa0efab156ec1e573a62ea67",
  "kmicroisa_hash": "f685bc60ee1f43a0e9b62e49c716fb4ef1e97669d0d58849ff08ae7c5cf798a3",
  "rtl_hash": "7c98031d27c62ee2743849f50dc7667dcf49c403309a47313a48e7e174b04313",
  "bitstream_hash": "4a73229b19dfb08bb2569f4ff6ff4cb1999dae133c94f1c1f5d6bb3ae74c4314",
  "board_id": "XILINX-KCU116-KINTEX-USPLUS-REV1",
  "fpga_part": "xcku5p-ffvb676-2-e",
  "clock_frequency": 500000000,
  "measurement_boundary": "FPGA_BOARD",
  "sensor_id": "INA226_I2C_CALIBRATED",
  "calibration_hash": "a1b2c3d4e5f60718293a4b5c6d7e8f90123456789abcdef0123456789abcdef0",
  "S_before": { "compute": 0.45, "energy": 0.45, "latency": 0.5, "memory": 0.6, "queue": 0.3, "thermal": 0.4 },
  "S_predicted": { "compute": 0.47, "energy": 0.42, "latency": 0.52, "memory": 0.6, "queue": 0.32, "thermal": 0.41 },
  "S_after": { "compute": 0.468, "energy": 0.418, "latency": 0.519, "memory": 0.6, "queue": 0.319, "thermal": 0.411 },
  "residual_vector": { "compute": -0.002, "energy_dynamic_per_op_uj": -0.002, "latency": -0.001, "memory": 0.0, "queue": -0.001, "thermal": 0.001 },
  "normalized_residual_vector": { "compute": -0.02, "energy_dynamic_per_op_uj": -0.04, "latency": -0.0192, "memory": 0.0, "queue": -0.02, "thermal": 0.02 },
  "memory_residual_mode": "EXACT",
  "memory_epsilon": 0.0,
  "memory_exact_pass": true,
  "latency_actual_ns": 207.6,
  "raw_energy_board_j": 0.8694,
  "idle_corrected_energy_j": 0.1944,
  "raw_energy_dynamic_j": 0.1944,
  "energy_per_validated_op_j": 1.944e-7,
  "raw_energy_op_j": 1.944e-7,
  "p_idle_before_w": 3.25,
  "p_idle_after_w": 3.25,
  "p_idle_mean_w": 3.25,
  "p_active_mean_w": 4.18,
  "sample_count": 3080,
  "sample_interval_ns": 100000,
  "measurement_duration_ns": 308000000,
  "burst_duration_ns": 208000000,
  "actual_sampling_rate_hz": 10000.0,
  "idle_sample_count": 1000,
  "active_sample_count": 2080,
  "dropped_samples": 0,
  "sensor_overflow_count": 0,
  "clock_sync_error_ns": 0.0,
  "operations_in_run": 1000000,
  "validated_operations": 1000000,
  "failed_operations": 0,
  "policy_hash": "PARETO_POLICY_K_OPT_THROUGHPUT",
  "invariant_set_hash": "772f122479e0a8d67280387b9fb541f48ba6d539ee647ce576395ec6be134015",
  "ina226_vbusct": "140_US",
  "ina226_vshct": "140_US",
  "ina226_avg": 1,
  "ina226_mode": "SHUNT_AND_BUS_CONTINUOUS",
  "sensor_poll_rate_hz": 10000.0,
  "sensor_conversion_rate_hz": 3571.43,
  "estimated_conversion_period_ns": 280000,
  "raw_read_count": 3080,
  "independent_conversion_count": 1100,
  "independent_sample_count": 1100,
  "duplicate_read_count": 1980,
  "active_independent_conversion_count": 742,
  "idle_independent_conversion_count": 358,
  "independent_sample_method": "CONVERSION_READY_OBSERVED",
  "conversion_ready_events_count": 1100,
  "acquisition_harness_hash": "f520b171e29ea8b85b5d11d65c2ad7dd4baa9e0019ecd4eb56fee8d585f7a064",
  "transition_oracle_hash": "3c87c20d94855b78f5ebcb92067376dca9dbc5699a723e7207ef8399c4fcbf42",
  "sensor_contract_hash": "257637962a9ff799222323732d938036e94b14f00d1ed2d55b4af944316934e9",
  "requirements_lock_hash": "3b00d587bf644fe1372aa72d3d9753111d3e1ace386c241ef94b57bfaa08bd8e",
  "acquisition_software_identity": "e5c8578cbe3382dd059a319ad915567859b18c6cb1d0fe61ed3abbbc0c7d3fd0",
  "provenance_pass": true,
  "first_physical_sample_timestamp": "2026-10-28T10:00:00Z",
  "acquisition_valid": true,
  "transition_pass": true,
  "receipt_integrity": "VERIFIED",
  "evidence_class": "FPGA_BOARD_MEASURED",
  "semantic_verdict": "COMMITTED",
  "vital_max_hp": 6,
  "hmac_signature": "..."
}
```

---

### 5.5 Autoritatívna Preregistrácia a Reťazec Proveniencie

#### 5.5.1 Nemennosť Preregistrovaného Manifestu
Preregistrovaný experimentálny manifest (`artifacts/hte001/HTE001_RUN_MANIFEST.json`) zmrazuje kompletný experimentálny kontrakt:
- **Kanonický SHA-256 Digest:**
  $$H_M = \texttt{ae6a1add5fcf1d3cb36c36859d3f38ef48a27df340b8aaf39daa3b47ea17f9cf}$$
- **Predchádzajúca revízia:** $\texttt{3a2003ce8101b3fb1ce0568f86f148ef37d41b4b4bca7fd4ba18783e2a44a468}$ (nahradená pred prvým fyzickým odberom z dôvodu integrácie INA226 ADC kadencie a dvojvrstvovej identity kódu).
- **Záväzné pravidlo nemennosti:** Manifest je trvalo uzamknutý. Akonáhle prebehne prvý fyzický odber na doske KCU116, manifest sa **nesmie spätne modifikovať**. Pole `first_physical_sample_timestamp` patrí výhradne do receiptu a akvizičného denníka (`RUN-0001`), v samotnom manifeste zostáva `null`.

#### 5.5.2 Dvojvrstvová Identita Softvéru (Dual-Layer Software Identity)
Pre zamedzenie rozchodu medzi Git commitom a skutočne vykonávaným akvizičným skriptom je definovaná zložená identita:
$$\boxed{H_{\text{software}} = \text{SHA256}\Big(\text{Commit} \parallel H(\text{Harness}) \parallel H(\text{Oracle}) \parallel H(\text{SensorContract}) \parallel H(\text{Lock})\Big)}$$
Akvizičný gate striktne overuje:
$$\text{PASS}_{\text{provenance}} \iff H_{\text{software,runtime}} == H_{\text{software,manifest}}$$

---

### 5.6 Experimentálna Štatistika a Dvojúrovňová Evaluácia

#### 5.6.1 Oddelenie Platnosti Akvizície a Splnenia Prechodu
$$\text{AcquisitionValid} \not\equiv \text{TransitionPass}$$
- $\text{AcquisitionValid}$: Meranie prebehlo na autoritatívnom hardvéri, bez straty vzoriek, s overenou provenienciou a validným HMAC.
- $\text{TransitionPass}$: Namerané fyzikálne hodnoty sa zmestili do tolerančnej obálky predikovaného modelu.

Minimálna vzorka pre štatistickú evaluáciu:
$$N_{\text{acquisition\_valid}} \ge 30$$

#### 5.6.2 Dvojúrovňové Akceptačné Kritérium
Experiment HTE-001 je úspešný ($\text{PASS}_{\text{HTE001}} == \text{True}$) práve vtedy, keď súčasne prejdú obe úrovne:

1. **Úroveň 1: Striktná Kompatibilita Obálky (Strict Envelope Compliance):**
   Žiadny akvizične platný beh nesmie porušiť tolerančnú obálku:
   $$\boxed{\sum_{i=1}^{N_{\text{valid}}} \mathbf{1}[\neg \text{TransitionPass}_i] == 0}$$
   Každý jednotlivý beh $i$ musí spĺňať $R_{i, \text{memory}} \equiv 0$ a $\max_{j \in I_+} |r_{i,j}| \le 1.0$.
2. **Úroveň 2: Test Skreslenia a Presnosti (Accuracy / Bias Test):**
   Pre každé normalizované pole $j \in I_+$ musí neparametrický 95% Bootstrap interval spoľahlivosti pre priemer ležať striktne vnútri jednotkového intervalu:
   $$\boxed{\forall j \in I_+: \quad \text{CI}_{95\%}(\bar{r}_j) \subset [-1.0, 1.0]}$$

#### Prečo Samotné Priemerovanie Nestačí
Aritmetický priemer rezíduí je voči osciláciám slepý: beh s chybou $+2.0$ a beh s chybou $-2.0$ majú priemerné rezíduum $0.0$, hoci oba individuálne zlyhali a hrubo porušili bezpečnostnú a výkonnostnú obálku. Preto KSL vyžaduje kombináciu nulového počtu porušení obálky ($\sum \mathbf{1}[\dots] == 0$) s bootstrap testom vychýlenia.

---

## 6. Záver a Epistemický Verdikt

S touto revíziou je kompletná metodologická, formálna aj softvérová vrstva KSL v0.2 uzamknutá a pripravená na fyzické vykonanie RUN-0001:
1. **RFC 8785 / ECMAScript 6 serializácia** je zavedená a overená voči V8/Node.js naprieč 406 testovacími prípadmi.
2. **128B ABI portability layer** obsahuje nielen `sizeof` a `alignof`, ale aj explicitné `_Static_assert(offsetof(...) == ...)` overené Clangom s `-Wall -Wextra -Werror`.
3. **Benchmarková nekonzistencia je odstránená** s presným oddelením historického 1.5 % baseline od aktuálneho 20.4 % / 39.9 % podielu pri $N = 1000$ s bootstrap CI pre P99.
4. **Oracle provenance** je viazaná na kompozitný hash sémantiky, bajtkódu a schémy.
5. **KSL v0.2 Hardware Transition Experiment 001** disponuje zmrazeným manifestom $H_M = \texttt{ae6a1add...}$, exaktným particionovaním $I_0/I_+$, dekouplovaným 3-osovým modelom receiptu a dvojúrovňovou štatistickou evaluáciou.
6. **Fyzický epistemický status:** Pred spustením RUN-0001 na doske KCU116 zostáva fyzická evidencia striktne označená ako `[PENDING]`. Žiadny simulačný beh ani syntetický test sa nepovyšuje na hardvérový dôkaz.

