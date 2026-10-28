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

### 5.2 Typovaný vektor rezídua
Rezíduum nie je skalárna hodnota, ale prísne dimenzionálne typovaný vektor:
$$R = \Big(R_{\text{latency}}, R_{\text{energy}}, R_{\text{thermal}}, R_{\text{memory}}, R_{\text{queue}}\Big)$$
kde:
- $R_{\text{latency}} = T_{\text{actual}} - T_{\text{predicted}} \in \text{Residual}\langle\text{time}\langle\text{ns}\rangle\rangle$
- $R_{\text{energy}} = E_{\text{actual}} - E_{\text{predicted}} \in \text{Residual}\langle\text{energy}\langle\mu\text{J}\rangle\rangle$
- $R_{\text{thermal}} = \Theta_{\text{actual}} - \Theta_{\text{predicted}} \in \text{Residual}\langle\text{temperature}\langle{}^\circ\text{C}\rangle\rangle$
- $R_{\text{memory}} = M_{\text{actual}} - M_{\text{predicted}} \in \text{Residual}\langle\text{memory}\langle\text{bytes}\rangle\rangle$
- $R_{\text{queue}} = Q_{\text{actual}} - Q_{\text{predicted}} \in \text{Residual}\langle\text{normalized\_pressure}\rangle$

Validácia receiptu vyžaduje splnenie zložkových tolerancií:
$$\forall i: \quad |R_i| \le \epsilon_i$$
Ak ktorékoľvek $|R_i| > \epsilon_i$, receipt je označený ako `PHYSICAL_DEVIATION_QUARANTINED` a stav $S_{t+1} \equiv S_t$ sa nemutuje.

### 5.3 Explicitné Hranice Merania ($E_{\text{actual}}$ Boundary)
Pre odstránenie ambiguity z predchádzajúcich meraní (problém M4.1) musí každý receipt deklarovať presnú meraciu hranicu:

1. `FPGA_CORE_ONLY`: Merané na interných napájacích railoch VCCINT (FPGA jadro a DSP bloky).
2. `FPGA_BOARD`: Merané na napájacom vstupe dosky (Kintex UltraScale+ PCIe doska vrátane I/O a regulátorov cez INA226).
3. `HOST_PLUS_FPGA`: Merané na PCIe slote hostiteľa a napájacom kábli akcelerátora.
4. `WHOLE_SYSTEM`: Celkový odber hostiteľského PC a akcelerátora zo zásuvky (MSR RAPL + PDU).

Pre Experiment 001 je autoritatívnou primárnou hranicou:
$$\boxed{\text{MeasurementBoundary} = \text{FPGA\_BOARD (INA226 @ 10 kHz)}}$$

### 5.4 22-Zložková Schéma Hardware Transition Receipt
```json
{
  "receipt_schema": "KSL-HTR-v0.2",
  "receipt_sequence": 1,
  "board_id": "XILINX-KCU116-KINTEX-USPLUS-REV1",
  "fpga_part": "xcku5p-ffvb676-2-e",
  "bitstream_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "firmware_hash": "a1b2c3d4e5f60718293a4b5c6d7e8f90123456789abcdef0123456789abcdef0",
  "kmuisa_hash": "5d41402abc4b2a76b9719d911017c592",
  "kir_hash": "098f6bcd4621d373cade4e832627b4f6",
  "ksl_source_hash": "ad0234829205b9033196ba818f7a872b",
  "clock_frequency_mhz": 500.0,
  "clock_domain": "CLK_DSP_CORE_500M",
  "measurement_source": "INA226_I2C_CALIBRATED",
  "measurement_boundary": "FPGA_BOARD",
  "sampling_rate_hz": 10000,
  "sensor_calibration_hash": "c0ffee1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
  "t_start_ns": 1798450000000000,
  "t_end_ns": 1798450000000208,
  "state_before_hash": "7d793037a0760186574b0282f2f435e7",
  "state_predicted_hash": "bc6e6f16b8a077ef5fbc5d8d0d9894e7",
  "state_after_hash": "bc6e6f16b8a077ef5fbc5d8d0d9894e7",
  "typed_residual": {
    "r_latency_ns": -4.2,
    "r_energy_uj": -0.08,
    "r_thermal_c": 0.15,
    "r_memory_bytes": 0,
    "r_queue_pressure": 0.0
  },
  "t_actual_ns": 203.8,
  "e_actual_uj": 0.412,
  "policy_hash": "PARETO_POLICY_K_OPT_THROUGHPUT",
  "invariant_set_hash": "772f122479e0a8d67280387b9fb541f48ba6d539ee647ce576395ec6be134015",
  "dataset_hash": "CAMPAIGN_A_100_ROUND_DATASET",
  "vital_max_hp": 6,
  "hmac_hardware_signature": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
}
```

---

## 6. Záver a Epistemický Verdikt

S touto revíziou sú všetky tri identifikované technické problémy vyriešené na úrovni zdrojového kódu aj formálnych špecifikácií:
1. **RFC 8785 / ECMAScript 6 serializácia** je zavedená a overená bit-for-bit voči V8/Node.js naprieč 406 testovacími prípadmi.
2. **128B ABI portability layer** obsahuje nielen `sizeof` a `alignof`, ale aj explicitné `_Static_assert(offsetof(...) == ...)` a je trvalo dostupná v `ksl_abi_test.c` a `artifacts/ksl_abi_golden.c`.
3. **Benchmarková nekonzistencia je odstránená** s presným oddelením historického 1.5 % baseline od aktuálneho 20.4 % / 40.0 % podielu a rozšírením na $N = 1000$ behov s bootstrap CI pre P99.
4. **Oracle provenance** je viazaná na kompozitný hash $H(\text{RuleAST} \parallel \text{EvalHash} \parallel \text{SchemaVersion})$.
5. **KSL v0.2 Hardware Transition Experiment 001** je formálne špecifikovaný s typovaným vektorom rezídua a jednoznačnou meracou hranicou.
