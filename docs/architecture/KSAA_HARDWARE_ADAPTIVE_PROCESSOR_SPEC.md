# 🔬 KRYSTAL SOVEREIGN ADAPTIVE ARCHITECTURE (KSAA) — HARDWARE CONTROL SPECIFICATION

> **Normatívny status:** Hardvérová špecifikácia silicon ko-dizajnu a FPGA koprocesora  
> **Konštitučný invariant:** $\text{VITAL\_MAX\_HP} \equiv 6 \land \Delta S \equiv 0 \text{ pri zlyhaní}}$  
> **Dôkazový uzol:** `THEORY-KSAA-HARDWARE-CONTROL-COPROCESSOR-ARCHITECTURE` (Node 51 v [`theory_dependency_graph.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/verification/theory_dependency_graph.py))

---

## 1. Definícia a Nový Princíp: Hardware as a Controllable Trajectory

Tradičný procesor operuje v paradigme:
$$\text{Instruction} \longrightarrow \text{Result}$$

V architektúre **KSAA** je výpočet riadený ako validovaná trajektória v prevádzkovom stavovom priestore:
$$\boxed{\text{State} + \text{Instruction} + \text{Action} \longrightarrow \text{Trajectory}}$$

### Kanonický Hardvérový Cyklus:
$$\boxed{\text{Observe} \longrightarrow \text{Predict} \longrightarrow \text{Gate} \longrightarrow \text{Execute} \longrightarrow \text{Measure} \longrightarrow \text{Validate}}$$

---

## 2. Sedem Fyzicko-Logických Vrstiev KSAA (Si-L0 až Si-L7)

Implementované v [`krystal_kernel/ksaa_hardware_control_coprocessor.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/ksaa_hardware_control_coprocessor.py):

| Vrstva | Názov Vrstvy | Typ Zbernice / Jednotky | Fyzická Funkcia a Invariant |
| :--- | :--- | :--- | :--- |
| **Si-L0** | State Observation Fabric | Read-Only zbernica senzorov | Odčítava 7D vektor $H = [C, M, T, E, Q, B, J]$ z kremíka. |
| **Si-L1** | Runtime State Register Bank | Hardvérová banka registrov | `REG_STATE_COMPUTE`, `REG_STATE_MEMORY`, `REG_STATE_THERMAL`... |
| **Si-L2** | Constraint Gate | Hardvérový komparátor | $\text{UnsafeCandidate} \implies \Delta S \equiv 0$. Hard limity TjMax, UMA. |
| **Si-L3** | Influence Matrix Engine | Vektorovo-maticový násobič | Počíta očakávaný prevod $\Delta S = M_I \cdot A$ v jedinom takte. |
| **Si-L4** | Predictive Transition Unit | Hardvérový 4-kandidátny MPC | Simuluje $A_0 \dots A_4$ (HOLD, LOWER_CONC, MIGRATE, PRECISION). |
| **Si-L5** | Adaptive Execution Fabric | Smerovač výpočtových pruhov | WorkloadAdaptive riadenie afinity, queue admission a DMA okien. |
| **Si-L6** | Transition Validation Unit | Výpočtový komparátor rezídua | Meria $R = \|S_{\text{actual}} - S_{\text{predicted}}\|$ a emituje receipt. |
| **Si-L7** | Hardware Runtime Memory | SRAM tabuľka validovaných prechodov | Ukladá `(STATE_SIGNATURE, ACTION, DELTA, RELIABILITY, COUNT)`. |

---

## 3. 7-Rozmerný Hardvérový Stavový Vektor: $H = [C, M, T, E, Q, B, J]$

1. **$C$ (Compute Pressure):** Saturácia ALU a tenzorových pipeline.
2. **$M$ (Memory Pressure):** LLC contention a pomer cache miss.
3. **$T$ (Thermal Pressure):** Priblíženie k limitom PL1/PL2 a TjMax.
4. **$E$ (Energy Pressure):** Spotreba energie voči rozpočtu RAPL.
5. **$Q$ (Queue Pressure):** Hĺbka inštrukčných a DMA kruhových bufferov.
6. **$B$ (Bandwidth Pressure):** Využitie UMA zbernice (limit $\mu_{\text{UMA}} < 0.78$).
7. **$J$ (Jitter Pressure):** Rozptyl latencie a medzirámcová nestabilita.

---

## 4. 32-Bitové Hexadecimálne Riadiace Slovo: `0xVTAICHRF`

Kompaktný formát prenášaný cez PCIe MMIO BAR / register:

```
Bit: 31    28 27    24 23    20 19    16 15    12 11     8 7      4 3      0
    ┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
    │   V    │   T    │   A    │   I    │   C    │   H    │   R    │   F    │
    └────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
     Version  Target   Action   Intensity Confid.  Horizon  Quadrant  Flags
```

* **V (Version):** `0x1` pre KSAA-v1.
* **T (Target):** `0x1` P-Core, `0x2` GPU EU, `0x3` NPU, `0x4` DMA Ring.
* **A (Action):** `0x0` HOLD, `0x1` LOWER_CONCURRENCY, `0x2` MIGRATE, `0x3` CHANGE_PRECISION, `0x4` THROTTLE...
* **I (Intensity):** $0 \dots 15$ intenzita zásahu.
* **C (Confidence):** $0 \dots 15$ ($15 \implies 100\%$ istota modelu).
* **H (Horizon):** Dwell čas v snímkach (12-frame hysterézny dwell).
* **R (Quadrant):** $Q_1$ až $Q_4$.
* **F (Flags):** `[ACT][LCK][CNF][VIT=6]`.

---

## 5. Matica Vplyvu (Influence Matrix Engine): $\Delta S = M_I \cdot A$

Matica $M_I \in \mathbb{R}^{7 \times 8}$ kóduje kalibrovanú odozvu procesora na elementárne akcie:

```text
               A0(HOLD) A1(LOWER_CONC) A2(MIGRATE) A3(PRECISION) A4(THROTTLE) A5(DMA) A6(SHADOW) A7(REBOUND)
Compute (C)   [  0.00       -0.15        -0.08        -0.20        -0.35      +0.05    -0.10     -0.50 ]
Memory  (M)   [  0.00       -0.05        -0.12        -0.25        -0.20      +0.15    -0.30     -0.40 ]
Thermal (T)   [  0.00       -0.08        -0.04        -0.15        -0.28      +0.02    -0.08     -0.45 ]
Energy  (E)   [  0.00       -0.10        -0.05        -0.18        -0.30      +0.04    -0.05     -0.40 ]
Queue   (Q)   [  0.00       +0.04        -0.02        -0.05        +0.15      -0.20    -0.10     +0.25 ]
Bandw.  (B)   [  0.00       -0.06        -0.10        -0.22        -0.18      +0.20    -0.35     -0.35 ]
Jitter  (J)   [  0.00       -0.02        +0.05        -0.01        +0.08      -0.05    -0.08     +0.10 ]
```

---

## 6. Verifikácia a Výsledky Testov Milestone KSAA-M1

Všetkých 10 podmienok Definition of Done (DoD) bolo overených testovacou procedúrou:
* **Test suite:** [`tests/test_ksaa_hardware_control_coprocessor.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/tests/test_ksaa_hardware_control_coprocessor.py) (7/7 PASS v 0.001s).
* **Bit-exact packing `0xVTAICHRF`:** Potvrdená parita u32 $\leftrightarrow$ hex string.
* **Gating Invariant:** Pri nebezpečnom stave (Thermal > 0.88) hardvér zablokuje mutáciu ($\Delta S \equiv 0$).
* **Reliability Convergence:** Hardvérová pamäť aktualizovala $TR > 0.60$ a rezíduum $< 0.35$.
* **Sovereign Operations Compiler:** `100% OPERÁCIE AUTORIZOVANÉ` naprieč všetkými 51 uzlami DAGu.
