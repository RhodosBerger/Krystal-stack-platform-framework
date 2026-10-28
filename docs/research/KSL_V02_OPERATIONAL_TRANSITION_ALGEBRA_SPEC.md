# KSL v0.2 — OPERATIONAL TRANSITION ALGEBRA SPECIFICATION

**Status:** APPROVED ENGINEERING SPECIFICATION  
**Scope:** KSL v0.2 Compiler, UMA Control Plane, Hardware Actuation, and Gameplay Semantics  
**Constitutional Rule:** **Zero New Theoretical Nodes** (56 nodes remain strictly frozen; these 12 transitions form an operational transition algebra $T: S \times A \to S'$, NOT Node 57)  
**Invariant:** $\text{VITAL\_MAX\_HP} \equiv 6$, $\Delta S_{\text{authoritative}} \equiv 0$ on breach  

---

## 1. Executive Summary & Epistemic Framing

The 12 operational transitions ($T_1 \dots T_{12}$) defined herein do not construct new theoretical nodes in the project epistemic graph. Rather, they define the formal **Operational Transition Algebra**:
$$T: S \times A \longrightarrow S'$$
stratified into four strictly isolated and decoupled planes:
1. **Plane 1: Static Dimensional KSL Algebra ($T_1 \dots T_4$)** — Static, abelian group dimension algebra implemented within [`krystal_kernel/ksl/ksl_type_checker.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/ksl/ksl_type_checker.py).
2. **Plane 2: UMA Control Plane & Scheduler Exclusivity ($T_5 \dots T_8$)** — Dynamic runtime resource multiplexing and invariant preservation.
3. **Plane 3: Hardware Transition Contract & Evidence Path ($T_{12}$ / HTE-001)** — Physical FPGA actuation, empirical residual vector, and authoritative commit quarantine.
4. **Plane 4: Gameplay Semantics Consumer Plane ($T_9 \dots T_{11}$)** — Consumer application layer consuming the transition semantics without contaminating hardware proof invariants.

```
       +-----------------------------------------------------------+
       |   PLANE 1: STATIC DIMENSIONAL ALGEBRA (T1-T4, Typecheck)  |
       |   Energy/Time=Power | Memory/Time=Bandwidth | 1/T=Freq    |
       +-----------------------------+-----------------------------+
                                     |
                                     v
       +-----------------------------------------------------------+
       |   PLANE 2: UMA CONTROL PLANE & SCHEDULER (T5-T8, Runtime) |
       |   RPE Quota Handshake | Hysteresis Switching | Phase Excl |
       +-----------------------------+-----------------------------+
                                     |
                    +----------------+----------------+
                    |                                 |
                    v                                 v
   +---------------------------------+   +---------------------------------+
   |   PLANE 3: HARDWARE CONTRACT    |   |   PLANE 4: GAMEPLAY SEMANTICS   |
   |   (T12, HTE-001 Actuation)      |   |   (T9-T11, Consumer Layer)      |
   |   Physical Actuation != Rollback|   |   Dwell Timing | Symplectic Form|
   |   Authoritative Quarantine      |   |   S = S_prot (+) S_sanction     |
   |   r_i = R_i / eps_i <= 1.0      |   |   Zero Mutation on Target       |
   +---------------------------------+   +---------------------------------+
```

---

## 2. Plane 1: Static Dimensional KSL Algebra ($T_1 \dots T_4$)

### 2.1 Formal 7-Dimensional Vector Space
A physical quantity is defined as the 5-tuple:
$$\text{Quantity} = \langle \text{DimensionVector}, \text{DimensionName}, \text{UnitName}, \text{Scale}, \text{Offset} \rangle$$
where `DimensionVector` is an element of $\mathbb{Z}^7$:
$$\mathbf{D} = [M, L, T, I, \Theta, \text{memory}, \text{work}] \in \mathbb{Z}^7$$

Computational quantities treat `memory` (bytes) and `work` (operations / cycles) as distinct orthogonal dimensions, ensuring:
$$\frac{\text{Memory}}{\text{Time}} = [\text{memory} \cdot T^{-1}] \neq \frac{\text{Work}}{\text{Time}} = [\text{work} \cdot T^{-1}]$$

### 2.2 Abelian Group Operations
1. **Addition & Subtraction:** Requires identical dimension vectors:
   $$\mathbf{D}_{A \pm B} = \mathbf{D}_A \iff \mathbf{D}_A = \mathbf{D}_B$$
2. **Multiplication:** Sum of dimension vectors:
   $$\mathbf{D}_{A \cdot B} = \mathbf{D}_A + \mathbf{D}_B$$
3. **Division:** Difference of dimension vectors:
   $$\mathbf{D}_{A / B} = \mathbf{D}_A - \mathbf{D}_B$$
4. **Power ($n \in \mathbb{Z}$):**
   $$\mathbf{D}_{A^n} = n \mathbf{D}_A$$
5. **Dimensionless Scalar:** $\mathbf{D}_{\text{scalar}} = [0, 0, 0, 0, 0, 0, 0]$.

### 2.3 The Affine Normalization Invariant
Affine units with non-zero offsets (such as Celsius with $\text{Offset} = 273.15$) cannot participate directly in multiplication or division:
$$\forall Q_A, Q_B: (Q_A * Q_B \lor Q_A / Q_B) \implies \text{Offset}(Q_A) = 0 \land \text{Offset}(Q_B) = 0$$
Attempting to multiply or divide affine units directly raises `DimensionalDerivationError`:
$$\text{temperature<celsius>} * \text{time<second>} \xrightarrow{\text{Compile Error}} \text{DimensionalDerivationError}$$
Affine quantities must be explicitly converted to an absolute scale ($\text{Celsius} \to \text{Kelvin}$) prior to algebraic derivation.

### 2.4 Derivation Table ($T_1 \dots T_4$)
| Transition | Algebraic Form | Dimension Vector Calculation | Inferred Result Type |
|---|---|---|---|
| **$T_1$** | $\frac{\text{Energy}}{\text{Time}}$ | $[M, L^2, T^{-2}] - [T] = [M, L^2, T^{-3}]$ | `power<watt>` |
| **$T_2$** | $\frac{\text{Memory}}{\text{Time}}$ | $[\text{memory}] - [T] = [\text{memory} \cdot T^{-1}]$ | `bandwidth<bytes_per_second>` |
| **$T_3$** | $\text{Time}^{-1}$ | $[0] - [T] = [T^{-1}]$ | `frequency<hertz>` |
| **$T_4$** | $\frac{\text{Throughput}}{\text{Power}}$ | $[\text{work} \cdot T^{-1}] - [M, L^2, T^{-3}] = [\text{work} \cdot M^{-1} \cdot L^{-2} \cdot T^2]$ | `efficiency<work_per_joule>` |

---

## 3. Plane 2: UMA Control Plane & Scheduler Exclusivity ($T_5 \dots T_8$)

### 3.1 $T_5$: RPE Multi-Resource Handshake, $T_{5, MC}$ Subtransition & Batch Invariant
The Resource-Performance-Energy (RPE) handshake coordinates `TIME`, `ENERGY`, and `MEMORY` quotas.
- **Hard Invariant ($\Delta Q_{\text{batch}} \equiv 0$):** Quotas cannot be mutated mid-batch. Any detected bottleneck transition updates the quota for the *subsequent* batch:
  $$\text{BottleneckChange} \implies Q_{\text{next\_batch}} \quad (\text{not } Q_{\text{current\_batch}})$$
  This guarantees bit-exact determinism of the compiled MicroOp DAG during execution.

#### Operational Subtransition $T_{5, MC}$ (Memory Controller Pressure & Cause Diagnosis)
While $RPE_{\text{memory}} = \frac{\sum \mu_i}{Q_{\text{memory}}}$ quantifies the *volume* of memory quota used, the **Memory Controller Pressure Vector ($\mathbf{MCP}$)** diagnoses *why* the controller is bottlenecked:
$$\mathbf{MCP} = \langle Q, B, R, F, A, W, H \rangle$$
where $Q$ is queue occupancy, $B$ is bank conflict rate, $R$ is row-miss pressure, $F$ is refresh blocking pressure ($t_{\text{RFC}}$), $A$ is request-age / service-debt pressure ($D_{MC}$), $W$ is write-drain turnaround, and $H$ is bandwidth headroom.

Subtransition $T_{5, MC}$ operates as an internal semantic sub-mapping of $T_5$ (strictly preserving the 56-node graph freeze, 0 cycles, without adding Node 57 or T13):
$$\boxed{T_{5, MC}: (RPE_m, \mathbf{MCP}) \longrightarrow (\text{MemoryCause}, \text{Action})}$$

- **Case A (High Volume, Healthy Controller):** $RPE_m = 0.90, \mu_{MC} = 0.35 \implies$ Sequential streaming without bank conflicts. Quota is sustained without throttling (`SUSTAIN_NOMINAL`).
- **Case B (Low Volume, High Contention):** $RPE_m = 0.45, \mu_{MC} = 0.91 \implies$ Bank conflicts or row misses dominate. Governor applies targeted cause-aware intervention:
  - $\text{BANK\_CONFLICT} \implies \text{REORDER\_REMAP\_DEFER}$
  - $\text{QUEUE\_CONGESTION} \implies \text{REDUCE\_PRODUCERS\_INCREASE\_QUANTUM}$
  - $\text{SERVICE\_DEBT\_BREACH} \; (D_{MC} > 1.0) \implies \text{PRIORITY\_INHERITANCE\_PROMOTION}$
  - $\text{REFRESH\_BLOCKING} \implies \text{HOLD\_BURST\_FOR\_REFRESH}$
  - $\text{WRITE\_DRAIN\_CONTENTION} \lor \text{BANDWIDTH\_SATURATION} \implies \text{DRAIN\_AND\_RECONCILE}$

### 3.2 $T_6$: State Switch $K \leftrightarrow S1$ with Asymmetric Hysteresis Guards
To eliminate policy flapping between full performance ($K$) and power-conserving shadow mode ($S1$):
- **Downscale ($K \longrightarrow S1$):** Triggers if any safety constraint is breached:
  $$\text{Battery} < 20\% \lor \mu_{\text{UMA}} \ge 0.78 \lor T \ge T_{\text{high}} \; (75^\circ\text{C})$$
- **Upscale ($S1 \longrightarrow K$):** Triggers strictly under joint hysteresis recovery:
  $$\text{Battery} > 25\% \land \mu_{\text{UMA}} < 0.60 \land T < T_{\text{low}} \; (65^\circ\text{C}) \land \text{RealtimeDemand} == \text{HIGH}$$
- **Thermal Headroom Formalism:** Thermal headroom is explicitly defined as a relative margin, not an absolute temperature:
  $$\text{ThermalHeadroom} = T_{\text{limit}} - T_{\text{actual}}$$

### 3.3 $T_7$: Chronometric Shadow Multiplexing as Phase Exclusivity
The 60 Hz frame cycle ($16.67\text{ ms}$) is apportioned chronometrically:
$$12.80\text{ ms (Render/Scanout)} + 0.13\text{ ms (Barrier/Blanking)} + 3.74\text{ ms (Shadow Window)} = 16.67\text{ ms}$$
- **Phase Model:** This is an intentional scheduler mutual-exclusion invariant, not a physical impossibility of GPU concurrency:
  $$\boxed{\text{SharedResourceConflict}(\text{Scanout}, \text{ShadowCompute}) \equiv 0}$$
  during the scheduler-granted shadow window. Epistemic reasoning and background microcode execute strictly when memory bus contention with scanout is zero.

### 3.4 $T_8$: Zero-Copy SPSC Ring Transfer
Critical path frames strictly follow the acquire/release barrier contract:
$$\text{Publish}(F) \implies \text{FullyInitialized}(F) \land \text{Size}(F) \equiv 128\text{B} \land \text{Align}(F) \equiv 128\text{B}$$

---

## 4. Plane 3: Hardware Transition Contract & Evidence Path ($T_{12}$ / HTE-001)

### 4.1 $T_{12}$: Physical Hardware Actuation Contract
Physical hardware actuation fundamentally alters physical reality:
$$\Delta S_{\text{physical}} \neq 0 \quad (\text{energy dissipated, flip-flops toggled, heat transferred})$$
Therefore, an actuation failure *cannot* magically roll back physical entropy.

### 4.2 Separation of Physical State and Authoritative State
The transactional guarantee $\Delta S \equiv 0$ strictly governs the **authoritative state machine**:
$$\langle S_{\text{authoritative}, t}, S_{\text{physical}, t} \rangle \xrightarrow{\text{Actuation}} \langle S_{\text{physical}, \text{after}}, \mathbf{R} \rangle$$
- **If Valid ($\forall i: |R_i| \le \epsilon_i$):**
  $$\text{COMMIT}(S_{\text{authoritative}, t+1} \leftarrow S_{\text{projected}})$$
- **If Invalid ($\exists i: |R_i| > \epsilon_i$):**
  $$\boxed{S_{\text{authoritative}, t+1} \equiv S_{\text{authoritative}, t}}$$
  $$S_{\text{physical}, \text{after}} \text{ and } \mathbf{R} \xrightarrow{} \text{QuarantineReceipt} \quad (\text{Evidence Captured})$$

### 4.3 Normalized Residual Verdict
The 5-dimensional residual vector $\mathbf{R} = (R_L, R_E, R_T, R_M, R_Q)$ is normalized against task-specified tolerances $\boldsymbol{\epsilon}$:
$$r_i = \frac{R_i}{\epsilon_i} \implies \boxed{\text{Valid} \iff \max_{i} |r_i| \le 1.0}$$

### 4.4 Reversibility Triage
When an actuation fails verification:
1. **$\text{REVERSIBLE}$:** Issue inverse compensating hardware command (e.g. restore frequency/voltage state).
2. **$\text{COMPENSATABLE}$:** Apply corrective delta in subsequent batch cycle.
3. **$\text{IRREVERSIBLE}$:** Isolate fault domain, enter safe fail-stop state, quarantine ring buffer.

---

## 5. Plane 4: Gameplay Semantics Consumer Plane ($T_9 \dots T_{11}$)

Gameplay semantics operate strictly as an application-level consumer of the transition system, decoupled from the hardware proof chain.

### 5.1 $T_9$: Dwell Timing & Gameplay Charge Multiplier
- **Timing Reconciliation:**
  $$12 \times 16.667\text{ ms} \approx 200.0\text{ ms} \quad (\text{12-frame completed dwell})$$
  Release occurs at $t \approx 200\text{ ms}$, corresponding exactly to the entry of the 13th frame.
- **Terminology Invariant:** Replaced "150% accumulated energy" with:
  $$\text{Gameplay Charge} = 1.5\times \quad \text{or} \quad \text{Damage Multiplier} = 1.5\times$$
  preventing semantic conflation with physical thermodynamic conservation of energy.

### 5.2 $T_{10}$: Canonical Symplectic Phase-Space Invariant
In Hamiltonian phase space $(q, p)$, the Liouville volume preservation is formally governed by the canonical symplectic matrix condition:
$$\boxed{\mathbf{J}^T \boldsymbol{\Omega} \mathbf{J} = \boldsymbol{\Omega}} \quad \text{where } \boldsymbol{\Omega} = \begin{pmatrix} 0 & \mathbf{I} \\ -\mathbf{I} & 0 \end{pmatrix}$$
While $\det(\mathbf{J}) = 1$ is a necessary condition (and sufficient for 2D area preservation), the symplectic form $\mathbf{J}^T \boldsymbol{\Omega} \mathbf{J} = \boldsymbol{\Omega}$ is the true general invariant. Kinetic energy preservation is not an automatic property of symplecticity alone; it additionally requires:
$$H(q', p') = H(q, p) \quad (\text{Conservative Hamiltonian})$$

### 5.3 $T_{11}$: State Split for Authoritative Invariant Preservation
To resolve the apparent contradiction between "$\text{InvalidAction} \implies \Delta S \equiv 0$" and "$\text{SoulWell} += \text{Aether}$", the game state space is partitioned into orthogonal direct-sum subspaces:
$$S = S_{\text{protected}} \oplus S_{\text{sanction}}$$
- **Protected State ($S_{\text{protected}}$):** Target player HP, vital attributes, position, and entity locks:
  $$\text{InvalidAction} \implies \Delta S_{\text{protected}} \equiv 0$$
- **Sanction State ($S_{\text{sanction}}$):** Disciplinary sinks, penalty accounts, and telemetry ledgers:
  $$\Delta S_{\text{sanction}} = \text{AuthorizedPenaltyTransition}(\text{AetherPenalty})$$
Thus:
$$\boxed{\text{InvalidAction} \implies \text{NoMutation}(S_{\text{target}}) + \text{AuthorizedPenaltyTransition}(S_{\text{sanction}})}$$
The constitutional law $\text{VITAL\_MAX\_HP} \equiv 6$ is strictly upheld without mathematical paradox.

---

## 6. Implementation Architecture

The operational transition algebra is implemented cleanly within existing modules:
- [`krystal_kernel/ksl/ksl_type_checker.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/ksl/ksl_type_checker.py):
  - `DimensionVector` (7-D abelian vector space)
  - `CANONICAL_PHYSICAL_QUANTITIES` & `DERIVED_DIMENSION_REGISTRY`
  - `DimensionalDerivationError` & Affine Normalization Enforcement
- [`krystal_kernel/ksl/ksl_ast.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/ksl/ksl_ast.py):
  - Unit-aware `LiteralExpr(value, literal_type, unit)`
- [`krystal_kernel/ksl/ksl_parser.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/ksl/ksl_parser.py):
  - Generic unit lookahead (`12.0<J>`, `0.5<s>`, `1000<ops>`)
  - Chained binary expression evaluation
- [`tests/test_ksl_compiler_v01.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/tests/test_ksl_compiler_v01.py):
  - Complete test verification of $T_1 \dots T_4$, affine normalization, and derivation error detection.

```
+-------------------------------------------------------------------------------+
| KSL v0.2 VERIFIED TRANSITION PIPELINE: T1-T4 -> T5-T8 -> T12 | (T9-T11 Decoupled) |
+-------------------------------------------------------------------------------+
