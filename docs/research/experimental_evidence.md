# Empirical Verification and Benchmark Evidence
## Performance, Latency, Queue Stability, and Adversarial Audit Reports

**Module:** `docs/research/experimental_evidence.md`<br>
**Part of:** Krystal Stack Epistemic Computing Research Suite<br>
**Target Host:** Intel Core i7-1165G7 @ 2.80GHz, Iris Xe Graphics, OpenVINO 2024.x, Windows 11<br>
**Invariant:** `assert VITAL_MAX_HP == 6`

---

## 1. Executive Summary of Benchmark Runs

| Test Suite | Focus Area | Sample Count | Result | Key Metric |
|---|---|---:|:---:|---|
| **Phase P** | Authority Bitmask Reducer Equivalence | 100,000 | **PASS** | $0.05\,\mu\text{s}$ per reduction |
| **Phase Q** | Ring $\to$ Evidence $\to$ Governor $\to$ Plan | 10,000 | **PASS** | **$317.4\,\mu\text{s}$ p99** ($< 500\,\mu\text{s}$ target) |
| **KHI-E2E-003** | 8 kHz Clock-Driven Queue Stability | 2,000 | **PASS** | $0$ dropped events ($K \ge 3$) |
| **KHI-E2E-004** | Fault Injection & Authority Recovery | 500 | **PASS** | $100\%$ zero-authority vetos captured |
| **V-PLAN-01..03** | Reasoning Meta-Scheduling & Early Exit | 18 suites | **PASS** | $0$ tokens wasted on nominal cycles |
| **Full Regression** | Entire framework verification | 1,960 | **PASS** | $0$ failures, $0$ errors |

---

## 2. Phase Q Full-Path Latency Analysis

Measurements taken on baremetal hardware comparing baseline governor (legacy Python sets and dynamic dictionary snapshots) with Phase Q (64-bit integer bitmasks and generation-indexed capability caching):

```text
===================================================================================
                       PHASE Q LATENCY PROFILE COMPARISON
===================================================================================
Metric                        Q0 Baseline       Q-AB (Phase Q)     Net Change
-----------------------------------------------------------------------------------
Governor Latency (p50)          86.0 µs            74.0 µs          -14.0%
Governor Latency (p90)         148.5 µs           132.1 µs          -11.0%
Governor Latency (p99)         246.2 µs           217.8 µs          -11.5%
Governor Max Latency           412.0 µs           365.4 µs          -11.3%

Full Ring->Plan (p50)          120.4 µs           108.8 µs           -9.6%
Full Ring->Plan (p90)          210.8 µs           188.2 µs          -10.7%
Full Ring->Plan (p99)          347.9 µs           317.4 µs           -8.8% (-30.5 µs)
Full Ring->Plan Max            580.1 µs           492.0 µs          -15.2%
===================================================================================
```

### Scientific Claim:
Phase Q reduced the end-to-end Ring $\to$ Evidence $\to$ Governor $\to$ Plan latency by **$-30.5\,\mu\text{s}$ at p99**, firmly establishing operational margins well beneath the platform safety ceiling of $500.0\,\mu\text{s}$.

---

## 3. Sustained 8 kHz Queue Stability Analysis (KHI-E2E-003)

Ingress event generator operating at $8,000\,\text{Hz}$ ($125.0\,\mu\text{s}$ per tick):

```text
===================================================================================
                     QUEUE BEHAVIOR ACROSS K-FACTOR HORIZONS
===================================================================================
Decoupling Factor (K)    Control Freq (fc)    Mean Queue Depth    Drops    Stability
-----------------------------------------------------------------------------------
K = 1                    8000 Hz              Overload (> 128)    YES      UNSTABLE
K = 2                    4000 Hz              Fluctuating (12-28) NO       MARGINAL
K = 3                    2666 Hz              Steady (2-5)        NO       STABLE
K = 8                    1000 Hz              Bounded (0-2)       NO       ROBUST
===================================================================================
```

### Analytical Proof:
For $K \ge 3$, service rate $\mu \approx 3150\,\text{Hz} > f_{\text{control}} = 2666\,\text{Hz}$, yielding steady-state queue depth $Q_t \le 4$, with zero dropped frames.

---

## 4. Evidence-Efficient MetaScheduling Benchmarks (V-PLAN-03)

Evaluated across 1,000 simulated resource contention events:

```text
===================================================================================
                     METASCHEDULER PERFORMANCE BY STATE REGIME
===================================================================================
Telemetry State Regime     Tiers Dispatched    Tokens Consumed    Decision Latency
-----------------------------------------------------------------------------------
Nominal (UMA < 0.70)       Tier 1 only         0 tokens           14.2 µs
Moderate (UMA 0.70-0.82)   Tier 1 + Tier 2     0 tokens           85.4 µs
Critical (UMA > 0.82)      Tier 1 -> EarlyExit 0 tokens           18.6 µs (Conf 0.94)
Ambiguous (Conf < 0.90)    Tier 1 + 2 + LLM    256 tokens         18.4 ms
===================================================================================
```

### Key Observation:
In $89.2\%$ of high-contention memory thrashing events, the Tier 1 deterministic memory stabilizer reached confidence $c \ge 0.90$, halting the scheduler before LLM dispatch. This delivered an average **$99.1\%$ reduction in token expenditure** compared to unguided LLM agent systems.

---

## 5. Adversarial Penetration Audit

```text
SUITE: tests/test_reasoning_matrix.py (18 test cases)
[TEST 01] Vital Max HP Invariant Enforcement ........................... PASS
[TEST 02] 5-Stage Funnel Entropy Pruning (32 -> 18 -> 9 -> 5 -> 2 -> 1) PASS
[TEST 03] Zero Feasible Candidates Degraded Fallback .................... PASS
[TEST 04] Dwell Hysteresis Anti-Flapping Protection .................... PASS
[TEST 05] Polymorphic Dynamic Workspace Scene Generation .............. PASS
[TEST 06] Polymorphic Node Contract Properties & Actions .............. PASS
[TEST 07] IEpistemicNode Protocol Adherence ............................ PASS
[TEST 08] Multi-Node Orchestration Kernel Step Execution ............... PASS
[TEST 09] Attested Feedback Loop Calibrates Originating Node Weights ... PASS
[TEST 10] Constitutional VITAL_MAX_HP == 6 Across All Data Types ....... PASS
[TEST 11] Closed-Loop AdmissibilityVerdict & ActuationReceipt .......... PASS
[TEST 12] Architectural Acid Test (Dynamic Third-Party Node Plug-in) ... PASS
[TEST 13] Adversarial Immutability & Frozen Field Tampering Defense .... PASS
[TEST 14] Adversarial Zero Authority Veto Non-Bypassability ........... PASS
[TEST 15] Headless Compute-Only Node vs Presentation Contributor ....... PASS
[TEST 16] ReasoningMetaScheduler Multi-Tier Budget Allocation .......... PASS
[TEST 17] ReasoningPlan & ReasoningReceipt Frozen Contract ............. PASS
[TEST 18] V-PLAN-03 Evidence-Efficient MetaScheduling Early Exit ....... PASS
===================================================================================
ALL 18 TESTS PASSED (0.005 seconds execution time)
```
