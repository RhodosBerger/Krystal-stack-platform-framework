# AEP WORKLOAD: GAME & TACTICAL DUEL (M1)
## The West 1876 Authoritative State & Combat Execution

```text
Workload Type:      Tactical Turn-Based Combat & Ballistics
Specification:      M1 Vertical Slice
Classification:     [IMPLEMENTED] & [MEASURED]
Authoritative Engine: krystal_web_hub.economic_engine.game_service
```

---

## 1. Workload Overview

The Game Workload represents the real-time and turn-based tactical arena of **Poslední Kmen: The West 1876**. It operates over a 19-hex deterministic grid where opposing archetypes engage in duels, construct sapper trenches, and compute mortar ballistics.

Under the **Adaptive Execution Processor (AEP)** model, the Game is not a standalone bespoke application, but a standard payload executing across layers L0 through L8.

---

## 2. AEP Layer Mapping for Game Workload

```text
┌────────────────────────────────────────────────────────────────────────┐
│ L0: CONTRACT PROCESSOR                                                 │
│     Validates /api/game/duel_round and /api/game/mortar_strike schemas.│
│     Enforces match_id and action type (DUEL, MORTAR, TRENCH, RESET).   │
├────────────────────────────────────────────────────────────────────────┤
│ L1: STATE FABRIC                                                       │
│     MatchRegistry resolves MatchState instance by match_id UUID.       │
│     Guarantees zero bleed between concurrent matches.                  │
├────────────────────────────────────────────────────────────────────────┤
│ L2: DETERMINISTIC LAW CORE                                             │
│     Invariants:                                                        │
│     • VITAL_MAX_HP == 6 (0 <= HP <= 6)                                 │
│     • Trench Cover: exactly 85% damage mitigation                      │
│     • Mortar Dispersion: Sadovsky overpressure radius [1, 2] hexes     │
├────────────────────────────────────────────────────────────────────────┤
│ L3: OBSERVATION ENGINE                                                 │
│     M1TelemetryRecord logs latency, damage delta, shot accuracy.       │
│     Pure observation: GET /api/game/status produces zero state delta.  │
├────────────────────────────────────────────────────────────────────────┤
│ L4: POLICY PROCESSOR                                                   │
│     Selects interactive low-latency profile (sub-50ms HTTP turnaround).│
├────────────────────────────────────────────────────────────────────────┤
│ L5: WORKLOAD SCHEDULER                                                 │
│     Assigned to Realtime Lane for fast player feedback.                │
├────────────────────────────────────────────────────────────────────────┤
│ L6: EXECUTION FABRIC                                                   │
│     GameRules & PosledniKmenGameService evaluate transition S' = T(S,A)│
├────────────────────────────────────────────────────────────────────────┤
│ L7: BACKPRESSURE CONTROLLER                                            │
│     Client request rate-limiter prevents spamming duel turns.          │
├────────────────────────────────────────────────────────────────────────┤
│ L8: COMMIT / AUDIT ENGINE                                              │
│     Generates state hashes H(S_before) -> H(S_after).                  │
│     Commits S' if valid; discards S' if HP invariant violated.         │
│     Emits AepReceipt.                                                  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Concrete Invariant Contracts [ARCHITECTURAL]

### 1. Invariant HP Bound:
$$\forall s \in S_{\text{game}}, \forall p \in \{\text{player}, \text{opponent}\} : 0 \le s.\text{hp}_p \le 6$$
If any transition candidate yields $s'.\text{hp} < 0$, it is clamped to $0$; if $s'.\text{hp} > 6$, the candidate is rejected.

### 2. Sapper Trench Protection:
When a combatant occupies a completed trench hex:
$$\text{Damage}_{\text{effective}} = \text{round}\Big(\text{Damage}_{\text{raw}} \times (1.0 - 0.85)\Big)$$
Cover provides exactly an $85\%$ damage reduction.

### 3. Transition Atomicity:
$$\neg\text{ValidAction}(A) \implies S_{t+1} \equiv S_t \land \text{Receipt.verdict} = \text{ABORTED}$$

---

## 4. Verification & Evidence Status [MEASURED]

The Game Workload is fully verified in the local and remote test suites:
- **Unit Suite:** [`tests/test_posledni_kmen_game.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/tests/test_posledni_kmen_game.py) (Duel mechanics, mortar dispersion, trench digging).
- **Transport Suite:** [`tests/test_posledni_kmen_server_api.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/tests/test_posledni_kmen_server_api.py) (Live HTTP E2E transport, multi-tenant session isolation, zero-mutation status verification).
- **Benchmark Evidence:**
  - `BENCH-GAME-M1-01`: 1,000 simulated combat rounds completed with **0 invariant violations**.
  - `BENCH-GAME-M1-02`: Multi-tenant session isolation verified across 10 concurrent HTTP sessions.
