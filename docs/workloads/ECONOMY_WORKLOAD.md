# AEP WORKLOAD: TRANSACTIONAL ECONOMY & LEDGER (M2)
## Double-Entry Conservation & Transactional Crafting Architecture

```text
Workload Type:      Transactional Ledger & Economy Engine
Specification:      M2 Generalizability Acid Test
Classification:     [ARCHITECTURAL] & [IMPLEMENTED]
Authoritative Engine: krystal_kernel.aep.economy_engine
```

---

## 1. Workload Overview

The **Economy Workload (M2)** serves as the primary acid test for the generalizability of the **Adaptive Execution Processor (AEP)**. Rather than being treated as a casual game feature ("crafting mini-game"), it is designed as a **mission-critical transactional financial engine** governed by strict double-entry conservation laws.

If the exact same AEP cycle (L0–L8) that governs game combat (M1) can execute financial settlements, item crafting, and ledger balancing without altering the architectural core, then AEP is proven to be a true **general-purpose execution architecture**.

---

## 2. AEP Layer Mapping for Economy Workload

```text
┌────────────────────────────────────────────────────────────────────────┐
│ L0: CONTRACT PROCESSOR                                                 │
│     Validates /api/economy/craft and /api/economy/transfer schemas.    │
│     Enforces caller signature, idempotency token, and payload format.  │
├────────────────────────────────────────────────────────────────────────┤
│ L1: STATE FABRIC                                                       │
│     RuntimeRegistry resolves isolated AccountState and InventoryState  │
│     by account_id or wallet_id. Zero shared mutable state across users.│
├────────────────────────────────────────────────────────────────────────┤
│ L2: DETERMINISTIC LAW CORE                                             │
│     Invariants:                                                        │
│     • Double-Entry Conservation: ∑ Debits ≡ ∑ Credits (per tx)        │
│     • Non-Negative Balance: Balance >= 0                               │
│     • Recipe Invariant: Consumed resources exactly equal recipe inputs │
│     • Inventory Capacity: Total items <= MAX_INVENTORY_CAPACITY        │
├────────────────────────────────────────────────────────────────────────┤
│ L3: OBSERVATION ENGINE                                                 │
│     Telemetry monitors transaction throughput (TPS), settlement        │
│     latency, and market velocity without state modification.           │
├────────────────────────────────────────────────────────────────────────┤
│ L4: POLICY PROCESSOR                                                   │
│     Selects settlement policy:                                         │
│     - INSTANT_ISOLATED: immediate single transaction commit            │
│     - BATCH_CLEARING: aggregated settlement during low-pressure window │
├────────────────────────────────────────────────────────────────────────┤
│ L5: WORKLOAD SCHEDULER                                                 │
│     Dispatched to Transactional Economy Lane (isolated from Realtime). │
├────────────────────────────────────────────────────────────────────────┤
│ L6: EXECUTION FABRIC                                                   │
│     CraftingEngine & DoubleEntryLedger compute candidate state:        │
│     S' = T(S, Action)                                                  │
├────────────────────────────────────────────────────────────────────────┤
│ L7: BACKPRESSURE CONTROLLER                                            │
│     Monitors trading rate. Injects pacing delays (rate-limiting) under │
│     high market turbulence instead of dropping ledger transactions.    │
├────────────────────────────────────────────────────────────────────────┤
│ L8: COMMIT / AUDIT ENGINE                                              │
│     Validates I(S'). If conserved, atomically commits candidate S' to  │
│     the append-only ledger and emits standardized AepReceipt.          │
│     If unbalanced, S' is discarded (ΔS ≡ 0).                           │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Constitutional Economy Invariants [ARCHITECTURAL]

### 1. Double-Entry Accounting Invariant
For every financial and resource mutation $\tau$, the sum of debits across all accounts must strictly equal the sum of credits:

$$\sum_{k=1}^{n} \text{Debits}_k \equiv \sum_{k=1}^{n} \text{Credits}_k$$

No currency, resource token, or item may be created or destroyed *ex nihilo*. System fees must be explicitly credited to the system vault account.

### 2. Solvent Asset Balance Invariant
No participant account may ever enter an uncollateralized negative balance:

$$\forall a \in \text{Accounts}, \forall r \in \text{Resources} : \text{Balance}(a, r) \ge 0$$

### 3. Conservation of Crafting Transformation
Crafting recipe conversion requires precise atomic balance:

$$\text{Inputs}(R) \xrightarrow{\quad\tau\quad} \text{Output}(R) + \text{Byproduct}(R)$$

If an account possesses $m < \text{Inputs}(R)$, the candidate state generation fails at L2, and authoritative state remains completely untouched:

$$\Delta S \equiv 0$$

---

## 4. Canonical Crafting Sequence (L0–L8)

```text
POST /api/economy/craft { user_id: "u-42", recipe_id: "recipe-black-powder" }
        │
        ▼
   L0: Contract Check ──► Valid format & signature?
        │ Yes
        ▼
   L1: State Binding ──► Acquire AccountState(u-42) from RuntimeRegistry
        │
        ▼
   L2: Invariant Check ──► Check: Sulfur >= 2, Saltpeter >= 3, Charcoal >= 1?
        │ Yes
        ▼
   L3: Observe Load ──► Record telemetry: queue depth, trade latency
        │
        ▼
   L4: Select Policy ──► INSTANT_ISOLATED policy
        │
        ▼
   L5: Scheduling ──► Schedule to Transactional Economy Lane
        │
        ▼
   L6: Execute ──► Compute S':
                   Debits: [Sulfur: 2, Saltpeter: 3, Charcoal: 1]
                   Credits: [BlackPowder: 1, SystemFee: 0]
                   Sum(Debits) == Sum(Credits) in equivalent atomic valuation
        │
        ▼
   L7: Backpressure ──► Check rate limit for user u-42 (OK)
        │
        ▼
   L8: Commit Gate:
       - Validate: Balance(Sulfur) >= 0? (True)
       - Compute: H(S') = HMAC-SHA256(S')
       - Commit: S_{t+1} = S'
       - Append to Ledger
       - Emit AepReceipt
```
