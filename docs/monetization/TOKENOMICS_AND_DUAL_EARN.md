# Krystal Stack: Dual-Earn Ledger & Economic Compute Trading

![Krystal Stack Dual-Earn Ledger](images/dual_earn_tokenomics.jpg)

> **Document Classification:** Economic Architecture, Tokenomics & Compute Cycle Arbitrage  
> **Status:** APPROVED COMMERCIAL SPECIFICATION  
> **Core Invariant:** `assert VITAL_MAX_HP == 6` (Inviolable State Gate)

---

## 1. The Dual-Earn Paradigm

In standard compute infrastructures, hardware operators face idle costs during off-peak hours, while AI startups and developers face exorbitant on-demand cloud pricing.

**Krystal Stack's Dual-Earn Ledger** introduces a decentralized peer-to-compute network operating on two synchronized, atomic rails:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              THE DUAL-EARN LEDGER ARCHITECTURE                         │
├────────────────────────────────────────┬───────────────────────────────────────────────┤
│ RAIL 1: COMPUTE ASSET TRANSACTIONS     │ RAIL 2: DISTRIBUTED REWARD ENGINE             │
├────────────────────────────────────────┼───────────────────────────────────────────────┤
│ • Real-time order book for GPU cycles  │ • Liquidity staking & node validation         │
│ • Microsecond NPU inference routing    │ • Proof-of-Useful-Compute verification       │
│ • Spot bandwidth & zero-copy cache     │ • Dynamic fee distribution to node providers  │
│ • Zero-loss atomic transaction protocol│ • Invariant guard against malicious overdraw  │
└────────────────────────────────────────┴───────────────────────────────────────────────┘
```

---

## 2. Market Mechanics: The Cross-Resource Order Book

Hardware providers list idle computational capacity into a standardized order book normalized by hardware compute coefficients:

$$\mathcal{C}_{\text{unit}} = w_{\text{gpu}} \cdot \text{TFLOPS}_{\text{FP16}} + w_{\text{npu}} \cdot \text{TOPS}_{\text{INT8}} + w_{\text{bw}} \cdot \text{GB/s}_{\text{UMA}}$$

### 2.1 The Two Participant Classes
1. **Buyers (AI Startups, Game Developers, Research Labs):**
   - Submit inference batches, procedural generation requests, or simulation frames.
   - Pay via KST (Krystal Stack Tokens) or automated fiat-backed stablecoin balances.
   - Enjoy up to **60% discount** compared to legacy hyperscaler spot instances.

2. **Sellers (Edge Kiosks, Machine Shops, Datacenters, Idle Workstations):**
   - Monetize unused night-time GPU cycles or idle NPU execution slots.
   - Earn automated payouts logged immutably into the local audit trail.
   - Retain 100% priority preemptibility: if local workload spikes, guest workloads are paused or migrated within $1.2\,\text{ms}$.

---

## 3. Dynamic Lookahead Hysteresis & Anti-Flapping Invariant

To prevent rapid oscillation between compute allocation states (ping-pong effect), the Dual-Earn engine enforces **Dynamic Lookahead Hysteresis**:

$$h \ge 2 \hat{\delta}_{\text{workload}}$$

Transactions must pass the **Dual-Gate Safety Arbiter**:
1. **Hardware Integrity Gate:** Silicon temperature $T_j \le 78^\circ\text{C}$, thermal headroom conserved.
2. **Constitutional Invariant Gate:** `VITAL_MAX_HP == 6` strictly verified on all transaction packets.

---

## 4. Staking, Yield, and Governance

- **Node Staking:** Minimum stake of 1,000 KST required to register as a verified Tier 1 or Tier 2 compute broker.
- **Slashing Conditions:** Malicious state drift, intentional packet dropping, or spoofed telemetry results in immediate forfeiture of the node's deposit.
- **Reward Frequency:** Block rewards distributed on a 1-second cadence via local vector clocks.
