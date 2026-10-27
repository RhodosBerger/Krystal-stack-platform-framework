# KRYSTAL STACK: UNIVERSAL HARDWARE TRAJECTORY RUNTIME & CRYPTO SCENARIO ENGINE SPECIFICATION

## 1. Context & Epistemic Purpose

This specification unites two complementary cyber-physical layers within the Krystal Stack:
1. **Universal Hardware Trajectory Runtime:** Moving from static chip tuning to an architecture profile modeling transitions across operational quadrants ($Q1..Q4$).
2. **Dual-Branch Quarterly Crypto Market & Farm Expansion Scenario Engine:** Disentangling external market demand (Spot ETF flows) from internal physical supply expansions (adding $N$ ASICs/GPUs to a farm), computing marginal value per chip ($\text{ChipMarginalValue}$) and break-even intervals across Bear, Base, and Bull scenarios.

**Epistemic Status:** `[ARCHITECTURAL_SPEC]` and `[MODEL_SIMULATED]`.  
**Constitutional Invariant:** $\text{VITAL\_MAX\_HP} \equiv 6 \land (\neg\text{Valid}(A) \implies \Delta S \equiv 0)$.

---

## 2. Universal Hardware Operating Trajectory Runtime

### 2.1 Canonical Architecture Profile
Instead of a legacy hardware BIOS, Krystal Stack defines an **Architecture Profile**:
$$\boxed{\text{ArchitectureProfile} = \langle \text{Topology}, \text{Limits}, \text{ResponseCurves}, \text{TransitionRules}, \text{Evidence} \rangle}$$

- **Topology:** Core counts (P/E), memory hierarchies (L1/L2/L3/UMA), bus interconnects (PCIe/NVLink/CXL).
- **Limits:** $T_{j,\max}$, PL1/PL2 power caps, VRM inductive flyback bounds ($\Delta V_{\max} \le 0.15\text{ V}$).
- **ResponseCurves:** $\frac{d\text{Freq}}{d\text{Voltage}}$, $\frac{d\text{Throughput}}{d\text{Power}}$, Marginal Power Utility ($\text{MPU}$).
- **TransitionRules:** Safe trajectories preventing thermal tripping or bus lock contention.
- **Evidence:** Cryptographic receipts verifying physical observations.

### 2.2 The Four Operational Quadrants ($Q1..Q4$)
The state space is partitioned across normalized **Computational Load** ($L \in [0.0, 1.0]$) and **Available Headroom** ($H \in [0.0, 1.0]$):

```
Headroom (H)
  ▲
1.0 ┼───────────────────────┬───────────────────────┐
    │          Q1           │          Q2           │
    │      STABLE IDLE      │   OPTIMAL PRODUCTIVE  │
    │ (Low Load, High Hdrm) │ (High Load, High Hdrm)│
    │  - Pacing / Green     │  - Peak Useful Work   │
0.5 ┼───────────────────────┼───────────────────────┤
    │          Q3           │          Q4           │
    │     CONSTRAINED       │   SATURATION / RISK   │
    │  - Inefficient stall  │  - Thermal throttling │
    │  - Memory/IO choked   │  - Emergency backpres.│
0.0 ┴───────────────────────┴───────────────────────┘
   0.0                     0.5                     1.0  ► Load (L)
```

- **Quadrant 1 (Q1 - Stable Idle):** Low load, high headroom. Safe for background consolidation and speculative prefill.
- **Quadrant 2 (Q2 - Optimal Productive):** High load, high headroom. Maximum verified useful work per grid Joule ($\eta_{\text{facility}}$).
- **Quadrant 3 (Q3 - Constrained):** Low compute load, low headroom (e.g. stalled on memory bus contention or thermal saturation from background jobs).
- **Quadrant 4 (Q4 - Saturation / Risk):** High load, exhausted headroom. Immediate backpressure downscaling (Raymarch LOD, batch throttling) required to prevent silicon thermal trip.

### 2.3 Trajectory-Aware Control Principle
Static governors observe only $(L, H)$ at time $t$. Krystal Stack evaluates the **State Trajectory Vector** $\mathbf{\tau}(t) = \frac{d}{dt}(L(t), H(t))$:
- **Trajectory $\tau_A: Q1 \to Q2$:** Controlled, productive ramp-up.
- **Trajectory $\tau_B: Q1 \to Q2 \to Q4$:** Rapid saturation indicating impending bus contention; trigger proactive phase-power throttling.
- **Trajectory $\tau_C: Q3 \to Q2$:** Successful memory/thermal recovery.

### 2.4 The 4 MVP Core Services
1. **Predictive Runtime Compiler (PRC):** Evaluates candidate execution graphs against boundary limits before commit.
2. **Adaptive Inference Governor (AIG):** Dynamically modulates token batch sizes, worker counts, and quantization levels (FP32 $\to$ FP16 $\to$ INT8).
3. **Diagnostic Key Engine (DKE):** Rapid incident lookup using 3-tuple indexing: $\langle \text{IdealKey}, \text{DeviationKey}, \text{RecoveryKey} \rangle$.
4. **Validation and Calibration Service (VCS):** Closed-loop comparison of predicted state $\hat{S}_{t+1}$ against empirical measurement $S_{t+1}$.

---

## 3. Dual-Branch Quarterly Crypto Market & Farm Scenario Engine

### 3.1 Causal Separation: Demand vs. Supply
A single farm expanding hardware capacity does not dictate the spot price of Bitcoin. The model separates two orthogonal causal pathways:

```
[ Institutional / Retail Spot ETF Flows ]
                   │
                   ▼
       BRANCH 1: MARKET DEMAND
  ├── Net ETF Inflow Volume (Quarterly Aggregated)
  ├── Flow Velocity & Persistence
  └── Reference Market Order-Book Liquidity
                   │
                   ▼
         ETF Demand Pressure (P_demand)
                   │
                   ├─────────────────────────────┐
                   ▼                             ▼
       [ QUARTERLY SCENARIO SOLVER: BEAR / BASE / BULL ]
                   ▲                             ▲
                   │                             │
         Miner Supply Pressure                   │
                   │                             │
       BRANCH 2: FARM EXPANSION                  │
  ├── Existing Farm Hashrate (H_old)             │
  ├── New Chip Additions (Delta N * H_chip)      │
  ├── Total Farm Hashrate (H_farm)               │
  ├── Network Difficulty & Total Hashrate        │
  ├── Electricity Tariff (€/kWh) & Farm Power    │
  └── Miner BTC Sell Ratio                       │
```

### 3.2 Formal Mathematical Models

#### 1. ETF Demand Pressure:
$$\text{ETFPressure}_Q = \frac{\text{NetETFInflows}_Q}{\text{ReferenceMarketLiquidity}_Q}$$
$$\mathbf{S}_{\text{market}} = \left[ \text{NetFlow}_Q, \text{FlowVelocity}, \text{FlowPersistence}, \text{Volatility}, \text{AUMChange} \right]$$

#### 2. Farm Expansion & Production:
$$H_{\text{farm, new}} = H_{\text{farm, old}} + \Delta N \cdot H_{\text{chip}}$$
$$\text{Share}_{\text{farm}} = \frac{H_{\text{farm, new}}}{H_{\text{network}}}$$
$$\text{BTC}_{\text{farm}, Q} = \text{Share}_{\text{farm}} \cdot \text{BTC}_{\text{network}, Q} \cdot \text{Uptime} \cdot (1 - \text{PoolFee})$$

#### 3. Power, Energy & Electricity Cost:
$$P_{\text{farm, new}} = P_{\text{farm, old}} + \Delta N \cdot P_{\text{chip}} + P_{\text{overhead}}$$
$$\text{EnergyCost}_Q = \frac{P_{\text{farm, new}}}{1\,000\,000} \cdot 24 \cdot 91.25 \cdot \text{ElectricityPrice}_{\text{kWh}} \cdot \text{PUE}$$

#### 4. Mining Supply Pressure on Spot Market:
$$\text{MinerSellPressure}_Q = \text{BTC}_{\text{farm}, Q} \cdot \text{MinerSellRatio}$$

#### 5. Break-Even BTC Price:
$$\text{BreakEvenPrice} = \frac{\text{EnergyCost}_Q + \text{OpEx}_Q}{\text{BTC}_{\text{farm}, Q}}$$

#### 6. Marginal Value per Chip ($\text{ChipMarginalValue}$):
$$\boxed{\text{ChipMarginalValue} = \frac{\Delta \text{ExpectedProfit}_Q}{\Delta N} \quad \left[ \frac{\text{€}}{\text{chip}} \right]}$$

#### 7. Tri-Scenario Quarterly Price Envelope ($P_{Q+1}$):
$$P_{Q+1}^{(\text{scenario})} = P_Q \cdot \left( 1 + \Delta_{\text{macro}} + \alpha \cdot \text{ETFPressure}_Q - \beta \cdot \frac{H_{\text{network, new}}}{H_{\text{network, old}}} \right)$$

- **BEAR:** Low/negative ETF flows, network difficulty expansion, low multiplier.
- **BASE:** Median trailing ETF flow persistence, steady hashrate growth.
- **BULL:** Institutional ETF acceleration, supply deficit, favorable macro liquidity.

---

## 4. Input Form & Output Report Schema

### 4.1 Input Form Parameters
1. `current_btc_price_eur`: Current spot price (€).
2. `quarter_etf_net_inflow_eur`: Net quarterly ETF inflow (€).
3. `etf_flow_persistence`: Momentum factor $[0.0, 1.0]$.
4. `existing_farm_hashrate_th`: Existing farm hashrate (TH/s).
5. `new_chip_count`: Number of newly added chips $\Delta N$.
6. `hashrate_per_chip_th`: Hashrate per chip (TH/s).
7. `power_per_chip_w`: Power per chip (Watts).
8. `network_hashrate_eh`: Global network hashrate (EH/s).
9. `network_btc_per_quarter`: Expected global block rewards + fees in quarter (~41,000 BTC).
10. `electricity_price_eur_kwh`: Power cost (€/kWh).
11. `farm_uptime_ratio`: Operating uptime $[0.0, 1.0]$.
12. `miner_sell_ratio`: Percentage of mined BTC liquidated for OpEx $[0.0, 1.0]$.

### 4.2 Output Report Structure
- **Production Metrics:** Farm Hashrate, Network Share (%), Mined BTC, Electricity Cost (€).
- **Economic Thresholds:** Break-Even BTC Price (€), Farm Profitability Margin.
- **Marginal Chip Metrics:** Marginal Profit per Chip (€/chip/quarter), Marginal Payback Period.
- **Scenario Price Matrix (BEAR / BASE / BULL):** Predicted Price, Revenue, Net Profit, Confidence Score.
