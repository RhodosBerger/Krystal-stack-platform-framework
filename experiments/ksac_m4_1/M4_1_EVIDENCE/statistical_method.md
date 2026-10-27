# KSAC M4.1 — STATISTICAL PROTOCOL AND ESTIMATOR SPECIFICATION

## 1. Randomized Block Design (RBD) & Pair Definition

Random execution without pairing (`B K K B B K...`) does not naturally define pairs and exposes trials to thermal drift.
KSAC M4.1 enforces a strict **Randomized Block Design (RBD)**:

$$\text{Pair}_i = (B1^{(i)}, K^{(i)}) \quad \text{for } i \in \{1, \dots, 30\}$$

where both members of pair $i$ belong to the same experimental block ($Block_i$), with randomized intra-block execution order ($B1 \to K$ or $K \to B1$).
This design rigorously balances and eliminates thermal drift and background OS scheduling noise between treatments.

## 2. Verified Hardware Leverage Estimators

Two estimators are tracked to ensure complete mathematical transparency:

### 2.1 Pairwise Ratio Estimator (Mean of Block-Paired Ratios) — PRIMARY
$$\overline{\text{VHL}}_{HC} = \frac{1}{N} \sum_{i=1}^{N} \frac{K^{(i)}}{B1^{(i)}} = \mathbf{1.1516}$$
Where $K^{(i)}$ and $B1^{(i)}$ are the verified efficiencies ($\text{ops}/\text{J}$) of the $i$-th paired trial under the identical block.

### 2.2 Ratio of Means Estimator — SECONDARY AUDIT
$$\widehat{\text{VHL}}_{\text{means}} = \frac{\overline{K}}{\overline{B1}} = \frac{22.04033}{19.14418} = \mathbf{1.1513}$$

### 2.3 Estimator Delta & Consistency
$$\Delta_{\text{est}} = |\overline{\text{VHL}}_{HC} - \widehat{\text{VHL}}_{\text{means}}| = |1.1516 - 1.1513| = \mathbf{0.0003} \quad (0.028\%)$$
Given standard error $\text{SE} = \frac{s}{\sqrt{N}} = 0.005280$, $\Delta_{\text{est}} \ll \text{SE}$. Both estimators are statistically congruent.

## 3. Confidence Interval Derivation
$$\text{CI}_{95\%} = \overline{\text{VHL}}_{HC} \pm t_{0.025, 29} \cdot \text{SE} = 1.1516 \pm (2.0452 \times 0.00528) = [\mathbf{1.1408}, \mathbf{1.1624}]$$
Because $\text{CI}_{95\%, \text{lower}} = 1.1408 > 1.10$, the harness benchmark satisfies the pre-registered `STRONG_PASS` threshold.

## 4. Control Leverage Factor (CLF_N)
- **Active Decision Window ($E_{\text{ctrl}} = 4.2194\text{ J}$)**:
  $$\mathbf{CLF}_{N,\text{active}} = \frac{\Delta W / W_B}{E_{\text{ctrl}} / E_B} = \frac{13.7943\%}{0.4824\%} = \mathbf{28.59}$$
- **Continuous Gross Board Energy ($E_{\text{FPGA,gross}} = 23.1194\text{ J}$)**:
  $$\mathbf{CLF}_{N,\text{gross}} = \frac{\Delta W / W_B}{E_{\text{FPGA,gross}} / E_B} = \frac{13.7943\%}{2.6432\%} = \mathbf{5.22}$$
