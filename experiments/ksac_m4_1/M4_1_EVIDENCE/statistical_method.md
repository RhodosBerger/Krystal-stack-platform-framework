# KSAC M4.1 — STATISTICAL PROTOCOL AND ESTIMATOR SPECIFICATION

## 1. Verified Hardware Leverage (VHL_F) Estimators

Two estimators are tracked to ensure complete transparency and prevent auditor ambiguity:

### 1.1 Pairwise Ratio Estimator (Mean of Ratios) — PRIMARY
$$\overline{\text{VHL}}_F = \frac{1}{N} \sum_{i=1}^{N} \frac{K_i}{B1_i} = \mathbf{1.1516}$$
Where $K_i$ and $B1_i$ are the verified efficiencies ($\text{ops}/\text{J}$) of the $i$-th paired trial under randomized execution.

### 1.2 Ratio of Means Estimator — SECONDARY AUDIT
$$\widehat{\text{VHL}}_{\text{means}} = \frac{\overline{K}}{\overline{B1}} = \frac{22.04033}{19.14418} = \mathbf{1.1513}$$

### 1.3 Estimator Equivalence & Consistency
$$\Delta_{\text{est}} = |\overline{\text{VHL}}_F - \widehat{\text{VHL}}_{\text{means}}| = |1.1516 - 1.1513| = \mathbf{0.0003} \quad (0.028\%)$$
Given standard error $\text{SE} = \frac{s}{\sqrt{N}} = 0.0053$, $\Delta_{\text{est}} \ll \text{SE}$. Both estimators are statistically congruent.

## 2. Confidence Interval Derivation
$$\text{CI}_{95\%} = \overline{\text{VHL}}_F \pm t_{0.025, 29} \cdot \text{SE} = 1.1516 \pm (2.0452 \times 0.0053) = [\mathbf{1.1408}, \mathbf{1.1624}]$$
Because $\text{CI}_{95\%, \text{lower}} = 1.1408 > 1.10$, the milestone meets the pre-registered `STRONG_PASS` threshold.

## 3. Normalized Control Leverage Factor (CLF_N)
$$\mathbf{CLF}_N = \frac{\Delta W / W_B}{E_{\text{control}} / E_B} = \frac{(19052.73 - 16743.13) / 16743.13}{4.2197 / 874.668} = \frac{0.137943}{0.004824} = \mathbf{28.59}$$
- Relative useful work gain: $+13.79\%$
- Relative control energy investment: $0.4824\%$
- Leverage ratio: $\mathbf{28.59\times}$
