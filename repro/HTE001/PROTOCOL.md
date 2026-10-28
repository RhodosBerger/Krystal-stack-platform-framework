# HTE-001 REPRODUCIBILITY & INDEPENDENT AUDIT PROTOCOL
## Third-Party Verification Guide for KSL v0.2 Hardware Transition Experiment 001

---

### 1. Overview & Purpose
This package provides a self-contained, zero-external-dependency reproduction and evaluation harness for **Hardware Transition Experiment 001 (HTE-001)**. Any external auditor, peer reviewer, or certification engineer can verify experimental manifests, validate raw execution receipts, and reproduce statistical tests using standard Python 3.

---

### 2. Package Contents

| File | Description |
|---|---|
| [`HTE001_RUN_MANIFEST.json`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/repro/HTE001/HTE001_RUN_MANIFEST.json) | Frozen preregistered experiment manifest (SHA-256: `ae6a1add...`) |
| [`standalone_evaluator.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/repro/HTE001/standalone_evaluator.py) | Self-contained Python 3 evaluator (standard library only) |
| [`RECEIPT_TEMPLATE.json`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/repro/HTE001/RECEIPT_TEMPLATE.json) | Canonical schema-compliant receipt template for hardware acquisition |
| [`schemas/hte001_manifest_schema.json`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/repro/HTE001/schemas/hte001_manifest_schema.json) | JSON Schema draft 2020-12 defining manifest structure |
| [`schemas/hte001_receipt_schema.json`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/repro/HTE001/schemas/hte001_receipt_schema.json) | JSON Schema draft 2020-12 defining hardware receipt structure |

---

### 3. Canonical Preregistration Anchors

The experiment is cryptographically pinned by the following hashes:

- **Manifest RFC 8785 Digest:**
  $$\boxed{H_M = \texttt{ae6a1add5fcf1d3cb36c36859d3f38ef48a27df340b8aaf39daa3b47ea17f9cf}}$$
- **Acquisition Harness Source Hash:**
  $$\boxed{H_{\text{harness}} = \texttt{f520b171e29ea8b85b5d11d65c2ad7dd4baa9e0019ecd4eb56fee8d585f7a064}}$$
- **Acquisition Software Identity:**
  $$\boxed{H_{\text{software}} = \texttt{e5c8578cbe3382dd059a319ad915567859b18c6cb1d0fe61ed3abbbc0c7d3fd0}}$$
- **Transition Oracle Hash:**
  $$\boxed{H_{\text{oracle}} = \texttt{3c87c20d94855b78f5ebcb92067376dca9dbc5699a723e7207ef8399c4fcbf42}}$$
- **Sensor Contract Hash (INA226):**
  $$\boxed{H_{\text{sensor}} = \texttt{257637962a9ff799222323732d938036e94b14f00d1ed2d55b4af944316934e9}}$$

---

### 4. Step-by-Step Verification Instructions

#### Step 1: Verify Manifest Integrity
Verify that the local manifest file has not been altered:
```bash
python standalone_evaluator.py --manifest HTE001_RUN_MANIFEST.json
```
**Expected Output:**
```text
Manifest Digest: ae6a1add5fcf1d3cb36c36859d3f38ef48a27df340b8aaf39daa3b47ea17f9cf [MATCHES PREREGISTERED ANCHOR]
OVERALL AUDIT VERDICT: COMPLIANT
```

#### Step 2: Audit a Hardware Execution Receipt
Audit a candidate receipt file against the preregistered manifest:
```bash
python standalone_evaluator.py --manifest HTE001_RUN_MANIFEST.json --receipt RECEIPT_TEMPLATE.json
```
The evaluator verifies:
1. Manifest hash match.
2. Constitutional invariant: `vital_max_hp == 6`.
3. Hard zero memory delta: $R_{\text{memory}} == 0.0$ bytes ($\Delta S_{\text{protected}} \equiv 0$).
4. Envelope compliance: $|r_j| \le 1.0$ across all normalized fields.
5. HMAC signature integrity (if `--secret-key-hex` is provided).

#### Step 3: Audit a Batch of 30+ Acquisition Runs
When auditing a directory containing 30 or more runs:
```bash
python standalone_evaluator.py --manifest HTE001_RUN_MANIFEST.json --receipt path/to/run_receipts/
```
The evaluator performs:
1. Individual compliance on each receipt.
2. Verification of $N_{\text{acquisition\_valid}} \ge 30$.
3. Exact-zero defect count: $\sum \mathbf{1}(R_{\text{memory}} \neq 0) == 0$.
4. 1,000-iteration bootstrap 95% Confidence Interval for each normalized residual $\bar{r}_j$:
   $$\text{CI}_{95\%}(\bar{r}_j) \subset [-1.0, 1.0]$$

---

### 5. Constitutional Rules for External Reviewers

1. **Anti-Promotion Barrier:**
   Preflight checks, simulations, or software receipts NEVER promote physical claims `KS-CLAIM-019` or `KS-CLAIM-020`. They strictly remain `PENDING` until physical execution on the KCU116 board.
2. **Epistemic Decoupling Principle:**
   $$\text{ReceiptIntegrity} \neq \text{EvidenceClass} \neq \text{SemanticVerdict}$$
   A run that fails hypothesis testing is NOT discarded. It retains `ReceiptIntegrity == VERIFIED` and `EvidenceClass == FPGA_BOARD_MEASURED`, with `SemanticVerdict == FAIL`.
3. **No Retroactive Re-tuning:**
   Tolerances ($\epsilon_j$) cannot be expanded after observing a failure. A failure terminates RUN-0001 as `FAIL`. Any parameter change requires an explicitly registered RUN-0002.
