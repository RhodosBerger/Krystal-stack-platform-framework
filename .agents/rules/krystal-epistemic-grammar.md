---
trigger: always_on
description: KRYSTAL ENLG — Epistemic Compiler (Tri-Plane Parity & 10 Cognitive Patterns)
---

# KRYSTAL ENLG — EPISTEMIC COMPILER

For every non-trivial conclusion:

CLAIM:
State exactly one atomic claim.

SCOPE:
Define system, environment, version, population/domain and exclusions.

ROLE:
Identify PROPOSER / VERIFIER / AUTHORITY / EXECUTOR / OBSERVER.

EVIDENCE:
Classify every supporting item:
DERIVED / CODE-PRESENT / SIMULATED / LOCAL-MEASURED /
REMOTE-CI / PHYSICAL-MEASURED / EXTERNAL-REPLICATED / UNKNOWN.

ANTI-PROMOTION:
Never infer:
Simulation -> Measurement
Hash integrity -> Authenticity
CI green -> Correctness
Correlation -> Mechanism
Proposal -> Authority
Coverage -> Proof

NULLS:
Generate:
M0_NO_EFFECT
M1_NOISE
M2_MEASUREMENT_OR_CONFOUNDER

FALSIFIER:
State the smallest observation capable of falsifying CLAIM.

VALIDITY:
Separate construct, internal and external validity.

VERDICT:
Use only:
UNKNOWN
CONFLICT
FALSIFIED
SUPPORTED_WITH_SCOPE
MEASURED_WITH_SCOPE
VERIFIED_WITH_SCOPE

RESIDUAL:
Preserve unresolved variables and contradictory evidence.

NEXT_EXPERIMENT:
Generate the smallest reversible experiment with:
measurement,
success criterion,
failure criterion,
rollback,
stop condition.

Never increase epistemic status without new evidence.

---

## The Tri-Plane Parity Invariants
$$\begin{array}{rcccl}
\textbf{Zero-Copy Memory Model:} & \text{Publish} & \implies & \text{FullyInitialized} & \text{(Acquire/Release memory barrier)} \\
\textbf{Governed Execution Model:} & \text{Execute} & \implies & \text{Authorized} & \text{(Capability contract gate)} \\
\textbf{Epistemic Language Model (ENLG):} & \text{AssertAsKnowledge} & \implies & \text{EvidentiallyAuthorized} & \text{(Anti-promotion barrier)}
\end{array}$$

## ArtifactGate Contract
$$\begin{array}{rcl}
\text{Release}(x) & \iff & \text{ReceiptValid}(x) \land \text{AuthorityValid}(x) \land \text{StatePreconditionValid}(x) \\
\neg\text{ReceiptValid}(x) & \implies & \text{Quarantine}(x) \land \Delta S = 0
\end{array}$$
