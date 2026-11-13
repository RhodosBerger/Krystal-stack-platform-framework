# Sovereign OpenVINO Companion v2 (Fine-Tuned & Adaptive Resource Spending)

Smarter, fine-tuned iteration over `projects/sovereign_openvino_companion`:
- `BufferedComputationLogSystem`: SHA-256 hash-chained ledger with write-coalesced flushing (`flush_every=16`).
- `AdaptiveSpendingOptimizer`: EWMA latency-calibrated `adaptive_crossover_bytes`, predictive UMA velocity pre-braking (`k_spec=4` soft throttle), and early P5 parasite cutoff (`0.0180 mJ` vs `0.284 mJ`).

```powershell
python -m projects.sovereign_openvino_companion_v2.cli --run-session --frames 16 --ab-benchmark 40 --reset-ledger
```
