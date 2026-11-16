# Consistency bot report (full)

Generated: 2026-11-16T09:58:14+0100

| Check | Claim | Verdict | Evidence | Time (s) |
| --- | --- | --- | --- | --- |
| C01 | Sync connection graph obeys authority rules | **VERIFIED_WITH_SCOPE** | STATIC_VALIDATION | 0.0 |
| C02 | Adaptive rollback has zero desync on 3 latency profiles | **SUPPORTED_WITH_SCOPE** | SIMULATED | 0.05 |
| C03 | OpenVINO conv pipeline matches NumPy reference on every device | **MEASURED_WITH_SCOPE** | LOCAL_MEASURED | 0.6 |
| C04 | Transposed conv is the exact adjoint; linear layer is invertible | **VERIFIED_WITH_SCOPE** | LOCAL_MEASURED_NUMERICAL | 0.39 |
| C05 | Summation-order fingerprint pattern holds over profiles | **CONFLICT** | LOCAL_MEASURED | 5.71 |
| C06 | Locate K where NumPy BLAS leaves sequential-FMA order | **MEASURED_WITH_SCOPE** | LOCAL_MEASURED | 0.55 |
| C07 | Card grid respects the Krystal UI design manual bans | **VERIFIED_WITH_SCOPE** | STATIC_VALIDATION | 0.01 |
| C08 | Deployment configs agree on the server port | **CONFLICT** | STATIC_VALIDATION | 0.0 |

Falsifiers:

- C01: any edge bypasses STEP_GATE_PROXY
- C02: any checksum desync
- C03: rel error >= 1e-3 on any device
- C04: adjoint gap >= 1e-5 or inverse error >= 1e-3
- C05: any hypothesis fails on a profile
- C06: no threshold in bracket
- C07: shadow / blur / radius > 4px present
- C08: nginx upstream port != app port
