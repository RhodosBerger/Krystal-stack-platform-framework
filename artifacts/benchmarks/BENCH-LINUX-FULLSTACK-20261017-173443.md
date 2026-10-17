# Krystal Stack Benchmark Report: BENCH-LINUX-FULLSTACK-20261017-173443
- **Timestamp:** 2026-10-17T15:34:43Z
- **Environment:** 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz (8 cores, 11809MB RAM, Linux 6.18.33.2-microsoft-standard-WSL2 x86_64)

### 1. CPU Stress Capacity
- **Metric:** 2720.16 events/sec

### 2. GPU Capacity (GLMark2)
- **Status:** EXECUTED
- **Score:** 4200

### 3. OpenVINO AEP Inference Workload
```json
{"status": "OK", "total_inferences": 100, "committed": 100, "total_duration_ms": 11.1, "throughput_tps": 9008.6, "latency_p50_us": 86.6, "latency_p95_us": 189.0, "latency_p99_us": 436.6, "backend": "OpenVINO-DirectSim-Intel-NPU"}
```

### 4. AEP Candidate State Machine Integrity
```json
{"total_transactions": 500, "committed": 333, "discarded_with_zero_mutation": 167, "duration_ms": 9.92, "transactions_per_sec": 50387.7, "final_state_hp": 0, "integrity_verified": true}
```
