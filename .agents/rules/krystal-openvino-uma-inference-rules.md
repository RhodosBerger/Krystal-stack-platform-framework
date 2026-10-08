# KRYSTAL-STACK: OPENVINO UMA & INFERENCE COEXISTENCE RULES
**Mandatory Architectural Rules for Tiger Lake (11th Gen) & Modern Intel Architectures**

### Rule 1: Dynamic UMA Aperture Partitioning (No Page Paging)
- System memory under UMA must be partitioned as:
  $$\text{UMA}_{\text{Total}} = \text{UMA}_{\text{Inference (70\%)}} + \text{UMA}_{\text{Graphics (30\%)}}$$
- Buffer pools must be pinned host-coherent (`is_zero_copy = true`, `is_host_coherent = true`). Under no circumstances may active OpenVINO weight matrices or rendering framebuffers be paged out to Windows pagefile on SSD.

### Rule 2: Interleaved Inference-Render Slicing (VSync Pause Harvesting)
- Real-time rendering holds absolute presentation deadline (8.33 ms for 120 Hz, 16.66 ms for 60 Hz).
- The remaining delta time in the frame deadline must be harvested by the OpenVINO tokenizer/inference engine for token micro-slicing.
- Model generation must be executed in asynchronous single-token increments so that frame drops never occur.

### Rule 3: Enforced DP4A INT8 & U8 KV-Cache Precision
- On Intel Iris Xe (80/96 EU) and newer Intel Arc Xe-cores, dense GEMM and Attention projections must execute in `INT8` using native `DP4A` instructions (`ov::hint::inference_precision(ov::element::i8)`).
- Attention KV-cache must be quantized to `u8` to preserve at least 70% of memory bandwidth for concurrent graphics texture sampling.

### Rule 4: Arrhenius Voltage Clamping & 32W Package Coexistence
- The combined power consumption of CPU cores, Iris Xe GPU, and uncore memory controller must remain within the unblocked package envelope:
  $$P_{\text{package}} \le 32.0\,\text{W}$$
- Operating voltage must be clamped to $V_{\text{core}} \le 1.020\,\text{V}$ and $T_{\text{junction}} \le 85.0^\circ\text{C}$ with an Arrhenius aging acceleration factor $\text{AF} \le 1.05$.

### Rule 5: Non-Negotiable System Invariant
- **`VITAL_MAX_HP = 6`** must be verified on every instruction dispatch, telemetry report, and benchmark assertion.
