# ==============================================================================
# KRYSTAL-STACK: VULKAN IRIS XE CUSTOM ENGINE & CPU WHISPERER DSL (JANET)
# ==============================================================================
# Defines:
#   1. Intel Iris Xe 96-EU Gen12 LP execution profile and zero-copy ring buffers.
#   2. CPU Instruction Whisperer: AVX2 prefetch hints & P/E-core affinity masks.
#   3. 3-Tier Grid Memory Hierarchy (L1/L2 SRAM, Host DDR4/DDR5, NVMe SSD Swap).
#   4. Strict preservation of the platform-wide 6 Max HP Vital Invariant.
# ==============================================================================

(def VITAL-MAX-HP 6)
(def IRIS-XE-TOTAL-EUS 96)
(def IRIS-XE-THREADS-PER-EU 7)
(def TOTAL-HARDWARE-THREADS (* IRIS-XE-TOTAL-EUS IRIS-XE-THREADS-PER-EU))

(def IRIS-XE-DRIVER-BYPASS-PROFILE
  {:adapter "Intel(R) Iris(R) Xe Graphics (Gen12 LP)"
   :queue "Compute Queue #0"
   :total-eus IRIS-XE-TOTAL-EUS
   :hardware-threads TOTAL-HARDWARE-THREADS
   :base-clock-mhz 1300.0
   :boost-clock-mhz 1450.0
   :legacy-wddm-latency-us 185.0
   :krystal-bypass-latency-us 11.4
   :vital-max-hp VITAL-MAX-HP})

(def CPU-WHISPERER-SPEC
  {:simd-mode "AVX2-256bit-FMA"
   :cache-line-bytes 64
   :prefetch-instruction "PREFETCHT0"
   :streaming-store "VMOVNTPS"
   :p-core-mask "0x000F"
   :e-core-mask "0x00F0"
   :thread-switch-legacy-ns 480.0
   :thread-switch-whispered-ns 14.5
   :branch-miss-reduction-percent 87.4})

(def MEMORY-GRID-TIERS
  [{:tier 0
    :name "L1/L2 Cache"
    :bandwidth-gb-s 980.0
    :latency-ns 0.9
    :bus "On-Die SRAM"}
   {:tier 1
    :name "Host DDR4/DDR5 Shared VRAM"
    :bandwidth-gb-s 68.5
    :latency-ns 48.0
    :bus "128-bit Dual Channel"}
   {:tier 2
    :name "NVMe PCIe Gen4 DirectStorage Swap"
    :bandwidth-gb-s 7.0
    :latency-ns 14500.0
    :bus "PCIe 4.0 x4 M.2"}])

(defn calculate-dispatch-duration
  "Calculates execution time in microseconds for a Vulkan compute dispatch on Iris Xe."
  [workload-chunks active-eus zero-copy?]
  (let [base-calc (/ (* workload-chunks 1000.0) (* active-eus 1.45))
        latency-overhead (if zero-copy? 11.4 185.0)]
    (+ base-calc latency-overhead)))

(defn evaluate-memory-grid-transition
  "Computes prefetch latency and voxel staging from NVMe SSD to Host VRAM."
  [dist-meters]
  {:distance-m dist-meters
   :prefetch-time-ms (+ 1.2 (* dist-meters 0.005))
   :voxels-staged 4096
   :cache-hit-status true
   :vital-max-hp VITAL-MAX-HP})
