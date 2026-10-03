# Krystal-Stack Janet DSL // Execution Architecture & UI Style Contract
# Stratifikácia metrík: MEASURED vs MODELED vs TARGET & 14 UI Primitív.
# Invariant: VITAL-MAX-HP = 6

(def VITAL-MAX-HP 6)
(def GOLDEN-RATIO 1.61803398875)

(def CANONICAL-EXECUTION-PIPELINE
  [:continuous-world-math
   :sdf-fbm-kernels
   :execution-governor
   :backend-dispatch
   :framebuffer
   :telemetry
   :feedback-loop])

(def CANONICAL-METRICS
  @{:vm-queue-throughput
    @{:name "FastRingBuffer VM Queue" :val 1724554 :unit "pps" :state :measured :speedup 4.39 :vital-max-hp VITAL-MAX-HP}
    :cpu-sdf-eval-rate
    @{:name "CPU SDF Kernel" :val 997000 :unit "eval/s" :state :measured :speedup 1.0 :vital-max-hp VITAL-MAX-HP}
    :terrain-sample-latency
    @{:name "Terrain Kernel Latency" :val 89.36 :unit "us/sample" :state :measured :speedup 1.0 :vital-max-hp VITAL-MAX-HP}
    :visual-entropy-telemetry
    @{:name "Telemetry Overhead" :val 1.07 :unit "ms/frame" :state :measured :speedup 4.21 :vital-max-hp VITAL-MAX-HP}
    :simd-terrain-projection
    @{:name "AVX2 SIMD Terrain" :val 13.14 :unit "us/sample" :state :modeled :speedup 6.80 :vital-max-hp VITAL-MAX-HP}
    :iris-xe-driver-bypass
    @{:name "Iris Xe Driver Bypass" :val 11.4 :unit "us/call" :state :modeled :speedup 16.23 :vital-max-hp VITAL-MAX-HP}
    :neural-sdf-eval-latency
    @{:name "Neural SDF Manifold" :val 0.28 :unit "ns/query" :state :target :speedup 3580.0 :vital-max-hp VITAL-MAX-HP}
    :webgpu-frame-rate
    @{:name "WebGPU Compute FPS" :val 120.0 :unit "FPS" :state :target :speedup 5.0 :vital-max-hp VITAL-MAX-HP}
    :rust-native-kernel
    @{:name "Rust Native Raymarcher" :val 45.0 :unit "x" :state :target :speedup 45.0 :vital-max-hp VITAL-MAX-HP}})

(def UI-PRIMITIVES-14
  [:KSNav :KSWorldHero :KSEntitySelector :KSStatBar :KSRadar
   :KSHudStatus :KSRuntimeMetric :KSActionButton :KSWorldButton
   :KSTerminalPanel :KSRenderViewport :KSBiomeIndicator
   :KSChapterLabel :KSArtifactCard])

(defn get-metrics-by-state
  "Filters execution metrics by verification tier (:measured, :modeled, :target)."
  [state-key]
  (let [matched @[]]
    (eachp [k m] CANONICAL-METRICS
      (when (= (get m :state) state-key)
        (array/push matched m)))
    matched))

(defn validate-execution-vital-invariant
  "Ensures all execution metrics strictly preserve VITAL-MAX-HP = 6."
  []
  (var valid true)
  (eachp [k m] CANONICAL-METRICS
    (unless (= (get m :vital-max-hp 0) VITAL-MAX-HP)
      (set valid false)))
  valid)
