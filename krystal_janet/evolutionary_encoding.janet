# ==============================================================================
# KRYSTAL-STACK: JANET EVOLUTIONARY ENCODING & CONGESTION DSL
# ==============================================================================
# Implements:
#   1. Variable frame-rate and sampling frequency schedule generator.
#   2. Frame stream encoding protocol header and metadata packer.
#   3. Evolutionary physics and motion trajectory fitness evaluator.
#   4. Event stream congestion window and flow control (AIMD).
#   5. Multi-dimensional hero matrix dimensional resolution (5x2, 4x5, 30x20, 90x120).
# ==============================================================================

(def *ml-evolution-version* "KRYSTAL-EVOLUTIONARY-ENCODING-1.0")

# ── Supported Matrix Dimensional Presets ──────────────────────────────────────
(def HERO-MATRIX-DIMS
  {:action-phase-5x2    {:cols 5  :rows 2   :cells 10    :hz-base 30}
   :stance-zone-4x5     {:cols 4  :rows 5   :cells 20    :hz-base 60}
   :spatial-field-30x20 {:cols 30 :rows 20  :cells 600   :hz-base 120}
   :highres-tensor-90x120 {:cols 90 :rows 120 :cells 10800 :hz-base 240}})

# ── Sampling Frequency Schedule ──────────────────────────────────────────────
(defn create-sampling-frequency-schedule
  "Computes variable sampling frequency burst schedule based on combat intensity."
  [intensity-factor base-hz peak-hz]
  (let [norm-intensity (math/max 0.0 (math/min 1.0 intensity-factor))
        effective-hz (+ base-hz (* (- peak-hz base-hz) norm-intensity))
        bullet-time-active (>= norm-intensity 0.85)]
    @{:intensity norm-intensity
      :base-hz base-hz
      :peak-hz peak-hz
      :effective-sampling-hz (math/round effective-hz)
      :bullet-time-burst bullet-time-active
      :recommended-matrix-dim (if bullet-time-active :highres-tensor-90x120 :spatial-field-30x20)}))

# ── Frame Stream Encoding Header ─────────────────────────────────────────────
(defn encode-frame-stream-header
  "Packs binary stream encoding envelope metadata for replay recording."
  [match-id frame-count avg-hz unique-snapshot-hash]
  @{:magic-signature "0x4B525953"
    :match-id match-id
    :total-frames frame-count
    :nominal-sampling-hz avg-hz
    :duration-sec (/ frame-count (math/max 1.0 avg-hz))
    :level-snapshot-hash unique-snapshot-hash
    :compression-mode :aether-quantized-tensor})

# ── Evolutionary Physics Trajectory Fitness ──────────────────────────────────
(defn evaluate-trajectory-fitness
  "Evaluates chromosome fitness for evolutionary movement and kinematic physics."
  [distance-covered energy-consumed jerk-penalty evasion-successes]
  (let [score (+ (* distance-covered 2.0)
                 (* evasion-successes 25.0)
                 (- (* energy-consumed 0.5))
                 (- (* jerk-penalty 1.2)))
        clamped-fitness (math/max 0.0 score)]
    @{:fitness-score clamped-fitness
      :distance distance-covered
      :efficiency (/ distance-covered (math/max 1.0 energy-consumed))
      :viable (> clamped-fitness 15.0)}))

# ── Event Stream Congestion Window (AIMD) ────────────────────────────────────
(defn step-congestion-window
  "Adjusts event dispatch rate using Additive Increase / Multiplicative Decrease."
  [current-window-size queue-depth max-queue-threshold]
  (if (> queue-depth max-queue-threshold)
    # Congestion detected: Multiplicative Decrease
    (let [new-size (math/max 5 (math/floor (* current-window-size 0.5)))]
      @{:window-size new-size :action :multiplicative-decrease :congestion true})
    # Normal flow: Additive Increase
    (let [new-size (+ current-window-size 2)]
      @{:window-size new-size :action :additive-increase :congestion false})))
