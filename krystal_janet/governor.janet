# ============================================================================
# Krystal-Stack Janet Engine: Visual Entropy & Economic Governor
# ============================================================================
# Calculates spatial/temporal frame entropy, coherence, and governs system
# compute budgets (GAMESA dual-earn protocol) with dynamic backpressure.

(defn create-governor [&opt init-budget max-budget]
  "Initializes the Economic Governor and Entropy state."
  @{:budget (or init-budget 850.0)
    :max-budget (or max-budget 1000.0)
    :state :OPTIMAL
    :thermal-penalty 0.0
    :entropy-threshold 0.70
    :spatial-entropy 0.0
    :temporal-entropy 0.0
    :total-entropy 0.0
    :coherence 1.0
    :backpressure false
    :prev-luma @[]})

(defn compute-entropy [gov ascii-frame cols rows]
  "Calculates spatial and temporal visual entropy across character luminance."
  (let [luma @[]
        len (length ascii-frame)]
    # Extract luminance weights from characters
    (for i 0 len
      (let [ch (ascii-frame i)]
        (cond
          (= ch (chr " ")) (array/push luma 0.0)
          (= ch (chr "░")) (array/push luma 0.25)
          (= ch (chr "▒")) (array/push luma 0.50)
          (= ch (chr "▓")) (array/push luma 0.75)
          (= ch (chr "█")) (array/push luma 1.0)
          (= ch (chr "\n")) nil
          (array/push luma 0.35))))

    (let [total-pixels (length luma)]
      (if (<= total-pixels 1)
        {:spatial 0.0 :temporal 0.0 :total 0.0 :coherence 1.0}
        (do
          # 1. Spatial variance across adjacent pixels
          (var spatial-diff 0.0)
          (for idx 0 (- total-pixels 1)
            (let [d (math/abs (- (luma idx) (luma (+ idx 1))))]
              (set spatial-diff (+ spatial-diff d))))
          (let [es (/ spatial-diff total-pixels)]

            # 2. Temporal difference against previous frame
            (var temp-diff 0.0)
            (let [prev (gov :prev-luma)
                  prev-len (length prev)]
              (when (> prev-len 0)
                (let [compare-len (min total-pixels prev-len)]
                  (for p 0 compare-len
                    (set temp-diff (+ temp-diff (math/abs (- (luma p) (prev p))))))
                  (set temp-diff (/ temp-diff compare-len)))))
            (let [et temp-diff
                  etot (+ (* 0.6 es) (* 0.4 et))
                  coh (max 0.0 (- 1.0 etot))]

              (put gov :spatial-entropy es)
              (put gov :temporal-entropy et)
              (put gov :total-entropy etot)
              (put gov :coherence coh)
              (put gov :prev-luma (array/slice luma))

              # Check backpressure trigger
              (if (> etot (gov :entropy-threshold))
                (do
                  (put gov :backpressure true)
                  (put gov :state :THROTTLED)
                  (put gov :thermal-penalty (+ (gov :thermal-penalty) 0.15)))
                (do
                  (put gov :backpressure false)
                  (put gov :state :OPTIMAL)
                  (put gov :thermal-penalty (max 0.0 (- (gov :thermal-penalty) 0.05)))))

              {:spatial es :temporal et :total etot :coherence coh})))))))

(defn update-budget [gov dt step-cost]
  "Updates economic budget based on frame computation cost."
  (let [cost (if (gov :backpressure) (* step-cost 1.8) step-cost)
        cur-b (gov :budget)
        new-b (max 50.0 (min (gov :max-budget) (- cur-b (* cost dt))))]
    (put gov :budget new-b)
    (when (< new-b 200.0)
      (put gov :state :CRITICAL_DEFICIT))
    new-b))

(defn replenish-budget [gov amount]
  "Replenishes governor budget with additional computational credits."
  (let [cur (gov :budget)
        new-val (min (gov :max-budget) (+ cur amount))]
    (put gov :budget new-val)
    (when (>= new-val 300.0)
      (put gov :state :OPTIMAL))
    new-val))
