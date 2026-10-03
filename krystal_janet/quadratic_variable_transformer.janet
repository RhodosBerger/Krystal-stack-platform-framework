# Krystal-Stack Janet DSL // Quadratic Variable Transformer & Latent x Bridge
# Converts heterogeneous variables (RAM, GPU clock, Hamiltonian H, Market Price, Vital HP)
# through latent parameter x = (-b +- sqrt(Delta)) / 2a.
# Invariant: VITAL-MAX-HP = 6

(def VITAL-MAX-HP 6)
(def GOLDEN-RATIO 1.61803398875)
(def INV-GOLDEN-RATIO 0.61803398875)

(def CANONICAL-QUADRATIC-DOMAINS
  {:memory-latency-ns
   {:name "Memory Latency L1/L2 and DDR"
    :unit "ns"
    :a 85.0
    :b 15.0
    :c 5.2
    :v-min 5.0
    :v-max 120.0}
   :vram-allocation-mb
   {:name "Shared VRAM Allocation (Host DDR)"
    :unit "MB"
    :a 3200.0
    :b 4800.0
    :c 256.0
    :v-min 256.0
    :v-max 8256.0}
   :gpu-clock-mhz
   {:name "Intel Iris Xe Clock Frequency"
    :unit "MHz"
    :a -400.0
    :b 1050.0
    :c 800.0
    :v-min 800.0
    :v-max 1450.0}
   :hamiltonian-h
   {:name "System Hamiltonian Energy H"
    :unit "energy"
    :a 0.85
    :b 0.20
    :c 0.05
    :v-min 0.05
    :v-max 1.20}
   :vital-hp
   {:name "Vital Health Points"
    :unit "HP"
    :a -3.5
    :b -1.5
    :c 6.0
    :v-min 0.0
    :v-max 6.0}})

(defn evaluate-quadratic-v
  "Calculates V(x) = a*x^2 + b*x + c for given coefficients and x."
  [a b c x]
  (+ (* a (* x x)) (* b x) c))

(defn solve-quadratic-x
  "Solves a*x^2 + b*x + (c - v) = 0 using the quadratic discriminant."
  [a b c v]
  (let [c-eff (- c v)
        delta (- (* b b) (* 4.0 (* a c-eff)))]
    (if (< delta 0.0)
      {:x (if (= a 0.0) 0.0 (/ (- b) (* 2.0 a)))
       :delta delta
       :real false}
      (let [sqrt-d (math/sqrt delta)
            x1 (/ (+ (- b) sqrt-d) (* 2.0 a))
            x2 (/ (- (- b) sqrt-d) (* 2.0 a))]
        {:x (if (and (>= x1 0.0) (<= x1 1.0)) x1 x2)
         :delta delta
         :real true}))))

(defn convert-variable-via-x
  "Converts variable value from source domain to target domain via latent parameter x."
  [source-key target-key source-val]
  (let [s-dom (get CANONICAL-QUADRATIC-DOMAINS source-key)
        t-dom (get CANONICAL-QUADRATIC-DOMAINS target-key)]
    (if (and s-dom t-dom)
      (let [sol (solve-quadratic-x (get s-dom :a) (get s-dom :b) (get s-dom :c) source-val)
            latent-x (get sol :x)
            clamped-x (min 1.0 (max 0.0 latent-x))
            target-val (evaluate-quadratic-v (get t-dom :a) (get t-dom :b) (get t-dom :c) clamped-x)
            final-target (if (= target-key :vital-hp) (min 6.0 (max 0.0 target-val)) target-val)]
        {:source-val source-val
         :latent-x clamped-x
         :target-val final-target
         :vital-hp-rule VITAL-MAX-HP})
      nil)))
