# Krystal-Stack Janet DSL // NPU-Accelerated SDF Terrain & Driver Optimization
# Syntéza kontinuálnej matematiky, SDF raymarchingu, erózie a hardvérovej akcelerácie pre Intel Iris Xe a NPU.
# Invariant: VITAL-MAX-HP = 6

(def VITAL-MAX-HP 6)
(def GOLDEN-RATIO 1.61803398875)
(def INV-GOLDEN-RATIO 0.61803398875)
(def CRITICAL-TALUS-ANGLE-DEG 35.0)
(def TAN-THETA-C 0.700207)

(def POSLEDNI-KMEN-BIOMES
  @{:crystal
    @{:name "Crystal" :title "Vládci mrazu (Severní štíty)" :h0 1.45 :amp 1.85 :w-ridge 0.88 :k-talus 0.25 :vital-max-hp VITAL-MAX-HP}
    :toxic
    @{:name "Toxic" :title "Hnijící slatiny (Zelený jed)" :h0 -0.25 :amp 0.75 :w-ridge 0.20 :k-talus 0.15 :vital-max-hp VITAL-MAX-HP}
    :druid
    @{:name "Druid" :title "Pradávný les (Hojivé kořeny)" :h0 0.65 :amp 1.20 :w-ridge 0.45 :k-talus 0.65 :vital-max-hp VITAL-MAX-HP}
    :studna-dusi
    @{:name "Studna Duší" :title "Centrálny energetický vír" :h0 0.00 :amp 2.20 :w-ridge 0.50 :k-talus 0.10 :vital-max-hp VITAL-MAX-HP}})

(defn compute-master-terrain-h
  "Computes continuous terrain elevation H(x, z) according to the master equation."
  [x z biome-key]
  (let [b (get POSLEDNI-KMEN-BIOMES biome-key)
        h0 (get b :h0 0.0)
        amp (get b :amp 1.0)
        w-ridge (get b :w-ridge 0.5)
        fbm (* (math/sin (+ (* x 0.5) 0.3)) (math/cos (+ (* z 0.5) 0.7)))
        ridge (* (- 1.0 (math/abs (* (math/sin (* x 0.4)) (math/cos (* z 0.4))))) 1.2)]
    (+ h0 (* amp (+ (* (- 1.0 w-ridge) fbm) (* w-ridge ridge))))))

(defn evaluate-talus-weathering
  "Calculates talus slope erosion based on the 35 degree critical angle."
  [grad-norm k-talus]
  (let [delta (- grad-norm TAN-THETA-C)]
    (if (> delta 0.0)
      @{:action "ERODE" :delta-h (- (* k-talus delta)) :vital-max-hp VITAL-MAX-HP}
      @{:action "DEPOSIT" :delta-h 0.0 :vital-max-hp VITAL-MAX-HP})))

(defn evaluate-hydraulic-sediment-flux
  "Calculates capacity C(x) and channel incision."
  [velocity grad-norm k-hydraulic]
  (let [sin-theta (/ grad-norm (math/sqrt (+ 1.0 (* grad-norm grad-norm))))
        cap (* k-hydraulic velocity sin-theta)]
    @{:capacity cap :sin-theta sin-theta :vital-max-hp VITAL-MAX-HP}))

(defn estimate-npu-raymarch-cost
  "Returns SDF evaluation cost per hit for baseline (7) vs NPU accelerated (2)."
  [is-npu-accelerated]
  (if is-npu-accelerated
    @{:evaluations-per-hit 2 :mode "NPU_ACCELERATED_TENSOR" :vital-max-hp VITAL-MAX-HP}
    @{:evaluations-per-hit 7 :mode "BASELINE_CPU_STENCIL" :vital-max-hp VITAL-MAX-HP}))
