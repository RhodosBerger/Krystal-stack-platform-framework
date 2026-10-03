# Krystal-Stack Janet DSL // Evolved High-Detail SVG Vector Engine
# Vektorová syntéza blueprintov, biómov Poslední Kmen a raymarching anotácií.
# Invariant: VITAL-MAX-HP = 6

(def VITAL-MAX-HP 6)
(def GOLDEN-RATIO 1.61803398875)
(def INV-GOLDEN-RATIO 0.61803398875)

(def CANONICAL-SVG-PRESETS
  @{:master-blueprint
    @{:name "Master Blueprint Poslední Kmen" :width 1600 :height 900 :format "SVG" :vital-max-hp VITAL-MAX-HP}
    :crystal-crest
    @{:name "Crystal Tribe Crest" :size 512 :accent "#00f0ff" :vital-max-hp VITAL-MAX-HP}
    :toxic-crest
    @{:name "Toxic Tribe Crest" :size 512 :accent "#00ff88" :vital-max-hp VITAL-MAX-HP}
    :druid-crest
    @{:name "Druid Tribe Crest" :size 512 :accent "#f59e0b" :vital-max-hp VITAL-MAX-HP}
    :studna-dusi-crest
    @{:name "Studna Duší Vortex Crest" :size 512 :accent "#bf5af2" :vital-max-hp VITAL-MAX-HP}})

(defn get-svg-preset-info
  "Retrieves metadata and dimensions for an SVG preset."
  [preset-key]
  (let [p (get CANONICAL-SVG-PRESETS preset-key)]
    (if p p @{:error "Preset not found" :vital-max-hp VITAL-MAX-HP})))

(defn compute-vector-raymarch-step
  "Calculates geometric sphere-tracing circle radius for SVG vector diagram."
  [step-index total-steps max-radius]
  (let [t (/ step-index total-steps)
        r (+ (* (- 1.0 t) (- max-radius 4.0)) 4.0)]
    @{:step step-index :radius r :vital-max-hp VITAL-MAX-HP}))

(defn validate-svg-vital-invariant
  "Ensures strict compliance with the platform 6 Max HP rule."
  [entity]
  (= (get entity :vital-max-hp 0) VITAL-MAX-HP))
