# ==============================================================================
# KRYSTAL-STACK PLATFORM: TOTEM ANOMALIES, SPELL PROJECTIONS & VISUAL PHENOMENA
# ==============================================================================
# Procedural definitions for:
#   1. Spatial spell cone & ray casting intersections on hex sectors.
#   2. Sensor anomaly field gradient divergence and frequency flux.
#   3. Totem resonance modulation, overcharge thresholds, and polarity shifts.
#   4. Visual shader parameters (chromatic aberration, St. Elmo's plasma).
# ==============================================================================

(def TOTEM-BASE-FREQUENCIES
  {:crystal 432.0   # Aetheric harmonic
   :toxic 216.5     # Miasma low resonance
   :druid 528.0     # Solfeggio life frequency
   :sulfur 360.0    # Combustion thermal pitch
   :nexus 864.0})   # Cosmic citadel crown

(defn calculate-cone-hit
  "Determines if target coordinates [tx ty] fall within a spell cone from [ox oy] with heading and half-angle"
  [ox oy tx ty heading-rad half-angle-rad max-dist]
  (let [dx (- tx ox)
        dy (- ty oy)
        dist (math/sqrt (+ (* dx dx) (* dy dy)))]
    (if (or (> dist max-dist) (<= dist 0.0001))
      false
      (let [target-angle (math/atan2 dy dx)
            diff (math/abs (- target-angle heading-rad))
            norm-diff (if (> diff math/pi) (- (* 2.0 math/pi) diff) diff)]
        (<= norm-diff half-angle-rad)))))

(defn compute-anomaly-gradient
  "Computes spatial scalar field gradient magnitude from 4-point cardinal differential"
  [val-c val-n val-s val-e val-w delta]
  (let [gx (/ (- val-e val-w) (* 2.0 delta))
        gy (/ (- val-n val-s) (* 2.0 delta))]
    (math/sqrt (+ (* gx gx) (* gy gy)))))

(defn evaluate-totem-resonance-shift
  "Calculates updated totem resonance and status based on anomaly magnitude and spell energy"
  [base-res anomaly-mag spell-potency is-aligned]
  (let [charge-delta (if is-aligned
                       (* spell-potency 12.5)
                       (- (* anomaly-mag 18.0) (* spell-potency 5.0)))
        new-res (math/max 0.0 (math/min 100.0 (+ base-res charge-delta)))
        status (cond
                 (>= new-res 90.0) :overcharged
                 (<= new-res 15.0) :nullified
                 (and (not is-aligned) (> anomaly-mag 0.6)) :corrupted
                 :attuned)]
    {:resonance new-res
     :status status
     :aura-boost (if (= status :overcharged) 1.5 1.0)}))

(defn generate-visual-shader-uniforms
  "Derives volumetric visual phenomena uniforms from totem resonance and anomaly field"
  [resonance anomaly-type]
  {:chroma-shift (math/round (* (/ resonance 100.0) 0.08) 4)
   :plasma-intensity (math/round (+ 0.2 (* (/ resonance 100.0) 0.8)) 3)
   :aurora-phase (math/round (* resonance 0.0628) 3)
   :distortion-scale (if (= anomaly-type :void-rift) 0.45 0.12)})
