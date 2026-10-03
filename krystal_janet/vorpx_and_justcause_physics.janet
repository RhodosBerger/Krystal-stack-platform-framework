# ==============================================================================
# KRYSTAL-STACK: VORPX VR INJECTION & JUST CAUSE KINETIC PHYSICS DSL (JANET)
# ==============================================================================
# Defines:
#   1. VorpX VR profiles: HTC Vive Pro, Meta Quest, Ncon by Korrado (130° FOV).
#   2. Stereoscopic 3D projection & IPD eye separation matrix.
#   3. Just Cause dual grappling hook tension & slingshot momentum formulas.
#   4. Wingsuit aerodynamic lift/drag glide ratio equations.
#   5. Borderlands cel-shaded ink contour & quantized shading specs.
# ==============================================================================

(def VORPX-HEADSET-PROFILES
  {:htc-vive-pro
   {:id "htc_vive_pro"
    :name "HTC Vive Pro"
    :resolution [1440 1600]
    :fov-degrees 110.0
    :refresh-rate-hz 90
    :ipd-default-mm 63.5
    :distortion-k1 0.22
    :distortion-k2 0.24}

   :meta-quest
   {:id "meta_quest"
    :name "Meta Quest 3 / Low-Cost"
    :resolution [2064 2208]
    :fov-degrees 110.0
    :refresh-rate-hz 120
    :ipd-default-mm 64.0
    :distortion-k1 0.18
    :distortion-k2 0.15}

   :ncon-by-korrado
   {:id "ncon_by_korrado"
    :name "Ncon by Korrado Ultra-VR"
    :resolution [3840 2160]
    :fov-degrees 130.0
    :refresh-rate-hz 144
    :ipd-default-mm 62.0
    :distortion-k1 0.08
    :distortion-k2 0.05
    :haptic-trigger-latency-ms 1.2
    :direct-drive-imu true}})

(defn calculate-stereo-eye-offset
  "Calculates horizontal eye shift given IPD in millimeters."
  [ipd-mm]
  (/ (* ipd-mm 0.001) 2.0))

(defn compute-tether-tension
  "Computes spring-damper tension force for Just Cause grappling cable."
  [rest-length current-length relative-velocity spring-k damper-c]
  (let [delta-l (max 0.0 (- current-length rest-length))]
    (+ (* spring-k delta-l) (* damper-c relative-velocity))))

(defn calculate-wingsuit-glide
  "Computes forward aerodynamic glide distance from drop height and airspeed."
  [drop-height-m airspeed-mps]
  (let [glide-ratio 3.5
        horizontal-distance (* drop-height-m glide-ratio)
        lift-coeff 1.25
        drag-coeff 0.35]
    {:horizontal-distance-m horizontal-distance
     :lift-to-drag (/ lift-coeff drag-coeff)
     :terminal-velocity-mps (* airspeed-mps 1.15)}))

(defn evaluate-slingshot-momentum
  "Calculates kinetic speed burst when releasing grapple tension."
  [mass-kg tension-n angle-degrees]
  (let [force-forward (* tension-n (math/cos (/ (* angle-degrees 3.14159) 180.0)))
        accel (/ force-forward mass-kg)]
    {:acceleration-mps2 accel
     :boost-impulse-ns (* tension-n 0.45)}))

(def BORDERLANDS-CEL-SHADING-SPEC
  {:ink-outline-thickness-px 2.5
   :sobel-depth-threshold 0.22
   :quantized-light-bands 4
   :background-yellow-hex "#facc15"
   :halftone-dot-frequency 32.0})
