# Krystal Stack Platform - CNC Drawing and Machining Simulator DSL
# ===================================================================
# Janet specifications for CAD vector entities, CAM toolpaths,
# feeds/speeds calculation, and 6 Max HP machining safety invariants.

(def VITAL-MAX-HP 6)
(def GOLDEN-RATIO 1.61803398875)
(def INV-GOLDEN-RATIO 0.61803398875)

(def CNC-TOOL-LIBRARY
  {:t1-endmill-3mm
   {:name "3.0mm Flat End Mill"
    :type :endmill
    :diameter 3.0
    :flutes 2
    :max-rpm 24000
    :vital-hp VITAL-MAX-HP}
   :t2-endmill-6mm
   {:name "6.0mm Roughing Mill"
    :type :endmill
    :diameter 6.0
    :flutes 3
    :max-rpm 18000
    :vital-hp VITAL-MAX-HP}
   :t3-ballnose-3mm
   {:name "3.175mm Ball Nose"
    :type :ballnose
    :diameter 3.175
    :flutes 2
    :max-rpm 24000
    :vital-hp VITAL-MAX-HP}
   :t4-vbit-60deg
   {:name "60-deg V-Bit Engraver"
    :type :vbit
    :diameter 6.0
    :flutes 1
    :max-rpm 20000
    :vital-hp VITAL-MAX-HP}
   :t5-drill-3-2mm
   {:name "3.2mm Tap Pilot Drill"
    :type :drill
    :diameter 3.2
    :flutes 2
    :max-rpm 8000
    :vital-hp VITAL-MAX-HP}
   :t6-facemill-25mm
   {:name "25mm Fly Cutter"
    :type :facemill
    :diameter 25.0
    :flutes 4
    :max-rpm 6000
    :vital-hp VITAL-MAX-HP}})

(def MATERIAL-DATABASE
  {:al-6061
   {:name "Aluminum 6061-T6"
    :vc 150.0
    :fz 0.035
    :max-stepdown 1.5
    :coolant true}
   :brass-c360
   {:name "Brass C360"
    :vc 180.0
    :fz 0.040
    :max-stepdown 2.0
    :coolant false}
   :steel-1018
   {:name "Mild Steel 1018"
    :vc 75.0
    :fz 0.020
    :max-stepdown 0.8
    :coolant true}
   :delrin-acetal
   {:name "Delrin / POM"
    :vc 240.0
    :fz 0.060
    :max-stepdown 3.5
    :coolant false}
   :krystal-resin
   {:name "Krystal Composite"
    :vc 200.0
    :fz 0.045
    :max-stepdown 2.2
    :coolant true}})

(defn compute-spindle-rpm
  "Calculates spindle RPM from surface speed (m/min) and tool diameter (mm)."
  [vc dia]
  (let [d (if (< dia 0.5) 0.5 dia)
        raw (/ (* vc 1000.0) (* 3.14159265359 d))]
    (math/floor (math/max 1200.0 (math/min 24000.0 raw)))))

(defn calculate-feed-rate
  "Calculates linear feed rate Vf (mm/min) = n * z * fz."
  [rpm flutes fz]
  (* rpm flutes fz))

(defn evaluate-toolpath-safety
  "Verifies tool deflection and feed rate against the 6 Max HP vital bounds."
  [feed-rate max-feed]
  (let [ratio (/ feed-rate (if (<= max-feed 0) 1.0 max-feed))
        risk-penalty (if (> ratio 1.0) (* (- ratio 1.0) 6.0) 0.0)
        remaining-hp (- VITAL-MAX-HP (math/min 5.0 risk-penalty))]
    {:safe (<= ratio 1.0)
     :vital-hp-score (math/max 1.0 remaining-hp)
     :ratio ratio}))

(defn generate-gcode-header
  "Produces the standardized preamble block for ISO/RS-274D programs."
  [program-name]
  (string "% \n"
          "O" (string/slice (string (math/abs (hash program-name))) 0 4) " (" program-name ")\n"
          "G21 (Metric Units)\n"
          "G90 (Absolute Distance Mode)\n"
          "G17 (XY Plane)\n"
          "G54 (Work Offset 1)\n"
          "G00 Z" (string VITAL-MAX-HP ".000 (Safe Clearance Z)\n")))
