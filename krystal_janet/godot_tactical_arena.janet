# ==============================================================================
# KRYSTAL-STACK // GODOT TACTICAL ARENA JANET DSL
# Parity with posledni_kmen_arena.tscn and godot_canvas_arena_engine.py
# Invariant: VITAL_MAX_HP = 6
# ==============================================================================

(def vital-max-hp 6)
(def golden-ratio 1.61803398875)

(def pbr-materials
  {:cryst {:albedo [0.4 0.98 0.94 0.85]
           :metallic 0.85
           :roughness 0.15
           :emission [0.0 1.0 1.0 1.0]
           :emission-energy 2.4}
   :slime {:albedo [0.22 1.0 0.08 0.9]
           :metallic 0.1
           :roughness 0.05
           :emission [0.22 1.0 0.08 1.0]
           :emission-energy 1.6}
   :bark {:albedo [0.55 0.27 0.07 1.0]
          :metallic 0.05
          :roughness 0.85
          :emission [1.0 0.84 0.0 1.0]
          :emission-energy 0.8}
   :basalt {:albedo [0.12 0.14 0.18 1.0]
            :metallic 0.45
            :roughness 0.65
            :emission [0.4 0.8 1.0 1.0]
            :emission-energy 1.2}})

(defn calculate-hex-distance [q1 r1 q2 r2]
  (let [s1 (- 0 q1 r1)
        s2 (- 0 q2 r2)
        dq (math/abs (- q1 q2))
        dr (math/abs (- r1 r2))
        ds (math/abs (- s1 s2))]
    (math/floor (/ (+ dq dr ds) 2))))

(defn calculate-ballistic-apex [distance elevation-deg]
  (let [rad (* elevation-deg (/ math/pi 180))
        h (* distance (math/sin rad))]
    (* h 0.61803398875)))

(defn evaluate-arena-vitality [current-hp]
  (min vital-max-hp (max 0 current-hp)))
