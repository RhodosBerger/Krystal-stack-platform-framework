# ==============================================================================
# KRYSTAL-STACK: AERIAL BALLOONS, ROGALLO WINGS & PLUNGING MORTAR DSL (JANET)
# ==============================================================================
# Defines:
#   1. Aerostat & balloon buoyancy, payload lift and altitude equilibrium.
#   2. Rogallo hang glider aerodynamics, glide ratios (7.5:1), and thermal lift.
#   3. Steerable parachute canopy drag, terminal descent velocity (5.2 m/s).
#   4. High-altitude plunging mortar ballistics with elevation gravity bonus.
#   5. Skybridge conduits connecting floating balloon-borne island platforms.
#   6. Strict 6 Max HP vital invariant compliance on explosive impact.
# ==============================================================================

(def VITAL-MAX-HP 6)

(def AEROSTAT-PROFILES
  {:thermal-hot-air
   {:id "thermal_hot_air"
    :name "Thermal Sun-Furnace Balloon"
    :gas-density 0.95
    :envelope-volume-m3 3200.0
    :max-payload-kg 1200.0
    :cruising-altitude-m 220.0
    :ascent-rate-mps 3.5}

   :aether-gas-balloon
   {:id "aether_gas_balloon"
    :name "Crystalline Aether Buoyancy Cell"
    :gas-density 0.18
    :envelope-volume-m3 6500.0
    :max-payload-kg 4800.0
    :cruising-altitude-m 380.0
    :ascent-rate-mps 6.0}

   :armored-siege-island
   {:id "armored_siege_island"
    :name "Iron-Clad Floating Island Platform"
    :gas-density 0.08
    :envelope-volume-m3 18500.0
    :max-payload-kg 15000.0
    :cruising-altitude-m 280.0
    :ascent-rate-mps 2.2
    :mortar-mount-capable true}})

(defn compute-aerostat-buoyancy
  "Computes net buoyant upward force (Newtons) given gas density and envelope volume."
  [envelope-volume-m3 gas-density-kgm3 payload-mass-kg]
  (let [air-density 1.225
        gravity 9.80665
        gross-lift (* (- air-density gas-density-kgm3) envelope-volume-m3 gravity)
        payload-weight (* payload-mass-kg gravity)]
    (- gross-lift payload-weight)))

(defn calculate-parachute-terminal-velocity
  "Computes steerable parachute steady-state descent speed in m/s."
  [total-mass-kg canopy-area-m2 drag-coefficient]
  (let [air-density 1.225
        gravity 9.80665
        denom (* 0.5 air-density canopy-area-m2 drag-coefficient)]
    (if (<= denom 0.0)
      45.0
      (math/sqrt (/ (* total-mass-kg gravity) denom)))))

(defn calculate-rogallo-glide-distance
  "Calculates forward flight distance of a Rogallo hang glider with thermal boost."
  [drop-height-m glide-ratio thermal-updraft-mps flight-time-sec]
  (let [effective-height (+ drop-height-m (* thermal-updraft-mps flight-time-sec))]
    (* (max 0.0 effective-height) glide-ratio)))

(defn compute-plunging-mortar-trajectory
  "Calculates plunging ballistic range and impact velocity from high-altitude floating island."
  [elevation-m muzzle-velocity-mps pitch-angle-rad wind-speed-mps]
  (let [g 9.80665
        v0 muzzle-velocity-mps
        sin-theta (math/sin pitch-angle-rad)
        cos-theta (math/cos pitch-angle-rad)
        vy0 (* v0 sin-theta)
        vx0 (* v0 cos-theta)
        discriminant (+ (* vy0 vy0) (* 2.0 g elevation-m))
        t-flight (/ (+ vy0 (math/sqrt (max 0.0 discriminant))) g)
        impact-vx (+ vx0 (* wind-speed-mps 0.5))
        impact-vy (- vy0 (* g t-flight))
        range-m (* impact-vx t-flight)
        impact-speed-mps (math/sqrt (+ (* impact-vx impact-vx) (* impact-vy impact-vy)))]
    {:flight-time-sec t-flight
     :effective-range-m range-m
     :impact-speed-mps impact-speed-mps
     :elevation-advantage-m elevation-m}))

(defn calculate-mortar-island-damage
  "Computes plunging mortar explosion damage strictly bounded by 6 Max HP vital invariant."
  [base-mortar-caliber-mm elevation-m distance-to-epicenter-m target-current-hp]
  (let [caliber-factor (/ base-mortar-caliber-mm 100.0)
        elevation-boost (min 1.5 (+ 1.0 (/ elevation-m 1000.0)))
        falloff (/ 1.0 (+ 1.0 (* distance-to-epicenter-m 0.25)))
        raw-damage (* 3.5 caliber-factor elevation-boost falloff)
        clamped-damage (min (- VITAL-MAX-HP 1) (max 1 (math/round raw-damage)))
        new-hp (max 0 (- target-current-hp clamped-damage))]
    {:damage-inflicted clamped-damage
     :target-remaining-hp new-hp
     :is-lethal (<= new-hp 0)
     :vital-invariant-verified (<= target-current-hp VITAL-MAX-HP)}))
