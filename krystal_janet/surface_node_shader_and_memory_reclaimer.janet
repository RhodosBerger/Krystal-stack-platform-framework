# Krystal-Stack Janet DSL // Surface Node Shader & Reverse Memory Slot Scavenger
# Simulates multi-layer surface adjustments, shadow occlusion, and combinatorial
# scenario growth via reverse memory hole reclamation.
# Invariant: VITAL-MAX-HP = 6

(def VITAL-MAX-HP 6)
(def GOLDEN-RATIO 1.61803398875)
(def INV-GOLDEN-RATIO 0.61803398875)
(def DEFAULT-SLOTS 64)

(def SURFACE-FUNCTIONALITIES
  @{:0 {:name "Micro-Faceted Specular Refraction" :uniform "u_specular_refraction"}
    :1 {:name "Parallax Self-Shadowing Occlusion" :uniform "u_self_shadow_density"}
    :2 {:name "Subsurface Chromatic Dispersion" :uniform "u_subsurface_scatter"}
    :3 {:name "Toxic Slime Caustic Flux" :uniform "u_caustic_flux"}
    :4 {:name "Druidic Crevice Weathering" :uniform "u_crevice_occlusion"}
    :5 {:name "Golden Mean Fresnel Rim" :uniform "u_fresnel_rim"}
    :6 {:name "Anisotropic Metallic Patina" :uniform "u_anisotropy_flow"}
    :7 {:name "Volumetric Absorption Gradient" :uniform "u_volumetric_absorb"}})

(defn create-memory-slab
  "Creates a memory slab array with num-slots capacity."
  [num-slots]
  (def slots @[])
  (for i 0 num-slots
    (array/push slots @{:id i
                        :occupied false
                        :reclaimed false
                        :entropy (math/abs (math/sin (* (+ i 1) 0.61803398875)))}))
  @{:total-slots num-slots :slots slots})

(defn allocate-slab-slot
  "Marks a slot in the slab as occupied by owner."
  [slab slot-id owner]
  (let [slots (get slab :slots)]
    (if (and (>= slot-id 0) (< slot-id (length slots)))
      (let [s (get slots slot-id)]
        (put s :occupied true)
        (put s :owner owner)
        (put s :reclaimed false)
        true)
      false)))

(defn free-slab-slot
  "Frees an occupied slot to create a memory hole for scavenging."
  [slab slot-id]
  (let [slots (get slab :slots)]
    (if (and (>= slot-id 0) (< slot-id (length slots)))
      (let [s (get slots slot-id)]
        (put s :occupied false)
        (put s :reclaimed false)
        true)
      false)))

(defn reverse-scavenge-empty-slots
  "Scans the memory slab in reverse order and reclaims vacant slots as procedural permutation seeds."
  [slab]
  (def reclaimed @[])
  (def slots (get slab :slots))
  (def len (length slots))
  (for idx 0 len
    (let [reverse-i (- (- len 1) idx)
          s (get slots reverse-i)]
      (when (not (get s :occupied))
        (put s :reclaimed true)
        (let [feature-mod (mod reverse-i 8)
              seed (math/abs (math/sin (+ (* reverse-i 13.37) 1.618)))]
          (array/push reclaimed @{:slot-id reverse-i
                                 :permutation-seed seed
                                 :feature-channel feature-mod})))))
  reclaimed)

(defn calculate-combinatorial-scenarios
  "Calculates combinatorial scenario expansion factor driven by reclaimed slots count k."
  [k-reclaimed]
  (let [k-clamped (min 8 k-reclaimed)
        # Factorial approximation for P(8, k)
        p-val (cond
                (= k-clamped 0) 1
                (= k-clamped 1) 8
                (= k-clamped 2) 56
                (= k-clamped 3) 336
                (= k-clamped 4) 1680
                (= k-clamped 5) 6720
                (= k-clamped 6) 20160
                (= k-clamped 7) 40320
                40320)
        exp-mult (math/pow 2 (/ (min 16 k-reclaimed) 4.0))]
    (math/floor (* p-val exp-mult))))

(defn evaluate-surface-node-graph
  "Computes final surface shader parameters from node adjustments and memory reclamation."
  [base-roughness base-metallic self-shadow-depth reclaimed-count]
  (let [scenarios (calculate-combinatorial-scenarios reclaimed-count)
        fresnel-rim (* INV-GOLDEN-RATIO 1.25)
        effective-shadow (min 1.0 (* self-shadow-depth 1.15))]
    @{:vital-max-hp VITAL-MAX-HP
      :scenarios scenarios
      :roughness base-roughness
      :metallic base-metallic
      :self-shadow-occlusion effective-shadow
      :fresnel-rim fresnel-rim
      :status :compiled}))
