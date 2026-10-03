# Krystal-Stack Janet DSL // High-Fidelity 3D Display Patterns & WSL DNF Model Importer
# Mappuje zobrazovacie PBR vzory, normálové mapovanie, ambientnú oklúziu a import 3D modelov cez WSL.
# Invariant: VITAL-MAX-HP = 6

(def VITAL-MAX-HP 6)
(def GOLDEN-RATIO 1.61803398875)
(def INV-GOLDEN-RATIO 0.61803398875)

(def CANONICAL-DISPLAY-PRESETS
  @{:preset-pbr-ultra
    @{:name "Ultra PBR Cook-Torrance GGX" :exposure 1.15 :roughness 0.25 :metallic 0.85
      :ao-intensity 0.75 :bloom 0.35 :fresnel 1.618 :vital-max-hp VITAL-MAX-HP}
    :preset-normals-tangent
    @{:name "Tangenciálne Vektory & Normály" :exposure 1.0 :roughness 0.0 :metallic 0.0
      :ao-intensity 0.0 :bloom 0.0 :fresnel 1.0 :vital-max-hp VITAL-MAX-HP}
    :preset-ambient-occlusion
    @{:name "Kontaktné Samotienenie (AO)" :exposure 0.9 :roughness 0.1 :metallic 0.0
      :ao-intensity 1.2 :bloom 0.05 :fresnel 1.0 :vital-max-hp VITAL-MAX-HP}
    :preset-wireframe-golden
    @{:name "Zlatá Topológia & Hrany Mriežky" :exposure 1.2 :roughness 0.1 :metallic 0.9
      :ao-intensity 0.4 :bloom 0.5 :fresnel 1.618 :vital-max-hp VITAL-MAX-HP}})

(def HIGH-FIDELITY-3D-MODELS
  @{:crystal-dragon-sanctuary
    @{:title "Svätyňa Kryštálového Draka" :format "OBJ" :spires 12 :vital-max-hp VITAL-MAX-HP}
    :zodiac-celestial-astrolabe
    @{:title "Nebeský Astroláb Čínskeho Zverokruhu" :format "OBJ" :branches 12 :vital-max-hp VITAL-MAX-HP}
    :cybernetic-titan-mech
    @{:title "Kybernetický Titan Mech" :format "OBJ" :pauldrons 2 :vital-max-hp VITAL-MAX-HP}
    :biomorphic-tree-of-life
    @{:title "Biomorfný Strom Života" :format "OBJ" :lobes 5 :vital-max-hp VITAL-MAX-HP}})

(defn get-display-preset
  "Returns display configuration dictionary for given preset key."
  [key]
  (get CANONICAL-DISPLAY-PRESETS key))

(defn evaluate-model-mesh-topology
  "Calculates estimated Euler characteristic and surface fidelity."
  [v-cnt f-cnt]
  (let [e-est (math/floor (* f-cnt 1.5))
        euler (- (+ v-cnt f-cnt) e-est)]
    @{:vertex-count v-cnt
      :face-count f-cnt
      :estimated-edges e-est
      :euler-characteristic euler
      :vital-max-hp VITAL-MAX-HP
      :is-manifold (= euler 2)}))

(defn calculate-cook-torrance-brdf
  "Calculates GGX specular microfacet attenuation."
  [n-dot-l n-dot-v roughness]
  (let [alpha (* roughness roughness)
        denom (max 0.001 (* 4.0 n-dot-l n-dot-v))]
    (/ alpha denom)))
