# Krystal-Stack Janet DSL // Godot Interior Surface & Procedural World Nodes
# Encodes interior design patterns, biophilic surface characteristics,
# and macro-scale pre-generation rules for trees, buildings, cities, and soil horizons.
# Strict Invariant: VITAL-MAX-HP = 6

(def VITAL-MAX-HP 6)
(def GOLDEN-RATIO 1.61803398875)
(def INV-GOLDEN-RATIO 0.61803398875)
(def DEFAULT-MULTIMESH-BUDGET 15000)

(def INTERIOR-MATERIALS
  @{:oak_herringbone_parquet
    @{:name "Prémiové Dubové Rybie Parkety"
      :category "wood"
      :albedo [0.68 0.48 0.32 1.0]
      :roughness 0.38
      :metallic 0.02
      :biophilic-resonance 0.88}
    :roman_travertine_stone
    @{:name "Rímsky Pórovitý Travertín"
      :category "stone"
      :albedo [0.86 0.82 0.73 1.0]
      :roughness 0.55
      :metallic 0.04
      :biophilic-resonance 0.76}
    :venetian_terrazzo
    @{:name "Benátske Terazzo"
      :category "composite"
      :albedo [0.92 0.90 0.88 1.0]
      :roughness 0.18
      :metallic 0.08
      :biophilic-resonance 0.65}
    :biophilic_vertical_moss
    @{:name "Biofilná Živá Machová Stena"
      :category "biophilic"
      :albedo [0.18 0.65 0.22 1.0]
      :roughness 0.92
      :metallic 0.0
      :biophilic-resonance 0.98}
    :acoustic_slat_oak_felt
    @{:name "Akustické Dubové Lamely"
      :category "wood"
      :albedo [0.45 0.32 0.22 1.0]
      :roughness 0.60
      :metallic 0.02
      :biophilic-resonance 0.84}
    :brushed_champagne_brass
    @{:name "Kefovaná Mosadz Šampanské Zlato"
      :category "metal"
      :albedo [0.88 0.78 0.52 1.0]
      :roughness 0.25
      :metallic 0.94
      :biophilic-resonance 0.55}
    :architectural_microcement
    @{:name "Architektonický Mikrocement"
      :category "mineral"
      :albedo [0.58 0.60 0.62 1.0]
      :roughness 0.48
      :metallic 0.05
      :biophilic-resonance 0.60}})

(def SOIL-HORIZONS
  @{:A @{:name "Humózna Ornica & Biofilný Substrát" :thickness-m 0.35 :porosity 48.5}
    :B @{:name "Minerálna Pôdna Vrstva" :thickness-m 0.85 :porosity 38.0}
    :C @{:name "Zvetralé Kamenisté Podložie" :thickness-m 1.60 :porosity 24.5}
    :R @{:name "Kryštalická Materská Skala" :thickness-m 10.0 :porosity 4.2}})

(defn calculate-golden-module
  "Computes an architectural dimension scaled by the Golden Ratio."
  [base-dim exponent]
  (* base-dim (math/pow GOLDEN-RATIO exponent)))

(defn compute-tree-canopy-volume
  "Calculates procedural tree canopy ellipsoid volume in cubic meters."
  [radius height]
  (* (/ 4.0 3.0) math/pi radius radius (* height 0.5)))

(defn evaluate-biophilic-resonance
  "Computes composite biophilic well-being factor blending material resonance and foliage coverage."
  [mat-resonance foliage-pct]
  (let [f-factor (min 1.0 (/ foliage-pct 100.0))]
    (min 1.0 (+ (* mat-resonance 0.6) (* f-factor 0.4)))))

(defn validate-godot-multimesh-instance-budget
  "Ensures instanced vegetation does not exceed GPU rendering budget."
  [instance-count]
  (let [budget DEFAULT-MULTIMESH-BUDGET]
    @{:valid (<= instance-count budget)
      :instance-count instance-count
      :max-budget budget
      :utilization-pct (* (/ instance-count budget) 100.0)}))
