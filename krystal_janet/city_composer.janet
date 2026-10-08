# ==============================================================================
# KRYSTAL-STACK: PROCEDURAL CITY COMPOSITION & MULTI-ASSET CANVAS ARTISTRY (JANET)
# ==============================================================================
# Reframes procedural city generation as an artistic multi-asset canvas painting:
# "Mesto ako Kreslená Kompozícia Assetov"
#
# Seven Compositional Layers:
#   Layer 0: Macro Skyline Dominants (Kyber Spires, Clocktowers, Citadels)
#   Layer 1: Midground Architectural Blocks (Tenements, Stepped Terraces, Plinths)
#   Layer 2: Connective Infrastructure (Grand Boulevards, Bridges, Canal Basins)
#   Layer 3: Biophilic Infill (Avenue Linden Trees, Marble Fountains, Rooftop Gardens)
#   Layer 4: Micro-Props & Urban Furniture (Ornate Cast-Iron Lamps, Neon Cyber Panes)
#   Layer 5: Kinetic Cruiser Vehicles (Panther Cruisers, Headlights, HP <= 6)
#   Layer 6: Atmospheric Tonal Wash (ACES Filmic Fog, Twilight Horizon Glow)
#
# Enforces Immutable Invariant: VITAL-MAX-HP = 6
# Enforces Golden Ratio Phi = 1.61803398875
# ==============================================================================

(def VITAL-MAX-HP 6)
(def GOLDEN-RATIO 1.61803398875)
(def INV-GOLDEN-RATIO (/ 1.0 GOLDEN-RATIO))
(def BASE-MODULE-M0 (* 3.0 GOLDEN-RATIO)) # ~4.854m

(def ASSET-BRUSHES
  @{:CYBER_SPIRE_MONOLITH
    @{:name "Kryštálový Kyber-Monolit"
      :layer 0
      :width-m (* BASE-MODULE-M0 4.0)
      :height-m 120.0
      :ascii-char "▲"
      :color-hex "#06b6d4"
      :vital-hp VITAL-MAX-HP}
    :ALCHEMICAL_CLOCKTOWER
    @{:name "Alchymistická Veža s Orlojom"
      :layer 0
      :width-m (* BASE-MODULE-M0 3.0)
      :height-m 68.0
      :ascii-char "Ω"
      :color-hex "#f59e0b"
      :vital-hp VITAL-MAX-HP}
    :DATA_CITADEL_ZIGGURAT
    @{:name "Dátová Zikkurat Citadela"
      :layer 0
      :width-m (* BASE-MODULE-M0 6.0)
      :height-m 48.0
      :ascii-char "█"
      :color-hex "#8b5cf6"
      :vital-hp VITAL-MAX-HP}
    :MODULAR_TENEMENT_BLOCK
    @{:name "Modulárny Mestský Blok"
      :layer 1
      :width-m (* BASE-MODULE-M0 3.0)
      :height-m 24.0
      :ascii-char "■"
      :color-hex "#d4d4d8"
      :vital-hp VITAL-MAX-HP}
    :STEPPED_TERRACE_RESIDENCE
    @{:name "Kaskádový Terasový Dom"
      :layer 1
      :width-m (* BASE-MODULE-M0 3.5)
      :height-m 18.0
      :ascii-char "≡"
      :color-hex "#e4e4e7"
      :vital-hp VITAL-MAX-HP}
    :GRAND_BOULEVARD_CONDUIT
    @{:name "Centrálny Bulvár"
      :layer 2
      :width-m (* BASE-MODULE-M0 4.0)
      :height-m 0.3
      :ascii-char "═"
      :color-hex "#27272a"
      :vital-hp VITAL-MAX-HP}
    :CANAL_BASIN_WATERWAY
    @{:name "Zrkadliaci Kanál"
      :layer 2
      :width-m (* BASE-MODULE-M0 3.0)
      :height-m 0.0
      :ascii-char "≈"
      :color-hex "#0284c7"
      :vital-hp VITAL-MAX-HP}
    :AVENUE_LINDEN_TREE
    @{:name "Alejová Lipa Malolistá"
      :layer 3
      :width-m BASE-MODULE-M0
      :height-m 7.5
      :ascii-char "♣"
      :color-hex "#22c55e"
      :vital-hp VITAL-MAX-HP}
    :URBAN_PLAZA_FOUNTAIN
    @{:name "Mramorová Fontána"
      :layer 3
      :width-m (* BASE-MODULE-M0 2.0)
      :height-m 2.8
      :ascii-char "○"
      :color-hex "#38bdf8"
      :vital-hp VITAL-MAX-HP}
    :ORNATE_STREET_LAMP
    @{:name "Liatinová Pouličná Lampa"
      :layer 4
      :width-m 0.8
      :height-m 4.8
      :ascii-char "†"
      :color-hex "#f59e0b"
      :vital-hp VITAL-MAX-HP}
    :PANTHER_CRUISER_2D
    @{:name "Panther Kinetic Cruiser"
      :layer 5
      :width-m 2.2
      :height-m 1.4
      :ascii-char "►"
      :color-hex "#f472b6"
      :vital-hp VITAL-MAX-HP}})

(defn compute-golden-focal-position
  "Calculates horizontal focal anchor position on canvas according to Golden Ratio split."
  [canvas-width use-right-split]
  (let [factor (if use-right-split INV-GOLDEN-RATIO (- 1.0 INV-GOLDEN-RATIO))]
    (- (* canvas-width factor) (* canvas-width 0.5))))

(defn validate-composition-vital-hp
  "Strictly verifies that no asset instance in the composition exceeds 6 Max HP."
  [instances]
  (all (fn [inst] (<= (get inst :vital-hp 0) VITAL-MAX-HP)) instances))

(defn generate-composition-summary
  "Summarizes composition layer counts and confirms mathematical invariants."
  [comp-name seed asset-count is-valid-hp]
  @{:name comp-name
    :seed seed
    :total-assets asset-count
    :vital-invariant is-valid-hp
    :max-hp-limit VITAL-MAX-HP
    :golden-ratio-used GOLDEN-RATIO})

(def DISTRICT-BIOMES
  @{:downtown_cyber_spires
    @{:name "Centrálne Kybernetické Jadro"
      :dominant :CYBER_SPIRE_MONOLITH
      :color "#06b6d4"}
    :historic_gothic_quarter
    @{:name "Historická Gotická Štvrť"
      :dominant :ALCHEMICAL_CLOCKTOWER
      :color "#f59e0b"}
    :industrial_docks_canal
    @{:name "Priemyselné Doky & Plavebný Kanál"
      :dominant :DATA_CITADEL_ZIGGURAT
      :color "#8b5cf6"}
    :residential_terraces
    @{:name "Rezidenčné Terasové Záhrady"
      :dominant :STEPPED_TERRACE_RESIDENCE
      :color "#ec4899"}
    :biophilic_central_park
    @{:name "Biofilný Centrálny Park"
      :dominant :URBAN_PLAZA_FOUNTAIN
      :color "#22c55e"}})

(defn compute-sector-world-origin
  "Computes world space origin (X, Z) for sector at grid coordinate (gx, gz)."
  [gx gz grid-cols grid-rows sector-size]
  (let [half-cols (* (- grid-cols 1) 0.5)
        half-rows (* (- grid-rows 1) 0.5)
        ox (* (- gx half-cols) sector-size)
        oz (* (- gz half-rows) sector-size)]
    [ox oz]))

