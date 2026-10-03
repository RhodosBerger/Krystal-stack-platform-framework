# ==============================================================================
# KRYSTAL-STACK: MRP HARMONIC ENGINE HIERARCHY & STREET MAP DSL (JANET)
# ==============================================================================
# Defines:
#   1. MRP manufacturing hierarchy transformed into 5-tier engine hierarchy:
#      - Level 0: Axiomatic Root & Pink Panther Color Tint Harmonies (Phi = 1.618)
#      - Level 1: Macro Metropolises (12 Global Cities/Regions)
#      - Level 2: Meso Street Grids (Google Street Map / OSM Vector Conduits)
#      - Level 3: Micro 2D Vehicle Cruisers (Top-down motion & drift)
#      - Level 4: Execution & Strict 6 Max HP Vital Invariant
#   2. Canonical regions: Gotham City, Arkham City, Sin City, Las Vegas,
#      Alabama, Ohio, Florida, Australia, Canary Islands, Bolivia, Ecuador, Peru.
# ==============================================================================

(def VITAL-MAX-HP 6)
(def GOLDEN-RATIO 1.6180339887)

(def PINK-PANTHER-COLOR-AXIOMS
  {:primary-pink "#f472b6"
   :deep-magenta "#db2777"
   :soft-blush "#fbcfe8"
   :champagne-cream "#fef3c7"
   :noir-charcoal "#18181b"
   :neon-cyan-accent "#06b6d4"
   :amber-gold "#f59e0b"
   :base-hue-degrees 330.0
   :harmonic-tint-ratio GOLDEN-RATIO})

(def MRP-ENGINE-LEVELS
  [:level-0-harmonic-axioms
   :level-1-macro-metropolises
   :level-2-meso-street-network
   :level-3-micro-vehicle-agents
   :level-4-vital-execution])

(def CANONICAL-WORLD-REGIONS
  {:gotham-city {:name "Gotham City" :biome "noir-gothic" :elevation-m 15.0 :grid-type "orthogonal-canyon"}
   :arkham-city {:name "Arkham City" :biome "noir-industrial" :elevation-m 8.0 :grid-type "fortified-perimeter"}
   :sin-city {:name "Sin City" :biome "stark-monochrome" :elevation-m 45.0 :grid-type "high-contrast-grid"}
   :las-vegas {:name "Las Vegas" :biome "neon-desert" :elevation-m 610.0 :grid-type "boulevard-strip"}
   :alabama {:name "Alabama" :biome "rural-pines" :elevation-m 120.0 :grid-type "highway-crossroads"}
   :ohio {:name "Ohio" :biome "rust-belt-river" :elevation-m 230.0 :grid-type "industrial-bend"}
   :florida {:name "Florida" :biome "coastal-palms" :elevation-m 2.0 :grid-type "ocean-drive-causeway"}
   :australia {:name "Australia" :biome "red-outback" :elevation-m 310.0 :grid-type "continental-highway"}
   :canary-islands {:name "Canary Islands" :biome "volcanic-atlantic" :elevation-m 480.0 :grid-type "oceanic-switchbacks"}
   :bolivia {:name "Bolivia" :biome "andean-yungas" :elevation-m 3650.0 :grid-type "death-road-serpentine"}
   :ecuador {:name "Ecuador" :biome "equatorial-volcanoes" :elevation-m 2850.0 :grid-type "avenue-of-volcanoes"}
   :peru {:name "Peru" :biome "sacred-valley" :elevation-m 2430.0 :grid-type "inca-stone-serpentine"}})

(defn calculate-harmonic-color-tint
  "Computes RGB tint shift based on Golden Ratio harmonic axioms and Pink Panther hue."
  [base-intensity step-factor]
  (let [phi-step (* step-factor (/ 1.0 GOLDEN-RATIO))
        r-tint (min 255 (math/round (* base-intensity (+ 1.0 (* phi-step 0.35)))))
        g-tint (min 255 (math/round (* base-intensity (+ 0.45 (* phi-step 0.15)))))
        b-tint (min 255 (math/round (* base-intensity (+ 0.72 (* phi-step 0.22)))))]
    {:r r-tint :g g-tint :b b-tint :hex (string/format "#%02x%02x%02x" r-tint g-tint b-tint)}))

(defn compute-mrp-bill-of-materials
  "Translates MRP levels into procedural component requirements for a city sector."
  [region-key block-count]
  (let [region (get CANONICAL-WORLD-REGIONS region-key)
        streets-required (* block-count 4)
        intersections (* block-count 2)
        cruiser-vehicles (max 2 (math/round (/ block-count 2)))
        elevation (get region :elevation-m 50.0)]
    {:region-id region-key
     :region-name (get region :name "Unknown")
     :elevation-m elevation
     :mrp-level-1-blocks block-count
     :mrp-level-2-street-conduits streets-required
     :mrp-level-2-intersections intersections
     :mrp-level-3-cruisers cruiser-vehicles
     :vital-max-hp-limit VITAL-MAX-HP}))

(defn simulate-2d-vehicle-vector
  "Simulates 2D top-down vehicle motion along road conduit with steering and drift."
  [pos-x pos-y velocity-mps heading-rad steering-input delta-time]
  (let [turn-rate (* steering-input 1.85)
        new-heading (+ heading-rad (* turn-rate delta-time))
        vx (* velocity-mps (math/cos new-heading))
        vy (* velocity-mps (math/sin new-heading))
        new-x (+ pos-x (* vx delta-time))
        new-y (+ pos-y (* vy delta-time))]
    {:x new-x
     :y new-y
     :heading-rad new-heading
     :velocity-mps velocity-mps
     :vital-hp VITAL-MAX-HP}))
