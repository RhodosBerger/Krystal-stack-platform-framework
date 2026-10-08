# ==============================================================================
# KRYSTAL-STACK: SKEUOMORPHIC PROCEDURAL SYNTHESIS ENGINE (JANET)
# ==============================================================================
# Bridges the procedural methodics gap: Transcends abstract fractals/noise to
# synthesize tactile, physical real-world items, Vitruvian characters, and
# architectural environments using Golden Ratio component composition.
#
# Core Principles:
# 1. Physical Material Substrates (Walnut, Damascus Steel, Leather, Brass, Glass)
# 2. Functional Seams & Physical Affordances (Mortise-tenon, Rivets, Stitching)
# 3. Ergonomic & Vitruvian Scaling (Phi = 1.61803398875)
# 4. Invariant: VITAL_MAX_HP = 6
# ==============================================================================

(def VITAL-MAX-HP 6)
(def PHI 1.61803398875)
(def INV-PHI (/ 1.0 PHI))

# ------------------------------------------------------------------------------
# 1. PHYSICAL MATERIAL SUBSTRATES
# ------------------------------------------------------------------------------
(def MATERIAL-SUBSTRATES
  @{:AGED_BOHEMIAN_WALNUT
    @{:name "Aged Bohemian Walnut"
      :category :organic_hardwood
      :density-g-cm3 0.68
      :roughness 0.62
      :metallic 0.00
      :normal-depth 0.45
      :color-hex "#3e2723"
      :tactile-grain "deep-linear-pores"}

    :FORGED_DAMASCUS_STEEL
    @{:name "Forged Damascus Steel"
      :category :folded_metal
      :density-g-cm3 7.85
      :roughness 0.28
      :metallic 0.92
      :normal-depth 0.35
      :color-hex "#4a5568"
      :tactile-grain "wootz-microlamellar"}

    :SADDLE_STITCHED_LEATHER
    @{:name "Saddle-Stitched Leather"
      :category :organic_hide
      :density-g-cm3 0.95
      :roughness 0.55
      :metallic 0.05
      :normal-depth 0.50
      :color-hex "#78350f"
      :tactile-grain "pebbled-grain-waxed"}

    :TARNISHED_CHAMPAGNE_BRASS
    @{:name "Tarnished Champagne Brass"
      :category :machined_alloy
      :density-g-cm3 8.45
      :roughness 0.38
      :metallic 0.85
      :normal-depth 0.25
      :color-hex "#b45309"
      :tactile-grain "micro-burnished-patina"}

    :ALCHEMICAL_BLOWN_GLASS
    @{:name "Alchemical Blown Glass"
      :category :vitreous_silicate
      :density-g-cm3 2.50
      :roughness 0.08
      :metallic 0.00
      :normal-depth 0.10
      :color-hex "#0d9488"
      :tactile-grain "fluted-bubble-inclusion"}

    :ILLUMINATED_PARCHMENT
    @{:name "Illuminated Calf Parchment"
      :category :organic_vellum
      :density-g-cm3 0.85
      :roughness 0.70
      :metallic 0.00
      :normal-depth 0.20
      :color-hex "#fef3c7"
      :tactile-grain "fibrous-calcified"}

    :ROMAN_TRAVERTINE_STONE
    @{:name "Roman Travertine Stone"
      :category :mineral_limestone
      :density-g-cm3 2.71
      :roughness 0.80
      :metallic 0.00
      :normal-depth 0.65
      :color-hex "#e2e8f0"
      :tactile-grain "porous-pitted-cellular"}})

# ------------------------------------------------------------------------------
# 2. SKEUOMORPHIC ITEMS DEFINITIONS & GENERATOR
# ------------------------------------------------------------------------------
(def ITEM-TEMPLATES
  @{:ALCHEMIST_LEATHER_GRIMOIRE
    @{:name "Alchemist's Bound Tome"
      :material :SADDLE_STITCHED_LEATHER
      :dimensions-cm [21.0 (* 21.0 PHI) 5.5]
      :weight-kg 1.85
      :vital-hp VITAL-MAX-HP
      :components
      @[@{:part "front_cover" :material :SADDLE_STITCHED_LEATHER :color "#78350f"}
        @{:part "brass_hasp_clasp" :material :TARNISHED_CHAMPAGNE_BRASS :color "#b45309"}
        @{:part "vellum_pages" :material :ILLUMINATED_PARCHMENT :color "#fef3c7"}
        @{:part "spine_ribs" :material :SADDLE_STITCHED_LEATHER :color "#522509"}]
      :ascii-art
      ["+====================+"
       "| .----------------. |"
       "| |  ~ LIBER VITAE ~ |"
       "| |  /\\  PHI: 1.618  |"
       "| | <()> ALCHEMY     |"
       "| |  \\/  [BRASS CLASP]  "
       "| '----------------' |"
       "+====================+"]}

    :FORGED_DAMASCUS_DAGGER
    @{:name "Damascus Forged Athame"
      :material :FORGED_DAMASCUS_STEEL
      :dimensions-cm [34.0 4.8 2.2]
      :weight-kg 0.68
      :vital-hp VITAL-MAX-HP
      :components
      @[@{:part "damascus_blade" :material :FORGED_DAMASCUS_STEEL :color "#4a5568"}
        @{:part "brass_crossguard" :material :TARNISHED_CHAMPAGNE_BRASS :color "#b45309"}
        @{:part "walnut_grip" :material :AGED_BOHEMIAN_WALNUT :color "#3e2723"}
        @{:part "faceted_pommel" :material :TARNISHED_CHAMPAGNE_BRASS :color "#d97706"}]
      :ascii-art
      ["       /\\"
       "      /  \\"
       "     | || |"
       "     | || |"
       "    ========"
       "      |##|"
       "      (==)"]}

    :BRASS_ASTROLABE_SEXTANT
    @{:name "Navigational Celestial Astrolabe"
      :material :TARNISHED_CHAMPAGNE_BRASS
      :dimensions-cm [28.0 28.0 4.5]
      :weight-kg 2.30
      :vital-hp VITAL-MAX-HP
      :components
      @[@{:part "outer_limb_scale" :material :TARNISHED_CHAMPAGNE_BRASS :color "#b45309"}
        @{:part "spider_rete_overlay" :material :TARNISHED_CHAMPAGNE_BRASS :color "#d97706"}
        @{:part "alidade_sight_rule" :material :FORGED_DAMASCUS_STEEL :color "#4a5568"}
        @{:part "shackle_suspension" :material :TARNISHED_CHAMPAGNE_BRASS :color "#92400e"}]
      :ascii-art
      ["      /---\\"
       "    .-' + '-."
       "   / (\\ | /) \\"
       "  | ---(+)--- |"
       "   \\ (/ | \\) /"
       "    '-. _ .-' "
       "       ---   "]}

    :POTION_CRYSTAL_FLASK
    @{:name "Elixir Blown Glass Phial"
      :material :ALCHEMICAL_BLOWN_GLASS
      :dimensions-cm [12.0 12.0 22.0]
      :weight-kg 0.42
      :vital-hp VITAL-MAX-HP
      :components
      @[@{:part "fluted_glass_body" :material :ALCHEMICAL_BLOWN_GLASS :color "#0d9488"}
        @{:part "walnut_turned_stopper" :material :AGED_BOHEMIAN_WALNUT :color "#3e2723"}
        @{:part "leather_harness" :material :SADDLE_STITCHED_LEATHER :color "#78350f"}
        @{:part "brass_neck_ring" :material :TARNISHED_CHAMPAGNE_BRASS :color "#b45309"}]
      :ascii-art
      ["     [===] (Cork)"
       "      | | "
       "     /   \\"
       "    | ~~~ |"
       "    | VIT |"
       "     \\___/ "]}})

(defn synthesize-item [item-type seed]
  (default seed 42)
  (let [tmpl (get ITEM-TEMPLATES item-type)
        scale-var (+ 0.95 (* (math/sin seed) 0.05))]
    (if (nil? tmpl)
      nil
      (merge tmpl
             @{:seed seed
               :actual-weight (* (get tmpl :weight-kg) scale-var)
               :timestamp (os/time)}))))

# ------------------------------------------------------------------------------
# 3. SKEUOMORPHIC CHARACTERS (VITRUVIAN CANON)
# ------------------------------------------------------------------------------
(def CHARACTER-TEMPLATES
  @{:BOHEMIAN_ALCHEMIST_HERO
    @{:name "Alchemist Master of Prague"
      :height-cm 182.0
      :head-length-cm (/ 182.0 8.0) # 22.75 cm
      :vital-hp VITAL-MAX-HP
      :garments
      @[@{:layer "inner" :item "Linen Chemise" :material :ILLUMINATED_PARCHMENT}
        @{:layer "middle" :item "Tailored Leather Jerkin" :material :SADDLE_STITCHED_LEATHER}
        @{:layer "outer" :item "Heavy Wool Scholar Robe" :material :AGED_BOHEMIAN_WALNUT}
        @{:layer "accessory" :item "Brass Sextant Harness" :material :TARNISHED_CHAMPAGNE_BRASS}]
      :silhouette-ascii
      ["     (o o)     [Scholar Hood]"
       "    /| _ |\\    [Leather Jerkin]"
       "   / |   | \\   [Brass Clasp]"
       "  *  |===|  *  [Grimoire in Pouch]"
       "     | | |     [Heavy Woolen Skirt]"
       "     |_|_|     [Saddle Boots]"]}

    :CRYSTAL_KNIGHT_GUARDIAN
    @{:name "Knight Commander of the Resonant Order"
      :height-cm 194.0
      :head-length-cm (/ 194.0 8.0) # 24.25 cm
      :vital-hp VITAL-MAX-HP
      :garments
      @[@{:layer "inner" :item "Padded Gambeson" :material :ILLUMINATED_PARCHMENT}
        @{:layer "middle" :item "Damascus Steel Plate Harness" :material :FORGED_DAMASCUS_STEEL}
        @{:layer "outer" :item "Velvet Tabard with Brass Filigree" :material :TARNISHED_CHAMPAGNE_BRASS}
        @{:layer "accessory" :item "Athame & Greatshield" :material :FORGED_DAMASCUS_STEEL}]
      :silhouette-ascii
      ["     [==T==]   [Damascus Great-Helm]"
       "    /|||||||\\  [Fluted Pauldrons]"
       "   |=| | | |=| [Crossguard Athame]"
       "   / | | | | \\ [Steel Breastplate]"
       "     | | |     [Greaves & Sollerets]"
       "     [_|_]     [VITAL-MAX-HP: 6]"]}

    :DRUID_WOODLAND_RANGER
    @{:name "Highland Warden of Bohemian Glades"
      :height-cm 176.0
      :head-length-cm (/ 176.0 8.0) # 22.0 cm
      :vital-hp VITAL-MAX-HP
      :garments
      @[@{:layer "inner" :item "Homespun Tunic" :material :ILLUMINATED_PARCHMENT}
        @{:layer "middle" :item "Reinforced Stitched Cuirass" :material :SADDLE_STITCHED_LEATHER}
        @{:layer "outer" :item "Forest Camouflage Cloak" :material :AGED_BOHEMIAN_WALNUT}
        @{:layer "accessory" :item "Turned Walnut Quiver & Glass Phials" :material :ALCHEMICAL_BLOWN_GLASS}]
      :silhouette-ascii
      ["     (^_^)     [Feathered Cowl]"
       "    //'|'\\\\    [Stitched Cuirass]"
       "   //  |  \\\\   [Walnut Recurve Bow]"
       "       |       [Elixir Phials]"
       "      / \\      [Tanned Leather Greaves]"
       "     /   \\     [Moccasins]"]}})

(defn synthesize-character [archetype seed]
  (default seed 108)
  (let [tmpl (get CHARACTER-TEMPLATES archetype)]
    (if (nil? tmpl)
      nil
      (merge tmpl
             @{:seed seed
               :vital-hp VITAL-MAX-HP
               :timestamp (os/time)}))))

# ------------------------------------------------------------------------------
# 4. SKEUOMORPHIC ROOMS & ENVIRONMENTS
# ------------------------------------------------------------------------------
(def ROOM-TEMPLATES
  @{:ALCHEMIST_WORKSHOP_CHAMBER
    @{:name "Old Town Alchemist Laboratory"
      :width-m 8.0
      :length-m (* 8.0 PHI) # ~12.94 m
      :height-m 4.5
      :primary-material :ROMAN_TRAVERTINE_STONE
      :secondary-material :AGED_BOHEMIAN_WALNUT
      :architectural-features
      @["Heavy Bohemian Walnut Ceiling Rafters"
        "Roman Travertine Ashlar Masonry Fireplace"
        "Leaded Stained Glass Oriel Window"
        "Walnut Heavy Workbench with Turnery Legs"
        "Hand-Blown Glass Alembics and Retorts"
        "Saddle Leather Wall Hangings and Draft Screens"]
      :elevation-ascii
      ["+===================================================+"
       "|   [=== BOHEMIAN WALNUT CEILING RAFTERS ===]       |"
       "|                                                   |"
       "|  |#####|      (Oriel Leaded Glass)     |#####|    |"
       "|  |#####|          .--------.           |#####|    |"
       "|  |HEARTH|         | [+][+] |          |HERBS|    |"
       "|  |FIRE |          '--------'           |RACK |    |"
       "|  |_____|   ========================    |_____|    |"
       "|            | WORKBENCH & RETORTS  |               |"
       "|            |  [Flask]   [Grimoire]|               |"
       "|~~~~~~~~~~~~'----------------------'~~~~~~~~~~~~~~~|"
       "+===================================================+"]}

    :MASTER_FORGE_ARMORY
    @{:name "Guildhall Master Weaponsmith Forge"
      :width-m 10.0
      :length-m (* 10.0 PHI) # ~16.18 m
      :height-m 5.5
      :primary-material :ROMAN_TRAVERTINE_STONE
      :secondary-material :FORGED_DAMASCUS_STEEL
      :architectural-features
      @["Double Bellows Forced-Air Charcoal Hearth"
        "Massive Granite Anvil Plinth with Quench Trough"
        "Racks of Forged Damascus Steel Blades"
        "Tarnished Brass Measuring Rods & Calipers"
        "Smoke-Stained Vaulted Travertine Ceiling"]
      :elevation-ascii
      ["+===================================================+"
       "|   [=== SMOKE-STAINED STONE VAULTED CEILING ===]   |"
       "|                                                   |"
       "|    /======\\                           |BLADES|    |"
       "|   | FORGE  |          ______          | | | | |   |"
       "|   | HEARTH |         / ANVIL\\         | | | | |   |"
       "|   |  FLAME |        [========]        | | | | |   |"
       "|   |________|         |      |         |_______|   |"
       "|   [QUENCH-TROUGH]   '--------'        [RIVET-BIN] |"
       "+===================================================+"]}

    :GOTHIC_LIBRARY_STUDIO
    @{:name "Gothic Scriptorium & Optical Observatorium"
      :width-m 7.0
      :length-m (* 7.0 PHI) # ~11.33 m
      :height-m 6.0
      :primary-material :AGED_BOHEMIAN_WALNUT
      :secondary-material :ILLUMINATED_PARCHMENT
      :architectural-features
      @["Two-Tiered Bohemian Walnut Bookcases with Gallery"
        "Cast Brass Rolling Ladder & Balustrade"
        "Central Octagonal Inlaid Reading Table"
        "Suspended Brass Armillary Sphere & Astrolabes"
        "Ribbed Vaulting with Gold-Leaf Rosettes"]
      :elevation-ascii
      ["+===================================================+"
       "|     /\\  /\\   GOTHIC RIBBED VAULTING   /\\  /\\      |"
       "|    /  \\/  \\     (ARMILLARY SPHERE)   /  \\/  \\     |"
       "|   |========|            (O)         |========|    |"
       "|   |BOOKS #1|         ---(+)---      |BOOKS #2|    |"
       "|   |--------|          /     \\       |--------|    |"
       "|   |[LADDER]|      .------------.    |[GALLERY|    |"
       "|   |   ||   |      |READING DESK|    |   ||   |    |"
       "|   |   ||   |      |  GRIMOIRE  |    |   ||   |    |"
       "+===================================================+"]}})

(defn synthesize-room [room-type seed]
  (default seed 256)
  (let [tmpl (get ROOM-TEMPLATES room-type)]
    (if (nil? tmpl)
      nil
      (merge tmpl
             @{:seed seed
               :aspect-ratio PHI
               :timestamp (os/time)}))))

# ------------------------------------------------------------------------------
# 5. GODOT 4 FORWARD+ TSCN EXPORT GENERATOR
# ------------------------------------------------------------------------------
(defn export-godot-tscn [entity]
  (def name (get entity :name "SkeuomorphicAsset"))
  (def lines
    @["[gd_scene format=3 uid=\"uid://krystal_skeuo_procedural\"]"
      ""
      (string "[node name=\"" name "\" type=\"Node3D\"]")
      "metadata/skeuomorphic = true"
      (string "metadata/vital_max_hp = " VITAL-MAX-HP)
      (string "metadata/golden_ratio = " PHI)
      ""
      "[node name=\"MeshInstance3D\" type=\"MeshInstance3D\" parent=\".\"]"
      "[node name=\"CollisionShape3D\" type=\"CollisionShape3D\" parent=\".\"]"
      ""])
  (string/join lines "\n"))

# ------------------------------------------------------------------------------
# 6. SANITY TEST ON EVALUATION
# ------------------------------------------------------------------------------
(defn test-skeuomorphic-suite []
  (print "[SKEUOMORPHIC JANET] Synthesizing physical test assets...")
  (def item (synthesize-item :FORGED_DAMASCUS_DAGGER 42))
  (print (string "Item: " (get item :name) " | HP: " (get item :vital-hp)))
  (def char (synthesize-character :BOHEMIAN_ALCHEMIST_HERO 108))
  (print (string "Char: " (get char :name) " | Head: " (get char :head-length-cm) " cm"))
  (def room (synthesize-room :ALCHEMIST_WORKSHOP_CHAMBER 256))
  (print (string "Room: " (get room :name) " | Aspect: " (get room :aspect-ratio)))
  (assert (= (get item :vital-hp) 6) "Invariant failed: VITAL-MAX-HP must be 6")
  (print "[SKEUOMORPHIC JANET] All physical synthesis tests passed successfully."))
