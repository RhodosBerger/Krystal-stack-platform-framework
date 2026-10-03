# ==============================================================================
# KRYSTAL-STACK: POSLEDNÍ KMEN - CARDS & 3D MESH LIBRARY (JANET AST)
# ==============================================================================
# Hardcoded card specifications, game mechanics, and procedural 3D scene mappings.
# Rules: Max 6 HP, Ledger Mana Accounting, 3 Core Tribes.
# Meshes link directly to baked Wavefront .obj models in res://godot_assets/
# ==============================================================================

(def *kmen-cards* @{})

(defn register-card [id name tribe cost hp-delta armor-delta status-effect mesh-fn]
  (put *kmen-cards* id 
       @{:id (string id)
         :name name 
         :tribe tribe 
         :cost cost 
         :hp_delta hp-delta 
         :armor_delta armor-delta 
         :status_effect status-effect 
         :mesh mesh-fn}))

# ------------------------------------------------------------------------------
# 1. KRYŠTÁLOVÝ KMEŇ (Severní Štíty) - Cyan, Sharp, Tech-Magic, Kinetic
# ------------------------------------------------------------------------------

# Card: Kryštálový Meteor
(defn mesh-crystal-meteor [tx ty tz]
  (create-node "Spatial" "Spell_CrystalMeteor" @{:position [tx ty tz]}
    (create-node "MeshInstance" "CoreShard" 
      @{:mesh "res://godot_assets/crystal_shard.obj" 
        :scale [1.6 1.6 1.6] 
        :material "CyanGlowingShader" 
        :color "#00ffff"}
      (create-node "AnimationPlayer" "DropAnim" @{:anim "meteor_strike" :duration 1.2}))
    (create-node "Particles" "ManaDust" @{:amount 180 :color "#66fcf1" :velocity [0 8 0]})
    (create-node "OmniLight" "ImpactGlow" @{:color "#00ffff" :energy 2.8 :range 8.0})))

(register-card :crystal_meteor "Kryštálový Meteor" :crystal 3 -2 0 :none mesh-crystal-meteor)

# Card: Kryštálový Štít
(defn mesh-crystal-shield [tx ty tz]
  (create-node "Spatial" "Spell_CrystalShield" @{:position [tx ty tz]}
    (create-node "MeshInstance" "HexBarrier" 
      @{:mesh "res://godot_assets/crystal_shield.obj" 
        :scale [1.2 1.2 1.2] 
        :material "GlassCrystalline" 
        :color "#66fcf1" 
        :opacity 0.85})
    (create-node "OmniLight" "ShieldAura" @{:color "#00ffff" :energy 1.6 :range 5.0})))

(register-card :crystal_shield "Kryštálový Štít" :crystal 2 0 2 :shielded mesh-crystal-shield)

# Card: Rezonančný Pylón
(defn mesh-crystal-pylon [tx ty tz]
  (create-node "Spatial" "Building_CrystalPylon" @{:position [tx ty tz]}
    (create-node "MeshInstance" "PylonSpire" 
      @{:mesh "res://godot_assets/crystal_shard.obj" 
        :scale [2.2 2.8 2.2] 
        :material "PureAether" 
        :color "#e0ffff"})
    (create-node "Particles" "ResonanceRings" @{:amount 80 :color "#00ffff" :spread 3.0})))

(register-card :crystal_pylon "Rezonančný Pylón" :crystal 4 0 0 :mana_regen mesh-crystal-pylon)

# ------------------------------------------------------------------------------
# 2. JEDOVATÝ KMEŇ (Pustina) - Acid Green, Purple, Slime, Degradation
# ------------------------------------------------------------------------------

# Card: Kyslý Sliz
(defn mesh-acid-slime [tx ty tz]
  (create-node "Spatial" "Spell_AcidSlime" @{:position [tx ty tz]}
    (create-node "MeshInstance" "SlimePool" 
      @{:mesh "res://godot_assets/acid_slime.obj" 
        :scale [1.8 1.0 1.8] 
        :material "ViscousAcidGreen" 
        :color "#39ff14"}
      (create-node "AnimationPlayer" "BubbleAnim" @{:anim "bubbling" :speed 1.5}))
    (create-node "Particles" "AcidVapor" @{:amount 120 :color "#39ff14" :spread 2.5})))

(register-card :acid_slime "Kyslý Sliz" :toxic 1 0 0 :rooted mesh-acid-slime)

# Card: Toxický Totem
(defn mesh-toxic-totem [tx ty tz]
  (create-node "Spatial" "Building_ToxicTotem" @{:position [tx ty tz]}
    (create-node "MeshInstance" "SporeColumn" 
      @{:mesh "res://godot_assets/toxic_totem.obj" 
        :scale [1.4 1.8 1.4] 
        :material "OrganicChitin" 
        :color "#7fff00"})
    (create-node "Particles" "PoisonMist" @{:amount 240 :color "#8a2be2" :radius 4.5})
    (create-node "Area" "DamageZone" @{:radius 4.0})))

(register-card :toxic_cloud "Toxický Oblak" :toxic 4 -1 0 :poison_aoe mesh-toxic-totem)

# ------------------------------------------------------------------------------
# 3. DRUIDI (Hlboký Les) - Amber, Emerald, Bark, Ancient Stems
# ------------------------------------------------------------------------------

# Card: Korene Zeme
(defn mesh-earth-roots [tx ty tz]
  (create-node "Spatial" "Spell_EarthRoots" @{:position [tx ty tz]}
    (create-node "MeshInstance" "GnarledRoots" 
      @{:mesh "res://godot_assets/earth_roots.obj" 
        :scale [1.5 1.5 1.5] 
        :material "AncientBark" 
        :color "#8b4513"})
    (create-node "Particles" "FloatingLeaves" @{:amount 60 :color "#2e8b57"})))

(register-card :earth_roots "Korene Zeme" :druid 2 -1 0 :stunned mesh-earth-roots)

# Card: Posvätný Monolit / Požehnanie Prírody
(defn mesh-druid-monolith [tx ty tz]
  (create-node "Spatial" "Building_DruidMonolith" @{:position [tx ty tz]}
    (create-node "MeshInstance" "RuneStone" 
      @{:mesh "res://godot_assets/druid_monolith.obj" 
        :scale [1.3 1.6 1.3] 
        :material "MossyGranite" 
        :color "#556b2f"})
    (create-node "OmniLight" "SanctuaryLight" @{:color "#ffd700" :energy 2.2 :range 6.0})
    (create-node "Particles" "LifeSparks" @{:amount 100 :color "#ffd700"})))

(register-card :nature_bless "Požehnanie Prírody" :druid 3 2 0 :healed mesh-druid-monolith)

# ------------------------------------------------------------------------------
# 4. ARENA GRID GENERATION
# ------------------------------------------------------------------------------
(defn mesh-hex-tile [x y z tile-type]
  (create-node "MeshInstance" (string "GridTile_" x "_" z) 
    @{:position [x y z] 
      :mesh "res://godot_assets/hex_tile.obj" 
      :tile_type tile-type
      :material (case tile-type
                  :crystal "TileCrystalCyan"
                  :toxic "TileToxicGreen"
                  :druid "TileDruidEarth"
                  "TileNeutralGrey")}))

(print "[LIBRARY] Loaded 7 Core Cards and Procedural Grid Tile Definitions.")
