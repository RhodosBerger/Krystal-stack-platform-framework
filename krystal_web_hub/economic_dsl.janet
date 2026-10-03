# ==============================================================================
# KRYSTAL-STACK: JANET ECONOMIC DSL & 3D AST PREFABS
# ==============================================================================
# Declarative Lisp specifications for economic structures, round gradation,
# and Godot SceneTree node compilation.
# ==============================================================================

(dofile "krystal_web_hub/krystal_3d_cms.janet")

# 1. ECONOMIC BUILD PREFABS
(defn prefab-aether-conduit [x y z]
  (create-node "Spatial" "Building_AetherConduit" @{:position [x y z] :tier 1}
    (create-node "MeshInstance" "ConduitBase" 
      @{:mesh "res://godot_assets/aether_conduit.obj" 
        :scale [1.2 1.2 1.2] 
        :material "IndustrialRefineryCyan" 
        :color "#00ffff"})
    (create-node "OmniLight" "AetherReactor" @{:color "#00ffff" :energy 2.4 :range 7.0})
    (create-node "Particles" "ManaVent" @{:amount 100 :color "#66fcf1" :velocity [0 5 0]})))

(defn prefab-slime-pit [x y z]
  (create-node "Spatial" "Building_SlimePit" @{:position [x y z] :tier 1}
    (create-node "MeshInstance" "VatStructure" 
      @{:mesh "res://godot_assets/slime_pit.obj" 
        :scale [1.3 1.1 1.3] 
        :material "CausticGreenVat" 
        :color "#39ff14"})
    (create-node "OmniLight" "SlimeGlow" @{:color "#39ff14" :energy 2.0 :range 6.0})
    (create-node "Particles" "AcidBubbles" @{:amount 80 :color "#39ff14"})))

(defn prefab-world-tree [x y z]
  (create-node "Spatial" "Building_WorldTree" @{:position [x y z] :tier 1}
    (create-node "MeshInstance" "CanopyCrown" 
      @{:mesh "res://godot_assets/world_tree.obj" 
        :scale [1.5 1.7 1.5] 
        :material "LivingCanopyGreen" 
        :color "#2e8b57"})
    (create-node "OmniLight" "NatureSanctuary" @{:color "#ffd700" :energy 2.8 :range 9.0})
    (create-node "Particles" "SunlitPollen" @{:amount 90 :color "#ffd700"})))

(defn prefab-capacitor-tower [x y z]
  (create-node "Spatial" "Building_CapacitorTower" @{:position [x y z] :tier 2}
    (create-node "MeshInstance" "PylonRings" 
      @{:mesh "res://godot_assets/capacitor_tower.obj" 
        :scale [1.2 1.5 1.2] 
        :material "CapacitorCoil" 
        :color "#66fcf1"})
    (create-node "OmniLight" "CoilSpark" @{:color "#00f2fe" :energy 3.2 :range 8.0})))

# 2. MACRO: COMPILE ECONOMIC MATCH BATTLEFIELD
(defn compile-economic-battlefield [round-num escalation player-buildings enemy-buildings]
  (def root (create-node "Node" "EconomicMatchRoot" 
              @{:round round-num :escalation escalation}))
  (def sun (create-node "DirectionalLight" "TacticalSun" 
             @{:color "#ffffff" :energy 1.1 :rotation [-45 30 0]}))
  (def cam (create-node "Camera" "TacticalCamera" 
             @{:position [0 16 18] :fov 60}))
  (def grid (prefab-tactical-hex-grid 2))

  (array/push (root :children) sun)
  (array/push (root :children) cam)
  (array/push (root :children) grid)

  # Attach player buildings
  (def player-group (create-node "Spatial" "PlayerInfrastructure" @{}))
  (each b player-buildings
    (def p (b :position))
    (def b-type (b :type))
    (cond
      (= b-type "aether_conduit") (array/push (player-group :children) (prefab-aether-conduit (p 0) (p 1) (p 2)))
      (= b-type "slime_pit") (array/push (player-group :children) (prefab-slime-pit (p 0) (p 1) (p 2)))
      (= b-type "world_tree") (array/push (player-group :children) (prefab-world-tree (p 0) (p 1) (p 2)))
      (= b-type "capacitor_tower") (array/push (player-group :children) (prefab-capacitor-tower (p 0) (p 1) (p 2)))))
  (array/push (root :children) player-group)

  # Attach enemy buildings
  (def enemy-group (create-node "Spatial" "EnemyInfrastructure" @{}))
  (each b enemy-buildings
    (def p (b :position))
    (def b-type (b :type))
    (cond
      (= b-type "aether_conduit") (array/push (enemy-group :children) (prefab-aether-conduit (p 0) (p 1) (p 2)))
      (= b-type "slime_pit") (array/push (enemy-group :children) (prefab-slime-pit (p 0) (p 1) (p 2)))
      (= b-type "world_tree") (array/push (enemy-group :children) (prefab-world-tree (p 0) (p 1) (p 2)))
      (= b-type "capacitor_tower") (array/push (enemy-group :children) (prefab-capacitor-tower (p 0) (p 1) (p 2)))))
  (array/push (root :children) enemy-group)

  (register-scene :active_economic_match "Active Economic Match" root)
  root)

(print "--- KRYSTAL JANET ECONOMIC DSL & 3D PREFABS LOADED ---")
