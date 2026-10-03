# ==============================================================================
# KRYSTAL-STACK: JANET 3D CMS CORE (Oxygen Builder Analog for Godot Engine)
# ==============================================================================
# Implements tree-based DOM -> Godot SceneTree compilation.
# Features:
# 1. Structural Tree Node Management (Oxygen Tree Model)
# 2. Semantic Prompt Parser (AI Bot procedural generation)
# 3. Hexagonal Grid & Tribal Prefab Assembly
# 4. Godot Scene (.tscn) & JSON AST Exporters
# ==============================================================================

(def *krystal-scene-registry* @{})

# Load Cards and Baked Meshes
(dofile "krystal_web_hub/kmen_cards_library.janet")

# 1. COMPONENT SYSTEM
(defn create-node [node-type name properties & children]
  @{:type node-type
    :name name
    :properties (if properties properties @{})
    :children (if children (array ;children) @[])})

# 2. CMS SCENE REGISTRY
(defn register-scene [scene-id title root-node]
  (put *krystal-scene-registry* scene-id 
       @{:title title 
         :tree root-node
         :status :published}))

# 3. PROCEDURAL ARENA PREFABS
(defn prefab-tactical-hex-grid [radius]
  (def tiles @[])
  (def hex-w 1.732)
  (def hex-h 1.5)
  (for q (- radius) (+ radius 1)
    (def r1 (max (- radius) (- (- q) radius)))
    (def r2 (min radius (- (- q) (- radius))))
    (for r r1 (+ r2 1)
      (def x (* hex-w (+ q (* r 0.5))))
      (def z (* hex-h r))
      (def tile-type (cond
                       (and (> x 1.0) (< z 1.0)) :crystal
                       (and (< x -1.0) (< z 1.0)) :toxic
                       (> z 2.0) :druid
                       :neutral))
      (array/push tiles (mesh-hex-tile (string/format "%.2f" x) 0.0 (string/format "%.2f" z) tile-type))))
  (create-node "Spatial" "TacticalHexGrid" @{:tile_count (length tiles)} ;tiles))

(defn prefab-crystal-sanctum [x y z]
  (create-node "Spatial" "CrystalSanctum" @{:position [x y z]}
    (create-node "MeshInstance" "CoreCrystal" 
      @{:mesh "res://godot_assets/crystal_shard.obj" 
        :scale [2.5 3.5 2.5] 
        :material "GlowingCyanShader"})
    (create-node "OmniLight" "SanctumBeacon" @{:color "#66fcf1" :energy 3.0 :range 10.0})
    (create-node "Particles" "AetherSwarm" @{:amount 120 :color "#00ffff"})
    (create-node "Area" "SanctumTerritory" @{:radius 6.0})))

(defn prefab-toxic-wasteland [x y z]
  (create-node "Spatial" "ToxicWasteland" @{:position [x y z]}
    (create-node "MeshInstance" "SlimeMother" 
      @{:mesh "res://godot_assets/acid_slime.obj" 
        :scale [3.0 1.2 3.0] 
        :material "ViscousGreenSlime"})
    (create-node "MeshInstance" "TotemSpire" 
      @{:mesh "res://godot_assets/toxic_totem.obj" 
        :position [0 0.5 0]
        :scale [1.6 2.2 1.6]})
    (create-node "OmniLight" "SporeGaze" @{:color "#39ff14" :energy 2.2 :range 8.0})
    (create-node "Particles" "PoisonCloud" @{:amount 250 :color "#8a2be2"})))

(defn prefab-druid-grove [x y z]
  (create-node "Spatial" "DruidGrove" @{:position [x y z]}
    (create-node "MeshInstance" "WorldRoots" 
      @{:mesh "res://godot_assets/earth_roots.obj" 
        :scale [2.0 2.2 2.0] 
        :material "LivingBark"})
    (create-node "MeshInstance" "SunAltar" 
      @{:mesh "res://godot_assets/druid_monolith.obj" 
        :position [2.0 0 1.5]
        :scale [1.4 1.8 1.4]})
    (create-node "OmniLight" "SolarBlessing" @{:color "#ffd700" :energy 2.5 :range 9.0})
    (create-node "Particles" "GoldenSpores" @{:amount 80 :color "#ffd700"})))

# 4. MCP BOT GENERATOR (Oxygen Analog)
(defn bot-generate-scene [prompt]
  (print (string "[AI BOT] Parsing Directive: " prompt))
  (def p-lower (string/ascii-lower prompt))
  
  (def scene-root 
    (create-node "Node" "WorldRoot" @{}
      (create-node "DirectionalLight" "GlobalSun" @{:color "#ffffff" :energy 0.9 :rotation [-45 30 0]})
      (create-node "Camera" "TacticalCamera" @{:position [0 14 16] :rotation [-40 0 0] :fov 65})
      (prefab-tactical-hex-grid 2)))
      
  # Conditional additions based on prompt semantics
  (if (or (string/find "kryst" p-lower) (string/find "crystal" p-lower) (string/find "sever" p-lower))
    (array/push (scene-root :children) (prefab-crystal-sanctum 0 0 -5)))
    
  (if (or (string/find "tox" p-lower) (string/find "jed" p-lower) (string/find "pustin" p-lower) (string/find "sliz" p-lower))
    (array/push (scene-root :children) (prefab-toxic-wasteland -6 0 2)))
    
  (if (or (string/find "druid" p-lower) (string/find "les" p-lower) (string/find "koren" p-lower) (string/find "nature" p-lower))
    (array/push (scene-root :children) (prefab-druid-grove 6 0 2)))
    
  # Default full arena if generic request
  (if (= (length (scene-root :children)) 3)
    (do
      (array/push (scene-root :children) (prefab-crystal-sanctum 0 0 -5))
      (array/push (scene-root :children) (prefab-toxic-wasteland -6 0 2))
      (array/push (scene-root :children) (prefab-druid-grove 6 0 2))))
      
  (register-scene :generated_arena "Poslední Kmen Procedural Arena" scene-root)
  (print "[AI BOT] Procedural 3D Scene compiled successfully into Janet AST.")
  scene-root)

# 5. EXPORT TO JSON AST
(defn export-scene-to-json [scene-id]
  (def scene (get *krystal-scene-registry* scene-id))
  (if scene
    (string/format `{"id": "%s", "title": "%s", "tree": %j}` scene-id (scene :title) (scene :tree))
    `{"error": "Scene not found"}`))

# 6. EXPORT TO GODOT .TSCN (Godot Scene File)
(defn node-to-tscn [node parent-path idx]
  (def lines @[])
  (def current-path (if (= parent-path "") "." (string parent-path "/" (node :name))))
  (def parent-str (if (= parent-path "") "." parent-path))
  
  (array/push lines (string/format `[node name="%s" type="%s" parent="%s"]` (node :name) (node :type) parent-str))
  (def props (node :properties))
  (eachp [k v] props
    (cond
      (= k :position) (array/push lines (string/format `transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, %s, %s, %s)` (v 0) (v 1) (v 2)))
      (= k :scale) (array/push lines (string/format `scale = Vector3(%s, %s, %s)` (v 0) (v 1) (v 2)))
      (string? v) (array/push lines (string/format `%s = "%s"` (string k) v))
      (number? v) (array/push lines (string/format `%s = %s` (string k) v))))
  (array/push lines "")
  
  (def ch (node :children))
  (if ch
    (for i 0 (length ch)
      (array/concat lines (node-to-tscn (ch i) current-path (+ idx i 1)))))
  lines)

(defn export-scene-to-tscn [scene-id]
  (def scene (get *krystal-scene-registry* scene-id))
  (if scene
    (do
      (def header @[`[gd_scene format=3 uid="uid://krystal_posledni_kmen"]` ""])
      (def body (node-to-tscn (scene :tree) "" 0))
      (string/join (array/concat header body) "\n"))
    "; Scene not found"))

(print "--- KRYSTAL JANET 3D CMS (OXYGEN ANALOG) READY ---")
