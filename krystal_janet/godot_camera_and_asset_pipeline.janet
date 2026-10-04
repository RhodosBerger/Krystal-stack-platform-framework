# Krystal-Stack Janet DSL // Godot Camera & 3D Animated Asset Pipeline
# Encodes camera perspective & orthographic projection parameters,
# animation clip registries, and seamless dual-rig transition mathematics.
# Strict Invariant: VITAL-MAX-HP = 6

(def VITAL-MAX-HP 6)
(def GOLDEN-RATIO 1.61803398875)
(def INV-GOLDEN-RATIO 0.61803398875)

(def ANIMATED-MODELS-CATALOG
  @{:robot_expressive
    @{:name "Krystal Cybernetic Robot Expressive"
      :filename "RobotExpressive.glb"
      :category "Character"
      :file-size 463988
      :vital-hp 6
      :animations ["Idle" "Walking" "Running" "Dance" "Death" "Sitting" "Standing" "Jump" "Yes" "No" "Wave" "Punch" "ThumbsUp"]
      :description "Rigged humanoid robot with 13 rich skeletal animation clips."}
    :cesium_man
    @{:name "Cesium Humanoid Explorer"
      :filename "CesiumMan.glb"
      :category "Character"
      :file-size 490956
      :vital-hp 6
      :animations ["Anim_0_Walk"]
      :description "Rigged humanoid walker with standard bone hierarchy and walking cycle."}
    :fox_quadruped
    @{:name "Sovereign Fox Quadruped"
      :filename "Fox.glb"
      :category "Creature"
      :file-size 162852
      :vital-hp 6
      :animations ["Survey" "Walk" "Run"]
      :description "Rigged quadruped mammal with Survey, Walk, and Run animation cycles."}})

(def CAMERA-PRESETS-CATALOG
  @{:perspective_action_follow
    @{:name "3rd Person Action Perspective (Orbital SpringArm)"
      :projection "PERSPECTIVE"
      :pitch-deg -20.0
      :yaw-deg 0.0
      :distance 8.0
      :fov 75.0
      :vital-hp 6
      :script "res://scripts/PerspectiveCameraController.gd"
      :template "res://scenes/templates/CameraPerspectiveRig.tscn"}
    :perspective_cinematic_wide
    @{:name "Cinematic Wide-Angle Perspective"
      :projection "PERSPECTIVE"
      :pitch-deg -12.0
      :yaw-deg 15.0
      :distance 10.5
      :fov 85.0
      :vital-hp 6
      :script "res://scripts/PerspectiveCameraController.gd"
      :template "res://scenes/templates/CameraPerspectiveRig.tscn"}
    :orthographic_true_isometric
    @{:name "True Isometric Tactical Arena (35.264° / 45°)"
      :projection "ORTHOGRAPHIC"
      :pitch-deg -35.264
      :yaw-deg 45.0
      :distance 20.0
      :ortho-size 14.0
      :vital-hp 6
      :script "res://scripts/OrthographicCameraController.gd"
      :template "res://scenes/templates/CameraOrthographicRig.tscn"}
    :orthographic_dimetric_military
    @{:name "Military Dimetric Projection (30° / 45°)"
      :projection "ORTHOGRAPHIC"
      :pitch-deg -30.0
      :yaw-deg 45.0
      :distance 20.0
      :ortho-size 14.0
      :vital-hp 6
      :script "res://scripts/OrthographicCameraController.gd"
      :template "res://scenes/templates/CameraOrthographicRig.tscn"}
    :orthographic_topdown_tactical
    @{:name "Top-Down Tactical Hex Grid (89.9°)"
      :projection "ORTHOGRAPHIC"
      :pitch-deg -89.9
      :yaw-deg 0.0
      :distance 25.0
      :ortho-size 18.0
      :vital-hp 6
      :script "res://scripts/OrthographicCameraController.gd"
      :template "res://scenes/templates/CameraOrthographicRig.tscn"}
    :dual_mode_rig
    @{:name "Unified DualCameraRig3D (Seamless Persp <-> Ortho)"
      :projection "DUAL_HYBRID"
      :pitch-deg -25.0
      :yaw-deg 0.0
      :distance 8.0
      :fov 75.0
      :ortho-size 14.0
      :vital-hp 6
      :script "res://scripts/DualCameraRig3D.gd"
      :template "res://scenes/templates/DualCameraRig3D.tscn"}})

(defn calculate-ortho-size-from-fov
  "Calculates equivalent orthographic size from perspective FOV (deg) and distance: size = 2 * dist * tan(fov / 2)"
  [fov-deg distance]
  (let [rad (* fov-deg (/ 3.141592653589793 180.0))
        half-rad (/ rad 2.0)
        tan-val (math/tan half-rad)]
    (* 2.0 distance tan-val)))

(defn calculate-perspective-fov-from-ortho
  "Calculates equivalent perspective FOV (deg) from orthographic size and distance: fov = 2 * atan(size / (2 * dist))"
  [ortho-size distance]
  (if (<= distance 0.001)
    75.0
    (let [val (/ ortho-size (* 2.0 distance))
          atan-val (math/atan val)
          deg (* (* 2.0 atan-val) (/ 180.0 3.141592653589793))]
      deg)))

(defn evaluate-camera-projection-aspect
  "Computes projection focal length scales for width and height"
  [fov-or-size aspect is-perspective]
  (if is-perspective
    (let [rad (* fov-or-size (/ 3.141592653589793 180.0))
          tan-half (math/tan (/ rad 2.0))]
      @{:focal-y (/ 1.0 tan-half)
        :focal-x (/ 1.0 (* aspect tan-half))
        :vital-hp VITAL-MAX-HP})
    @{:scale-y (/ 2.0 fov-or-size)
      :scale-x (/ 2.0 (* aspect fov-or-size))
      :vital-hp VITAL-MAX-HP}))

(defn validate-camera-vital-invariant
  "Enforces that camera entity vital HP cannot exceed VITAL-MAX-HP (6)"
  [hp]
  (if (> hp VITAL-MAX-HP)
    VITAL-MAX-HP
    (if (< hp 1) 1 hp)))
