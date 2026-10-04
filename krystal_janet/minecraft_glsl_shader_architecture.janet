# ==============================================================================
# KRYSTAL-STACK: MINECRAFT GLSL SHADER ARCHITECTURE & PIPELINE DSL (JANET)
# ==============================================================================
# Defines:
#   1. Canonical Shader Pack Profiles (BSL, Complementary, SEUS PTGI, Continuum, Iris).
#   2. Deferred G-Buffer Pipeline Stages (gbuffers, shadows, voxel-gi, SSR, bloom).
#   3. LabPBR 1.3 Specular & Normal channel decoding rules.
#   4. Recommended Godot 4.x Plugins (Terrain3D, PhantomCamera, Zylann Voxel, LimboAI).
#   5. Invariants: VITAL-MAX-HP = 6, GOLDEN-RATIO = 1.61803398875.
# ==============================================================================

(def VITAL-MAX-HP 6)
(def GOLDEN-RATIO 1.6180339887)
(def INV-GOLDEN-RATIO (/ 1.0 GOLDEN-RATIO))

(def CANONICAL-SHADER-PACKS
  {:bsl-v8
   {:name "BSL Shaders v8.2" :author "Capt Tatsu" :target-fps 58 :vital-max-hp VITAL-MAX-HP}
   :complementary-reimagined
   {:name "Complementary Reimagined" :author "EminGT" :target-fps 54 :vital-max-hp VITAL-MAX-HP}
   :seus-ptgi-hrr
   {:name "SEUS PTGI HRR (Path Traced GI)" :author "Sonic Ether" :target-fps 38 :vital-max-hp VITAL-MAX-HP}
   :continuum-cinematic
   {:name "Continuum 2.0 RT" :author "Continuum Graphics" :target-fps 32 :vital-max-hp VITAL-MAX-HP}
   :iris-vanilla-plus
   {:name "Iris Vanilla+ Ultralight" :author "Krystal Stack Team" :target-fps 75 :vital-max-hp VITAL-MAX-HP}})

(def CANONICAL-DEFERRED-STAGES
  {:gbuffers-terrain {:vram-mb 64.0 :gpu-ms 2.8 :is-compute false}
   :gbuffers-water   {:vram-mb 32.0 :gpu-ms 1.2 :is-compute false}
   :shadow-cascades  {:vram-mb 48.0 :gpu-ms 3.1 :is-compute false}
   :voxel-gi-dda     {:vram-mb 96.0 :gpu-ms 4.5 :is-compute true}
   :volumetric-fog   {:vram-mb 24.0 :gpu-ms 2.1 :is-compute true}
   :ssr-reflections  {:vram-mb 32.0 :gpu-ms 2.4 :is-compute false}
   :post-composite   {:vram-mb 16.0 :gpu-ms 1.6 :is-compute false}})

(def CANONICAL-GODOT-PLUGINS
  {:terrain-3d
   {:name "Terrain3D" :category "terrains" :asset-lib "TokisanGames/Terrain3D"}
   :phantom-camera
   {:name "Phantom Camera" :category "cameras" :asset-lib "ramok/phantom-camera"}
   :zylann-voxel
   {:name "Godot Voxel Tools" :category "voxels" :asset-lib "Zylann/godot_voxel"}
   :limbo-ai
   {:name "LimboAI" :category "ai-npcs" :asset-lib "limbonaut/limboai"}
   :gpu-particles
   {:name "GPUParticles3D Sub-Emitters" :category "vfx" :asset-lib "Godot Core"}})

(defn decode-labpbr-smoothness
  "Converts perceptual smoothness [0..1] to physical linear roughness."
  [smoothness]
  (let [clamped (max 0.0 (min 1.0 smoothness))
        one-minus (- 1.0 clamped)]
    (* one-minus one-minus)))

(defn compute-gerstner-frequency
  "Computes deep water wave frequency based on wavelength."
  [wavelength]
  (let [safe-len (max 0.01 wavelength)]
    (/ (* 2.0 3.14159265) safe-len)))

(defn evaluate-pipeline-budget
  "Evaluates total frame time and VRAM usage against 60 FPS target."
  [total-gpu-ms total-vram-mb]
  (let [target-budget-ms 16.666
        fps (/ 1000.0 (max 0.001 total-gpu-ms))
        is-60fps (>= fps 59.5)]
    {:fps fps
     :is-60fps is-60fps
     :vram-mb total-vram-mb
     :vital-max-hp VITAL-MAX-HP}))
