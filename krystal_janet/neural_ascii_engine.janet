# ============================================================================
# Krystal-Stack Janet Engine: Neural ASCII Engine & 3D Raymarcher
# ============================================================================
# Implements real-time 3D Signed Distance Field (SDF) sphere tracing,
# numerical surface normals, Phong lighting, and TrueColor ANSI ASCII rasterization.

(import ./krystal_sdf :as sdf)

(def PALETTE-CYBERPUNK [" " "░" "▒" "▓" "█"])
(def PALETTE-BLUEPRINT ["-" "/" "|" "\\\\"])
(def PALETTE-DENSITY " .:-=+*#%@")
(def PALETTE-MATRIX "0123456789ABCDEF")

(defn create-camera [&opt pos target fov]
  "Constructs a 3D pinhole camera."
  (let [p (or pos [0.0 0.0 -3.2])
        tgt (or target [0.0 0.0 0.0])
        f (or fov 60.0)
        forward (sdf/vec3-normalize (sdf/vec3-sub tgt p))
        world-up [0.0 1.0 0.0]
        right (sdf/vec3-normalize
                [(- (* (forward 1) (world-up 2)) (* (forward 2) (world-up 1)))
                 (- (* (forward 2) (world-up 0)) (* (forward 0) (world-up 2)))
                 (- (* (forward 0) (world-up 1)) (* (forward 1) (world-up 0)))])
        up (sdf/vec3-normalize
             [(- (* (right 1) (forward 2)) (* (right 2) (forward 1)))
              (- (* (right 2) (forward 0)) (* (right 0) (forward 2)))
              (- (* (right 0) (forward 1)) (* (right 1) (forward 0)))])]
    @{:pos p
      :forward forward
      :right right
      :up up
      :fov f}))

(defn rotate-y [p theta]
  (let [c (math/cos theta)
        s (math/sin theta)]
    [(+ (* (p 0) c) (* (p 2) s))
     (p 1)
     (- (* (p 2) c) (* (p 0) s))]))

(defn rotate-x [p theta]
  (let [c (math/cos theta)
        s (math/sin theta)]
    [(p 0)
     (- (* (p 1) c) (* (p 2) s))
     (+ (* (p 1) s) (* (p 2) c))]))

(defn scene-sdf [p t]
  "Evaluates composite SDF scene (rotating torus + pulsating core)."
  (let [p-rot (rotate-x (rotate-y p (* t 1.2)) (* t 0.8))
        d-torus (sdf/sdf-torus p-rot 1.0 0.38)
        core-r (+ 0.55 (* 0.12 (math/sin (* t 3.5))))
        d-core (sdf/sdf-sphere p core-r)]
    (sdf/smin d-torus d-core 0.28)))

(defn calc-normal [p t]
  "Estimates 3D surface normal gradient via central differences."
  (let [eps 0.003
        d (scene-sdf p t)
        nx (- (scene-sdf [(+ (p 0) eps) (p 1) (p 2)] t) d)
        ny (- (scene-sdf [(p 0) (+ (p 1) eps) (p 2)] t) d)
        nz (- (scene-sdf [(p 0) (p 1) (+ (p 2) eps)] t) d)]
    (sdf/vec3-normalize [nx ny nz])))

(defn raymarch [ro rd t max-steps max-dist]
  "Performs numerical sphere-tracing along ray ro + dist * rd."
  (var dist 0.0)
  (var hit false)
  (var steps-taken 0)
  (for s 0 max-steps
    (let [p (sdf/vec3-add ro (sdf/vec3-scale rd dist))
          d (scene-sdf p t)]
      (set steps-taken (+ s 1))
      (if (< d 0.002)
        (do (set hit true) (break))
        (do
          (set dist (+ dist d))
          (when (> dist max-dist) (break))))))
  {:hit hit :dist dist :steps steps-taken})

(defn render-frame [cols rows t mode &opt max-steps]
  "Renders a full 2D grid into an ASCII string buffer with TrueColor ANSI codes."
  (let [steps (or max-steps 36)
        ro [0.0 0.0 -3.2]
        aspect (/ (* cols 0.5) rows)
        light-dir (sdf/vec3-normalize [0.577 0.577 -0.577])
        buffer @[]]

    (for y 0 rows
      (let [v (- 1.0 (/ (* 2.0 y) rows))
            row-chars @[]]
        (for x 0 cols
          (let [u (* (- (/ (* 2.0 x) cols) 1.0) aspect)
                rd (sdf/vec3-normalize [u v 1.8])
                res (raymarch ro rd t steps 10.0)]
            (if (res :hit)
              (let [hit-p (sdf/vec3-add ro (sdf/vec3-scale rd (res :dist)))
                    n (calc-normal hit-p t)
                    diff (max 0.0 (sdf/vec3-dot n light-dir))
                    norm-y (max 0.0 (min 1.0 (+ (* (n 1) 0.5) 0.5)))

                    # TrueColor lighting based on mode
                    r (math/floor (min 255 (* (+ 0.1 (* diff 0.9)) (if (= mode :CYBERPUNK) 0 255))))
                    g (math/floor (min 255 (* (+ 0.2 (* diff 0.8)) (if (= mode :CYBERPUNK) 240 (* norm-y 200)))))
                    b (math/floor (min 255 (* (+ 0.3 (* diff 0.7)) 255)))

                    # Character glyph selection
                    char-idx (math/floor (* diff 4.99))
                    glyph (if (= mode :CYBERPUNK)
                            (PALETTE-CYBERPUNK char-idx)
                            (string/slice PALETTE-DENSITY char-idx (+ char-idx 1)))]
                (array/push row-chars (string/format "\e[38;2;%d;%d;%dm%s\e[0m" r g b glyph)))
              # Background empty space
              (array/push row-chars " "))))
        (array/push buffer (string/join row-chars ""))))
    (string/join buffer "\n")))
