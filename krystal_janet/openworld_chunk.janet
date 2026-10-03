# ============================================================================
# Krystal-Stack Janet Engine: Open-World Chunk Evaluator & Fiber Streaming
# ============================================================================
# Implements multi-frequency terrain sampling, cooperative fiber chunk streaming,
# and high-speed ASCII camera raymarching.

(import ./krystal_sdf :as sdf)
(import ./antigravity_peg :as peg)

# --- Deterministic Math & Noise Primitives in Janet ---

(defn hash21 [x z seed]
  (let [xi (+ (math/floor x) seed)
        zi (+ (math/floor z) (* seed 31))
        n (band (bxor (* xi 374761393) (* zi 668265263)) 0x7fffffff)]
    (/ (% n 1000000) 1000000.0)))

(defn smoothstep [t]
  (let [tc (max 0.0 (min 1.0 t))]
    (* tc tc (- 3.0 (* 2.0 tc)))))

(defn value-noise-2d [x z seed]
  (let [x0 (math/floor x)
        z0 (math/floor z)
        x1 (+ x0 1.0)
        z1 (+ z0 1.0)
        tx (smoothstep (- x x0))
        tz (smoothstep (- z z0))
        v00 (hash21 x0 z0 seed)
        v10 (hash21 x1 z0 seed)
        v01 (hash21 x0 z1 seed)
        v11 (hash21 x1 z1 seed)
        vx0 (+ v00 (* tx (- v10 v00)))
        vx1 (+ v01 (* tx (- v11 v01)))]
    (+ vx0 (* tz (- vx1 vx0)))))

(defn fbm-terrain [x z octaves ridge-w seed]
  (var total 0.0)
  (var amp 1.0)
  (var freq 1.0)
  (var max-val 0.0)
  (for i 0 octaves
    (let [n (value-noise-2d (* x freq) (* z freq) (+ seed (* i 19)))
          r (- 1.0 (math/abs (- (* 2.0 n) 1.0)))
          sig (+ (* n (- 1.0 ridge-w)) (* r r ridge-w))]
      (set total (+ total (* sig amp)))
      (set max-val (+ max-val amp))
      (set amp (* amp 0.5))
      (set freq (* freq 2.0))))
  (if (> max-val 0.0) (/ total max-val) 0.0))

# --- Chunk Definition & Fiber Streaming ---

(def CHUNK-SIZE 16)

(defn create-chunk [chunk-x chunk-z params]
  "Evaluates a single terrain chunk grid inside a cooperative fiber."
  (let [grid @[]
        h-scale (params :height-scale)
        ridge-w (params :ridge-weight)
        oct (params :octaves)
        freq 0.08
        start-x (* chunk-x CHUNK-SIZE)
        start-z (* chunk-z CHUNK-SIZE)]
    (for lz 0 CHUNK-SIZE
      (let [row @[]]
        (for lx 0 CHUNK-SIZE
          (let [world-x (+ start-x lx)
                world-z (+ start-z lz)
                raw-h (fbm-terrain (* world-x freq) (* world-z freq) oct ridge-w 42)
                h (* (- (* raw-h 2.0) 1.0) h-scale)]
            (array/push row h)))
        (array/push grid row)
        # Yield to scheduler after processing each row to allow concurrent rendering!
        (yield :row-complete)))
    {:chunk-x chunk-x
     :chunk-z chunk-z
     :grid grid}))

(defn stream-chunks-cooperative [chunk-coords params]
  "Streams multiple chunks using Janet fibers without blocking the engine."
  (let [completed-chunks @[]]
    (each coord chunk-coords
      (let [f (fiber/new (fn [] (create-chunk (coord 0) (coord 1) params)))]
        # Resume fiber until completion
        (while (= (fiber/status f) :alive)
          (let [res (resume f)]
            (if (dictionary? res)
              (array/push completed-chunks res))))))
    completed-chunks))

# --- ASCII Raymarching Projection in Janet ---

(def ASCII-PALETTE " .:-=+*#%@")

(defn sample-world-height [x z params]
  (let [h-scale (params :height-scale)
        ridge-w (params :ridge-weight)
        oct (params :octaves)
        freq 0.08
        raw-h (fbm-terrain (* x freq) (* z freq) oct ridge-w 42)]
    (* (- (* raw-h 2.0) 1.0) h-scale)))

(defn render-ascii-view [params width height cam-x cam-y cam-z]
  (let [lines @[]
        aspect (* (/ width height) 0.5)]
    (for y 0 height
      (let [row @[]
            sy (- 1.0 (/ (* 2.0 y) (- height 1)))]
        (for x 0 width
          (let [sx (* (- (/ (* 2.0 x) (- width 1)) 1.0) aspect)
                # Primary ray
                rdx sx
                rdy (- sy 0.2)
                rdz 1.0
                rd-len (math/hypot rdx (math/hypot rdy rdz))
                nd-x (/ rdx rd-len)
                nd-y (/ rdy rd-len)
                nd-z (/ rdz rd-len)
                # March ray
                hit-char (var hc " ")]
            (var t 1.0)
            (var hit false)
            (while (and (< t 35.0) (not hit))
              (let [px (+ cam-x (* nd-x t))
                    py (+ cam-y (* nd-y t))
                    pz (+ cam-z (* nd-z t))
                    terrain-h (sample-world-height px pz params)
                    dist (- py terrain-h)]
                (if (< dist 0.05)
                  (do
                    (set hit true)
                    (let [norm-h (/ (+ terrain-h 3.0) 6.0)
                          idx (math/floor (* (max 0.0 (min 1.0 norm-h)) 9))]
                      (set hc (string/slice ASCII-PALETTE idx (+ idx 1)))))
                  (set t (+ t (max 0.2 (* dist 0.7)))))))
            (if (not hit)
              (set hc "·"))
            (array/push row hc)))
        (array/push lines (string/join row))))
    (string/join lines "\n")))

(defn main [& args]
  (print "=== Krystal-Stack Janet Engine: Open-World Procedural Generator ===")
  (let [prompt "vytvor kaňon s lávou, 6 rekurzií a veže"
        _ (print (string/format "Compiling prompt via Janet PEG: '%s'" prompt))
        ast (peg/parse-antigravity-prompt prompt)
        math-ir (peg/ast->mathematical-ir ast)]
    (print (string/format "-> Topography: %q" (ast :topography)))
    (print (string/format "-> Atmosphere: %q" (ast :atmosphere)))
    (print (string/format "-> Height Scale: %.2f" (math-ir :height-scale)))
    (print (string/format "-> Octaves: %d" (math-ir :octaves)))
    (print "\nStreaming 2x2 Open-World Chunks via Cooperative Fibers...")
    (let [chunks (stream-chunks-cooperative [[0 0] [0 1] [1 0] [1 1]] math-ir)]
      (print (string/format "Successfully evaluated %d chunks concurrently.\n" (length chunks))))
    (print "Rendering ASCII Camera Projection (64x14):")
    (let [frame (render-ascii-view math-ir 64 14 0.0 5.0 -12.0)]
      (print frame))))
