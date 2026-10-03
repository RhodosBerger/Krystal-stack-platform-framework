# ============================================================================
# Krystal-Stack Janet Engine: Signed Distance Functions & Blender Modifiers
# ============================================================================
# Implements 3D analytical primitives, polynomial smooth CSG operations,
# and procedural modifier stacks using immutable tuples and functional macros.

# --- Vector Operations on Immutable 3D Tuples [x y z] ---

(defn vec3-add [a b]
  [(+ (a 0) (b 0))
   (+ (a 1) (b 1))
   (+ (a 2) (b 2))])

(defn vec3-sub [a b]
  [(- (a 0) (b 0))
   (- (a 1) (b 1))
   (- (a 2) (b 2))])

(defn vec3-scale [v s]
  [(* (v 0) s)
   (* (v 1) s)
   (* (v 2) s)])

(defn vec3-dot [a b]
  (+ (* (a 0) (b 0))
     (* (a 1) (b 1))
     (* (a 2) (b 2))))

(defn vec3-length [v]
  (math/sqrt (vec3-dot v v)))

(defn vec3-normalize [v]
  (let [len (vec3-length v)]
    (if (> len 1e-6)
      (vec3-scale v (/ 1.0 len))
      [0.0 1.0 0.0])))

# --- Polynomial Smooth Minimum & CSG Operators ---

(defn smin [d1 d2 k]
  (let [h (max 0.0 (min 1.0 (+ 0.5 (/ (* 0.5 (- d2 d1)) k))))]
    (- (+ (* d2 (- 1.0 h)) (* d1 h)) (* k h (- 1.0 h)))))

(defn smax [d1 d2 k]
  (- (smin (- d1) (- d2) k)))

(defn smooth-diff [d1 d2 k]
  (smax d1 (- d2) k))

# --- Analytical 3D Signed Distance Fields ---

(defn sdf-sphere [p r]
  (- (vec3-length p) r))

(defn sdf-box [p b]
  (let [qx (- (math/abs (p 0)) (b 0))
        qy (- (math/abs (p 1)) (b 1))
        qz (- (math/abs (p 2)) (b 2))
        outside [(max qx 0.0) (max qy 0.0) (max qz 0.0)]
        d-out (vec3-length outside)
        d-in (min (max qx (max qy qz)) 0.0)]
    (+ d-out d-in)))

(defn sdf-cylinder [p r h]
  (let [d-radial (- (math/hypot (p 0) (p 2)) r)
        d-axial  (- (math/abs (p 1)) (* h 0.5))
        d-out (math/hypot (max d-radial 0.0) (max d-axial 0.0))
        d-in (min (max d-radial d-axial) 0.0)]
    (+ d-out d-in)))

(defn sdf-torus [p r-major r-minor]
  (let [q [( - (math/hypot (p 0) (p 2)) r-major) (p 1)]
        d (- (math/hypot (q 0) (q 1)) r-minor)]
    d))

(defn sdf-octahedron [p s]
  (let [px (math/abs (p 0))
        py (math/abs (p 1))
        pz (math/abs (p 2))]
    (* (- (+ px py pz) s) 0.57735027)))

# --- Procedural Blender Modifier Stack Pipeline ---

(defn mod-mirror-x [p]
  [(math/abs (p 0)) (p 1) (p 2)])

(defn mod-mirror-z [p]
  [(p 0) (p 1) (math/abs (p 2))])

(defn mod-array-radial [p count radius]
  (let [angle-step (/ (* 2.0 math/pi) count)
        theta (math/atan2 (p 2) (p 0))
        sector (math/floor (+ (/ theta angle-step) 0.5))
        folded-angle (- theta (* sector angle-step))
        r (math/hypot (p 0) (p 2))
        rx (- (* r (math/cos folded-angle)) radius)
        rz (* r (math/sin folded-angle))]
    [rx (p 1) rz]))

(defn mod-twist-y [p rate]
  (let [theta (* (p 1) rate)
        cos-t (math/cos theta)
        sin-t (math/sin theta)
        rx (- (* (p 0) cos-t) (* (p 2) sin-t))
        rz (+ (* (p 0) sin-t) (* (p 2) cos-t))]
    [rx (p 1) rz]))

(defn mod-displace-harmonic [p amp freq]
  (let [harmonic (+ (math/sin (* (p 0) freq))
                    (math/cos (* (p 1) freq))
                    (math/sin (* (p 2) freq)))]
    (* harmonic amp)))

# --- Composite Artifact Example: Alchemical Monolith in Janet ---

(defn evaluate-alchemical-monolith [p t]
  # Stack: 1. Plinth (box), 2. Spire (twisted octahedron), 3. Levitating Capstone
  (let [p-twist (mod-twist-y p 0.5)
        d-spire (sdf-octahedron [(p-twist 0) (- (p-twist 1) 0.4) (p-twist 2)] 1.2)
        d-plinth (sdf-box [p 0] [1.4 0.25 1.4])
        capstone-y (+ 2.2 (* (math/sin (* t 2.0)) 0.15))
        d-capstone (sdf-octahedron [p 0 (- (p 1) capstone-y) (p 2)] 0.45)
        d-merged (smin d-spire d-plinth 0.2)]
    (smin d-merged d-capstone 0.1)))
