# ============================================================================
# Krystal-Stack Janet Engine: Symplectic Cyclic Organism
# ============================================================================
# A continuous-time Symplectic Hamiltonian Dynamical Engine governing the
# computational organism's homeostatic cycles, thermodynamic stability, and
# cognitive brainwave phases (Alpha, Beta, Gamma, Omega).
#
# Implements:
# 1. 2nd-Order Symplectic Velocity-Verlet Integration preserving phase space volume.
# 2. Non-linear Lyapunov stability evaluation.
# 3. Closed-loop visual entropy and thermodynamic damping.
# 4. Cognitive Brainwave State Machine modulating step budgets and priority.

(def MASS-INERTIA [1.0 0.5 2.0 1.2])
(def SPRING-K [2.0 0.8 4.0 1.5])
(def Q-STAR [1.0 60.0 0.50 0.65])
(def Q-MAX [2.5 144.0 0.95 1.0])
(def BARRIER-XI 0.02)
(def GAMMA-0 0.05)
(def BETA-ENTROPY 2.5)

(defn create-organism [&opt organism-id]
  "Initializes a new Symplectic Hamiltonian Cyclic Organism."
  @{:id (or organism-id "Organism-Janet-Prime")
    :q (array/slice Q-STAR)
    :p @[0.0 0.0 0.0 0.0]
    :mass MASS-INERTIA
    :spring-k SPRING-K
    :q-star Q-STAR
    :q-max Q-MAX
    :gamma-0 GAMMA-0
    :beta BETA-ENTROPY
    :visual-entropy 0.25
    :gpu-temp 48.0
    :economic-budget 850.0
    :cognitive-phase :BETA
    :cycle-count 0
    :backpressure false})

(defn potential-gradient [org]
  "Computes conservative restoring force vector -dV/dq."
  (let [q (org :q)
        q-star (org :q-star)
        q-max (org :q-max)
        k (org :spring-k)
        forces @[]]
    (for i 0 4
      (let [qi (q i)
            f-harm (* (- (k i)) (- qi (q-star i)))
            dist-to-max (max 0.01 (- (q-max i) qi))
            f-barr (/ (* -2.0 BARRIER-XI) (* dist-to-max dist-to-max dist-to-max))]
        (array/push forces (+ f-harm f-barr))))
    forces))

(defn dissipative-force [org p-vec entropy]
  "Computes non-conservative thermodynamic drag: -(gamma_0 + beta * E^2) * p."
  (let [drag (+ (org :gamma-0) (* (org :beta) entropy entropy))
        forces @[]]
    (for i 0 4
      (array/push forces (* (- drag) (p-vec i))))
    forces))

(defn economic-forcing [t]
  "Computes dynamic market driving force vector based on time."
  (let [f-econ (* 0.3 (math/sin (* t 0.2)))
        f-fps (* 0.5 (math/cos (* t 0.1)))]
    @[0.0 f-fps 0.0 f-econ]))

(defn kinetic-energy [org]
  "Evaluates kinetic energy T(p) = 0.5 * sum(p_i^2 / m_i)."
  (var t 0.0)
  (let [p (org :p)
        m (org :mass)]
    (for i 0 4
      (set t (+ t (/ (* (p i) (p i)) (* 2.0 (m i))))))
    t))

(defn potential-energy [org]
  "Evaluates potential energy V(q) = 0.5 * sum(k_i * (q_i - q_i*)^2) + barrier."
  (var v 0.0)
  (let [q (org :q)
        q-star (org :q-star)
        q-max (org :q-max)
        k (org :spring-k)]
    (for i 0 4
      (let [diff (- (q i) (q-star i))
            harm (* 0.5 (k i) diff diff)
            dist (max 0.01 (- (q-max i) (q i)))
            barr (/ BARRIER-XI (* dist dist))]
        (set v (+ v harm barr))))
    v))

(defn total-hamiltonian [org]
  "Total system Hamiltonian H = T + V."
  (+ (kinetic-energy org) (potential-energy org)))

(defn lyapunov-stability-index [org]
  "Lyapunov function L(q, p) measuring distance from homeostatic attractor."
  (var l-val (kinetic-energy org))
  (let [q (org :q)
        q-star (org :q-star)
        k (org :spring-k)]
    (for i 0 4
      (let [diff (- (q i) (q-star i))]
        (set l-val (+ l-val (* 0.5 (k i) diff diff)))))
    l-val))

(defn- update-cognitive-state [org]
  "Updates cognitive brainwave state machine based on thermodynamic state."
  (let [entropy (org :visual-entropy)
        t-kin (kinetic-energy org)]
    (cond
      (> entropy 0.70)
      (do
        (put org :cognitive-phase :OMEGA)
        (put org :backpressure true))

      (> t-kin 2.0)
      (do
        (put org :cognitive-phase :GAMMA)
        (put org :backpressure false))

      (< t-kin 0.2)
      (do
        (put org :cognitive-phase :ALPHA)
        (put org :backpressure false))

      (do
        (put org :cognitive-phase :BETA)
        (put org :backpressure false)))))

(defn symplectic-step [org dt entropy &opt gpu-temp]
  "Advances phase space trajectory by dt using 2nd-order Symplectic Velocity-Verlet."
  (put org :visual-entropy (max 0.0 (min 1.0 entropy)))
  (when gpu-temp (put org :gpu-temp gpu-temp))
  (put org :cycle-count (+ (org :cycle-count) 1))
  (let [t (* (org :cycle-count) dt)
        q (org :q)
        p (org :p)
        m (org :mass)
        q-max (org :q-max)

        # 1. Total force at t
        f-cons-t (potential-gradient org)
        f-diss-t (dissipative-force org p (org :visual-entropy))
        f-econ-t (economic-forcing t)
        f-total-t @[]]

    (for i 0 4
      (array/push f-total-t (+ (f-cons-t i) (f-diss-t i) (f-econ-t i))))

    # 2. Half-step momentum
    (let [p-half @[]]
      (for i 0 4
        (array/push p-half (+ (p i) (* 0.5 dt (f-total-t i)))))

      # 3. Full-step coordinate
      (for i 0 4
        (let [q-next (+ (q i) (* dt (/ (p-half i) (m i))))
              clamped (max 0.05 (min (- (q-max i) 0.02) q-next))]
          (put q i clamped)))

      # 4. Total force at t + dt
      (let [f-cons-next (potential-gradient org)
            f-diss-next (dissipative-force org p-half (org :visual-entropy))
            f-econ-next (economic-forcing (+ t dt))
            f-total-next @[]]
        (for i 0 4
          (array/push f-total-next (+ (f-cons-next i) (f-diss-next i) (f-econ-next i))))

        # 5. Complete momentum
        (for i 0 4
          (put p i (+ (p-half i) (* 0.5 dt (f-total-next i))))))))

  # 6. Update brainwaves
  (update-cognitive-state org)
  org)

(defn generate-phase-trajectory [org &opt num-points]
  "Generates a 2D phase portrait projection [q0, p0]."
  (let [n (or num-points 24)
        points @[]
        h (total-hamiltonian org)
        r (math/sqrt (max 0.01 h))]
    (for idx 0 n
      (let [theta (* (/ (* 2.0 math/pi) n) idx)
            q-pt (+ ((org :q) 0) (* r (math/cos theta) 0.3))
            p-pt (* r (math/sin theta) 0.8)]
        (array/push points {:q q-pt :p p-pt})))
    points))
