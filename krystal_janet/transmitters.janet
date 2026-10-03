# ==============================================================================
# KRYSTAL-STACK: JANET BOT TRANSMITTER & COSMOLOGICAL METRICS DSL
# ==============================================================================
# Implements:
#   1. Bot transmitter channel streaming and telemetry serialization.
#   2. Premium step-by-step travel mode with time-travel rewind debug functions.
#   3. Nocturnal sky atmospheric calculations and celestial spirit rays.
#   4. Thermodynamic weather entropy and cataclysm detection.
#   5. Five cosmological planes (Peklo, Jaskyne, Aréna, Aéter, Nebo).
#   6. Twelve Apostles and Angelic Guardians divine patronage queries.
# ==============================================================================

(def *transmitter-version* "KRYSTAL-JANET-TRANSMITTER-2.0")

# ── Cosmological Plane Definitions ───────────────────────────────────────────
(def COSMOLOGICAL-PLANES
  {:inferno-tartarus {:tier -2 :name "Peklo (Tartarus)" :entropy 0.95 :ambient-color "#ff1a1a"}
   :caves-necropolis {:tier -1 :name "Podzemné Jaskyne a Zrkadlá" :entropy 0.55 :ambient-color "#1c2b36"}
   :mortal-arena     {:tier  0 :name "Mortal Realm (Poslední Kmen)" :entropy 0.30 :ambient-color "#66fcf1"}
   :aether-sky       {:tier  1 :name "Oblačná Sféra (Aéterová Obloha)" :entropy 0.15 :ambient-color "#a8e6cf"}
   :empyrean-heaven  {:tier  2 :name "Nebeský Trón (Empyrean Heaven)" :entropy 0.00 :ambient-color "#ffd700"}})

# ── Bot Transmitter Constructor ──────────────────────────────────────────────
(defn create-bot-transmitter
  "Creates a stateful bot transmitter with telemetry buffer and step memory."
  [bot-id channel-freq]
  @{:bot-id bot-id
    :channel-freq channel-freq
    :step-index 0
    :premium-mode true
    :position [0.0 0.0 0.0]
    :waypoints []
    :history-stack @[]
    :spirit-light-intensity 1.0
    :cosmological-plane :mortal-arena
    :active-ward 6
    :status :connected})

# ── Step & Teleport Debug Functions ──────────────────────────────────────────
(defn step-forward
  "Executes one discrete step in premium travel mode."
  [transmitter delta-vec]
  (let [cur-pos (get transmitter :position)
        new-pos [(+ (get cur-pos 0) (get delta-vec 0))
                 (+ (get cur-pos 1) (get delta-vec 1))
                 (+ (get cur-pos 2) (get delta-vec 2))]
        prev-step (get transmitter :step-index)]
    # Push state to history for time-travel rewinds
    (array/push (get transmitter :history-stack)
                @{:step prev-step :position cur-pos :ward (get transmitter :active-ward)})
    (put transmitter :position new-pos)
    (put transmitter :step-index (+ prev-step 1))
    @{:success true :bot-id (get transmitter :bot-id) :step (get transmitter :step-index) :position new-pos}))

(defn rewind-step
  "Rewinds bot state to the previous recorded step."
  [transmitter]
  (let [hist (get transmitter :history-stack)]
    (if (> (length hist) 0)
      (let [last-state (array/pop hist)]
        (put transmitter :position (get last-state :position))
        (put transmitter :step-index (get last-state :step))
        (put transmitter :active-ward (get last-state :ward))
        @{:success true :rewound-to (get transmitter :step-index) :position (get transmitter :position)})
      @{:success false :error "HISTORY_EMPTY"})))

(defn debug-teleport
  "Teleports bot directly to target coordinates."
  [transmitter target-pos]
  (put transmitter :position target-pos)
  @{:success true :teleported-to target-pos :bot-id (get transmitter :bot-id)})

# ── Nocturnal Sky & Atmospheric Spirit Rays ──────────────────────────────────
(defn eval-nocturnal-sky
  "Evaluates nocturnal sky light beams and celestial atmospheric parameters."
  [moon-phase celestial-time]
  (let [scattering-factor (* 0.75 (+ 1.0 (math/sin celestial-time)))
        spirit-ray-count (math/floor (+ 3 (* 4 (math/cos (* celestial-time 0.5)))))
        aurora-hue (string/format "#%02x%02x%02x"
                                  (math/floor (* 80 (+ 1.0 (math/sin celestial-time))))
                                  (math/floor (* 120 (+ 1.0 (math/cos celestial-time))))
                                  (math/floor (* 100 (+ 1.0 (math/sin (* celestial-time 1.5))))))]
    @{:moon-phase moon-phase
      :scattering-factor scattering-factor
      :volumetric-rays spirit-ray-count
      :aurora-hue aurora-hue
      :celestial-status "NOCTURNAL_ASTRAL_HARMONY"}))

# ── Weather Entropy & Natural Catastrophe ────────────────────────────────────
(defn calculate-weather-entropy
  "Calculates atmospheric entropy and returns current storm/catastrophe status."
  [pressure-delta humidity wind-shear]
  (let [entropy-norm (math/min 1.0 (math/max 0.0 (/ (+ (* pressure-delta 0.4) (* humidity 0.3) (* wind-shear 0.3)) 100.0)))
        state (cond
                (< entropy-norm 0.20) :clear-starry-night
                (< entropy-norm 0.45) :druidic-verdant-rain
                (< entropy-norm 0.65) :acid-toxic-fog
                (< entropy-norm 0.85) :supercell-lightning-storm
                true                  :aetheric-cataclysm-rift)]
    @{:weather-entropy entropy-norm
      :weather-state state
      :catastrophe-active (>= entropy-norm 0.80)
      :storm-surge-power (math/round (* entropy-norm 10.0))}))

# ── Arbor & Mycorrhizal Network Query ────────────────────────────────────────
(defn query-tree-network
  "Queries interconnected mycorrhizal root network for mana transfer."
  [root-nodes-count total-sacred-groves]
  (let [mana-flow (* root-nodes-count 2.5)
        collective-ward (* total-sacred-groves 1.5)]
    @{:active-root-nodes root-nodes-count
      :sacred-groves total-sacred-groves
      :ley-line-mana-flow mana-flow
      :collective-ward-barrier collective-ward
      :network-status :intertwined-healthy}))
