# ==============================================================================
# KRYSTAL-STACK: JANET ZODIAC, GRIDS & OPTIC ZOOM DSL
# ==============================================================================
# Implements:
#   1. Celestial Zodiac constellations and sky rotation metrics.
#   2. Immunity system and status ailment damage soak procedures.
#   3. Grid dimensions (3x3, 2x5, 6x5, 12x9, 16:9, 21:19) and capacity metrics.
#   4. Dynamic optical zoom based on high-tier gear magnification.
#   5. Plus inventory expansion tokens and slot reservation algebra.
# ==============================================================================

(def *zodiac-dsl-version* "KRYSTAL-ZODIAC-GRIDS-1.0")

# ── Twelve Zodiac Constellations ─────────────────────────────────────────────
(def ZODIAC-HOUSES
  [:aries-baran
   :taurus-byk
   :gemini-blizenci
   :cancer-rak
   :leo-lev
   :virgo-panna
   :libra-vahy
   :scorpio-skorpion
   :sagittarius-strelec
   :capricorn-kozorozec
   :aquarius-vodnar
   :pisces-ryby])

# ── Grid Dimension Specifications ────────────────────────────────────────────
(def SUPPORTED-GRID-PRESETS
  {:grid-3x3   {:cols 3  :rows 3  :slots 9   :purpose "Core Runic Quickbelt"}
   :grid-2x5   {:cols 2  :rows 5  :slots 10  :purpose "Tactical Familiar Pouch"}
   :grid-6x5   {:cols 6  :rows 5  :slots 30  :purpose "Standard Backpack Grid"}
   :grid-12x9  {:cols 12 :rows 9  :slots 108 :purpose "Zodiac Grand Armory"}
   :grid-16x9  {:cols 16 :rows 9  :slots 144 :purpose "Widescreen Combat Arena"}
   :grid-21x19 {:cols 21 :rows 19 :slots 399 :purpose "Ultrawide Numerology Grid"}})

# ── Zodiac House by Celestial Angle ──────────────────────────────────────────
(defn calculate-zodiac-influence
  "Determines active celestial zodiac house based on nocturnal sky angle theta [0, 2*PI]."
  [sky-angle-rad]
  (let [norm-angle (math/abs (% sky-angle-rad (* 2.0 math/pi)))
        house-idx (math/floor (/ (* norm-angle 12.0) (* 2.0 math/pi)))
        clamped-idx (math/max 0 (math/min 11 house-idx))
        active-house (get ZODIAC-HOUSES clamped-idx)]
    @{:active-house active-house
      :angle-radians norm-angle
      :zenith-alignment (> (math/sin norm-angle) 0.85)
      :zodiac-power (+ 10 (math/round (* 15 (math/sin norm-angle))))}))

# ── Immunity Soak Procedure ──────────────────────────────────────────────────
(defn evaluate-immunity-soak
  "Mitigates ailment damage based on resistance percentage and ward status."
  [raw-potency resistance-pct has-immune-shield]
  (if has-immune-shield
    @{:status :immune :absorbed-potency raw-potency :final-potency 0}
    (let [soak-ratio (/ (math/max 0 (math/min 100 resistance-pct)) 100.0)
          mitigated (* raw-potency soak-ratio)
          final-pot (math/max 0 (math/round (- raw-potency mitigated)))]
      @{:status (if (= final-pot 0) :resisted :affected)
        :absorbed-potency (math/round mitigated)
        :final-potency final-pot})))

# ── Gear Optical Zoom Scaling ────────────────────────────────────────────────
(defn calculate-zoom-optic-magnification
  "Computes dynamic optical zoom factor and FOV reduction for better equipment."
  [gear-tier base-fov]
  (let [tier-clamped (math/max 1 (math/min 5 gear-tier))
        zoom-mult (case tier-clamped
                    1 1.0
                    2 1.5
                    3 2.5
                    4 4.0
                    5 8.0)
        effective-fov (/ base-fov zoom-mult)
        headshot-bonus (* (- zoom-mult 1.0) 8.0)]
    @{:gear-tier tier-clamped
      :zoom-magnification zoom-mult
      :effective-fov effective-fov
      :aim-precision-bonus (math/round headshot-bonus)
      :celestial-constellations-visible (>= tier-clamped 3)}))

# ── Plus Inventory Slots ─────────────────────────────────────────────────────
(defn calculate-inventory-plus-slots
  "Calculates total inventory slots including premium '+' slot expansion tokens."
  [base-capacity plus-tokens]
  (let [bonus-slots (* (math/max 0 plus-tokens) 5)
        total-slots (+ base-capacity bonus-slots)]
    @{:base-capacity base-capacity
      :plus-tokens plus-tokens
      :unlocked-plus-slots bonus-slots
      :total-capacity total-slots
      :aether-weightless-slots (math/min bonus-slots 20)}))
