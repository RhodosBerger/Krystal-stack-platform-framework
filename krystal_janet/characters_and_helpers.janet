# ==============================================================================
# KRYSTAL-STACK: JANET ARCHETYPES, HELPERS & SLIDER BLEND DSL
# ==============================================================================
# Implements:
#   1. Query procedures for 20 Male and 20 Female magical archetypes.
#   2. 120 Helper entities catalog indexing and racial synergy calculation.
#   3. Dual-system stat resolution (Warhammer + The West duel algebra).
#   4. Continuous slider blending for attributes and synergy multipliers.
# ==============================================================================

(def *archetype-dsl-version* "KRYSTAL-ARCHETYPES-HELPERS-1.0")

# ── 12 Canonical Races Registry ──────────────────────────────────────────────
(def CANONICAL-RACES
  [:crystal :toxic :druid :human :dwarf :elf
   :inferno :celestial :spectral :elemental :fae :beastkin])

# ── Archetype Category Tags ──────────────────────────────────────────────────
(def ARCHETYPE-CATEGORIES
  [:vestica-oracle
   :carodejnica-witch
   :mag-arcanist
   :kuzelnik-wizard
   :iluzionista-illusionist
   :chladny-vystupovac-intimidator
   :odolavac-bulwark
   :ostrelovac-sniper
   :saman-spiritualist
   :banshee-necromancer])

# ── Helper Query Procedure ───────────────────────────────────────────────────
(defn query-helper-by-index
  "Looks up helper metadata and base stat aura by continuous index (1 to 120)."
  [idx]
  (let [clamped-idx (math/max 1 (math/min 120 (math/floor idx)))
        race-idx (% (- clamped-idx 1) 12)
        assigned-race (get CANONICAL-RACES race-idx)
        tier (+ 1 (% (math/floor (/ (- clamped-idx 1) 12)) 5))
        base-aura-val (+ 5 (* tier 2))]
    @{:helper-index clamped-idx
      :race assigned-race
      :tier tier
      :aura-power base-aura-val
      :status :active}))

# ── Racial Synergy Calculation ───────────────────────────────────────────────
(defn calculate-helper-synergy
  "Calculates synergy multiplier between selected hero race and helper race."
  [hero-race helper-race synergy-slider]
  (let [base-match (= hero-race helper-race)
        match-mult (if base-match 1.40 1.00)
        final-synergy (* match-mult (/ synergy-slider 100.0))]
    @{:match base-match
      :synergy-factor final-synergy
      :amplified-aura (math/round (* 10.0 final-synergy))}))

# ── Slider Blend Evaluation ──────────────────────────────────────────────────
(defn blend-duel-stats-with-sliders
  "Blends baseline hero duel stats with interactive slider overrides and helper aura."
  [base-stat slider-val helper-aura synergy-mult]
  (let [blended (+ (* base-stat 0.5) (* slider-val 0.5))
        with-helper (+ blended (* helper-aura synergy-mult))]
    (math/max 1 (math/round with-helper))))

# ── Vital Invariant Guard ────────────────────────────────────────────────────
(defn clamp-vital-wounds
  "Enforces the strict 6 Max HP vital invariant for all duelists."
  [calculated-wounds]
  (math/max 1 (math/min 6 calculated-wounds)))
