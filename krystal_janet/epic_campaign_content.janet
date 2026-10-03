# ==============================================================================
# KRYSTAL-STACK: EPIC CAMPAIGN SAGA & MISSION GENERATION DSL (JANET)
# ==============================================================================
# Defines:
#   1. Canonical campaign chapters: Desert Caravan, Totem Tempest, Sky Citadel.
#   2. Threat scaling, hazard multipliers and procedural mission parameters.
#   3. Vital 6 Max HP invariant compliance across all mission encounters.
#   4. Faction codex descriptors and aerial tactical synergies.
# ==============================================================================

(def VITAL-MAX-HP 6)

(def CAMPAIGN-CHAPTERS
  [{:id "chapter_1_desert_caravan"
    :title "Kapitola 1: Krádež Aéterového Plameňa"
    :subtitle "Nočný výsadok na padákoch do kaňonov a prepad karavány"
    :recommended-tier 1
    :altitude-tier "canyon_ground"
    :base-reward-nuggets 250
    :vital-max-hp VITAL-MAX-HP}

   {:id "chapter_2_totem_tempest"
    :title "Kapitola 2: Búrka Totémov a Rozštiepenie Sektora"
    :subtitle "Letecký súboj na rogalách cez pole rozrezaných anomálií"
    :recommended-tier 2
    :altitude-tier "skybridge_midways"
    :base-reward-nuggets 450
    :vital-max-hp VITAL-MAX-HP}

   {:id "chapter_3_citadel_siege"
    :title "Kapitola 3: Obliehanie Balónovej Citadely"
    :subtitle "Strmá paľba 240mm mínometu z vznášajúceho sa ostrova a hák"
    :recommended-tier 3
    :altitude-tier "stratospheric_citadel"
    :base-reward-nuggets 1200
    :vital-max-hp VITAL-MAX-HP}])

(defn compute-mission-threat-reward
  "Calculates total gold and nuggets reward scaled by threat level and vital risk."
  [base-nuggets threat-level remaining-hp]
  (let [threat-clamped (min 5 (max 1 threat-level))
        threat-multiplier (+ 1.0 (* (- threat-clamped 1) 0.45))
        hp-risk-bonus (/ (- (+ VITAL-MAX-HP 1) remaining-hp) (double VITAL-MAX-HP))
        scaled-nuggets (math/round (* base-nuggets threat-multiplier (+ 1.0 (* hp-risk-bonus 0.25))))]
    {:threat-level threat-clamped
     :threat-multiplier threat-multiplier
     :scaled-nuggets scaled-nuggets
     :vital-max-hp VITAL-MAX-HP}))

(defn evaluate-aerial-squad-synergy
  "Computes tactical combat rating of combined sky-island mortar and Rogallo wing squad."
  [mortar-caliber-mm rogallo-count parachute-troopers]
  (let [artillery-power (* (/ mortar-caliber-mm 100.0) 3.2)
        mobility-power (* rogallo-count 2.4)
        drop-power (* parachute-troopers 1.8)
        total-power (+ artillery-power mobility-power drop-power)]
    {:combined-power-rating (math/round total-power)
     :artillery-share artillery-power
     :aerial-mobility-share mobility-power
     :drop-insertion-share drop-power}))

(defn validate-unit-vital-invariant
  "Ensures any unit's HP is bounded strictly between 0 and 6."
  [current-hp incoming-damage]
  (let [safe-initial (min VITAL-MAX-HP (max 0 current-hp))
        clamped-dmg (max 0 incoming-damage)
        remaining-hp (max 0 (- safe-initial clamped-dmg))]
    {:valid (and (<= safe-initial VITAL-MAX-HP) (<= remaining-hp VITAL-MAX-HP))
     :initial-hp safe-initial
     :damage-applied clamped-dmg
     :remaining-hp remaining-hp}))
