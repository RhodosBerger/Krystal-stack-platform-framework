# ==============================================================================
# KRYSTAL-STACK: COMBINATORIAL MOVES & MULLIGAN PHASE DSL (JANET)
# ==============================================================================
# Defines:
#   1. Mulligan phase card replacement & deck exchange rules.
#   2. Canonical Mulligan cards: Ladova Gula, Pohyb s Jednotkou, Spojovaci Krystal, Zamrznutie.
#   3. Combinatorial tactical move space (256,000 = 2^18 branches across hex map).
#   4. Connecting crystal conduit reachability & unit escort movement.
# ==============================================================================

(def COMBINATORIAL-TOTAL-PERMUTATIONS 262144) # 256k states (2^18)
(def VITAL-MAX-HP 6)

(def CANONICAL-MULLIGAN-CARDS
  {:ladova-gula
   {:id "ladova_gula"
    :name "ĽADOVÁ GUĽA"
    :cost 1
    :power 2
    :range-stat 1.5
    :element "crystal"
    :desc "Zaútoč na nepriateľa, ak už je spomalený, zakoreníš ho"
    :status-combo "slow-to-root"}

   :pohyb-s-jednotkou
   {:id "pohyb_s_jednotkou"
    :name "POHYB S JEDNOTKOU"
    :cost 1
    :power 2
    :range-stat 1.5
    :element "tactical"
    :desc "Pohni sa o 1 s hrdinom a môžeš zobrať so sebou 1 jednotku"
    :escort-capacity 1}

   :spojovaci-krystal
   {:id "spojovaci_krystal"
    :name "SPOJOVACÍ KRYŠTÁL"
    :cost 2
    :power 2
    :range-stat 1.5
    :element "crystal-conduit"
    :desc "Každý kryštál, ktorý má dosah na iný kryštál, môže svoju akciu zahrať na susedné pole ako spojovací kryštál"
    :conduit-hop-bonus 2}

   :zamrznutie
   {:id "zamrznutie"
    :name "ZAMRZNUTIE"
    :cost 2
    :power 2
    :range-stat 1.5
    :element "crystal-freeze"
    :desc "Znehybni kryštálové pole nepriateľa a zmraz jednotky v dosahu"
    :aoe-freeze-radius 2}})

(defn evaluate-mulligan-exchange
  "Calculates new hand composition after returning selected cards to bottom of deck."
  [hand-card-ids return-card-ids draw-pile-count]
  (let [keep-count (- (length hand-card-ids) (length return-card-ids))
        exchange-count (length return-card-ids)
        available-pool (max 0 (- draw-pile-count exchange-count))]
    {:keep-count keep-count
     :exchange-count exchange-count
     :remaining-draw-pool available-pool
     :mulligan-valid (<= exchange-count (length hand-card-ids))}))

(defn calculate-conduit-reach
  "Evaluates hex reach when connecting crystals daisy-chain across the sector map."
  [base-range crystal-nodes-count]
  (+ base-range (* crystal-nodes-count 2.0)))

(defn compute-combinatorial-score
  "Heuristic utility function scoring one move path out of the 256k space."
  [damage-potency control-hexes escort-safety-factor conduit-length]
  (+ (* damage-potency 2.5)
     (* control-hexes 1.8)
     (* escort-safety-factor 3.0)
     (* conduit-length 1.2)))
