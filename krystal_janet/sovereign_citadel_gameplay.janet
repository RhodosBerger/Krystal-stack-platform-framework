# ==============================================================================
# KRYSTAL-STACK JANET DSL: SOVEREIGN CITADEL TACTICAL DEFENSE SPECIFICATION
# ==============================================================================
# Derived from the Sovereign Cyber Fortress Artwork (wordpress_subdomain_security_shield.jpg)
# Invariant: VITAL-MAX-HP strictly preserved at 6 across all entities and shields.
# ==============================================================================

(def VITAL-MAX-HP 6)
(def GOLDEN-RATIO 1.61803398875)
(def INV-GOLDEN-RATIO 0.61803398875)

(def CITADEL-DEFENSIVE-PYLONS
  {:pylon-left
   {:id "pylon_left"
    :name "Perun's Slavic Mortar Pylon"
    :type :slavic-mortar
    :range-px 420.0
    :base-damage 3.0
    :cooldown-s 0.8
    :vital-max-hp VITAL-MAX-HP}
   :pylon-right
   {:id "pylon_right"
    :name "Athena's Tactical Beam Pylon"
    :type :hellenic-beam
    :range-px 480.0
    :base-damage 1.8
    :cooldown-s 0.4
    :vital-max-hp VITAL-MAX-HP}})

(def CANONICAL-CITADEL-CARDS
  {:card-aegis-wall
   {:name "Sovereign Aegis Wall"
    :mana-cost 2
    :archetype "Cryptographic Barrier"
    :effect-type :shield-boost
    :power 4.0}
   :card-perun-mortar
   {:name "Perun's Lightning Barrage"
    :mana-cost 3
    :archetype "Slavic Artillery"
    :effect-type :mortar-strike
    :power 4.0}
   :card-athena-freeze
   {:name "Athena's Tactical Freeze"
    :mana-cost 2
    :archetype "Hellenic Axiom"
    :effect-type :freeze-wave
    :power 3.5}
   :card-bbq-purge
   {:name "BBQ Firewall Purge"
    :mana-cost 4
    :archetype "WAF Core"
    :effect-type :firewall-purge
    :power 3.0}
   :card-ledger-surge
   {:name "Ledger Mana Surge"
    :mana-cost 1
    :archetype "Double-Entry Accounting"
    :effect-type :mana-surge
    :power 3.0}
   :card-vital-restore
   {:name "Cryptographic Repair"
    :mana-cost 3
    :archetype "Spinal Reflex"
    :effect-type :repair-vital
    :power 1.0}})

(def INVADER-ARCHETYPES
  {:sqli-ram {:base-hp 6.0 :speed 22.0 :dmg 2.0 :reward 2}
   :xss-swarm {:base-hp 2.5 :speed 55.0 :dmg 1.0 :reward 1}
   :rce-stalker {:base-hp 4.0 :speed 38.0 :dmg 2.5 :reward 3}
   :ddos-colossus {:base-hp 18.0 :speed 15.0 :dmg 4.0 :reward 5}})

(defn calculate-pylon-dps
  "Computes sustained damage per second delivered by a defensive bastion."
  [base-damage cooldown-s]
  (/ base-damage (max 0.1 cooldown-s)))

(defn evaluate-card-mana-efficiency
  "Calculates tactical value quotient per unit of mana expended."
  [power mana-cost]
  (/ power (max 1 mana-cost)))

(defn compute-wave-threat-budget
  "Calculates aggregate kinetic threat score for an incoming packet wave."
  [wave-num]
  (let [unit-count (+ 4 (* wave-num 2))
        base-weight (* (math/pow GOLDEN-RATIO 1.2) wave-num 10.0)]
    (+ (* unit-count 3.5) base-weight)))
