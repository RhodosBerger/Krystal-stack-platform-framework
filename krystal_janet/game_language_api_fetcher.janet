# ==============================================================================
# Krystal-Stack Janet DSL // Game Language API Fetcher & Tactical Controller
# ==============================================================================
# Encodes multilingual natural language command schemas, semantic intent
# categorizations, game API dispatch routing, and autonomous fetcher loop.
# Strict Invariant: VITAL-MAX-HP = 6
# ==============================================================================

(def VITAL-MAX-HP 6)
(def GOLDEN-RATIO 1.61803398875)
(def DEFAULT-HUB-PORT 8089)

(def COMMAND-TYPES
  [:CAST_CARD
   :TACTICAL_MOVE
   :WEATHER_ATMOSPHERE
   :CAMERA_CONTROL
   :ARTILLERY_FIRE
   :CITADEL_BUILD
   :SPAWN_ENTITY
   :ENVIRONMENT_HAZARD
   :MATCH_ESCALATION
   :QUERY_STATUS
   :UNKNOWN])

(def KNOWN-CARD-SYNONYMS
  @{"frost_shard" ["frost shard" "mrazivý črep" "mrazivy crep" "črep"]
    "crystal_shield" ["crystal shield" "kryštálový štít" "krystalovy stit" "štít"]
    "crystal_meteor" ["crystal meteor" "kryštálový meteor" "meteor"]
    "glacial_lance" ["glacial lance" "ľadovcová kopija" "ladovcova kopija" "kopija"]
    "orbital_hyper_lance" ["orbital hyper lance" "orbitálna hyperkopija" "hyperkopija"]
    "acid_slime" ["acid slime" "kyslý sliz" "kysly sliz" "sliz"]
    "venom_dart" ["venom dart" "jedovatá šípka" "jedovata sipka" "šípka"]
    "decay_strike" ["decay strike" "zuby rozkladu" "rozklad"]
    "toxic_cloud" ["toxic cloud" "toxický oblak" "oblak"]
    "bark_skin" ["bark skin" "dubová kôra" "dubova kora" "kôra"]
    "wild_regeneration" ["wild regeneration" "divoká regenerácia" "liečenie" "heal"]})

(def KNOWN-WEATHER-SYNONYMS
  @{"Clear" ["clear" "jasno" "slnečno" "slnecno" "sun"]
    "Overcast" ["overcast" "zamračené" "oblačno" "oblacno" "clouds"]
    "Rain" ["rain" "dážď" "dazd" "prší" "lejak"]
    "Thunderstorm" ["thunderstorm" "búrka" "burka" "blesky" "storm"]})

(def KNOWN-CAMERA-SYNONYMS
  @{"perspective_action_follow" ["action" "follow" "3rd person" "akčný" "sledovanie"]
    "perspective_cinematic_wide" ["cinematic" "wide" "filmový" "široký"]
    "orthographic_true_isometric" ["isometric" "izometrický" "izometria" "iso"]
    "orthographic_tactical_topdown" ["topdown" "top-down" "zhora" "taktický"]
    "dual_cinematic_hybrid" ["hybrid" "dual" "duálny"]})

(defn parse-language-prompt [prompt-str]
  "Parses raw prompt into semantic intent with parameters and command type."
  (let [p-low (string/ascii-lower prompt-str)
        is-sk (or (string/find "zahraj" p-low)
                  (string/find "presuň" p-low)
                  (string/find "prepnúť" p-low)
                  (string/find "počasie" p-low)
                  (string/find "stav" p-low)
                  (string/find "štít" p-low))]
    @{:raw-prompt prompt-str
      :language (if is-sk "sk" "en")
      :vital-max-hp VITAL-MAX-HP
      :command-type
      (cond
        (or (string/find "stav" p-low) (string/find "status" p-low)) :QUERY_STATUS
        (or (string/find "počasie" p-low) (string/find "weather" p-low) (string/find "búrka" p-low) (string/find "rain" p-low)) :WEATHER_ATMOSPHERE
        (or (string/find "kamera" p-low) (string/find "camera" p-low) (string/find "pohľad" p-low) (string/find "isometric" p-low)) :CAMERA_CONTROL
        (or (string/find "presuň" p-low) (string/find "move" p-low) (string/find "kráčaj" p-low)) :TACTICAL_MOVE
        (or (string/find "moždiar" p-low) (string/find "mortar" p-low) (string/find "delo" p-low)) :ARTILLERY_FIRE
        (or (string/find "postav" p-low) (string/find "citadel" p-low) (string/find "veža" p-low)) :CITADEL_BUILD
        (or (string/find "spawn" p-low) (string/find "vyvolaj" p-low)) :SPAWN_ENTITY
        (or (string/find "zahraj" p-low) (string/find "cast" p-low) (string/find "karta" p-low)) :CAST_CARD
        (or (string/find "kolo" p-low) (string/find "turn" p-low)) :MATCH_ESCALATION
        :UNKNOWN)}))

(defn format-battlefield-briefing [state-dict lang]
  "Generates articulate natural language battlefield status."
  (let [hp (min VITAL-MAX-HP (get state-dict :hp 6))
        mana (get state-dict :mana 8)
        rnd (get state-dict :round 1)
        weather (get state-dict :weather "Clear")]
    (if (= lang "sk")
      (string/format "🛡️ Poslední Kmen Kolo %d: Hrdina %d/%d HP, Mana %d/10. Počasie: %s."
                     rnd hp VITAL-MAX-HP mana weather)
      (string/format "🛡️ Poslední Kmen Round %d: Hero %d/%d HP, Mana %d/10. Weather: %s."
                     rnd hp VITAL-MAX-HP mana weather))))
