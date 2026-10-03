# ============================================================================
# Krystal-Stack Janet Engine: Antigravity Natural Language PEG Grammar
# ============================================================================
# Uses Janet's native Parsing Expression Grammars (PEGs) to parse multilingual
# natural language intent into structured mathematical world ASTs.

(def prompt-peg
  (peg/compile
    ~{:main (* (any (choice :topo-token
                            :atmo-token
                            :artifact-token
                            :octaves-token
                            :folds-token
                            :word
                            :s+)))

      :s+ (some (choice " " "\t" "\n" "," "." "!" "?"))
      :word (some (if-not (choice :s+ (set "0123456789")) 1))

      # Topography token matching
      :topo-token
      (group
        (* (constant :topography)
           (choice
             (* (choice "mountain" "mountains" "hory" "vrch" "štít" "alpy" "ridge" "hrebeň" "skaly")
                (constant :ALPINE_RIDGES))
             (* (choice "canyon" "kaňon" "rokliny" "priepasť" "chasm" "trench" "tiesňava")
                (constant :CANYON_TRENCHES))
             (* (choice "plain" "pláň" "pláne" "dunes" "duny" "púšť" "desert" "rovina")
                (constant :ROLLING_DUNES_PLAINS))
             (* (choice "crater" "kráter" "krátery" "wasteland" "pustatina" "ruiny")
                (constant :CRATERED_WASTELAND)))))

      # Atmosphere token matching
      :atmo-token
      (group
        (* (constant :atmosphere)
           (choice
             (* (choice "sulfur" "síra" "sírny" "dym" "toxic" "toxický" "acid" "kyslý")
                (constant :SULFUR_SMOG))
             (* (choice "neon" "neón" "cyberpunk" "cyber" "kyber" "syntetický" "matrix")
                (constant :NEON_CYBER_SMOG))
             (* (choice "crystal" "kryštál" "aether" "éter" "čistý" "jasný" "ice" "ľad")
                (constant :CRYSTALLINE_AETHER))
             (* (choice "lava" "láva" "vulkan" "volcano" "oheň" "fire" "popol" "ash")
                (constant :VOLCANIC_EMBER)))))

      # Artifact token matching
      :artifact-token
      (group
        (* (constant :artifact)
           (choice
             (* (choice "veže" "veža" "spire" "spires" "dáta")
                (constant :CYBERPUNK_DATA_SPIRE))
             (* (choice "obelisk" "monolit" "monolith" "svätyňa")
                (constant :ANCIENT_OBELISK_MONOLITH))
             (* (choice "turret" "vežičk" "delo" "obrana")
                (constant :CYBER_TURRET_MK4))
             (* (choice "mech" "titan" "robot" "kráčajúci")
                (constant :MECH_WALKER_TITAN))
             (* (choice "hniezdo" "xenobiotic" "giger" "biomech" "alien")
                (constant :BIOMECHANICAL_XENODRONE)))))

      # Numeric octaves matching
      :octaves-token
      (group
        (* (constant :octaves)
           (/ '(some :d) ,scan-number)
           (? (* (any :s+) (choice "octaves" "octave" "rekurzia" "rekurzií" "násobn")))))

      # Dihedral symmetry folds matching
      :folds-token
      (group
        (* (constant :mirror-folds)
           (/ '(some :d) ,scan-number)
           (? (* (any :s+) (choice "fold" "folds" "uholník" "uholníková" "hran")))))}))

(defn parse-antigravity-prompt [prompt-str]
  "Parses a natural language prompt string into a structured AST table."
  (let [matches (peg/match prompt-peg (string/ascii-lower prompt-str))
        ast @{:topography :ALPINE_RIDGES
              :atmosphere :NEON_CYBER_SMOG
              :octaves 5
              :mirror-folds 6
              :artifacts @[]}]
    (when matches
      (each token matches
        (case (token 0)
          :topography (put ast :topography (token 1))
          :atmosphere (put ast :atmosphere (token 1))
          :octaves (put ast :octaves (token 1))
          :mirror-folds (put ast :mirror-folds (token 1))
          :artifact (array/push (ast :artifacts) (token 1)))))
    ast))

(defn ast->mathematical-ir [ast]
  "Converts the AST table into mathematical parameters for the terrain manifold."
  (let [topo (ast :topography)
        atmo (ast :atmosphere)
        oct (ast :octaves)
        folds (ast :mirror-folds)
        h-scale (case topo
                  :ALPINE_RIDGES 4.5
                  :CANYON_TRENCHES 3.8
                  :ROLLING_DUNES_PLAINS 1.2
                  :CRATERED_WASTELAND 2.2
                  2.5)
        ridge-w (case topo
                  :ALPINE_RIDGES 0.85
                  :CANYON_TRENCHES 0.90
                  :ROLLING_DUNES_PLAINS 0.15
                  0.5)
        fog-d (case atmo
                :SULFUR_SMOG 0.08
                :NEON_CYBER_SMOG 0.05
                :CRYSTALLINE_AETHER 0.02
                :VOLCANIC_EMBER 0.09
                0.04)]
    {:hurst-exponent (- 1.0 (* 0.48 0.8))
     :height-scale h-scale
     :octaves oct
     :ridge-weight ridge-w
     :fog-density fog-d
     :mirror-folds folds
     :artifacts (ast :artifacts)}))
