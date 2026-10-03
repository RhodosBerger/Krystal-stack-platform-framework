# ==============================================================================
# KRYSTAL-STACK: WORDPRESS DOCKER SECURITY, PROJECTOR ANALOG ADC & JAVA TRANSPILER DSL
# ==============================================================================
# Defines:
#   1. WordPress Docker & Related Filesystem Security Specifications (SSL/TLS, BBQ Firewall,
#      Antispam Bee, Wordfence Brute Force Protection, Argon2id/PBKDF2 Hashing).
#   2. Projector Stream Analog Converter (ADC) & NPU Streamed AI Pipeline.
#   3. Parallel Java Transpiler & Custom Record Hierarchies.
#   4. Procedural Island Realms with Metaphorical Geographic Mapping:
#      - Denmark as Heaven (Dánsko ako Nebo)
#      - Finland as Slovakia (Fínsko ako Slovensko)
#      - Czechia as Latvia (Česko ako Lotyšsko)
#      - Germany as Poland (Nemecko ako Poľsko)
#      - France as America (Francúzsko ako Amerika)
#   5. Strict 6 Max HP Vital Invariant across all island garrisons and heroes.
# ==============================================================================

(def VITAL-MAX-HP 6)
(def ARGON2ID-MEMORY-KIB 65536)
(def ARGON2ID-ITERATIONS 3)
(def ARGON2ID-PARALLELISM 4)

(def BBQ-FIREWALL-PATTERNS
  {:sqli-union "(union.*select|into.*outfile|load_file)"
   :sqli-sleep "(benchmark\\(|sleep\\()"
   :traversal "((\\.\\./|\\.\\.\\\\|boot\\.ini|etc/passwd))"
   :rce-eval "(eval\\(|base64_decode\\(|passthru\\(|system\\()"
   :xss-tags "(<script|%3Cscript)"})

(def ANTISPAM-BEE-SPEC
  {:honeypot-field-name "krystal_trap_honey_bee"
   :min-submission-time-seconds 3.0
   :block-ip-on-spam true
   :zero-database-bloat true
   :trust-registered-users true})

(def WORDFENCE-BRUTEFORCE-LIMITS
  {:max-failed-attempts 5
   :lockout-duration-minutes 30
   :enforce-strong-passwords true
   :two-factor-totp-required-for-admin true})

(def PROJECTOR-ANALOG-CONVERTER-CONFIG
  {:analog-scanlines 1080
   :adc-bit-depth 10
   :sampling-rate-mhz 148.5
   :npu-stream-tensor-shape [1 3 1080 1920]
   :target-fps 60.0})

(def CANONICAL-ISLAND-REALMS
  {:denmark-heaven
   {:display-name "Dánsko ako Nebo"
    :latin-lore "Caelum Danicum // Celestial Sky Island"
    :biome "celestial-floating-cloudbanks"
    :elevation-m 4500.0
    :primary-color "#fef3c7"
    :vital-max-hp VITAL-MAX-HP
    :character-sort-category "seraphic-valkyries"
    :description "Floating ethereal archipelago supported by golden clouds and glowing sun-spires."}

   :finland-slovakia
   {:display-name "Fínsko ako Slovensko"
    :latin-lore "Silva Boralis // High Tatra Boreal Forest"
    :biome "alpine-glacial-tarn-taiga"
    :elevation-m 2655.0
    :primary-color "#10b981"
    :vital-max-hp VITAL-MAX-HP
    :character-sort-category "tatra-forest-druids"
    :description "Glacial granite pinnacles rising from deep ancient spruce forests and crystal lakes."}

   :czechia-latvia
   {:display-name "Česko ako Lotyšsko"
    :latin-lore "Ora Succinica // Baltic Bohemian Amber Island"
    :biome "baltic-amber-dune-gothic"
    :elevation-m 45.0
    :primary-color "#f59e0b"
    :vital-max-hp VITAL-MAX-HP
    :character-sort-category "amber-gothic-sentinels"
    :description "Coastal amber dunes flanked by gothic maritime fortresses and pine-crested shores."}

   :germany-poland
   {:display-name "Nemecko ako Poľsko"
    :latin-lore "Bastio Planitiei // Fortress Plain Realm"
    :biome "stone-bastion-fertile-plain"
    :elevation-m 180.0
    :primary-color "#64748b"
    :vital-max-hp VITAL-MAX-HP
    :character-sort-category "iron-bastion-knights"
    :description "Impregnable stone river bastions surrounded by expansive fertile plains and foundries."}

   :france-america
   {:display-name "Francúzsko ako Amerika"
    :latin-lore "Novus Mundus Libertatis // Revolutionary Canyon Island"
    :biome "revolutionary-red-canyons"
    :elevation-m 850.0
    :primary-color "#e11d48"
    :vital-max-hp VITAL-MAX-HP
    :character-sort-category "revolutionary-pioneers"
    :description "Epic red sandstone canyons intersected by grand avenues and monumental liberty spires."}})

(defn evaluate-bbq-query
  "Checks if a given query string triggers BBQ firewall bad query rules."
  [query-string]
  (let [q (string/ascii-lower query-string)
        is-sqli (or (string/find "union" q) (string/find "select" q) (string/find "sleep(" q))
        is-traversal (or (string/find "../" q) (string/find "..\\" q) (string/find "etc/passwd" q))
        is-rce (or (string/find "eval(" q) (string/find "base64_decode" q))]
    {:blocked (if (or is-sqli is-traversal is-rce) true false)
     :threat-type (cond
                    is-sqli "SQL_INJECTION"
                    is-traversal "PATH_TRAVERSAL"
                    is-rce "REMOTE_CODE_EXECUTION"
                    "CLEAN")}))

(defn evaluate-antispam-submission
  "Evaluates form submission against Antispam Bee honeypot and time trap."
  [honeypot-value elapsed-sec]
  (let [honey-filled (not (= honeypot-value ""))
        too-fast (< elapsed-sec 3.0)]
    {:is-spam (or honey-filled too-fast)
     :reason (cond
               honey-filled "HONEYPOT_TRIPPED"
               too-fast "SUBMISSION_TOO_FAST_BOT"
               "LEGITIMATE")}))

(defn compute-analog-projector-bandwidth-gbps
  "Computes raw digital throughput generated by analog projector ADC converter."
  [scanlines fps bit-depth]
  (let [pixels-per-frame (* scanlines (/ (* scanlines 16.0) 9.0))
        bits-per-second (* pixels-per-frame 3.0 bit-depth fps)
        gbps (/ bits-per-second 1000000000.0)]
    gbps))

(defn generate-java-record-signature
  "Generates modern Java 21 record signature for an island realm."
  [island-key]
  (let [realm (get CANONICAL-ISLAND-REALMS island-key)
        name (get realm :display-name "Unknown")
        elev (get realm :elevation-m 0.0)]
    (string/format "public record %sRealm(String name, double elevationMeters, int vitalMaxHp) {}"
                   (string/replace-all "-" "" (string island-key)))))

(def SUBDOMAIN-SECURITY-SPEC
  {:token-ttl-seconds 60.0
   :vital-max-hp VITAL-MAX-HP
   :allowed-subdomain-regex "^(krystal\\..+|localhost|127\\.0\\.0\\.1)"
   :bbq-firewall-active true
   :rate-limit-login-rpm 3})

(defn evaluate-subdomain-token-validity
  "Validates a simulated subdomain session token against TTL and vital HP rules."
  [issued-at-ts current-ts declared-hp]
  (let [elapsed (- current-ts issued-at-ts)
        is-expired (> elapsed 60.0)
        hp-valid (<= declared-hp VITAL-MAX-HP)]
    {:valid (and (not is-expired) hp-valid)
     :is-expired is-expired
     :vital-hp-preserved hp-valid
     :vital-max-hp VITAL-MAX-HP}))

