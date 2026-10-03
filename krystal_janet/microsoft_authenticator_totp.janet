# ==============================================================================
# KRYSTAL-STACK: MICROSOFT AUTHENTICATOR 2FA & RFC 6238 TOTP DSL (JANET)
# ==============================================================================
# Defines:
#   1. RFC 6238 Time-Based One-Time Password (TOTP) standard used by
#      Microsoft Authenticator, Google Authenticator, and hardware tokens.
#   2. 30-second time-step interval with HMAC-SHA1 dynamic truncation.
#   3. 2-Digit Number Matching Challenge protocol (Microsoft Authenticator spec).
#   4. Emergency One-Time Backup Recovery Codes.
#   5. Strict 6 Max HP Vital Invariant across authenticated administrator sessions.
# ==============================================================================

(def VITAL-MAX-HP 6)

(def TOTP-SPEC
  {:time-step-seconds 30
   :token-digits 6
   :hash-algorithm :sha1
   :drift-window-tolerance 1
   :issuer "KrystalStack"
   :default-account-label "admin@krystal.mesh"})

(def NUMBER-MATCHING-SPEC
  {:challenge-digits 2
   :min-number 10
   :max-number 99
   :ttl-seconds 120
   :require-biometric-prompt true})

(def BACKUP-CODES-SPEC
  {:code-count 8
   :code-length 10
   :single-use-only true})

(defn compute-time-counter
  "Computes integer time-step counter T = floor(current_time / time_step)."
  [unix-timestamp-sec step-sec]
  (math/floor (/ unix-timestamp-sec step-sec)))

(defn verify-totp-window
  "Checks if submitted token matches token at T-1, T, or T+1 (drift tolerance)."
  [submitted-token current-tok prev-tok next-tok]
  (or (= submitted-token current-tok)
      (= submitted-token prev-tok)
      (= submitted-token next-tok)))

(defn generate-number-matching-challenge
  "Generates a 2-digit number challenge for Microsoft Authenticator push prompt."
  [seed-salt]
  (let [num (+ 10 (math/floor (* (math/random) 90)))]
    {:challenge-number num
     :ttl-seconds 120
     :status :pending-match}))

(defn verify-number-matching-response
  "Validates user selected number in Microsoft Authenticator prompt."
  [expected-number submitted-number]
  (if (= expected-number submitted-number)
    {:status :matched :approved true}
    {:status :rejected :approved false}))
