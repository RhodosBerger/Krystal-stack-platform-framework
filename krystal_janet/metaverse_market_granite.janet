# ==============================================================================
# KRYSTAL-STACK PLATFORM: METAVERSE MARKET & 2B / 4B GRANITE LLM DSL
# ==============================================================================
# Defines procedures for:
#   1. AMM Constant Product invariant (x * y = k) swaps and slippage.
#   2. Order-book mid-price and spread derivation.
#   3. Model parameter budgeting: <= 2B Edge and 4B IBM Granite 2026 builds.
#   4. Market sentiment heuristic weighting and merchant barter scoring.
# ==============================================================================

(def GRANITE-MODEL-TIERS
  {:edge-2b {:name "IBM Granite 3.0 2B Instruct"
             :max-parameters 2000000000
             :vram-budget-mb 1250
             :context-window 8192
             :target "Edge / Local CPU & NPU"}
   :granite-4b-2026 {:name "IBM Granite 3.1 4B Enterprise Quant"
                     :max-parameters 4100000000
                     :vram-budget-mb 2450
                     :context-window 131072
                     :target "High-Throughput Local Market Reasoner"}})

(defn calculate-amm-output
  "Calculates output tokens for constant product AMM with 0.3% fee: (x * y = k)"
  [reserve-in reserve-out amount-in]
  (let [amount-with-fee (* amount-in 997)
        numerator (* amount-with-fee reserve-out)
        denominator (+ (* reserve-in 1000) amount-with-fee)]
    (if (= denominator 0)
      0
      (/ numerator denominator))))

(defn calculate-market-spread
  "Calculates bid-ask spread percentage given best bid and best ask"
  [best-bid best-ask]
  (if (<= best-bid 0)
    0.0
    (let [diff (- best-ask best-bid)
          mid (/ (+ best-ask best-bid) 2.0)]
      (* (/ diff mid) 100.0))))

(defn verify-llm-budget
  "Verifies whether parameter count conforms to <= 2B or 4B Granite limit"
  [tier-key param-count]
  (let [tier (get GRANITE-MODEL-TIERS tier-key)]
    (if (nil? tier)
      false
      (<= param-count (get tier :max-parameters)))))

(defn score-barter-viability
  "Calculates barter acceptance likelihood (0.0 to 1.0) based on value ratio and merchant greed"
  [offered-val requested-val merchant-greed]
  (if (<= requested-val 0)
    1.0
    (let [ratio (/ offered-val requested-val)
          threshold (+ 1.0 (* merchant-greed 0.25))]
      (if (>= ratio threshold)
        1.0
        (max 0.0 (/ ratio threshold))))))
