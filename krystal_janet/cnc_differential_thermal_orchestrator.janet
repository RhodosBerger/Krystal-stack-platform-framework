# ==============================================================================
# KRYSTAL-STACK: CNC DIFFERENTIAL THERMAL ORCHESTRATOR & FALLBACK (JANET DSL)
# ==============================================================================
# File: krystal_janet/cnc_differential_thermal_orchestrator.janet
# Description: Janet S-expression DSL orchestrating real-time cutting thermal
#              bounds, dynamic RPM modulation, and system collapse differentials.
#
# System Invariant: VITAL-MAX-HP = 6
# Author: Dušan Kopecký & Krystal-Stack Computational Manufacturing Council (2026)
# ==============================================================================

(def VITAL-MAX-HP 6)

(def THERMAL-LIMITS
  @{:ambient-temp-c 20.0
    :safe-carbide-temp-c 480.0
    :critical-carbide-temp-c 750.0
    :max-allowed-gradient-c-per-s 20.0
    :chatter-ssv-amplitude-pct 0.12
    :coolant-loss-rpm-reduction-pct 0.40})

(def TOOL-HEALTH-TIERS
  @{:tier-6 {:hp 6 :label :PRISTINE :max-vb 0.05 :action :FULL-SPEED-ROUGHING}
    :tier-5 {:hp 5 :label :BREAK-IN :max-vb 0.10 :action :NOMINAL-CUTTING}
    :tier-4 {:hp 4 :label :STEADY-WEAR :max-vb 0.18 :action :ACTIVE-SSV-MODULATION}
    :tier-3 {:hp 3 :label :ELEVATED-TEMP :max-vb 0.25 :action :MANDATORY-PECK-BREATHERS}
    :tier-2 {:hp 2 :label :SEVERE-WEAR :max-vb 0.32 :action :THROTTLE-FEED-35-PCT}
    :tier-1 {:hp 1 :label :CRITICAL-LIMIT :max-vb 0.40 :action :SCHEDULED-TOOL-CHANGE}
    :tier-0 {:hp 0 :label :TOOL-EXHAUSTED :max-vb 0.50 :action :AUTO-RETRACT-PARK}})

(defn cnc/deduce-thermal-rpm-damping
  "Calculates non-linear damping factor for spindle RPM based on thermal gradient."
  [current-temp-c gradient-c-per-s]
  (let [safe-temp (get THERMAL-LIMITS :safe-carbide-temp-c 480.0)
        crit-temp (get THERMAL-LIMITS :critical-carbide-temp-c 750.0)
        temp-excess (math/max 0.0 (- current-temp-c safe-temp))
        gradient-factor (if (> gradient-c-per-s 15.0) 0.82 1.0)
        temp-factor (math/max 0.45 (- 1.0 (/ temp-excess (- crit-temp safe-temp))))]
    (math/max 0.40 (* temp-factor gradient-factor))))

(defn cnc/deduce-collapse-fallback
  "Evaluates fault conditions and deduces the exact differential recovery mode and G-code splice."
  [fault-type]
  (case fault-type
    :sensor-dropout
    @{:mode :KALMAN-SENSOR-ESTIMATE
      :safety-throttle 0.80
      :gcode-action "(KATG FALLBACK: SWITCH TO KALMAN THERMAL ESTIMATOR)"
      :action-status :RECOVERED}

    :spindle-torque-stall
    @{:mode :SAFE-AXIS-RETRACT
      :safety-throttle 0.0
      :gcode-action "G91 G00 Z5.0\nM05\nG90"
      :action-status :RETRACTED}

    :coolant-pressure-loss
    @{:mode :COOLANT-LOSS-TROCHOIDAL
      :safety-throttle 0.60
      :gcode-action "M09\nS4800 M03\nG04 P300"
      :action-status :MODULATED}

    :critical-tool-wear
    @{:mode :EMERGENCY-SAFE-PARK
      :safety-throttle 0.0
      :gcode-action "G91 G28 Z0\nM05\nM00 (TOOL HP EXHAUSTED)"
      :action-status :PARKED}

    @{:mode :NOMINAL
      :safety-throttle 1.0
      :gcode-action "(NOMINAL CUTTING)"
      :action-status :RUNNING}))

(defn cnc/evaluate-tool-vitality-hp
  "Maps flank wear (VB in mm) to the 6-Tier Tool Structural Health Invariant."
  [flank-wear-vb]
  (let [clamped-vb (math/max 0.0 flank-wear-vb)]
    (cond
      (<= clamped-vb 0.05) 6
      (<= clamped-vb 0.10) 5
      (<= clamped-vb 0.18) 4
      (<= clamped-vb 0.25) 3
      (<= clamped-vb 0.32) 2
      (<= clamped-vb 0.40) 1
      0)))
