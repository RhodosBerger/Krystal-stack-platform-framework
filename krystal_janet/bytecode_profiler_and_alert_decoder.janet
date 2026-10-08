# ==============================================================================
# KRYSTAL-STACK: JANET BYTECODE PROFILER & BINARY ALERT DECODER
# ==============================================================================
# File: krystal_janet/bytecode_profiler_and_alert_decoder.janet
# Description: Explores and utilizes Janet language primitives (PEGs, byte buffers,
#              structs, fibers, vector string formatting) to:
#              1. Decode raw 64-bit aligned binary bytecode streams (KSYN format)
#                 into structured operational alerts ("hlásenia").
#              2. Emit dual-representation logs (Hexadecimal representation for audit,
#                 raw binary for hardware execution).
#              3. Render graphical execution profiles, command maps, and memory
#                 aperture layouts as SVG vector blueprints and ASCII dashboards.
#
# System Invariant: VITAL-MAX-HP = 6
# Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
# ==============================================================================

(def VITAL-MAX-HP 6)
(def KSYN-MAGIC "KSYN")

# ─── 1. OPCODES & COMMAND METADATA DICTIONARY ────────────────────────────────

(def OPCODES
  {0x00 {:name "OP_HALT" :desc "Zastavenie výpočtového vlákna" :domain :core :power-w 0.5}
   0x01 {:name "OP_VITAL_ASSERT_HP" :desc "Overenie invariantu VITAL_MAX_HP == 6" :domain :integrity :power-w 1.2}
   0x02 {:name "OP_TERRAIN_MULTIOCTAVE" :desc "Procedurálny terénny fraktál (fBm)" :domain :compute :power-w 18.5}
   0x03 {:name "OP_SDF_CHALICE" :desc "SDF Raymarching: Alchymistický kalich" :domain :graphics :power-w 14.0}
   0x04 {:name "OP_SDF_ATHAME" :desc "SDF Raymarching: Rituálna dýka" :domain :graphics :power-w 14.0}
   0x05 {:name "OP_URBAN_EXTRUDE_SPIRE" :desc "Morfologická extrúzia veží z máp" :domain :geometry :power-w 12.0}
   0x06 {:name "OP_BALLISTICS_MORTAR" :desc "Disperzná balistika mínometu" :domain :physics :power-w 9.5}
   0x07 {:name "OP_BULLET_TIME_DILATE" :desc "Časová dilatácia a Coxeterovo zrkadlo" :domain :spacetime :power-w 15.0}
   0x08 {:name "OP_BAYER_DITHER_SAMPLE" :desc "Bayerov poltónový raster a luma" :domain :raster :power-w 6.0}
   0x09 {:name "OP_COXETER_DIHEDRAL" :desc "Dihedrálne Coxeterovo zrkadlenie" :domain :math :power-w 11.0}
   0x0A {:name "OP_METABOLIC_PULSE" :desc "Metabolický pulz kmeňa (fáza Beta)" :domain :governor :power-w 3.5}
   0x0B {:name "OP_LEDGER_MINT_CREDITS" :desc "Emisia kreditov suverénneho kmeňa" :domain :economy :power-w 2.0}
   0x0C {:name "OP_PREFETCH_L1_STREAM" :desc "K-ISA Špekulatívny prefetch do L1" :domain :memory :power-w 4.0}
   0x0D {:name "OP_BARRIER_L3_SYNC" :desc "Pamäťová bariéra a synchronizácia L3" :domain :memory :power-w 5.0}
   0xA1 {:name "K_SPEC_PREFETCH_UMA" :desc "Odomknutá UMA tenzorová predpríprava" :domain :kisa :power-w 8.0}
   0xA2 {:name "K_SPEC_INTERPOLATE_FRAME" :desc "Špekulatívna interpolácia 120Hz medzisnímku" :domain :kisa :power-w 16.0}
   0xA3 {:name "K_SAFE_VOLT_CLAMP" :desc "Arrheniusov napäťový strop (<=1.02V)" :domain :kisa :power-w 1.0}
   0xA4 {:name "K_FALLBACK_REVERT" :desc "Rollback stavu s nulovou réžiou" :domain :kisa :power-w 1.5}
   0xA5 {:name "K_FUSE_INT8_DP4A" :desc "Zlúčený DP4A INT8 dot-product na Iris Xe" :domain :kisa :power-w 22.0}
   0xA6 {:name "K_VERIFY_INVARIANT_HP" :desc "Hardvérový zámok integrity HP = 6" :domain :kisa :power-w 0.8}})

# ─── 2. BINÁRNY PEG PARSER & BYTE BUFFER DECODER ────────────────────────────

(defn decode-binary-word
  "Decodes an 8-byte instruction word from a binary buffer into a structured Janet table."
  [buf offset]
  (let [opcode (get buf offset)
        flags (get buf (+ offset 1))
        param1 (bor (blshift (get buf (+ offset 2)) 8)
                    (get buf (+ offset 3)))
        param2 (bor (blshift (get buf (+ offset 4)) 24)
                    (blshift (get buf (+ offset 5)) 16)
                    (blshift (get buf (+ offset 6)) 8)
                    (get buf (+ offset 7)))
        info (get OPCODES opcode {:name (string/format "OP_UNKNOWN_0x%02X" opcode)
                                  :desc "Neznáma inštrukcia"
                                  :domain :unknown
                                  :power-w 1.0})]
    {:opcode opcode
     :opcode-hex (string/format "0x%02X" opcode)
     :name (get info :name)
     :description (get info :desc)
     :domain (get info :domain)
     :flags flags
     :param1 param1
     :param2 param2
     :estimated-power-w (get info :power-w)}))

(defn parse-ksyn-binary-stream
  "Parses a raw KSYN binary stream (magic header, version, seed, instruction words)."
  [binary-bytes]
  (let [len (length binary-bytes)]
    (if (< len 8)
      {:status :error :message "Binárka je príliš krátka (menej ako 8 bajtov hlavičky)"}
      (let [magic (string/slice binary-bytes 0 4)
            version (bor (blshift (get binary-bytes 4) 8) (get binary-bytes 5))
            seed (bor (blshift (get binary-bytes 6) 8) (get binary-bytes 7))]
        (if (not= magic KSYN-MAGIC)
          {:status :error :message (string/format "Neplatná hlavička: očakávané '%s', nájdené '%s'" KSYN-MAGIC magic)}
          (let [instructions @[]
                offset 8]
            (var cur-off offset)
            (while (<= (+ cur-off 8) len)
              (array/push instructions (decode-binary-word binary-bytes cur-off))
              (set cur-off (+ cur-off 8)))
            {:status :ok
             :magic magic
             :version (string/format "%d.%d" (brshift version 8) (band version 0xFF))
             :seed seed
             :total-words (length instructions)
             :instructions instructions}))))))

# ─── 3. FORMÁTOVANIE HLÁSENÍ A AUDITNÝCH SPRÁV (REPORTS & ALERTS) ───────────

(defn synthesize-alert-report
  "Decodes binary stream and produces a human/system readable status report ('hlásenie')."
  [parsed-data]
  (if (= (get parsed-data :status) :error)
    {:severity :CRITICAL
     :title "CHYBA DEKÓDOVANIA BINÁRNEHO STREAMU"
     :summary (get parsed-data :message)
     :vital-max-hp VITAL-MAX-HP}
    (let [insts (get parsed-data :instructions)
          total-pwr (reduce (fn [acc i] (+ acc (get i :estimated-power-w))) 0 insts)
          has-hp-assert (some (fn [i] (or (= (get i :opcode) 0x01) (= (get i :opcode) 0xA6))) insts)
          has-kisa-spec (some (fn [i] (= (get i :opcode) 0xA2)) insts)
          alerts @[]]

      # Validácia integrity
      (if has-hp-assert
        (array/push alerts {:severity :INFO
                            :code :INVARIANT_VERIFIED
                            :message (string/format "Systémový invariant VITAL_MAX_HP = %d úspešne overený v inštrukčnom toku." VITAL-MAX-HP)})
        (array/push alerts {:severity :WARNING
                            :code :MISSING_INVARIANT_ASSERT
                            :message "Inštrukčný tok neobsahuje explicitnú kontrolu VITAL_MAX_HP = 6!"}))

      # Analýza spotreby a napätia
      (if (> total-pwr 85.0)
        (array/push alerts {:severity :WARNING
                            :code :HIGH_POWER_ENVELOPE
                            :message (string/format "Kumulatívna spotreba inštrukčného bloku (%.1f W) prekračuje odporúčaný envelope. Odporúčaný K-SAFE-VOLT-CLAMP." total-pwr)})
        (array/push alerts {:severity :INFO
                            :code :POWER_ENVELOPE_SAFE
                            :message (string/format "Spotreba v bezpečnom pásme: %.1f W (< 85W ceiling)." total-pwr)}))

      # Špekulatívna akcelerácia
      (when has-kisa-spec
        (array/push alerts {:severity :OPTIMIZED
                            :code :KISA_SPECULATIVE_ACTIVE
                            :message "K-ISA Špekulatívna interpolácia medzisnímku aktívna: plynulých 120 Hz zabezpečených."}))

      {:severity (if (some (fn [a] (= (get a :severity) :WARNING)) alerts) :WARNING :NOMINAL)
       :title (string/format "HLÁSENIE KERNELU // JANET DEKÓDOVANÁ BINÁRKA (Verzia %s, Seed %d)"
                             (get parsed-data :version) (get parsed-data :seed))
       :instruction-count (length insts)
       :total-power-watts total-pwr
       :alerts alerts
       :decoded-commands (map (fn [i] (string/format "[%s] %-24s -> %s (P1:%d, P2:%d, %.1fW)"
                                                     (get i :opcode-hex)
                                                     (get i :name)
                                                     (get i :description)
                                                     (get i :param1)
                                                     (get i :param2)
                                                     (get i :estimated-power-w)))
                              insts)
       :vital-max-hp VITAL-MAX-HP})))

# ─── 4. VYKRESLENIE PROFILU & MAPE PRÍKAZOV (SVG VECTOR ENGINE) ─────────────

(defn render-execution-profile-svg
  "Generates an SVG vector graphic blueprint illustrating the decoded binary command profile."
  [parsed-data &opt width height]
  (default width 900)
  (default height 420)
  (let [insts (get parsed-data :instructions @[])
        count (max 1 (length insts))
        bar-w (/ (- width 120) count)
        bars @[]]

    (loop [idx :range [0 (length insts)]]
      (let [inst (get insts idx)
            pwr (get inst :estimated-power-w)
            bar-h (* (/ pwr 25.0) 180.0)
            x (+ 60 (* idx bar-w))
            y (- 280 bar-h)
            color (case (get inst :domain)
                    :kisa "#c084fc"
                    :compute "#00f0ff"
                    :graphics "#00ff88"
                    :integrity "#f59e0b"
                    :geometry "#38bdf8"
                    "#8892b0")]
        (array/push bars (string/format
          `<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s" opacity="0.85" rx="3" stroke="#ffffff" stroke-width="0.5"/>
           <text x="%.1f" y="300" fill="#94a3b8" font-family="monospace" font-size="9" transform="rotate(45 %.1f,300)">%s</text>`
          x y (- bar-w 4) bar-h color (+ x (/ bar-w 2)) (+ x (/ bar-w 2)) (get inst :opcode-hex)))))

    (string/format
      `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="100%%" height="100%%" style="background:#07090e; border:1px solid #1e293b; border-radius:8px;">
        <defs>
          <linearGradient id="grid" width="30" height="30" patternUnits="userSpaceOnUse">
            <path d="M 30 0 L 0 0 0 30" fill="none" stroke="rgba(255,255,255,0.03)" stroke-width="1"/>
          </linearGradient>
        </defs>
        <rect width="100%%" height="100%%" fill="url(#grid)" />
        <text x="30" y="36" fill="#00f0ff" font-family="sans-serif" font-weight="bold" font-size="16">KRYSTAL-STACK // JANET BINARY EXECUTION PROFILE (KSYN)</text>
        <text x="30" y="56" fill="#8892b0" font-family="monospace" font-size="11">Vykreslenie profilu inštrukčných slov zarovnaných na 64-bitov // VITAL_MAX_HP = 6</text>
        
        <!-- Y Axis Guidelines -->
        <line x1="50" y1="280" x2="%d" y2="280" stroke="#334155" stroke-width="1"/>
        <line x1="50" y1="190" x2="%d" y2="190" stroke="rgba(255,255,255,0.08)" stroke-dasharray="4"/>
        <line x1="50" y1="100" x2="%d" y2="100" stroke="rgba(255,255,255,0.08)" stroke-dasharray="4"/>
        <text x="15" y="284" fill="#64748b" font-family="monospace" font-size="10">0W</text>
        <text x="15" y="194" fill="#64748b" font-family="monospace" font-size="10">12W</text>
        <text x="15" y="104" fill="#64748b" font-family="monospace" font-size="10">25W</text>
        
        <!-- Instruction Bars -->
        %s

        <!-- Legend -->
        <circle cx="60" cy="385" r="4" fill="#c084fc"/><text x="70" y="388" fill="#cbd5e1" font-family="sans-serif" font-size="10">K-ISA Špekulácia</text>
        <circle cx="180" cy="385" r="4" fill="#00f0ff"/><text x="190" y="388" fill="#cbd5e1" font-family="sans-serif" font-size="10">Výpočet (fBm)</text>
        <circle cx="280" cy="385" r="4" fill="#00ff88"/><text x="290" y="388" fill="#cbd5e1" font-family="sans-serif" font-size="10">SDF Raymarching</text>
        <circle cx="410" cy="385" r="4" fill="#f59e0b"/><text x="420" y="388" fill="#cbd5e1" font-family="sans-serif" font-size="10">Integrita (Assert HP)</text>
      </svg>`
      width height (- width 40) (- width 40) (- width 40) (string/join bars "\n"))))

# ─── 5. TERMINÁLOVÉ ASCII VYKRESLENIE MAPY PRÍKAZOV ─────────────────────────

(defn render-terminal-ascii-profile
  "Renders a high-density ASCII table and opcode map for terminal visualization."
  [parsed-data]
  (let [insts (get parsed-data :instructions @[])
        lines @[]]
    (array/push lines "================================================================================")
    (array/push lines "  KRYSTAL-STACK: JANET BINÁRNE MAPOVANIE & DEKÓDOVANÉ HLÁSENIE (KSYN)")
    (array/push lines "================================================================================")
    (array/push lines (string/format "  Hlavička: %s | Verzia: %s | Seed: %d | Počet inštrukcií: %d"
                                     (get parsed-data :magic "KSYN")
                                     (get parsed-data :version "1.0")
                                     (get parsed-data :seed 0)
                                     (length insts)))
    (array/push lines "--------------------------------------------------------------------------------")
    (array/push lines "  IDX | HEX  | NÁZOV PRÍKAZU            | OBLASŤ     | VÝKON  | POPIS")
    (array/push lines "--------------------------------------------------------------------------------")
    (loop [i :range [0 (length insts)]]
      (let [w (get insts i)]
        (array/push lines (string/format "  %02d  | %-4s | %-24s | %-10s | %4.1f W | %s"
                                         i
                                         (get w :opcode-hex)
                                         (get w :name)
                                         (string (get w :domain))
                                         (get w :estimated-power-w)
                                         (get w :description)))))
    (array/push lines "================================================================================")
    (string/join lines "\n")))
