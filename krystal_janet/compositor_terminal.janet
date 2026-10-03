# ====================================================================
# KRYSTAL-STACK // Janet Compositor Terminal
# ====================================================================
# A terminal CLI that automatically generates JSON characteristics
# for Symposia compositions based on prompts and quota management.
# ====================================================================

(import spork/json)

(def *version* "1.0.0")
(def *available-quota* @{:generations 100})

(defn print-header []
  (print "====================================================")
  (print "  KRYSTAL-STACK JANET TERMINAL: COMPOSITOR AI")
  (print "  v" *version* " | Quota remaining: " (*available-quota* :generations))
  (print "====================================================\n")
  (print "Type a prompt to generate composition traits, or 'exit' to quit.\n"))

(defn parse-prompt-to-json [prompt]
  "Simulates an AI mapping a prompt to composition sliders and layer characteristics."
  (def base-adj @{:contrast 1.0 :saturation 1.0 :brightness 1.0})
  (def config @{:edge_opacity 0.8 
                :edge_blend "screen" 
                :texture_opacity 0.4
                :atmosphere_color [0 0 0 255]})
  
  # Basic heuristic matching to mimic AI extraction
  (if (string/find "dark" (string/ascii-lower prompt))
    (do 
      (put base-adj :brightness 0.6)
      (put config :atmosphere_color [10 10 30 255])))
      
  (if (string/find "neon" (string/ascii-lower prompt))
    (do 
      (put base-adj :saturation 1.5)
      (put config :edge_opacity 1.0)
      (put config :atmosphere_color [0 255 200 255])))
      
  (if (string/find "vintage" (string/ascii-lower prompt))
    (do 
      (put base-adj :contrast 0.8)
      (put base-adj :saturation 0.6)
      (put config :texture_opacity 0.7)
      (put config :atmosphere_color [150 100 50 255])))

  (put config :base_adjustments base-adj)
  config)

(defn dispatch-to-compositor [json-spec]
  "Writes the JSON to a file or dispatches it to the Python backend"
  (def filename "composition_request.json")
  (spork/json/write json-spec filename)
  (print "[Terminal] Emitted composition characteristics to " filename)
  (print "           (Ready for neural_photobank_compositor.py to consume)")
  (print "----------------------------------------------------\n"))

(defn run-terminal []
  (print-header)
  (forever
    (if (<= (*available-quota* :generations) 0)
      (do 
        (print "[!] QUOTA DEPLETED. Terminal entering Omega state.")
        (break)))
        
    (file/write stdout "compositor> ")
    (file/flush stdout)
    (def input-line (file/read stdin :line))
    
    (if (or (nil? input-line) (= (string/trim input-line) "exit"))
      (break))
      
    (def prompt (string/trim input-line))
    (if (not (empty? prompt))
      (do
        (put *available-quota* :generations (- (*available-quota* :generations) 1))
        (print "[Terminal] Parsing semantic prompt...")
        (def json-spec (parse-prompt-to-json prompt))
        (print "[Terminal] Generated Composition Characteristics:")
        (print (spork/json/encode json-spec))
        (dispatch-to-compositor json-spec)))))

# Execute the terminal loop
(run-terminal)
