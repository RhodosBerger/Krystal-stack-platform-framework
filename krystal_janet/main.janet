# ============================================================================
# Krystal-Stack Janet Engine: Standalone CLI Mission Control Entry Point
# ============================================================================
# Cloned subproject runner orchestrating:
# 1. Symplectic Hamiltonian Cyclic Organism (Velocity-Verlet Phase Orbits)
# 2. 3D Neural ASCII Raymarching Renderer with TrueColor ANSI stream
# 3. Visual Entropy and Economic Governor
# 4. Topological Virtual Machine queue pipelines using Janet fibers

(import ./krystal_sdf :as sdf)
(import ./cyclic_organism :as cyclic)
(import ./neural_ascii_engine :as renderer)
(import ./governor :as gov)
(import ./topological_vm :as tvm)
(import ./antigravity_peg :as peg)

(defn print-header []
  (print "\e[1;36m======================================================================\e[0m")
  (print "\e[1;32m [KRYSTAL-STACK] JANET NATIVE ENGINE // HETEROGENEOUS MISSION CONTROL\e[0m")
  (print "\e[1;36m======================================================================\e[0m")
  (print " >> Architecture: Symplectic Hamiltonian Organism + Topological Fibers")
  (print " >> Target:       Janet Lisp Subproject Engine")
  (print " >> Color Space:  24-bit TrueColor ANSI VT-100 Stream")
  (print "\e[1;36m======================================================================\e[0m\n"))

(defn run-engine [&opt max-frames is-test-mode]
  (let [limit (or max-frames 12)
        test-mode (or is-test-mode false)
        cols 64
        rows 24
        org (cyclic/create-organism "Organism-Janet-Localhost")
        governor (gov/create-governor 900.0 1000.0)
        vm (tvm/create-topological-vm)]

    # Setup K-TVM queues & pipeline
    (tvm/alloc-queue vm "InStream" 256 3 [-2.0 0.0 0.0])
    (tvm/alloc-queue vm "ProcessKernel" 512 2 [0.0 1.0 0.0])
    (tvm/alloc-queue vm "OutRaster" 256 1 [2.0 0.0 0.0])

    (tvm/connect-pipeline vm "Stage1" "InStream" "ProcessKernel"
      (fn [pkt] (merge pkt {:stage1-transformed true})))
    (tvm/connect-pipeline vm "Stage2" "ProcessKernel" "OutRaster"
      (fn [pkt] (merge pkt {:rasterized true})))

    # Inject test seed packets
    (tvm/inject-packet vm "InStream" {:packet-id 101 :manifold :GYROID})
    (tvm/inject-packet vm "InStream" {:packet-id 102 :manifold :TORUS_KNOT})

    (unless test-mode
      (print-header))

    (var last-entropy 0.22)
    (for frame 1 (+ limit 1)
      (let [t (* frame 0.08)

            # 1. Step Topological VM
            vm-metrics (tvm/step-vm vm 2)

            # 2. Render Neural ASCII Frame (3D Raymarch)
            ascii-output (renderer/render-frame cols rows t :CYBERPUNK 28)

            # 3. Calculate Visual Entropy & Update Economic Governor
            ent-metrics (gov/compute-entropy governor ascii-output cols rows)
            tot-entropy (ent-metrics :total)
            b-rem (gov/update-budget governor 0.033 1.2)

            # 4. Step Symplectic Hamiltonian Organism (2nd-Order Velocity-Verlet)
            _ (cyclic/symplectic-step org 0.033 tot-entropy 46.5)
            h-energy (cyclic/total-hamiltonian org)
            lyap (cyclic/lyapunov-stability-index org)
            c-phase (org :cognitive-phase)]

        (set last-entropy tot-entropy)

        (unless test-mode
          # Clear screen or move cursor home
          (print "\e[H")
          (print ascii-output)
          (print (string/format
                   "\e[1;33m[FRAME %03d]\e[0m | \e[1;32mPHASE: %s\e[0m | \e[1;36mH: %.4f\e[0m | \e[1;35mLYAPUNOV: %.4f\e[0m | \e[1;31mENTROPY: %.2f\e[0m | \e[1;34mBUDGET: %.0f\e[0m | \e[1;32mVM: %.0f pps\e[0m"
                   frame (string c-phase) h-energy lyap tot-entropy b-rem (vm-metrics :throughput-pps))))))

    (when test-mode
      (print (string/format "[TEST OK] Completed %d cycles. Final H=%.4f, Phase=%s, Entropy=%.2f"
                            limit (cyclic/total-hamiltonian org) (string (org :cognitive-phase)) last-entropy)))
    {:status :OK :cycles limit :final-h (cyclic/total-hamiltonian org) :phase (org :cognitive-phase)}))

(defn main [& args]
  (let [is-test (some (fn [a] (or (= a "--test") (= a "-t"))) args)
        steps (if is-test 6 12)]
    (run-engine steps is-test)))
