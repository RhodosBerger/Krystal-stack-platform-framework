# Krystal-Stack Janet DSL // Procedural Vegetation Strata & Master Phenomena Engine
# Definuje 6-úrovňovú botanickú stratifikáciu, fraktálne L-Systémy, Vogelovu fylotaxiu (137.508 deg)
# a 10 kanonických prírodných, atmosférických, optických a okultných úkazov.
# Invariant: VITAL-MAX-HP = 6

(def VITAL-MAX-HP 6)
(def GOLDEN-RATIO 1.61803398875)
(def INV-GOLDEN-RATIO 0.61803398875)
(def PHYLLOTAXIS-GOLDEN-ANGLE-DEG 137.507764)

# 1. 6-Úrovňová Botanická Stratifikácia
(def CANONICAL-BOTANICAL-STRATA
  @{:emergent-canopy
    @{:name "Poschodie Korún a Klenby"
      :tier 1
      :height-range [22.0 45.0]
      :light-affinity 1.0
      :multimesh-priority "high"
      :vital-max-hp VITAL-MAX-HP}
    :understory-subcanopy
    @{:name "Podrastové Dreviny a Mladé Stromy"
      :tier 2
      :height-range [8.0 20.0]
      :light-affinity 0.65
      :multimesh-priority "medium"
      :vital-max-hp VITAL-MAX-HP}
    :shrub-and-vines
    @{:name "Kríkové a Lianové Poschodie"
      :tier 3
      :height-range [1.0 6.0]
      :light-affinity 0.45
      :multimesh-priority "medium"
      :vital-max-hp VITAL-MAX-HP}
    :herbaceous-and-ferns
    @{:name "Bylinné a Papraďové Poschodie"
      :tier 4
      :height-range [0.2 1.8]
      :light-affinity 0.35
      :multimesh-priority "dense"
      :vital-max-hp VITAL-MAX-HP}
    :moss-and-lichens
    @{:name "Machové a Lišajníkové Koberce"
      :tier 5
      :height-range [0.005 0.1]
      :light-affinity 0.20
      :multimesh-priority "surface-decal"
      :vital-max-hp VITAL-MAX-HP}
    :rhizosphere-mycelium
    @{:name "Podzemná Rhizosféra a Mycélium"
      :tier 6
      :height-range [-3.0 0.0]
      :light-affinity 0.0
      :multimesh-priority "subterranean-tensor"
      :vital-max-hp VITAL-MAX-HP}})

# 2. 10 Kanonických Prírodných a Atmosférických Úkazov
(def CANONICAL-MASTER-PHENOMENA
  @{:aetheric-aurora-stream
    @{:title "Éterická Polárna Žiara a Magnetosféra"
      :category "atmospheric_optical"
      :frequency-hz 7.83
      :wavelength-nm 557.7
      :volumetric-density 0.45
      :vital-max-hp VITAL-MAX-HP}
    :st-elmos-plasma-fire
    @{:title "Eliášov Oheň na Listoch a Vetvách"
      :category "electromagnetic_plasma"
      :frequency-hz 144000.0
      :wavelength-nm 430.0
      :volumetric-density 0.70
      :vital-max-hp VITAL-MAX-HP}
    :chromatic-aberration-burst
    @{:title "Chromatické Lámanie cez Kryštalickú Hmlu"
      :category "atmospheric_optical"
      :frequency-hz 540000000000.0
      :wavelength-nm 589.0
      :volumetric-density 0.55
      :vital-max-hp VITAL-MAX-HP}
    :bioluminescent-spore-tempest
    @{:title "Bioluminiscenčná Spórová Búrka"
      :category "biological_spore"
      :frequency-hz 528.0
      :wavelength-nm 515.0
      :volumetric-density 0.82
      :vital-max-hp VITAL-MAX-HP}
    :ball-lightning-vortex
    @{:title "Guľový Blesk a Levitujúci Plazmový Vír"
      :category "electromagnetic_plasma"
      :frequency-hz 432.0
      :wavelength-nm 480.0
      :volumetric-density 0.90
      :vital-max-hp VITAL-MAX-HP}
    :geothermal-steam-fumarole
    @{:title "Geotermálny Gejzír a Sírna Fumarola"
      :category "geological_telluric"
      :frequency-hz 14.5
      :wavelength-nm 620.0
      :volumetric-density 0.75
      :vital-max-hp VITAL-MAX-HP}
    :cryo-crystallization-wave
    @{:title "Rázová Vlna Bleskovej Kryo-Kryštalizácie"
      :category "geological_telluric"
      :frequency-hz 256.0
      :wavelength-nm 460.0
      :volumetric-density 0.68
      :vital-max-hp VITAL-MAX-HP}
    :gravitational-microlens-warp
    @{:title "Gravitačné Mikrošošovkovanie Studne Duší"
      :category "cosmic_occult"
      :frequency-hz 3.14159
      :wavelength-nm 380.0
      :volumetric-density 0.85
      :vital-max-hp VITAL-MAX-HP}
    :crepuscular-zodiacal-rays
    @{:title "Zodiakálne Protisvetlo a Krepuskulárne Lúče"
      :category "atmospheric_optical"
      :frequency-hz 888.0
      :wavelength-nm 580.0
      :volumetric-density 0.50
      :vital-max-hp VITAL-MAX-HP}
    :ley-line-harmonic-pulse
    @{:title "Harmonický Pulz Dračích Žíl a Telúrnych Prúdov"
      :category "cosmic_occult"
      :frequency-hz 528.0
      :wavelength-nm 632.8
      :volumetric-density 0.80
      :vital-max-hp VITAL-MAX-HP}})

(defn calculate-phyllotaxis-coords
  "Calculates (r, theta) polar coordinates for n-th leaf or seed using golden angle."
  [n c-spread]
  (let [r (* c-spread (math/sqrt n))
        golden-rad (* PHYLLOTAXIS-GOLDEN-ANGLE-DEG (/ 3.1415926535 180.0))
        theta (* n golden-rad)
        x (* r (math/cos theta))
        z (* r (math/sin theta))]
    @{:n n
      :radius r
      :theta theta
      :x x
      :z z
      :vital-max-hp VITAL-MAX-HP}))

(defn evaluate-phenomenon-energy-flux
  "Computes resonant energy flux (W/m2) based on frequency and intensity."
  [phenom-key intensity]
  (let [spec (get CANONICAL-MASTER-PHENOMENA phenom-key)
        freq (if spec (get spec :frequency-hz) 100.0)
        norm-intensity (min 1.0 (max 0.1 intensity))
        flux (* freq norm-intensity 0.001618)]
    @{:phenom phenom-key
      :intensity norm-intensity
      :energy-flux flux
      :vital-max-hp VITAL-MAX-HP}))

(defn get-botanical-stratum
  "Retrieves botanical stratum specification."
  [stratum-key]
  (get CANONICAL-BOTANICAL-STRATA stratum-key))
