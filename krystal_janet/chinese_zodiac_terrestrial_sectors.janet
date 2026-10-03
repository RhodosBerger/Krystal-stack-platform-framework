# Krystal-Stack Janet DSL // Chinese Zodiac Terrestrial Phenomena & 12 Sectors
# Mappuje 12 pozemských vetiev (地支) a zvierat čínskeho zverokruhu na samostatné sektory
# a fyzikálne/terestriálne javy v mriežke sveta.
# Invariant: VITAL-MAX-HP = 6

(def VITAL-MAX-HP 6)
(def GOLDEN-RATIO 1.61803398875)
(def INV-GOLDEN-RATIO 0.61803398875)

(def CANONICAL-ZODIAC-SECTORS
  @{:sector-01-rat
    @{:name "Potkan (Rat / 鼠)" :branch "Zi" :element "Water" :polarity "Yang"
      :phenomenon "Nočná Hydrologická Spodná Voda" :direction "North" :max-hp VITAL-MAX-HP}
    :sector-02-ox
    @{:name "Byvol (Ox / 牛)" :branch "Chou" :element "Earth" :polarity "Yin"
      :phenomenon "Kryo-Tektonické Vrstvenie Žuly" :direction "NNE" :max-hp VITAL-MAX-HP}
    :sector-03-tiger
    @{:name "Tiger (Tiger / 虎)" :branch "Yin" :element "Wood" :polarity "Yang"
      :phenomenon "Ranné Blesky & Kinetická Lesná Búrka" :direction "ENE" :max-hp VITAL-MAX-HP}
    :sector-04-rabbit
    @{:name "Zajac (Rabbit / 兔)" :branch "Mao" :element "Wood" :polarity "Yin"
      :phenomenon "Ranná Hmla & Rast Fyto-Spór" :direction "East" :max-hp VITAL-MAX-HP}
    :sector-05-dragon
    @{:name "Drak (Dragon / 龙)" :branch "Chen" :element "Earth" :polarity "Yang"
      :phenomenon "Geomagnetická Búrka & Magmatický Gejzír" :direction "ESE" :max-hp VITAL-MAX-HP}
    :sector-06-snake
    @{:name "Had (Snake / 蛇)" :branch "Si" :element "Fire" :polarity "Yin"
      :phenomenon "Zemný Plyn & Geotermálna Štrbina" :direction "SSE" :max-hp VITAL-MAX-HP}
    :sector-07-horse
    @{:name "Kôň (Horse / 马)" :branch "Wu" :element "Fire" :polarity "Yang"
      :phenomenon "Poludňajší Solárny Žiar & Termálna Púšť" :direction "South" :max-hp VITAL-MAX-HP}
    :sector-08-goat
    @{:name "Koza (Goat / 羊)" :branch "Wei" :element "Earth" :polarity "Yin"
      :phenomenon "Sprašová Usadenina & Minerálne Ložisko" :direction "SSW" :max-hp VITAL-MAX-HP}
    :sector-09-monkey
    @{:name "Opica (Monkey / 猴)" :branch "Shen" :element "Metal" :polarity "Yang"
      :phenomenon "Horský Vzdušný Vír & Ionosférická Rezonancia" :direction "WSW" :max-hp VITAL-MAX-HP}
    :sector-10-rooster
    @{:name "Kohút (Rooster / 鸡)" :branch "You" :element "Metal" :polarity "Yin"
      :phenomenon "Zrkadlenie Rudných Žíl & Magnetit" :direction "West" :max-hp VITAL-MAX-HP}
    :sector-11-dog
    @{:name "Pes (Dog / 狗)" :branch "Xu" :element "Earth" :polarity "Yang"
      :phenomenon "Seizmická Hliadka & Pôdna Odolnosť Kôry" :direction "WNW" :max-hp VITAL-MAX-HP}
    :sector-12-pig
    @{:name "Prasa (Pig / 猪)" :branch "Hai" :element "Water" :polarity "Yin"
      :phenomenon "Aluviálna Bažina & Sedimentárne Úložisko" :direction "NNW" :max-hp VITAL-MAX-HP}})

(defn get-zodiac-sector
  "Returns sector data for a specific sector key."
  [sec-key]
  (get CANONICAL-ZODIAC-SECTORS sec-key))

(defn calculate-terrestrial-phenomenon-flux
  "Calculates resonance flux of an earthly phenomenon scaled by intensity."
  [base-intensity turn-idx]
  (let [stride (* turn-idx INV-GOLDEN-RATIO)
        wave (+ 1.0 (* 0.35 (math/sin stride)))]
    (min 3.0 (* base-intensity wave))))

(defn evaluate-ruling-earthly-branch
  "Evaluates the ruling sector and phenomenon for a given cycle turn."
  [turn]
  (let [idx (mod (- turn 1) 12)
        keys [:sector-01-rat :sector-02-ox :sector-03-tiger :sector-04-rabbit
              :sector-05-dragon :sector-06-snake :sector-07-horse :sector-08-goat
              :sector-09-monkey :sector-10-rooster :sector-11-dog :sector-12-pig]
        active-key (get keys idx)
        sec (get CANONICAL-ZODIAC-SECTORS active-key)]
    @{:turn turn
      :vital-max-hp VITAL-MAX-HP
      :active-sector active-key
      :zodiac-name (get sec :name)
      :branch (get sec :branch)
      :element (get sec :element)
      :phenomenon (get sec :phenomenon)}))
