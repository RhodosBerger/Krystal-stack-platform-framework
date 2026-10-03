# Krystal-Stack // Blender Core Modifiers & Mimicry Compositor Engine

> **Dátum:** 2. Október 2026  
> **Autor:** Architektonický tím Krystal-Stack & Dušan Kopecký  
> **Klasifikácia:** Procedurálna syntéza reálnych objektov, Blender Modifier Stack & Herné scény  

---

## 1. Exekutívne Zhrnutie

Tento dokument detailne popisuje návrh a implementáciu **Mimicry Compositor Engine** pre framework Krystal-Stack. Hlavnou úlohou enginu je:
1. **Mimikry reálnych objektov:** Schopnosť napodobniť a skladať komplexné reálne objekty sveta (bojové veže, bipedálne roboty, obelisky, organické bio-drony, mrakodrapy, vesmírne sondy) pomocou kombinácie fundamentálnych 3D primitív, Signed Distance Functions (SDF) a CSG operácií.
2. **Integrácia jadra Blenderu:** Prevzatie architektúry **Modifier Stacku** (vyhodnocovací orientovaný acyklický graf – DAG) s plnou podporou kľúčových modifikátorov:
   - `ArrayModifier` (lineárna aj radiálna multiplikácia objektov s transformáciami),
   - `MirrorModifier` (bilaterálna a osová symetria X/Y/Z),
   - `BooleanModifier` (zjednotenie, rozdiel, prienik s polynomickým hladkým prechodom $s_{\min}$),
   - `BevelModifier` (zaobľovanie hrán a faziet),
   - `DisplaceModifier` (harmonický 3D šum pre reliéfy a organické žily),
   - `DeformModifier` (krútenie Twist, skosenie Taper, ohyb Bend),
   - `SolidifyModifier` (duté steny škrupín).
3. **Osnovy a frameworky pre herné scény:** Šablóny a hierarchické grafy kombinujúce rôzne artefakty do hotových herných prostredí s exportom do **Godot 4.x** (`.tscn`), Vulkan GLSL shaderov a ASCII kamerového raymarchera.

---

## 2. Architektúra Vyhodnocovania Modifikátorov (Blender DAG)

V tradičných enginoch vyžaduje zmena geometrie deštruktívne prepočítavanie vrcholov (mesh editing). Krystal-Stack aplikuje **funkcionálny reťazec modifikátorov**, kde každý modifikátor obaľuje pôvodnú dištančnú funkciu $f_0(\mathbf{p})$:

```
┌────────────────────────────────────────────────────────┐
│               Základné 3D Primitívum                   │
│   Box / Cylinder / Sphere / Torus / Cone / HexPrism    │
│                  f_0(p) : R^3 -> R                     │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                    MODIFIER STACK                      │
│                                                        │
│  [1] DeformModifier (Twist / Taper / Bend)             │
│      p_1 = T_deform(p)                                 │
│                                                        │
│  [2] MirrorModifier (Symmetry Fold X/Y/Z)              │
│      p_2 = (|p_1.x|, p_1.y, p_1.z)                     │
│                                                        │
│  [3] ArrayModifier (Radial / Linear Multiplicity)      │
│      d_3 = min_i f(R_i * p_2 - offset_i)               │
│                                                        │
│  [4] DisplaceModifier (Procedural 3D Noise)            │
│      d_4 = d_3 + A * sin(f*x)*cos(f*y)*sin(f*z)        │
│                                                        │
│  [5] BevelModifier / Solidify                          │
│      d_final = |d_4| - thickness - radius              │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│               CSG Smooth Boolean (smin)                │
│       Kombinácia viacerých častí do celého tela        │
└────────────────────────────────────────────────────────┘
```

### Matematická formulácia polynomického hladkého minima ($s_{\min}$)

Pre plynulé organické aj mechanické spájanie jednotlivých častí (napr. prechod trupu veže do kĺbu gimbálu) sa namiesto ostrého minima $\min(a, b)$ používa Quilezov/Blender polynomický hladký minimum s polomerom prechodu $k$:

$$s_{\min}(a, b, k) = \min(a, b) - \frac{\max(k - |a - b|, 0)^2}{4k}, \quad k > 0$$

Pre hladké odčítanie (vyrezávanie priezorov, kokpitov a dutín):
$$\text{smooth\_diff}(a, b, k) = s_{\max}(a, -b, k) = \max(a, -b) + \frac{\max(k - |a - (-b)|, 0)^2}{4k}$$

---

## 3. Katalóg Reálnych Objektových Mimikier

Modul [`mimicry_engine/mimic_recipes.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/mimicry_engine/mimic_recipes.py) obsahuje 6 detailne zostavených procedurálnych receptov:

| ID Receptu | Názov | Kategória | Zložené Časti a Aplikované Modifikátory |
| :--- | :--- | :--- | :--- |
| **`CYBER_TURRET_MK4`** | Cyber Turret MK-4 Automated Sentry | Vojenský hardvér | **PedestalBase** (`HexPrism` + `Bevel`), **GimbalCore** (`Sphere` + `smin`), **TurretMantlet** (`RoundBox`), **DualBarrels** (`Cylinder` + `ArrayModifier` offset X), **AmmoPods** (`Torus` + `MirrorModifier` X). |
| **`MECH_WALKER_TITAN`** | Titan-IV Heavy Bipedal Mech | Robotika & Vozidlá | **CockpitPod** (`RoundBox` + `DeformModifier Taper`), **SensorVisor** (`Box` smooth subtraction), **PelvisGirdle** (`Cylinder`), **LegStruts** (`Capsule` + `MirrorModifier` X), **TractionPads** (`RoundBox` + `MirrorModifier` X). |
| **`ANCIENT_OBELISK_MONOLITH`** | Ancient Runic Obelisk Monolith | Staroveké monumenty | **SteppedPlinth** (`Box` + `ArrayModifier` 3 stupne), **ObeliskSpire** (`CappedCone` + `DisplaceModifier` runový reliéf), **FloatingCapstone** (`Octahedron` + `Bevel`). |
| **`BIOMECHANICAL_XENODRONE`** | Xenobiotic Swarm Drone | Biologické organizmy | **ChitinCranium** (`RoundBox` + `DisplaceModifier` žily), **SpinalVertebrae** (`Torus` + `ArrayModifier` 4 stavce pozdĺž osi), **Mandibles** (`CappedCone` + `MirrorModifier` X). |
| **`CYBERPUNK_DATA_SPIRE`** | Megacity Central Data Spire | Urbanistická architektúra | **CentralCore** (`TallBox`), **ServerDecks** (`RoundBox` + `ArrayModifier` 5 poschodí), **HoloRingCrown** (`Torus`), **CoolantFins** (`Box` + `MirrorModifier` X). |
| **`RETRO_SOLAR_EXPLORER`** | Solar Explorer Research Vessel | Vesmírne plavidlá | **CommandSphere** (`Sphere`), **IonNozzles** (`CappedCone` + `ArrayModifier Radial` 4 trysky), **SolarWings** (`Box` + `MirrorModifier` X), **HighGainDish** (`CappedCone` + `SolidifyModifier`). |

---

## 4. Osnovy Herných Scén (Game Scene Frameworks)

V module [`mimicry_engine/scene_composer.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/mimicry_engine/scene_composer.py) sú definované 4 ucelené herné prostredia:

### 4.1 `SCENE_CYBERPUNK_DISTRICT` (Kyberpunkový Megacity Distrikt)
- **Kompozícia:** Dominantný centrálny Data Spire (`scale=1.35`), sekundárny bočný spire (`scale=0.9`), perimetrové obranné veže Cyber Turret MK-4 na vyvýšených rampách.
- **Atmosféra:** Tmavomodrá hmla (`fog_color=(0.02, 0.03, 0.06)`), tyrkysovo-modré neónové osvetlenie, reflexný povrch vozovky.
- **Využitie:** Mestské RPG, taktické stealth misie, kyberpunkové prestrelky.

### 4.2 `SCENE_SACRED_ALCHEMICAL_RUINS` (Chrám Tisícich Rún)
- **Kompozícia:** Kardinálny kruh štyroch Obelisk Monolitov (`radius=2.4m`, rotácie $0^\circ, 90^\circ, 180^\circ, 270^\circ$) obklopujúcich centrálnu levitujúcu svätyňu.
- **Atmosféra:** Zlatistá jantárová žiara (`ambient_light=(0.3, 0.2, 0.08)`), runový prach.
- **Využitie:** Mystické hádanky, arény s bossmi, posvätné rituálne svätyne.

### 4.3 `SCENE_DEEP_SPACE_HANGAR` (Hangárový Dok Hviezdnej Základne)
- **Kompozícia:** Výskumná loď Solar Explorer v nulovej gravitácii, hliadkujúci bipedálny Titan-IV Mech Walker na pristávacej ploche, strážna veža.
- **Atmosféra:** Chladné vesmírne modro-šedé svetlo (`ambient_light=(0.15, 0.2, 0.35)`), priemyselné konštrukcie.
- **Využitie:** Sci-Fi vesmírne stanice, údržbárske doky, odletové zóny.

### 4.4 `SCENE_ALIEN_HIVE_CHAMBER` (Organická Liahňa Xenodronov)
- **Kompozícia:** Matriarchálny bio-dron (`scale=1.2`) na centrálnom kokóne, robotnícke drony visiace zo stien jaskyne.
- **Atmosféra:** Bio-luminiscenčná zelená hmla (`ambient_light=(0.1, 0.25, 0.12)`), organické rebrové piliere.
- **Využitie:** Survival horror, mimozemské liahne, biologické jaskyne.

---

## 5. Integrácia s Godot Engine 4.x & Vulkan

Všetky scény a kompozitné objekty sú natívne exportovateľné:
1. **Deklaratívny generátor `.tscn`:** Metóda `export_godot_tscn()` automaticky vytvorí kompletný textový súbor scény Godot 4 (s kamerou, svetlami, environmentom a prepojením na `KrystalHoloBridge.gd`).
   - Súbory sú okamžite generované v [`godot_project/scenes/`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/godot_project/scenes/):
     - `SCENE_CYBERPUNK_DISTRICT.tscn`
     - `SCENE_SACRED_ALCHEMICAL_RUINS.tscn`
     - `SCENE_DEEP_SPACE_HANGAR.tscn`
     - `SCENE_ALIEN_HIVE_CHAMBER.tscn`
     - `MimicryCompositorStage.tscn`
2. **GPU Screen-Space Raymarcher:** Súbor [`godot_project/shaders/mimicry_sdf_compositor.gdshader`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/godot_project/shaders/mimicry_sdf_compositor.gdshader) umožňuje Godot enginu vyhodnocovať kompozitné SDF objekty priamo na grafickej karte s nulovou záťažou na CPU.

---

## 6. Ovládanie v Localhost Web Hube (http://localhost:8080)

Do webového dashboardu boli integrované nové ovládacie prvky:
* **Mód `09: MIMICRY OBJ`:** Prepne hlavný terminálový canvas na raymarching zvoleného kompozitného objektu.
* **Mód `10: GAME SCENE`:** Prepne terminálový canvas na celú multi-objektovú 3D scénu s kamerovým orbitom.
* **Panel "BLENDER CORE MODIFIERS & MIMICRY":**
  * Výber reálneho objektu (Turret, Mech, Obelisk, Xenodrone, Spire, Explorer).
  * Výber hernej scény (Cyber District, Alchemical Ruins, Space Hangar, Alien Hive).
  * Dynamické zobrazenie aktívneho Modifier Stacku v reálnom čase (Array, Mirror, Bevel, Displace).
  * Tlačidlá: `RENDER OBJECT`, `RENDER SCENE`, `EXPORT GODOT .TSCN`.

---

## 7. Verifikácia a Výsledky Testov

Testovací skript [`tests/test_mimicry_compositor.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/tests/test_mimicry_compositor.py) overil:
- **Test 1:** Všetky 3D primitíva a Quilezov $s_{\min}$ CSG operátor – **PASSED**.
- **Test 2:** Blender modifier stack (Array, Mirror, Bevel, Displace, Deform) – **PASSED**.
- **Test 3:** 6 reálnych kompozitných receptov a ich ASCII raymarching projekcie – **PASSED**.
- **Test 4:** 4 herné scény a Godot 4 `.tscn` export – **PASSED**.
- **Test 5:** Live REST API na `http://127.0.0.1:8080` (`/api/mimicry/objects`, `/api/mimicry/scenes`, `/api/mimicry/select`, `/api/mimicry/export-godot`) – **PASSED**.

Systém je pripravený na produkčné nasadenie, generovanie herných levelov a experimentovanie v reálnom čase.
