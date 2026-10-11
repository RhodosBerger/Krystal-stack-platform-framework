# Krystal-Stack // Godot Engine 4.x Holographic ASCII & Recursive AR Mirror Integration

Tento balík integruje **Godot Engine 4.x** (Vulkan Forward+ / Compatibility) s platformou Krystal-Stack a Antigravity Prompt Engine.

---

## 💎 Architektúra integrácie

```
┌────────────────────────────────────────────────────────┐
│             ANTIGRAVITY PROMPT ENGINE                  │
│  "zrkadli 6-uholníkovú posvätnú geometriu v alchýmii"   │
└───────────────────────────┬────────────────────────────┘
                            │ Syntéza parametrov & D_N Foldov
                            ▼
┌────────────────────────────────────────────────────────┐
│               GODOT ENGINE 4.x (VULKAN)                │
│  • 3D Scény: HoloStage.tscn, RecursiveMirrorARStage.tscn│
│  • Shadery: ascii_hologram.gdshader,                   │
│             recursive_ar_mirror.gdshader               │
│  • Push Constants: u_mirror_folds, u_fresnel_factor,   │
│                    u_recursion_limit, u_scanlines      │
└───────────────────────────┬────────────────────────────┘
                            │ Screen-Space Post-Process
                            ▼
┌────────────────────────────────────────────────────────┐
│           RECURSIVE DIHEDRAL AR REFLECTION             │
│  • Dihedrálne skladanie rovín (D3 až D16)               │
│  • Schlick-Fresnel reflexné útlmy                      │
│  • Kvantizácia do ASCII matice v reálnom čase          │
└───────────────────────────┬────────────────────────────┘
                            │ GDScript Telemetry Bridge (`KrystalHoloBridge.gd`)
                            ▼
┌────────────────────────────────────────────────────────┐
│              KRYSTAL LOCALHOST HUB (8080)              │
│  • Web Dashboard & Antigravity Studio (127.0.0.1:8080) │
│  • Server-Sent Events (SSE) Stream (30+ FPS)           │
│  • Vizuálna entropia & Ekonomický guvernér Gamesa      │
└────────────────────────────────────────────────────────┘
```

---

## 🚀 Dostupné 3D scény v Godot

1. **`scenes/HoloStage.tscn`**:
   - Rotujúci diamantový Krystal hranol s objemovými laserovými interferenčnými pruhmi ($632.8\text{ nm}$).
   - Shader: `shaders/ascii_hologram.gdshader` (Stereoskopický anaglyf rozptyl cyan/magenta).

2. **`scenes/RecursiveMirrorARStage.tscn`**:
   - Posvätný toroidálny orbitál s rekurzívnym zrkadlením dihedrálnych rovín ($D_6$ symetria, 12-násobná symetria).
   - Shader: `shaders/recursive_ar_mirror.gdshader` (Rekurzívne zrkadlenie, Fresnel reflexie, spektrálna chromatická aberácia).

---

## 🎨 Vulkan Shader Uniformy (`recursive_ar_mirror.gdshader`)

V inšpektore uzla `RecursiveMirrorRect` alebo cez Antigravity API je možné riadiť tieto parametre:

| Parameter Uniformu | Typ | Rozsah | Popis |
| :--- | :--- | :--- | :--- |
| `u_mirror_folds` | `int` | $2 - 16$ | Počet dihedrálnych zrkadlových rovín ($D_N$) |
| `u_fold_angle` | `float` | $0.1 - 1.57$ | Uhol zrkadlenia: $\alpha = \frac{\pi}{N}$ (radiány) |
| `u_recursion_limit` | `int` | $1 - 16$ | Hĺbka rekurzívnych odrazov svetelných lúčov |
| `u_fresnel_factor` | `float` | $0.1 - 1.0$ | Základná odrazivosť povrchu podľa Schlickovej aproximácie |
| `u_chromatic_aberration` | `float` | $0.0 - 0.05$ | Chromatický posun spektra pri rekurzívnych odrazoch |
| `u_primary_color` | `Color` | RGB | Hlavná neónová paleta generovaná Antigravity promptom |
| `u_accent_color` | `Color` | RGB | Akcentová žiarivá farba pre hrany a runové symboly |
| `cell_size` | `Vector2` | $4\times 8$ až $16\times 32$ | Rozmery bunky ASCII znaku v pixeloch |

---

3. **`scenes/posledni_kmen_arena.tscn`**:
   - Taktická ťahová aréna pre *Poslední Kmen* (19 hexových polí, živly cryst/slime/bark/basalt).
   - Skript: `scripts/PosledniKmenDuelController.gd` s implementáciou The West súbojovej algebry (Head, Shoulders, Torso), ženijných zákopov (Sapper Trenches) a mínometnej plošnej dikcie (CEP dispersion).
   - Striktný invariant: `VITAL_MAX_HP = 6` (6 diskrétnych pips / sŕdc života).
   - Prepojené na HTTP REST API Krystal Hubu (`http://127.0.0.1:8080/api/game/...`) s automatickým offline núdzovým fallbackom.

---

## 🛠️ Ako spustiť a prepojiť s Localhost Hubom

1. Spustite Krystal Localhost Hub na Windows:
   ```cmd
   start_localhost.bat
   ```
2. Otvorte webovú hru na `http://127.0.0.1:8080/game` (alebo `/arena` či `/posledni-kmen`).
3. V Godot Editore otvorte projekt [godot_project/project.godot](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/godot_project/project.godot).
4. Otvorte a spustite scénu [scenes/posledni_kmen_arena.tscn](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/godot_project/scenes/posledni_kmen_arena.tscn) (stlačte **F6**).
5. Alebo spustite priamo cez Godot CLI:
   ```cmd
   godot --path godot_project res://scenes/posledni_kmen_arena.tscn
   ```

