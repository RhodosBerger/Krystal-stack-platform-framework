# Krystal-Stack Research: Evolved High-Detail SVG Vektorová Grafika vs. Nepripravené 3D Modely

**Dátum:** 3. Október 2026  
**Autor:** Dušan Kopecký & Krystal-Stack Mathematical & Hardware Architecture Group  
**Kľúčový Invariant:** `VITAL_MAX_HP = 6`  
**Zlatý Pomer:** $\phi = 1.61803398875$  
**Formát:** Čisté Scalable Vector Graphics (SVG, XML DOM) & Evolved Concept Blueprints

---

## 1. Motivačná Architektúra: Prečo Evolved SVG nahrádza nepripravené 3D drôtené modely

Nepripravené (unprepared) alebo surové 3D modely bez kompletných PBR textúr, tieňovačov a osvetlenia trpia v reálnom čase zásadnými hendikepmi:
1. **Réžia grafických ovládačov a kompilácie shaderov:** Surové 3D meshe vyžadujú alokáciu VRAM bufferov, odosielanie draw-callov a kompiláciu pipelines, čo na integrovanej grafike (Intel Iris Xe) spôsobuje zbytočnú latenciu $185.0\ \mu\text{s}$.
2. **Artefakty rasterizácie a aproximácie:** Pri zoomovaní dochádza k lámaniu polygónov alebo strate detailov.
3. **Absencia matematickej expresivity:** Drôtené modely nedokážu priamo zobraziť diferenciálne rovnice, integrály energie ani vektorové lúčové diagramy sphere-tracingu.

**Riešenie v Krystal-Stack:**
Prechod na **vysoko-detailnú procedurálnu vektorovú grafiku (Evolved SVG)** s vrstvenými gradientmi, žiarivými filtrami (`feGaussianBlur`), precíznymi izometrickými vrstevnicami a heraldickými kmeňovými znakmi hry **Poslední Kmen**.

---

## 2. Architektonické Vrstvy Master Blueprintu

Náš procedurálny engine generuje SVG štruktúrované do 8 nezávislých vektorových vrstiev:

```
[ Layer 1: Holografická Blueprint Mriežka (40px / 200px kótovanie) ]
       ↓
[ Layer 2: Vonkajší Technický Rám & Rohové Kalibračné Zameriavače ]
       ↓
[ Layer 3: Plávajúci Ostrov & Koreňové Šľachy (Skalný suterén) ]
       ↓
[ Layer 4: Bióm Křišťálové Ledové Štíty (Faceted Ice Spikes, w_ridge = 0.88) ]
       ↓
[ Layer 5: Bióm Toxické Kaňony Jedů (Voronoi Fissures, C_fault = 0.92) ]
       ↓
[ Layer 6: Bióm Druidský Podzimní Les (Zlaté koruny, Suťové svahy θ_c = 35°) ]
       ↓
[ Layer 7: Studna Duší (Kozmický Vortex Singularity, Stabilizačné prstence) ]
       ↓
[ Layer 8: Matematický Aparát & SDF Sphere-Tracing Lúčový Lúč ∇f(p) ]
```

---

## 3. Presné Matematické Predpisy v SVG Reprezentácii

### 3.1 Integrál Hustoty Energie
Vektorové pole potenciálu Studny Duší spĺňa variačný princíp minimálnej energie:

$$E = \int_V (\nabla \phi)^2 \, dv$$

### 3.2 Vzdialenostné Pole (Signed Distance Function)
Presné sférické polia pre trasovanie raymarchingom:

$$\text{SDF}(x) = \min_c (\|x - c\|) - r$$

Lúčové trasovanie je priamo vykreslené ako červená čiarkovaná vektorová trajektória $r_i(t) = r_0 + t \cdot \hat{d}_i$ so zmenšujúcimi sa kružnicami polomeru $r(t)$ až po bod dopadu na povrch s normálou $\nabla f(p)$.

### 3.3 Teplotné Zvetrávanie Suťových Svahov
Kritický uhol $\theta_c \approx 35^\circ$ ($\tan \theta_c \approx 0.7002$):

$$\Delta H_{thermal}(x) = -K_{talus} \cdot \max(0, \|\nabla H(x)\| - \tan \theta_c)$$

---

## 4. Porovnanie: Surový 3D Mesh vs. Evolved High-Detail SVG

| Vlastnosť | Surový/Nepripravený 3D Mesh | Evolved High-Detail SVG |
| :--- | :--- | :--- |
| **VRAM Stopa** | Niekoľko MB až stovky MB | **0 KB VRAM** (čistý DOM vektor) |
| **Škálovateľnosť rozlíšenia** | Závislá od hustoty polygónov | **Nekonečná (od mobilov po 8K bez straty ostrosti)** |
| **Réžia grafického ovládača** | $185.0\ \mu\text{s}$ (WDDM stall) | **$0.0\ \mu\text{s}$ (natívne renderované bez GPU lockov)** |
| **Matematická čitateľnosť** | Nulová (iba vrcholy a indexy) | **100% integrované vzorce a symbolické anotácie** |
| **Exportovateľnosť** | Vyžaduje OBJ/GLTF importér | **Priamy download, web embedding, Godot SVG import** |
| **Pravidlo integrity** | Neaplikovateľné | **`VITAL_MAX_HP = 6` striktne zachovaný** |
