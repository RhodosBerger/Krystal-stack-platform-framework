# Krystal-Stack Research: High-Fidelity 3D Zobrazovacie Vzory a Linux WSL DNF Model Import Pipeline

**Dátum:** 3. Október 2026  
**Autor:** Krystal-Stack 3D Graphics, Geometry Processing & Linux WSL Integration Group  
**Kľúčový Invariant:** `VITAL_MAX_HP = 6` (Pravidlo maximálnej celistvosti systému)  
**Zlatý Pomer:** $\phi = 1.61803398875$, $\phi^{-1} \approx 0.61803398875$

---

## 1. Úvod a Ciele Výskumu

Cieľom tejto implementácie je posunúť vizuálnu a geometrickú vernosť enginu Krystal-Stack na úroveň moderných AAA herných systémov (Godot 4.3+, Unreal Engine 5, Vulkan PBR):
1. **High-Fidelity Display Patterns (Zobrazovacie vzory vysokej vernosti):**
   - Fyzikálne presné osvetlenie (PBR Cook-Torrance GGX microfacet BRDF).
   - Tangenciálne normálové mapovanie s MikkTSpace ortogonalizáciou.
   - Kontaktné samotienenie (Screen-Space Horizon-based Ambient Occlusion - SSAO/HBAO).
   - ACES Filmic HDR tónové mapovanie eliminujúce prepálenie svetiel.
   - Viackanálové inšpekčné režimy (Normals heatmap, Roughness/Metallic kanály, Wireframe zlatá topológia).
2. **Nové procedurálne 3D modely s vysokou hustotou polygonálnej siete:**
   - `crystal_dragon_sanctuary.obj` (fraktálne kryštálové veže, viacstupňový sokel).
   - `zodiac_celestial_astrolabe.obj` (12 ramien čínskeho zverokruhu s Bagua zubami).
   - `cybernetic_titan_mech.obj` (kĺbový trup, hexagonálne náramenníky, hydraulické nohy).
   - `biomorphic_tree_of_life.obj` (dvojzávitnicový kmeň, 5 sférických korún).
3. **Linux WSL a balíčkovač DNF (Dandified YUM):**
   - Automatizácia prípravy prostredia WSL (Fedora / RHEL / Ubuntu).
   - Inštalácia 3D modelovacích nástrojov cez DNF: `assimp`, `assimp-tools`, `blender`, `vulkan-tools`.
   - Python-WSL most pre importovanie, validáciu manifoldnosti (Eulerova charakteristika $\chi = V - E + F = 2$) a konverziu formátov.

---

## 2. Matematické Modely PBR a Zobrazovacích Vzorov

### 2.1 Cook-Torrance Microfacet Spekulárna BRDF
$$f_r(l, v) = \frac{D(h, \alpha) \cdot F(v, h, F_0) \cdot G(l, v, h, \alpha)}{4(n \cdot l)(n \cdot v)}$$

- **GGX / Trowbridge-Reitz Normálová Distribúcia ($D$):**
  $$D(h) = \frac{\alpha^2}{\pi \left( (n \cdot h)^2 (\alpha^2 - 1) + 1 \right)^2}$$
- **Schlickova Fresnelova Aproximácia ($F$):**
  $$F(v, h) = F_0 + (1 - F_0)(1 - (v \cdot h))^5$$
- **Smithovo Geometrické Maskovanie ($G$):**
  $$G(l, v) = \frac{2(n \cdot l)(n \cdot v)}{(n \cdot v)\sqrt{\alpha^2 + (1-\alpha^2)(n \cdot l)^2} + (n \cdot l)\sqrt{\alpha^2 + (1-\alpha^2)(n \cdot v)^2}}$$

### 2.2 ACES Filmic Tónové Mapovanie
Krivka mapovania zamedzuje orezaniu sýtosti pri vysokých hodnotách expozície:
$$f(x) = \frac{x(2.51x + 0.03)}{x(2.43x + 0.59) + 0.14}$$

---

## 3. Linux WSL a DNF Balíčkovací Režim

DNF (Dandified YUM) je štandardný balíčkovač pre Fedora Linux a Red Hat Enterprise Linux (RHEL). V Krystal-Stack je WSL integrované ako konverzný a validačný procesor pre 3D modely:

```bash
# Inštalácia DNF a 3D nástrojov vo WSL:
sudo dnf install -y \
    assimp \
    assimp-tools \
    blender \
    vulkan-tools \
    mesa-vulkan-drivers \
    python3-numpy \
    python3-scipy
```

### 3.1 Kontrola Manifoldnosti a Topológie Siete
Pre uzavretú triangulovanú 3D sieť platí Eulerova-Poincaréova veta:
$$\chi = V - E + F = 2(1 - g)$$
kde $g$ je rod plochy (genus). Pre guľovú topológiu ($g=0$) platí $\chi = 2$.
Skript `scripts/wsl_dnf_setup.sh` a engine `high_fidelity_3d_display_and_wsl_importer.py` automaticky počítajú $\chi$ a overujú, či model nemá prevrátené alebo chýbajúce normály.

---

## 4. Architektúra Implementácie v Krystal-Stack

| Komponent | Súbor | Popis |
| :--- | :--- | :--- |
| **Python Engine** | [high_fidelity_3d_display_and_wsl_importer.py](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/economic_engine/high_fidelity_3d_display_and_wsl_importer.py) | Generovanie 3D modelov, katalóg presetov, WSL diagnostika a import. |
| **Janet DSL** | [high_fidelity_3d_display_and_wsl_importer.janet](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_janet/high_fidelity_3d_display_and_wsl_importer.janet) | Slovníky presetov `CANONICAL-DISPLAY-PRESETS` a BRDF výpočty. |
| **Linux WSL Setup Script** | [wsl_dnf_setup.sh](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/scripts/wsl_dnf_setup.sh) | Skript na inštaláciu DNF, Assimp a validačných nástrojov vo WSL. |
| **REST API (Port 8089)** | [krystal_engine_core.py](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/krystal_engine_core.py) | `/api/3d-models/catalog`, `/api/3d-models/wsl-diagnostic`, `/api/3d-models/bake-model`, `/api/3d-models/wsl-validate`. |
| **Interaktívne 3D Štúdio** | [high_fidelity_3d_studio.html](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/static/high_fidelity_3d_studio.html) | WebGL Three.js viewport, prepínač shaderov, PBR posuvníky, live WSL terminál. |
| **WebOS Desktop Integrácia** | [krystal_webos_desktop.html](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/static/krystal_webos_desktop.html) | Ikona `🎨 3D Displej & WSL`, položka v menu a samostatné okno. |
| **Platformový Invariant** | Všade | Striktné dodržanie `VITAL_MAX_HP = 6`. |
