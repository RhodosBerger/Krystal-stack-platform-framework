# Znalostná Báza: GDM Display Manager, GNOME Režimy Okien, WSL + Godot 4 WebGL2, Anno Modding → 60s Cyberpunk & Propedeutika Príbehu

> **Ústavná kotva:** `VITAL_MAX_HP == 6` (`\neg Valid \implies \Delta S = 0`) · **Zbernica UMA:** `< 0.78` (`30 %` šírky plochy pre živý graf UMA zbernice)

---

## 1. Nadstavba GDM (`GNOME Display Manager`), GNOME Shell & Moderné Linux Režimy Okien
- **GDM (`GNOME Display Manager`)** riadi grafické sedenia (`systemd-logind`, správa KMS/DRM lease, štart Wayland kompozítora `Mutter` a odovzdanie kontextu pre Xwayland/WSLg). Na seba viaže správu autentifikácie, prepínanie používateľských profilov, multi-seat priradenie GPU a výber režimu relácie (`.desktop` session súbory v `/usr/share/xsessions` a `/usr/share/wayland-sessions`).
- **5 Režimov Okien Inšpirovaných Modernými Linux Kompozítormi (`COSMIC`, `Niri`, `Hyprland`)**:
  1. `MODE_01_COSMIC_AUTO_TILING` (*System76 COSMIC*): Automatické dlaždicové delenie obrazovky jedným prepínačom bez zmenšovania čitateľnosti telemetrie,
  2. `MODE_02_NIRI_SCROLLABLE_STRIP` (*Niri / PaperWM*): Nekonečný horizontálny pás stĺpcov s plnou šírkou — otváranie nových kariet nezmenšuje existujúce okná,
  3. `MODE_03_HYPRLAND_DWINDLE_SHADER` (*Hyprland*): Dynamické špirálové delenie (`dwindle`) so živým hraničným shaderom Adrenalin (`#ed1c24` → `#00e5ff`),
  4. `MODE_04_STACKED_TABBED_DECK` (*COSMIC Stacked*): Vrstvenie viacerých analytických kariet do jedného dlaždicového slotu s rýchlym prepínaním záložiek,
  5. `MODE_05_FLOATING_HUD_OVERLAY`: Plávajúce priesvitné okná pre Free-Run a bojové priebehy.

---

## 2. Spúšťanie cez WSLg v Termináli & Zobrazenie WebGL2 / GLES3 Procedúr cez Godot 4
- Vo Windows 11 zabezpečuje **WSLg (`WSL GUI`)** priamy Wayland + Weston + Mesa `d3d12`/`zink` most medzi Linux terminálom a GPU (`Intel Iris Xe` / `Arc`).
- Modul [launch_wsl_gnome_godot.sh](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/deploy/gnome_gdm_wsl_godot/launch_wsl_gnome_godot.sh) spúšťa projekt [project.godot](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/deploy/gnome_gdm_wsl_godot/godot_project/project.godot) s profilom `gl_compatibility` (zdieľaným medzi natívnym **GLES3** oknom a **WebGL 2.0** HTML5 exportom) a harmonickým kosínusovým vykresľovačom [PanopticonProceduralWebGL.gd](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/deploy/gnome_gdm_wsl_godot/godot_project/scripts/PanopticonProceduralWebGL.gd).

---

## 3. Moderská Komunita Série Anno (Ubisoft) & Protokol Odpojenia do 60s Cyberpunk Estetiky
- **Kľúčové mechanizmy z moderskej komunity Anno (`Anno 1404 IAAM`, `Anno 2070 ARRC`, `Anno 2205`, `Anno 1800`)**:
  - **`.rda` archívy a `RDA Explorer`:** Kontajnerový systém pre textúry, rastrové dlaždice a konfigurácie,
  - **`assets.xml` + XPath `ModOp` záplaty (`add`, `merge`, `replace` podľa `GUID`):** Deklaratívne rozširovanie katalógu budov a výrobných reťazcov bez prepisovania pôvodných súborov hry,
  - **`.cfg` a `.ifo` súbory + `FileDBReader`:** Definujú kolíznu stopu mriežky (`build-blocker`) a **procedurálne doplnky (`procedural props`)**, ktoré oživujú okolie stavieb v opakujúcich sa slučkách.
- **Protokol odpojenia od stereotypu Anno 2070 (`PROTO_DECOUPLE_ANNO_TO_60S_CYBERPUNK`)**:
  - Preberáme čistú logistickú čitateľnosť mriežky, opakovacie rastrové slučky a deklaratívny `ModOp` katalóg objektov,
  - **Odmietame** sterilný korporátny bielo-modrý sci-fi stereotyp Anno 2070 a nahrádzame ho naším **Dizajn Manuálom (`60s Space-Age Retro-Futurizmus × Cyberpunk × AMD Adrenalin`)**: obsidiánový podklad `#0a0e17`, jantárové osciloskopy Apollo DSKY `#ff9e2c`, karmínová hrana `#ed1c24` a holografické kmene `#00e5ff` / `#34d399` / `#eab308`.
- **C-terminológia rastrového škálovania ([krystal_state_before_after_scaler.c](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/src/c/krystal_state_before_after_scaler.c))**:
  - Prechod `state_before` $\xrightarrow{\text{state\_a}}$ `state_after` škáluje obraz v `16.16` pevnej rádovej čiarke a vkladá procedurálne objekty pri zachovaní invariantu $\neg\text{Valid}(\text{state\_a}) \implies \text{state\_after} \equiv \text{state\_before}$.

---

## 4. Päť Kľúčových Vlastností pre Pipeline Monitoringu Propedeutiky Príbehových Aspektov Hry
1. **`PROP_01_COGNITIVE_ONBOARDING_SLOPE` ($\nabla C_k \le 0.618$):** Prírastok nových pravidiel, kariet a symbolov na jednu príbehovú kapitolu nesmie prekročiť inverzný zlatý rez voči už zvládnutým čankom.
2. **`PROP_02_CAUSAL_CHOICE_TRACEABILITY`:** Okamžitá čitateľnosť vzťahu *zahraná karta / dialóg $\to$ odomknutý quest $\to$ zmena v Panoptiku*.
3. **`PROP_03_PACING_TENSION_RELIEF_RATIO`:** Vyvážené striedanie intenzívnych bojových / Free-Run pasáží s exploratívnym zberom surovín a dejovým zásuvkovým oknom.
4. **`PROP_04_DIEGETIC_SPATIAL_SIGNPOSTING`:** Schopnosť hráča orientovať sa podľa samotnej estetiky priestoru (farba lomu, radarový kmitočet, hustota hmly) bez textových nápovied.
5. **`PROP_05_PANOPTICON_ATTENTION_COHERENCE`:** Pomer vnímaných živých udalostí v Panoptiku k celkovej vizuálnej entropii ($E_{\text{total}} \le 0.70$, zbernica `UMA < 0.78`).
