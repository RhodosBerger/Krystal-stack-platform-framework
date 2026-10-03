# Praktický interaktívny tutoriál: Krystal-Stack od Nuly po Produkciu
**Praktický sprievodca krok za krokom pre Windows, WSL2, NPU a Godot Engine**  
*Verzia: 1.0 (2026) | Autori: Dušan Kopecký & Krystal-Stack Engineering Team*

---

## 🎯 Čo sa v tomto tutoriáli naučíte:
1. Spustiť a ovládať produkčný Localhost Hub s reálnym ASCII streamom.
2. Využívať nový **Holografický 3D Volumetrický Projektor**.
3. Prepojiť Windows s linuxovým subsystémom **WSL2** pomocou mosta (`wsl_bridge`).
4. Prepojiť herný engine **Godot 4.x** s neurálnym kompozitorom.
5. Napísať vlastný kognitívny príkaz a sledovať reakciu Economic Governora.

---

## KROK 1: Spustenie Localhost Mission Control Hubu

Localhost hub je navrhnutý tak, aby fungoval na čistom Windowse bez nutnosti kompilácie zložitých C++ knižníc.

### 1. Spustenie servera
V koreňovom adresári spustite:
```powershell
# Jedným príkazom:
python start_localhost.py
```
*(Alebo dvakrát kliknite na `start_localhost.bat`).*

### 2. Overenie vo webovom prehliadači
Otvorte si stránku: **[http://localhost:8080/](http://localhost:8080/)**  
Uvidíte živý kyberpunkový terminál, kde beží 30–60 FPS stream, animované budíky entropie a ukazovateľ rozpočtu kreditov.

---

## KROK 2: Testovanie Holografického 3D Projektora

Vytvorili sme dedikovaný holografický projektor simulujúci laserové interferenčné prúžky a chromatickú anaglyfnú hĺbku.

### Ako prepnúť na Holografický režim:
1. Na webovom dashboarde kliknite na tlačidlo **`[07 HOLOGRAPHIC 3D]`**.
2. Alebo v termináli pošlite REST požiadavku:
   ```powershell
   Invoke-RestMethod -Uri "http://localhost:8080/api/control" -Method Post -ContentType "application/json" -Body '{"mode":"HOLOGRAPHIC_3D"}'
   ```
3. Sledujte, ako sa obraz pretransformuje na volumetrický kryštál s rotujúcim prstencom a interferenčnými bodkami (`⋄`, `◇`, `◈`, `◆`, `█`).

---

## KROK 3: Prepojenie s Microsoft WSL2 (Linux Subsystém)

Ak chcete využiť pôvodné linuxové moduly (Gamesa Cortex V2, RAPL sysfs, Linux Vulkan):

### 1. Príprava WSL prostredia
Otvorte PowerShell v koreňovom adresári a spustite:
```powershell
wsl bash wsl_bridge/setup_wsl_env.sh
```
Skript overí prítomnosť `/dev/dxg` (Direct3D 12 GPU passthrough) a nastaví potrebné knižnice.

### 2. Spustenie obojsmerného mosta
Spustite pripravený spúšťač:
```powershell
.\wsl_bridge\wsl_launcher.ps1
```
Most začne v reálnom čase čítať telemetriu Linuxového jadra (`/proc/stat`, `/proc/meminfo`) a preposielať ju do Windows rozhrania. Zároveň sprístupní unixový socket `/tmp/krystal_wsl.sock` pre aplikácie bežiace vo vnútri WSL.

---

## KROK 4: Integrácia s Godot Engine 4.x

V priečinku [`godot_project/`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/godot_project/) máte pripravený kompletný projekt pre Godot:

1. Spustite Godot 4.2+ a kliknite na **Import** $\rightarrow$ vyberte súbor `godot_project/project.godot`.
2. Otvorte scénu `scenes/HoloStage.tscn`.
3. Stlačte **F5**.
4. **Čo sa stane:**
   - 3D diamant v Godote sa vykreslí cez Vulkan Forward+ renderer.
   - Screen-space shader `shaders/ascii_hologram.gdshader` priamo na grafickej karte prepočíta pixely na holografické znaky.
   - Skript `scripts/KrystalHoloBridge.gd` zachytí počet draw-callov a FPS a pošle ich na `localhost:8080`.

---

## KROK 5: Práca s Kognitívnym Režisérom (SLM)

Kognitívny režisér prijíma pokyny v prirodzenom jazyku a prispôsobuje štýl zobrazenia hernej situácii:

1. Do poľa **`COGNITIVE SCENE DIRECTOR`** vľavo dole napíšte:
   ```text
   engage stealth mode
   ```
   *Reakcia:* Systém okamžite prepne štýl na `MATRIX_RAIN` (digitálny dážď).
2. Napíšte:
   ```text
   activate combat overload
   ```
   *Reakcia:* Systém prepne na `CYBERPUNK` s vysokým kontrastom.
3. Napíšte:
   ```text
   activate hologram
   ```
   *Reakcia:* Systém aktivuje volumetrický holografický lúč.

---

## KROK 6: Overenie cez Automatizovaný Testovací Balík

Kedykoľvek môžete spustiť kompletnú verifikáciu zdravia systému:
```powershell
python verify_localhost_env.py
```
Ak všetkých 7 testov prejde, prostredie je v 100 % prevádzkyschopnom stave!
