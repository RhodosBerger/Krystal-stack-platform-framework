# Krystal-Stack Research: WSL Emulačný Subsystém a Virtuálne Linuxové Runtime Prostredie

**Dátum:** 3. Október 2026  
**Autor:** Krystal-Stack Linux Interop & Virtualization Systems Group  
**Kľúčový Invariant:** `VITAL_MAX_HP = 6` (Pravidlo maximálnej celistvosti systému)  
**Zlatý Pomer:** $\phi = 1.61803398875$, $\phi^{-1} \approx 0.61803398875$

---

## 1. Úvod a Motivácia

Moderné herné enginy a výpočtové systémy vyžadujú heterogénne prostredie: Windows pre nízkoúrovňové grafické ovládače (DirectX 12, Vulkan na Intel Iris Xe, WDDM), a Linux pre automatizované build skripty, balíčkovače (DNF, RPM, APT) a 3D konverzné nástroje (`assimp`, `blender`, `meshoptimizer`).

V mnohých prostrediach (izolované kontajnery, CI/CD bežce bez hardvérovej virtualizácie Hyper-V, školské či firemné počítače s vypnutým WSL2) však **fyzické WSL2 nie je dostupné**. 

Preto Krystal-Stack zavádza **WSL Emulačný Subsystém (`WslEmulationSubsystem`)**:
- Plne sebestačný virtuálny Linux runtime v pamäti (In-Memory POSIX Rootfs).
- Virtuálny balíčkovač **DNF 4.24** s transakčným mechanizmom, riešením závislostí a repozitármi (Fedora, Updates, Krystal-Stack Repo).
- Hybridný mostík (`Auto_Hybrid_Fallback`): ak je fyzické WSL2 na hostiteľskom Windows systéme detekované, engine transparentne využíva natívne Linuxové jadro; ak fyzické WSL zlyhá alebo chýba, prepína sa bez pádu aplikácie do virtuálneho emulovaného režimu.

---

## 2. Architektúra WSL Emulačného Subsystému

```
+-------------------------------------------------------------------+
|               KRYSTAL-STACK PLATFORM (WINDOWS HOST)               |
+-------------------------------------------------------------------+
                                  |
                                  v
                    +---------------------------+
                    |     Hybrid WSL Bridge     |
                    |   (WslRuntimeMode.AUTO)   |
                    +---------------------------+
                                  |
                +-----------------+-----------------+
                |                                   |
         (Ak WSL2 dostupné)                  (Ak WSL2 chýba)
                v                                   v
+-------------------------------+   +-------------------------------+
|     Fyzické WSL2 (Linux)      |   |   WSL Emulačný Subsystém      |
|  wsl.exe bash -c "<príkaz>"   |   |   - Virtual POSIX VFS         |
|  - Real Kernel 6.6+           |   |   - Virtual DNF 4.24 Engine   |
|  - Real dnf / assimp / blender|   |   - Virtual /etc, /proc, /mnt |
+-------------------------------+   +-------------------------------+
                |                                   |
                +-----------------+-----------------+
                                  v
+-------------------------------------------------------------------+
|         UNIFIKOVANÉ JSON API & WEBOVÝ INTERAKTÍVNY TERMINÁL       |
|            GET /api/wsl/subsystem-status | POST /api/wsl/exec     |
|            POST /api/wsl/dnf-emulate     | /wsl-emulator          |
+-------------------------------------------------------------------+
```

### 2.1 Virtuálny Súborový Systém (In-Memory POSIX VFS)
Subsystém simuluje štruktúru:
- `/etc/os-release`: Informácie o distribúcii Fedora Linux 40 (Cloud Edition).
- `/proc/version`: Jadro `6.18.33.2-krystal-virtual-WSL2 #1 SMP PREEMPT_DYNAMIC x86_64`.
- `/etc/dnf/dnf.conf`: Konfigurácia repozitárov a cache politík.
- `/mnt/c/Krystal-stack-platform-framework`: Mapovanie hostiteľského Windows adresára (emulácia 9P/drvfs).

### 2.2 Virtuálny DNF Balíčkovací Engine
Implementuje transakčné riadenie balíkov:
- `dnf install <balík>`: Kontroluje existenciu v katalógu, rekurzívne doťahuje závislosti (`libstdc++`, `python3`, `mesa-vulkan-drivers`), aktualizuje zoznam inštalovaných balíkov a vytvára transakčný záznam s časovou pečiatkou.
- `dnf remove <balík>`: Vymazáva balík z inštalačnej databázy.
- `dnf repolist`: Vracia aktívne repozitáre (`fedora`, `updates`, `krystal-stack`).
- `dnf check-update`: Simuluje kontrolu metadát bez nutnosti internetového pripojenia.

---

## 3. Mapovanie v Architektúre Krystal-Stack

| Komponent | Súbor | Popis |
| :--- | :--- | :--- |
| **Python Core Engine** | [wsl_emulation_subsystem.py](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/economic_engine/wsl_emulation_subsystem.py) | Virtuálny kernel, VFS, DNF engine a transparentný POSIX spúšťač. |
| **Janet DSL** | [wsl_emulation_subsystem.janet](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_janet/wsl_emulation_subsystem.janet) | Distribučné profily, slovníky balíkov a overovanie invariantu. |
| **REST API (Port 8089)** | [krystal_engine_core.py](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/krystal_engine_core.py) | Endpoints `/api/wsl/subsystem-status`, `/api/wsl/virtual-fs`, `/api/wsl/exec`, `/api/wsl/dnf-emulate`. |
| **Webový Terminál** | [wsl_emulator_studio.html](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/static/wsl_emulator_studio.html) | Interaktívna Linux konzola, rýchle tlačidlá príkazov, VFS prehliadač a DNF monitor. |
| **WebOS Desktop Integrácia** | [krystal_webos_desktop.html](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/static/krystal_webos_desktop.html) | Ikona `🐧 WSL Emulátor`, položka v Štart menu a plávajúce okno. |
| **Platformový Invariant** | Všade | Striktné dodržanie `VITAL_MAX_HP = 6`. |
