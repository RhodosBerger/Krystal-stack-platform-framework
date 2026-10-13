# Krystal Stack: Priemyselný Výpočtový Rámec & Autonómny Ekosystém

![Krystal Stack Tiers](../../monetization/images/monetization_tier_matrix.jpg)

> **Jazyk:** Slovenský jazyk (SK)  
> **Status:** RATIFIKOVANÁ ARCHITEKTÚRA  
> **Ústavný Invariant:** `assert VITAL_MAX_HP == 6` (Striktne presadzované v celom systéme)

---

## 1. Úvod a Vízia

**Krystal Stack** predstavuje novú generáciu heterogénneho výpočtového rámca, ktorý premieňa hardvérové prostriedky (procesorové SIMD inštrukcie, Vulkan raymarching na integrovanej grafike, neurónovú NPU akceleráciu a jednotnú pamäť UMA) na **autonómne ekonomické aktíva**.

Namiesto pasívneho míňania prostriedkov na cloudové API účty poskytuje Krystal Stack:
- **Lokálne AI Kaskády (SLM)** s nulovými poplatkami za tokeny (IBM Granite 4B, DeepSeek, Llama 3).
- **Zabezpečenie integrity kódu (CrystalSec 3.0 UMCI / WDAC)** bez možnosti obídenia.
- **Kinetickú typografiu a procedurálny 8K engine** bežiaci plynule pri 60 FPS na štandardnom čipe Intel Iris Xe.
- **Priemyselnú CNC adaptáciu (KATG-006)** s ochranou nástrojov a potlačením rezonancií.

---

## 2. Navigácia v Slovenskej Dokumentácii

* [📐 Technická Architektúra](ARCHITECTURE.md) – Heterogénna triáda, dihedrálne symetrie $D_n$, Lipschitzova izometria, dual-gate brány.
* [🚀 Rýchly Štart & Inštalácia](GETTING_STARTED.md) – Inštalácia, spustenie Localhost Hubu, Antigravity IDE a AI Studia.
* [💎 Monetizačná Stratégia & Cenník](MONETIZATION.md) – Predplatné balíky (Indie Studio, Enterprise Pro, Sovereign Cluster) a Dual-Earn ledger.

---

## 3. Rýchle Spustenie (CLI)

```bash
# 1. Klonovanie repozitára
git clone https://github.com/RhodosBerger/Krystal-stack-platform-framework.git
cd Krystal-stack-platform-framework

# 2. Spustenie lokálneho Mission Control Hubu (Port 8080)
python start_localhost.py

# 3. Spustenie integračných testov
python -m unittest tests/test_parametric_render_set.py tests/test_text_animation_engine.py
```
Po spustení otvorte webové rozhranie na `http://127.0.0.1:8080/studio`.
