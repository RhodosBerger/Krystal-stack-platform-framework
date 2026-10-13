# Krystal Stack: Sprievodca Inštaláciou a Rýchlym Štartom

> **Jazyk:** Slovenský jazyk (SK)  
> **Určené pre:** Vývojárov, systémových inžinierov a prevádzkovateľov uzlov

---

## 1. Systémové Požiadavky

- **Operačný systém:** Windows 10/11 (odporúčané), Linux (Ubuntu 22.04+ / Arch), WSL2.
- **Procesor:** Intel 11th Gen+ (Tiger Lake / Alder Lake / Raptor Lake) s AVX2 / AVX-512 alebo ekvivalent AMD Zen 3/4.
- **Grafika:** Intel Iris Xe G7 (Vulkan 1.3 / OpenVINO) alebo dedikovaná GPU (NVIDIA RTX / AMD Radeon).
- **Python:** 3.10 až 3.14 (64-bit).
- **Pamäť:** Minimálne 8 GB RAM (odporúčané 16 GB).

---

## 2. Inštalácia v 3 Krokoch

```bash
# Krok 1: Klonovanie repozitára
git clone https://github.com/RhodosBerger/Krystal-stack-platform-framework.git
cd Krystal-stack-platform-framework

# Krok 2: Vytvorenie virtuálneho prostredia a inštalácia závislostí
python -m venv venv
.\venv\Scripts\activate      # Na Linuxe: source venv/bin/activate
pip install -r requirements.txt

# Krok 3: Spustenie lokálneho servera
python start_localhost.py
```

---

## 3. Prístup k Webovým Štúdiám

Po spustení sú na porte `8080` k dispozícii tieto rozhrania:
- **Cascade Model Studio:** `http://127.0.0.1:8080/static/cascade_model_studio.html` (4-panelový workbench, kaskády modelov, benchmarky a kinetická typografia).
- **Mission Control Hub:** `http://127.0.0.1:8080/` (prehľad hardvéru, telemetria a invarianty).
- **MCP Endpointy:** `http://127.0.0.1:8080/api/mcp/sse` a `http://127.0.0.1:8080/api/mcp/call`.

---

## 4. Spustenie Verifikačných Testov

```bash
python -m unittest tests/test_parametric_render_set.py tests/test_text_animation_engine.py tests/test_hybrid_mcp_antigravity_connector.py tests/test_model_cascade_runtime.py
```
Všetkých 37 testov by malo prebehnúť bez jedinej chyby s kódom 0.
