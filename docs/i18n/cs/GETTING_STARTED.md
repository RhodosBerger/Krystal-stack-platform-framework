# Krystal Stack: Průvodce Instalací a Rychlým Startem

> **Jazyk:** Český jazyk (CS)  
> **Určeno pro:** Vývojáře, systémové inženýry a provozovatele nodů

---

## 1. Systémové Požadavky

- **Operační systém:** Windows 10/11 (doporučeno pro plné UMCI/WDAC), Linux (Ubuntu 22.04+), WSL2.
- **Procesor:** Intel 11th Gen+ (Tiger Lake / Alder Lake) s AVX2 / AVX-512 nebo AMD Zen 3/4.
- **Grafická karta:** Intel Iris Xe G7 (Vulkan 1.3 / OpenVINO) nebo dedikovaná GPU (NVIDIA/AMD).
- **Python:** 3.10 až 3.14 (64-bit).
- **Paměť:** Minimálně 8 GB RAM (16 GB doporučeno).

---

## 2. Instalace ve 3 Krocích

```bash
# Krok 1: Klonování repozitáře
git clone https://github.com/RhodosBerger/Krystal-stack-platform-framework.git
cd Krystal-stack-platform-framework

# Krok 2: Vytvoření virtuálního prostředí a instalace balíčků
python -m venv venv
.\venv\Scripts\activate      # Linux: source venv/bin/activate
pip install -r requirements.txt

# Krok 3: Spuštění lokálního serveru
python start_localhost.py
```

---

## 3. Přístup k Webovým Studiím

Po spuštění jsou k dispozici tato rozhraní:
- **Cascade Model Studio:** `http://127.0.0.1:8080/static/cascade_model_studio.html`
- **Mission Control Hub:** `http://127.0.0.1:8080/`
- **MCP Rozhraní:** `http://127.0.0.1:8080/api/mcp/sse` a `http://127.0.0.1:8080/api/mcp/call`

---

## 4. Spuštění Testovací Sady

```bash
python -m unittest tests/test_parametric_render_set.py tests/test_text_animation_engine.py tests/test_hybrid_mcp_antigravity_connector.py tests/test_model_cascade_runtime.py
```
Všech 37 testů by mělo projít bez chyb během několika desítek milisekund.
