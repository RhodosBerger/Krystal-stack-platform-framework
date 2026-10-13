# Krystal Stack: Schnellstart & Installationsanleitung

> **Sprache:** Deutsch (DE)  
> **Zielgruppe:** Entwickler, Systemingenieure und Knotenbetreiber

---

## 1. Systemvoraussetzungen

- **Betriebssystem:** Windows 10/11 (empfohlen für volle UMCI/WDAC-Prüfung), Linux (Ubuntu 22.04+), WSL2.
- **Prozessor:** Intel 11th Gen+ (Tiger Lake / Alder Lake) mit AVX2 / AVX-512 oder AMD Zen 3/4.
- **Grafikkarte:** Intel Iris Xe G7 (Vulkan 1.3 / OpenVINO) oder dedizierte GPU.
- **Python:** 3.10 bis 3.14 (64-Bit).
- **Arbeitsspeicher:** Mindestens 8 GB RAM (16 GB empfohlen).

---

## 2. Schnellstart in 3 Schritten

```bash
# Schritt 1: Repository klonen
git clone https://github.com/RhodosBerger/Krystal-stack-platform-framework.git
cd Krystal-stack-platform-framework

# Schritt 2: Virtuelle Umgebung erstellen und Abhängigkeiten installieren
python -m venv venv
.\venv\Scripts\activate      # Linux: source venv/bin/activate
pip install -r requirements.txt

# Schritt 3: Lokalen Server starten
python start_localhost.py
```

---

## 3. Web-Benutzeroberflächen

Öffnen Sie nach dem Start:
- **Cascade Model Studio:** `http://127.0.0.1:8080/static/cascade_model_studio.html` (4-Pane-Workbench, Modell-Kaskaden, Benchmarks).
- **Mission Control Hub:** `http://127.0.0.1:8080/` (Hardware-Übersicht und Telemetrie).
- **MCP-Endpunkte:** `http://127.0.0.1:8080/api/mcp/sse` und `http://127.0.0.1:8080/api/mcp/call`.

---

## 4. Tests Ausführen

```bash
python -m unittest tests/test_parametric_render_set.py tests/test_text_animation_engine.py tests/test_hybrid_mcp_antigravity_connector.py tests/test_model_cascade_runtime.py
```
Alle 37 Tests müssen erfolgreich mit Code 0 abschließen.
