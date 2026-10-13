# Krystal Stack: Getting Started & Installation Guide

> **Language:** English (EN)  
> **Audience:** Developers, DevOps Engineers, and Node Operators

---

## 1. System Prerequisites

- **OS:** Windows 10/11 (Preferred for full UMCI/WDAC), Linux (Ubuntu 22.04+ / Arch), WSL2.
- **CPU:** Intel 11th Gen+ (Tiger Lake / Alder Lake) with AVX2 / AVX-512, or AMD Zen 3/4.
- **GPU:** Intel Iris Xe G7 (Vulkan 1.3 / OpenVINO) or Discrete NVIDIA/AMD GPU.
- **Python:** 3.10 to 3.14 (64-bit).
- **RAM:** Minimum 8 GB (16 GB recommended).

---

## 2. 3-Step Quickstart

```bash
# Step 1: Clone repo
git clone https://github.com/RhodosBerger/Krystal-stack-platform-framework.git
cd Krystal-stack-platform-framework

# Step 2: Create venv & install requirements
python -m venv venv
.\venv\Scripts\activate      # Linux: source venv/bin/activate
pip install -r requirements.txt

# Step 3: Start Mission Control Server
python start_localhost.py
```

---

## 3. Web Interfaces & Studio Dashboards

With the server running on port `8080`, navigate to:
- **Cascade Model Studio:** `http://127.0.0.1:8080/static/cascade_model_studio.html`
- **Mission Control Hub:** `http://127.0.0.1:8080/`
- **MCP Endpoints:** `http://127.0.0.1:8080/api/mcp/sse` and `http://127.0.0.1:8080/api/mcp/call`

---

## 4. Running the Test Suite

```bash
python -m unittest tests/test_parametric_render_set.py tests/test_text_animation_engine.py tests/test_hybrid_mcp_antigravity_connector.py tests/test_model_cascade_runtime.py
```
All 37 test cases should pass with code 0 in under 30 ms.
