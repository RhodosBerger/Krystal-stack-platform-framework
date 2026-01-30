# LOCALHOST SETUP GUIDE - FANUC RISE
## Spustenie systému na lokalnom počítači

---

## PREDPOKLADY

### A. Softvér ktorý musíte mať nainštalovaný:
- **Python 3.11+** ([stiahni tu](https://www.python.org/downloads/))
- **Docker Desktop** ([stiahni tu](https://www.docker.com/products/docker-desktop/))
- **Git** ([stiahni tu](https://git-scm.com/))
- **VS Code** (voliteľné, ale odporúčané)

### B. Hardvér:
- **CPU**: 4+ cores
- **RAM**: 8GB minimum (16GB odporúčané)
- **Disk**: 20GB voľného miesta

---

## KROK 1: KLONOVANIE REPOZITÁRA

```powershell
# Windows PowerShell
cd C:\Users\dusan\Documents\GitHub\Dev-contitional
git clone https://github.com/TVOJ-USERNAME/advanced_cnc_copilot.git
cd advanced_cnc_copilot
```

---

## KROK 2: KONFIGURÁCIA PROSTREDIA

### 2.1 Vytvorenie .env súboru
```powershell
# Skopíruj príklad konfigurácie
cp .env.example .env
```

### 2.2 Edituj .env (otvor v notepad alebo VS Code)
**Minimálna konfigurácia pre localhost:**
```env
# Ponechaj USE_MOCK_HAL=true ak nemáš fyzický CNC stroj
USE_MOCK_HAL=true

# Ak chceš testovať LLM funkcie, pridaj API kľúč
OPENAI_API_KEY=sk-tvoj-api-key
```

---

## KROK 3: INŠTALÁCIA (Dva spôsoby)

### MOŽNOSŤ A: Docker (Jednoduchšie - Odporúčané)

```powershell
# Spusti všetko naraz (PostgreSQL + Redis + API + Dashboard)
docker-compose up -d

# Skontroluj status
docker-compose ps

# Sleduj logy
docker-compose logs -f api
```

**Výsledok**: Systém beží na `http://localhost:8000`

### MOŽNOSŤ B: Python Virtuálne Prostredie (Pre development)

```powershell
# Vytvor virtuálne prostredie
python -m venv venv

# Aktivuj ho
.\venv\Scripts\Activate.ps1

# Nainštaluj závislosti
pip install -r requirements.txt

# Spusti databázu a Redis v Dockeri
docker-compose up -d postgres redis

# Spusti aplikáciu
uvicorn cms.fanuc_api:app --reload --host 0.0.0.0 --port 8000
```

---

## KROK 4: VERIFIKÁCIA ŽE TO BEŽÍ

### 4.1 API Endpoint Test
Otvor prehliadač: `http://localhost:8000/docs`  
→ Mala by sa zobraziť **Swagger UI** (interaktívna API dokumentácia)

### 4.2 Dashboard Test
Otvor: `http://localhost:8000/dashboard/hub.html`  
→ Mal by sa zobraziť **Portal Hub** s kartami

### 4.3 Health Check
```powershell
curl http://localhost:8000/health
```
**Očakávaný output**:
```json
{
  "status": "healthy",
  "database": "connected",
  "redis": "connected",
  "hal_mode": "mock"
}
```

---

## KROK 5: TESTOVANIE ZÁKLADNÝCH FUNKCIÍ

### 5.1 Mock Telemetria
```powershell
# Otvor nové okno PowerShell
curl http://localhost:8000/api/telemetry/mock
```

Otvor Dashboard (`index.html`) a sleduj ako sa **metríky menia v reáln čase**.

### 5.2 LLM Suggestion (ak máš API kľúč)
```powershell
curl -X POST http://localhost:8000/api/suggest `
  -H "Content-Type: application/json" `
  -d '{"material": "Aluminum", "complexity": 5}'
```

### 5.3 Logging System
Otvor: `http://localhost:8000/dashboard/logs.html`  
→ Mala by sa zobraziť lokálna telemetria.

---

## ŠTRUKTÚRA PROJEKTU

```
advanced_cnc_copilot/
├── cms/                          # Core modules
│   ├── sensory_cortex.py         # HAL abstraction
│   ├── impact_cortex.py          # Safety logic
│   ├── dopamine_engine.py        # Reward system
│   ├── fanuc_api.py              # FastAPI server
│   ├── logging_system.py         # Multi-level logs
│   ├── dashboard/                # Frontend
│   │   ├── index.html            # Live telemetry
│   │   ├── hub.html              # Portal
│   │   ├── logs.html             # Log viewer
│   │   └── style.css
│   └── theories/                 # Documentation
├── config.py                     # Configuration loader
├── requirements.txt              # Python dependencies
├── docker-compose.yml            # Localhost stack
├── Dockerfile                    # API container
├── .env.example                  # Config template
└── README.md                     # This file
```

---

## ČO FUNGUJE UŽ TERAZ (v Mock režime)

✅ **FastAPI Server** (REST + WebSocket)  
✅ **PostgreSQL** (databáza pripravená)  
✅ **Redis** (cache/session storage)  
✅ **Dashboard** (HTML5 frontend)  
✅ **Logging System** (3 úrovne: dev/tech/operator)  
✅ **Mock HAL** (simulované CNC dáta)  
✅ **Dopamine Engine** (reward scoring)  
✅ **LLM Integration** (ak máš API kľúč)

---

## ČO EŠTE CHÝBA / TREBA DOPLNIŤ

### PRIORITA 1: Fyzické Prepojenie (Ak máš CNC stroj)
- [ ] FOCAS knižnica pre Fanuc (`.dll` súbory)
- [ ] Network kábel: PC ↔ CNC
- [ ] IP konfigurácia (nastaviť `FANUC_IP` v `.env`)
- [ ] Zmeniť `USE_MOCK_HAL=false`

### PRIORITA 2: Database Migrácie
- [ ] Vytvoriť Alembic migrácie (schéma pre projekty/telemetriu)
- [ ] Seed data (inicializácia základných dát)

### PRIORITA 3: Authentifikácia
- [ ] JWT token generation (už nakódované, potrebuje secret key generovanie)
- [ ] User registration endpoint
- [ ] Login page (frontend)

### PRIORITA 4: LLM Training Pipeline
- [ ] Project harvester (scan existing `.nc` files)
- [ ] Feature extraction (G-code → vectors)
- [ ] Fine-tuning script (OpenAI/Anthropic)

### PRIORITA 5: Advanced Features
- [ ] Solidworks COM integration (potrebuje Windows + Solidworks)
- [ ] Multi-machine coordination (pre 2+ CNC)
- [ ] Predictive maintenance (RNN model)

---

## ODPORÚČANÝ PLÁN UČENIA

### Týždeň 1: Pochopenie základov
- [ ] Preštuduj `cms/theories/` dokumenty
- [ ] Skúmaj `cms/fanuc_api.py` (FastAPI routes)
- [ ] Otestuj všetky Dashboard stránky

### Týždeň 2: Modifikácia kódu
- [ ] Zmeň farby v `dashboard/style.css`
- [ ] Pridaj novú metriku do `sensory_cortex.py`
- [ ] Vytvor si vlastný log message typ

### Týždeň 3: Databázová integrácia
- [ ] Nauč sa Alembic migrations
- [ ] Vytvor model pre "Projects" tabuľku
- [ ] Test CRUD operácie

### Týždeň 4: API rozšírenie
- [ ] Vytvor nový endpoint `/api/custom`
- [ ] Integruj s externým API
- [ ] Napíš unit test (pytest)

---

## TROUBLESHOOTING

### Problém: Docker sa nespustí
**Riešenie**: Zapni Docker Desktop, počkaj 30s, skús znova.

### Problém: Port 8000 už používaný
**Riešenie**: Zmeň `APP_PORT=8001` v `.env`

### Problém: PostgreSQL connection error
**Riešenie**: 
```powershell
docker-compose down
docker-compose up -d postgres
# Počkaj 10s
docker-compose up -d api
```

### Problém: Dashboard sa nezobrazuje
**Riešenie**: Skontroluj `docker-compose logs nginx`

---

## UŽITOČNÉ PRÍKAZY

```powershell
# Zastaviť všetko
docker-compose down

# Vymazať databázu (fresh start)
docker-compose down -v

# Rebuild po zmene kódu
docker-compose up -d --build

# Vstúpiť do API containera
docker exec -it fanuc_rise_api bash

# Backup databázy
docker exec fanuc_rise_db pg_dump -U postgres fanuc_rise > backup.sql
```

---

## ĎALŠIE KROKY

1. **Prejdi cez tento guide** a spusti systém
2. **Otvor issue** ak niečo nefunguje
3. **Skús vytvoriť prvý Pull Request** (napr. pridaj slovenskú lokalizáciu do Dashboardu)

**Otázky?** Vytvor discussion na GitHub alebo pošli email.

---

*Happy hacking! 🚀*
