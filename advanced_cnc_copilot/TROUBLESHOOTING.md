# TROUBLESHOOTING & ALTERNATIVE SETUP
## Riešenie problémov s Docker + Natívny Python setup

---

## ❌ PROBLÉM: Docker Desktop nie je spustený

**Error**: `cannot find the file specified` pri `docker-compose ps`

### RIEŠENIE 1: Spusti Docker Desktop

1. **Nájdi Docker Desktop** vo Windows Start menu
2. **Klikni pravým** → Run as Administrator
3. **Počkaj 30-60 sekúnd** (Docker engine sa načíta)
4. **Overenie**: V system tray (vedľa hodín) by mal byť Docker icon
5. **Skús znova**: `docker-compose up -d`

---

## 🐍 RIEŠENIE 2: Natívny Python Setup (BEZ Dockeru)

Ak nechceš/nemôžeš použiť Docker, tu je natívny Windows setup:

### A. Inštaluj Požiadavky

```powershell
# 1. PostgreSQL (Database)
# Stiahni z: https://www.postgresql.org/download/windows/
# Počas inštalácie:
#   - Password: changeme123
#   - Port: 5432
#   - Database name: fanuc_rise

# 2. Redis (Cache) - Voliteľné
# Stiahni z: https://github.com/microsoftarchive/redis/releases
# Alebo preskočiť (bude warning, ale pojde to)

# 3. Python dependencies
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### B. Vytvor .env súbor

```powershell
# Skopíruj príklad
cp .env.example .env

# Edituj .env (otvor v Notepad):
DB_HOST=localhost
DB_PORT=5432
DB_NAME=fanuc_rise
DB_USER=postgres
DB_PASSWORD=changeme123  # Použite heslo z PostgreSQL inštalácie

REDIS_HOST=localhost  # Alebo zakomentuj ak nemáš Redis
USE_MOCK_HAL=true  # Dôležité! Mock mode bez fyzického CNC
```

### C. Inicializuj Databázu

```powershell
# Pripoj sa k PostgreSQL
psql -U postgres

# V psql konzole:
CREATE DATABASE fanuc_rise;
\q

# Alebo cez pgAdmin (GUI tool)
```

### D. Spusti API Server

```powershell
# Aktivuj venv
.\venv\Scripts\Activate.ps1

# Spusti server
uvicorn cms.fanuc_api:app --reload --host 0.0.0.0 --port 8000
```

### E. Otvor Dashboard

**Možnosť 1**: Priamo zo súborového systému
```
Otvor v Chrome: file:///C:/Users/dusan/Documents/GitHub/Dev-contitional/advanced_cnc_copilot/cms/dashboard/hub.html
```

**Možnosť 2**: Cez Python HTTP server (v druhom termináli)
```powershell
cd cms/dashboard
python -m http.server 8080

# Otvor: http://localhost:8080/hub.html
```

---

## 🔧 MINIMÁLNA KONFIGURÁCIA (Žiadna databáza potrebná)

Ak chceš len **quick demo**, môžeš spustiť:

```powershell
# Vytvor jednoduché .env
echo "USE_MOCK_HAL=true" > .env

# Spusti API (bez DB)
python -m cms.fanuc_api
```

Potom otvor Dashboard priamo zo súborového systému.

---

## ✅ OVERENIE ŽE TO FUNGUJE

### Test 1: API Endpoint
```powershell
curl http://localhost:8000/docs
# Očakávaný výsledok: Swagger UI v prehliadači
```

### Test 2: Mock Telemetria
```powershell
curl http://localhost:8000/api/telemetry/mock
# Očakávaný výsledok: JSON s RPM, load, vibration
```

### Test 3: Dashboard
```
Otvor: http://localhost:8000/dashboard/hub.html
# Očakávaný výsledok: Portal s 3 kartami
```

---

## 🚨 ČASTÉ PROBLÉMY

### "Port 8000 already in use"
```powershell
# Nájdi čo beží na porte 8000
netstat -ano | findstr :8000

# Zabij proces (replace PID s ID z vyššie)
taskkill /PID <PID> /F

# Alebo zmeň port v .env:
APP_PORT=8001
```

### "ModuleNotFoundError: No module named 'fastapi'"
```powershell
# Nie si vo venv, aktivuj ho:
.\venv\Scripts\Activate.ps1

# Overenie (mala by byť cesta k venv):
where python
```

### "PostgreSQL connection refused"
```powershell
# Skontroluj či PostgreSQL service beží
Get-Service postgresql*

# Ak nie je spustený:
Start-Service postgresql-x64-15  # Replace s tvojou verziou
```

### "ImportError: DLL load failed"
```powershell
# Chýbajú Visual C++ redistributables
# Stiahni z: https://aka.ms/vs/17/release/vc_redist.x64.exe
```

---

## 📋 QUICK START CHECKLIST

- [ ] Python 3.11+ nainštalovaný (`python --version`)
- [ ] venv vytvorené (`python -m venv venv`)
- [ ] venv aktivované (vidíš `(venv)` v príkazovom riadku)
- [ ] Dependencies nainstalované (`pip install -r requirements.txt`)
- [ ] .env súbor existuje (skopírovaný z .env.example)
- [ ] PostgreSQL beží (voliteľné pre quick demo)
- [ ] API server beží (`uvicorn cms.fanuc_api:app --reload`)
- [ ] Dashboard sa otvára (v prehliadači)

---

## 🎯 NAJJEDNODUCHŠÍ MOŽNÝ SETUP (2 minúty)

```powershell
# 1. Vytvor venv a aktivuj
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Inštaluj dependencies
pip install fastapi uvicorn pydantic

# 3. Vytvor minimálny .env
"USE_MOCK_HAL=true" | Out-File .env -Encoding utf8

# 4. Spusti
uvicorn cms.fanuc_api:app --reload

# 5. Otvor v Chrome
start http://localhost:8000/docs
```

**Pozor**: Toto je absolutné minimum. Pre plnú funkcionalitu potrebuješ všetky dependencies z `requirements.txt`.

---

## 💡 ODPORÚČANIE

**Pre development**: Natívny Python setup (flexibilnejšie, ľahšie debugging)  
**Pre production**: Docker (konzistentné prostredie, jednoduchšie nasadenie)

**Tvoje rozhodnutie**: Ak máš Docker Desktop, použite ho (lepšie). Ak nie, natívny Python je OK pre začiatok.

---

*Keď to rozbehneš, daj mi vedieť!* 🚀
