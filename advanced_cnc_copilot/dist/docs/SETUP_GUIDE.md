# COMPLETE SETUP GUIDE - FANUC RISE

## Quick Start (5 Minutes)

### 1. Install Dependencies

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install all packages
pip install -r requirements.txt
pip install -r flask_service/requirements.txt
```

### 2. Setup Database

```powershell
# Make sure PostgreSQL is running
# Create database
psql -U postgres -c "CREATE DATABASE fanuc_rise;"

# Run Django migrations
python manage.py makemigrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser
```

### 3. Start Services

```powershell
# Terminal 1: Flask Microservice (Real-time telemetry)
cd flask_service
python app.py
# Running on http://localhost:5000

# Terminal 2: Django Server (ERP/API)
python manage.py runserver
# Running on http://localhost:8000
```

### 4. Access Dashboards

- **Django Admin**: http://localhost:8000/admin
- **REST API**: http://localhost:8000/api/
- **Swagger Docs**: http://localhost:8000/api/ (with DRF)
- **Dashboard**: http://localhost:8000/dashboard/dynamic-panel.html
- **Flask Health**: http://localhost:5000/health

---

## Project Structure

```
advanced_cnc_copilot/
├── cms/                          # Core cognitive modules
│   ├── sensory_cortex.py
│   ├── impact_cortex.py
│   ├── dopamine_engine.py
│   ├── signaling_system.py
│   ├── demo_data_generator.py
│   ├── dynamic_form_builder.py
│   ├── protocol_conductor.py
│   └── dashboard/               # Frontend files
│       ├── dynamic-panel.html
│       ├── dynamic-panel.css
│       └── dynamic-panel.js
│
├── erp/                          # Django ERP app
│   ├── models.py                # Database models
│   ├── views.py                 # REST API views
│   ├── serializers.py
│   ├── urls.py
│   ├── admin.py
│   ├── oee_calculator.py
│   ├── economics.py
│   └── similarity.py
│
├── flask_service/               # Real-time microservice
│   ├── app.py
│   └── requirements.txt
│
├── fanuc_rise_django/           # Django project settings
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
├── requirements.txt             # Main dependencies
└── README.md
```

---

## API Endpoints

### Django REST API (Port 8000)

**Machines:**
- GET `/api/machines/` - List all machines
- POST `/api/machines/` - Create machine
- GET `/api/machines/{id}/` - Machine detail
- GET `/api/machines/{id}/telemetry/` - Get real-time telemetry
- GET `/api/machines/{id}/oee/` - Calculate OEE
- POST `/api/machines/{id}/dopamine_check/` - Check dopamine score

**Projects:**
- GET `/api/projects/` - List projects
- POST `/api/projects/` - Create project
- GET `/api/projects/{id}/similar/` - Find similar projects
- POST `/api/projects/{id}/suggest_params/` - Get LLM suggestions

**Tools:**
- GET `/api/tools/` - Tool inventory
- GET `/api/tools/needs_replacement/` - Tools needing replacement
- POST `/api/tools/{id}/log_usage/` - Update tool usage

**Jobs:**
- GET `/api/jobs/` - Job list
- GET `/api/jobs/schedule/` - Optimized schedule
- POST `/api/jobs/{id}/start/` - Start job
- POST `/api/jobs/{id}/complete/` - Complete job

**Analytics:**
- GET `/api/analytics/dashboard/` - Dashboard summary

**Configuration:**
- GET `/api/config/{type}/` - Get dynamic form config
- POST `/api/config/{type}/save/` - Save configuration

### Flask WebSocket API (Port 5000)

**REST:**
- GET `/health` - Health check
- GET `/api/telemetry/current` - Current snapshot
- GET `/api/telemetry/history?minutes=60` - Historical data
- POST `/api/dopamine/evaluate` - Evaluate dopamine
- POST `/api/signal/check` - Check semaphore signal

**WebSocket Events:**
- `connect` - Client connects
- `join_machine` - Subscribe to machine updates
- `telemetry_update` - Real-time telemetry (1Hz)

---

## Testing

```powershell
# Test Flask service
curl http://localhost:5000/health

# Test Django API
curl http://localhost:8000/api/machines/

# Test WebSocket (using wscat)
npm install -g wscat
wscat -c ws://localhost:5000
> {"event": "join_machine", "data": {"machine_id": "CNC_VMC_01"}}
```

---

## Production Deployment

### Using Docker Compose

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: fanuc_rise
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: changeme123
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7

  flask:
    build: ./flask_service
    ports:
      - "5000:5000"
    depends_on:
      - redis

  django:
    build: .
    command: gunicorn fanuc_rise_django.wsgi:application --bind 0.0.0.0:8000
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis

volumes:
  postgres_data:
```

Run: `docker-compose up -d`

---

## Troubleshooting

### "No module named 'cms'"
```powershell
# Add project root to PYTHONPATH
$env:PYTHONPATH = "C:\Users\dusan\Documents\GitHub\Dev-contitional\advanced_cnc_copilot"
```

### "psycopg2 not found"
```powershell
pip install psycopg2-binary
```

### "Port already in use"
```powershell
# Find process on port 8000
netstat -ano | findstr :8000
# Kill it
taskkill /PID <PID> /F
```

---

## Next Steps

1. ✅ **Run system locally** (completed above)
2. 📝 **Create sample data** via Django admin
3. 🧪 **Test API endpoints** with Postman
4. 🎨 **Customize dashboard** UI
5. 🤖 **Add OpenAI API key** for LLM features
6. 📊 **Import real G-code** projects
7. 🚀 **Deploy to production**

---

**System Status**: Phase 45 Complete - Full Stack Ready! 🎉
