# ČO EŠTE CHÝBA & PRIORITY DEVELOPMENT
## Analýza čo je hotové a kam ísť ďalej

---

## ✅ ČO JE UŽ HOTOVÉ (42 Fáz)

### 1. Kognitívny Core (Raw Python)
- [x] `sensory_cortex.py` - HAL abstrakcia
- [x] `impact_cortex.py` - Safety logika
- [x] `dopamine_engine.py` - Reward systém
- [x] `knowledge_graph.py` - Topológia problémov
- [x] `signaling_system.py` - Semafory (Green/Amber/Red)
- [x] `process_scheduler.py` - Plánovač úloh

### 2. API & Frontend
- [x] `fanuc_api.py` - FastAPI server (REST + WebSocket)
- [x] Dashboard HTML/CSS (index, hub, lab, docs, logs)
- [x] Multi-level logging system
- [x] WebSocket real-time telemetria

### 3. Integrácia & AI
- [x] `protocol_conductor.py` - LLM scenár generátor
- [x] `llm_action_parser.py` - Text → Command
- [x] `operation_queue.py` - Safety buffer
- [x] `fanuc_solidworks_bridge.py` - CAD integrácia (research)

### 4. Cloud & Auth
- [x] `cloud_auth_model.py` - RBAC modely
- [x] Multi-stack architecture plan
- [x] Cloud infrastructure research

### 5. Dokumentácia
- [x] 22+ theory documents v `cms/theories/`
- [x] Slovak marketing materials
- [x] Technical specification
- [x] Complete localhost setup

---

## ❌ ČO CHÝBA - PRIORITA 1 (Kritické pre spustenie)

### A. Database Schema & Migrations
**Status**: Moduly používajú DB, ale schéma neexistuje  
**Potrebné**:
```python
# Vytvoriť Alembic migrations pre:
- Users table (id, email, password_hash, role)
- Projects table (id, gcode, telemetry_json, outcome)
- Telemetry table (timestamp, rpm, load, vibration, machine_id)
- Sessions table (session_id, user_id, started_at, jwt_token)
```

**Akcia**:
1. Inicializuj Alembic: `alembic init alembic`
2. Vytvor modely v `cms/models.py`
3. Generate migration: `alembic revision --autogenerate`
4. Apply: `alembic upgrade head`

**Odhadovaný čas**: 4 hodiny

---

### B. Fyzické CNC Prepojenie (Real HAL)
**Status**: Mock mode funguje, real FOCAS neimplementované  
**Potrebné**:
- Fanuc FOCAS knižnica (`.dll` súbory)
- Wrapper pre ctypes volania
- Error handling pre network timeouts

**Stub kód existuje**:
```python
# cms/hal_fanuc.py - Line 15
# TODO: Implement real FOCAS connection
```

**Akcia**:
1. Získaj FOCAS SDK od Fanuc (license potrebná)
2. Test connection s jedným strojom
3. Verify telemetry accuracy (1kHz sampling)

**Odhadovaný čas**: 8 hodín (+ FOCAS license approval)

---

### C. Authentication Flow (Full Implementation)
**Status**: Models existujú, ale routes/frontend chýbajú  
**Potrebné**:
- `/auth/register` endpoint
- `/auth/login` endpoint (vráti JWT)
- `/auth/refresh` (refresh token logic)
- Login page (`dashboard/login.html`)
- Token validation middleware v FastAPI

**Akcia**:
1. Vytvor `cms/auth_routes.py`
2. Implementuj bcrypt hashing
3. JWT signing (RS256 s asymetric keys)
4. Frontend login form

**Odhadovaný čas**: 6 hodín

---

## ⚠️ ČO CHÝBA - PRIORITA 2 (Dôležité pre plnú funkciu)

### D. LLM Training Pipeline
**Status**: Conductor vie volať LLM, ale nie je trained na CNC data  
**Potrebné**:
- Project harvester (scan existing `.nc` files)
- Feature extractor (G-code → embeddings)
- Fine-tuning script (OpenAI/Claude API)
- Training dataset (500+ projects JSON)

**Akcia**:
1. Vytvor `cms/project_harvester.py` (scan filesystem)
2. Extract features: `calculate_complexity(gcode)`
3. Format pre fine-tuning: `{"prompt": ..., "completion": ...}`
4. Submit training job cez API

**Odhadovaný čas**: 12 hodín + €500 OpenAI fine-tuning cost

---

### E. Dashboard ↔ API Connection (Frontend JS)
**Status**: HTML existuje, ale JavaScript pre API calls chýba  
**Potrebné**:
- `dashboard/app.js` - Fetch telemetry cez WebSocket
- Update DOM s live dátami
- Click handlers pre "Apply Suggestion" button
- Chart.js integrácia pre grafy

**Akcia**:
1. WebSocket client: `const ws = new WebSocket('ws://localhost:8000/ws')`
2. Parse messages, update `<div id="rpm-value">`
3. Axios/Fetch pre REST calls
4. Error handling

**Odhadovaný čas**: 8 hodín

---

### F. Multi-Machine Support
**Status**: Systém predpokladá 1 stroj  
**Potrebné**:
- `machine_id` parameter všade
- Database foreignkey: `telemetry.machine_id`
- Frontend selector: `<select id="machine-picker">`
- Load balancing pre 10+ strojov

**Akcia**:
1. Extend DB models s `machine_id`
2. Update HAL: `sensory_cortex.connect(machine_id)`
3. API: `/api/telemetry/{machine_id}`

**Odhadovaný čas**: 6 hodín

---

## 🔮 ČO CHÝBA - PRIORITA 3 (Advanced Features)

### G. Predictive Maintenance (RNN Model)
**Nápad**: Predpovedať zlyhanie nástroja 30 min vopred  
**Potrebné**: TensorFlow/PyTorch model trained na vibrations

### H. Swarm Optimization
**Nápad**: 100 virtual agents testujú rôzne feed rates  
**Potrebné**: Genetic algorithm implementation

### I. AR Visualization
**Nápad**: HoloLens zobrazuje "ghost toolpath" nad strojom  
**Potrebné**: Unity3D + Mixed Reality Toolkit

---

## 📚 TÉMY NA ŠTÚDIUM (Pre zlepšenie development skills)

### Týždeň 1-2: Database Design
- **Potrebné**: Alembic, SQLAlchemy relationships
- **Resource**: [SQLAlchemy Tutorial](https://docs.sqlalchemy.org/tutorial/)
- **Cieľ**: Vytvoriť production-ready schému

### Týždeň 3-4: WebSocket Programming
- **Potrebné**: AsyncIO, FastAPI WebSockets
- **Resource**: [FastAPI WebSockets Guide](https://fastapi.tiangolo.com/advanced/websockets/)
- **Cieľ**: Real-time telemetria bez polling

### Týždeň 5-6: LLM Fine-Tuning
- **Potrebné**: OpenAI API, JSONL formatting
- **Resource**: [OpenAI Fine-tuning Docs](https://platform.openai.com/docs/guides/fine-tuning)
- **Cieľ**: Custom CNC-domain model

### Týždeň 7-8: Docker & Kubernetes
- **Potrebné**: docker-compose → K8s migration
- **Resource**: [Kubernetes Basics](https://kubernetes.io/docs/tutorials/)
- **Cieľ**: Multi-node deployment

---

## 🎯 ODPORÚČANÝ DEVELOPMENT PLÁN (Next 4 Weeks)

### Week 1: Database Foundation
- [ ] Day 1-2: Alembic setup, vytvor modely
- [ ] Day 3: Seed data (test users, mock projects)
- [ ] Day 4: CRUD endpoints (`/api/projects`)
- [ ] Day 5: Test queries performance

### Week 2: Authentication
- [ ] Day 1-2: Implementuj auth routes
- [ ] Day 3: JWT middleware
- [ ] Day 4: Login page frontend
- [ ] Day 5: Test auth flow (register → login → access protected route)

### Week 3: Real HAL + Dashboard
- [ ] Day 1-2: FOCAS integration (ak máš stroj)
- [ ] Day 3-4: Dashboard JavaScript (WebSocket)
- [ ] Day 5: End-to-end test (CNC → API → Dashboard)

### Week 4: LLM Pipeline
- [ ] Day 1-2: Project harvester
- [ ] Day 3: Feature extraction
- [ ] Day 4: Training data prep
- [ ] Day 5: Submit fine-tuning job

**Po týchto 4 týždňoch**: Máš MVP ready pre pilot deployment.

---

## 🚨 BLOCKING ISSUES (Riešiť najskôr)

### Issue #1: FOCAS License
**Problém**: Fanuc FOCAS SDK je proprietary  
**Workaround**: Použiť mock mode, alebo hľadať open-source alternatívy (MTConnect?)  
**Action**: Kontaktuj Fanuc distributor

### Issue #2: LLM API Costs
**Problém**: Fine-tuning = €500+, inference = €100/month  
**Workaround**: Použiť local LLM (Ollama + Llama 3)  
**Action**: Test Ollama performance

### Issue #3: Database Migration Conflicts
**Problém**: Ak viac devs robí migrations súčasne  
**Workaround**: Git branch per migration  
**Action**: Dokumentuj migration workflow

---

## 💡 QUICK WINS (Rýchle úspechy pre motiváciu)

### Win #1: Slovak Localization (2 hodiny)
Preklad Dashboard labels do slovenčiny.

### Win #2: Custom Theme (1 hodina)
Zmeň farby na firemné (napr. modrá → zelená).

### Win #3: Email Notifications (3 hodiny)
Pošli email ak Load > 95% (SMTP integration).

### Win #4: CSV Export (2 hodiny)
Button "Download Report" → Excel súbor.

---

## 📞 GDE HĽADAŤ POMOC

### Community:
- **Discord**: FastAPI server (Python help)
- **Reddit**: r/cnc, r/machining (CNC advice)
- **Stack Overflow**: Tag `fastapi`, `sqlalchemy`

### Dokumentácia:
- **FastAPI**: https://fastapi.tiangolo.com
- **SQLAlchemy**: https://docs.sqlalchemy.org
- **Docker**: https://docs.docker.com

### Firemné:
- **Fanuc Support**: Official FOCAS docs
- **OpenAI Forum**: Fine-tuning help

---

## ✅ CHECKLIST PRE PRODUCTION READY

- [ ] **Security**: HTTPS, JWT expiry, rate limiting
- [ ] **Testing**: 80%+ code coverage
- [ ] **Monitoring**: Prometheus + Grafana setup
- [ ] **Backups**: Automated DB backups (daily)
- [ ] **Documentation**: Updated README, API docs
- [ ] **CI/CD**: GitHub Actions pipeline
- [ ] **Load Testing**: Handle 50 concurrent users
- [ ] **Error Handling**: Graceful degradation

---

**Záver**: Máš solid foundation. Focus na Priority 1 (Database, Auth, Real HAL), potom Priority 2 (LLM, Dashboard JS). Priority 3 sú "nice to have" pre neskôr.

*Success metric*: Po 4 týždňoch máš 1 fyzický CNC pripojený a Dashboard zobrazuje live dáta. 🎯
