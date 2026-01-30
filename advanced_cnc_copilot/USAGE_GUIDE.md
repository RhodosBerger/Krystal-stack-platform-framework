# FANUC RISE v2.1 - Complete Usage Guide

## 🚀 System Status: READY

Your development environment is **fully operational** and running:
- ✅ Frontend UI: http://localhost:3000
- ✅ Backend API: http://localhost:8000
- ✅ Hot-reload active on both servers
- ✅ All animations refined (professional, no bounce)
- ✅ Responsive layouts for all personas

---

## Quick Start (5 Minutes)

### 1. Access the Interface
Open your browser and go to: **http://localhost:3000**

### 2. Navigate Between Personas
Click the persona switcher in the top-right corner:
- **Operator** 👷 - Shop floor HUD with real-time telemetry
- **Manager** 📊 - Fleet command dashboard with analytics
- **Creator** 🎨 - Generative design studio
- **Admin** ⚙️ - Configuration console

### 3. Explore Features
- View real-time telemetry data (simulated)
- Check manufacturing analytics
- Browse marketplace components
- Review system configuration

---

## Development Workflow

### Making Code Changes
1. Edit files in `frontend-react/src/` or `backend/`
2. Save your changes
3. Changes auto-reload in the browser (no manual refresh needed)

### Common Files to Edit
- **Layouts**: `frontend-react/src/layouts/OperatorLayout.jsx`
- **Components**: `frontend-react/src/components/NeuroCard.jsx`
- **Styles**: `frontend-react/src/index.css`
- **API Routes**: `backend/routers/`

---

## Available Commands

### Frontend (React + Vite)
```bash
cd frontend-react

# Start dev server (already running)
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### Backend (FastAPI)
```bash
# Start backend server (already running)
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# View API documentation
# Open: http://localhost:8000/docs
```

### Docker Deployment (When Docker Desktop is running)
```bash
# Build and start all containers
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop containers
docker-compose down
```

---

## System Architecture

### Frontend Stack
- **Framework**: React 18 + Vite
- **Styling**: TailwindCSS with custom design tokens
- **Animations**: Framer Motion (professional, no bounce)
- **State**: Context API + Custom Hooks
- **Icons**: Lucide React

### Backend Stack
- **API**: FastAPI (Python)
- **Database**: TimescaleDB (PostgreSQL time-series)
- **Cache**: Redis
- **Workers**: Celery
- **AI**: Integration-ready for LLM endpoints

### Key Features
1. **Multi-Persona Interface**: 4 distinct views for different roles
2. **Real-Time Telemetry**: WebSocket connection for live data
3. **Responsive Design**: Mobile → Tablet → Desktop
4. **Professional Animations**: Smooth fades, no distracting effects
5. **Design System**: Consistent glass-panel-pro styling

---

## Troubleshooting

### Frontend Not Loading
**Problem**: Blank screen or errors
**Solution**: 
```bash
cd frontend-react
rm -rf node_modules
npm install
npm run dev
```

### Backend Connection Errors
**Problem**: API proxy errors (ECONNREFUSED)
**Solution**: Make sure backend is running:
```bash
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Won't Start
**Problem**: "Cannot find file specified"
**Solution**: Docker Desktop must be running
1. Open Docker Desktop from Start Menu
2. Wait for green status
3. Retry `docker-compose up --build -d`

### Port Already in Use
**Problem**: Port 3000 or 8000 occupied
**Solution**:
```bash
# Windows: Find and kill process
netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

---

## API Endpoints

### Core Routes
- `GET /` - API status
- `GET /docs` - Interactive API documentation
- `GET /api/telemetry/stream` - WebSocket telemetry
- `GET /api/swarm/status` - Fleet status
- `GET /api/analytics/metrics` - Manufacturing metrics
- `POST /api/intelligence/ask` - LLM chat endpoint

### Authentication
- Auth system configured but optional for development
- See `backend/core/security.py` for configuration

---

## File Structure

```
advanced_cnc_copilot/
├── frontend-react/           # React frontend
│   ├── src/
│   │   ├── layouts/         # Persona layouts
│   │   ├── components/      # Reusable components
│   │   ├── context/         # State management
│   │   ├── hooks/           # Custom hooks
│   │   └── index.css        # Global styles
│   ├── Dockerfile           # Frontend container
│   └── package.json
├── backend/                 # FastAPI backend
│   ├── routers/            # API routes
│   ├── core/               # Configuration
│   └── main.py             # Entry point
├── cms/                    # CMS modules
│   └── thermal_biased_simulator.py
├── docker-compose.yml      # Container orchestration
└── requirements.txt        # Python dependencies
```

---

## Next Steps

### Immediate Actions
1. ✅ System is running locally
2. ✅ Visit http://localhost:3000
3. ✅ Explore all persona views
4. ✅ Make code changes and see live updates

### Production Deployment
When ready to deploy:
1. Start Docker Desktop
2. Run `docker-compose up --build -d`
3. Access via http://localhost:3000 (containerized)

### Further Development
- Add more components to the marketplace
- Customize telemetry data sources
- Integrate with real CNC machines
- Deploy to cloud (AWS, Azure, GCP)

---

## Support & Documentation

- 📁 **Full Walkthrough**: See `walkthrough.md` in your artifacts
- 📋 **Task Tracker**: See `task.md` in your artifacts
- 🏗️ **Architecture**: See `arch_manifest_v2_1.md`
- 🚢 **Docker Guide**: See `DOCKER_DEPLOYMENT.md`

**Need Help?** All systems are operational. You can start developing immediately!
