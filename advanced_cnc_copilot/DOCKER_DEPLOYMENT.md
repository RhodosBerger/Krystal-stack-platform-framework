# Docker Deployment Guide - FANUC RISE v2.1

## Prerequisites
1. **Start Docker Desktop** from Windows Start Menu
2. Wait for Docker engine to show **green status** in system tray
3. Verify Docker is running: `docker info`

## Quick Deploy (One Command)
```bash
docker-compose up --build -d
```

## What Gets Deployed
- **Frontend** (React + Nginx): Port 3000
- **Backend** (FastAPI): Port 8000
- **Database** (TimescaleDB): Port 5432
- **Redis Cache**: Port 6379
- **Celery Workers**: Background processing
- **Flower Monitor**: Port 5555

## Access Points After Deployment
- Frontend UI: http://localhost:3000
- Backend API: http://localhost:8000/docs
- Flower Monitor: http://localhost:5555

## Useful Commands
```bash
# View running containers
docker-compose ps

# View logs
docker-compose logs -f frontend
docker-compose logs -f backend

# Stop all containers
docker-compose down

# Rebuild and restart
docker-compose up --build -d

# Stop and remove all data
docker-compose down -v
```

## Troubleshooting

### "Cannot find file specified" Error
**Problem**: Docker Desktop is not running
**Solution**: Open Docker Desktop and wait for green status

### Port Already in Use
**Problem**: Port 3000 or 8000 is occupied
**Solution**: Stop local dev servers first:
- Stop `npm run dev` (Ctrl+C in terminal)
- Stop `uvicorn` backend (Ctrl+C in terminal)

### Build Failures
**Problem**: Dependencies fail to install
**Solution**: Check your internet connection and retry:
```bash
docker-compose down
docker-compose up --build -d --no-cache
```

## Current Status
✅ All animations refined (bounce removed, fade kept)
✅ Responsive layouts for all personas
✅ Docker configuration ready
⏳ Waiting for Docker Desktop to start

**Next Step**: Launch Docker Desktop, then run `docker-compose up --build -d`
