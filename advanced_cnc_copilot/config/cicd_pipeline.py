# CI/CD Pipeline Configuration
# Phase 1: Foundation - Automated Deployment

"""
Paradigm: CI/CD as Restaurant Kitchen

Traditional Deployment = Home Cooking:
- Manual process (chop, cook, plate)
- Inconsistent results
- Hard to scale
- Time-consuming

CI/CD = Professional Kitchen:
- Standardized recipes (scripts)
- Prep stations (stages)
- Quality checks (testing)
- Expeditor (automation)
- Consistent output (deployments)

Kitchen Brigade System:
Executive Chef  → Tech Lead
Sous Chef       → Senior Developer  
Line Cook       → Developer
Expeditor       → CI/CD Pipeline
Food Runner     → Deployment Bot
Dishwasher      → Cleanup Scripts
Inspector       → Quality Assurance
"""

# GitHub Actions Workflow
GITHUB_ACTIONS_WORKFLOW = '''
name: CNC Copilot CI/CD Pipeline
# Analogy: Kitchen Operations Schedule

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    # Nightly builds (like prep work before opening)
    - cron: '0 2 * * *'

env:
  PYTHON_VERSION: '3.11'
  NODE_VERSION: '18'

jobs:
  # ========================================
  # PREP STATION: Code Quality
  # ========================================
  lint:
    name: 🔍 Code Inspection (Mise en Place)
    runs-on: ubuntu-latest
    
    steps:
      - name: 📥 Checkout Code (Get Ingredients)
        uses: actions/checkout@v3
      
      - name: 🐍 Setup Python (Prep Tools)
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'
      
      - name: 📦 Install Dependencies (Gather Tools)
        run: |
          pip install flake8 black isort mypy
          pip install -r requirements.txt
      
      - name: 🎨 Format Check (Plating Standards)
        run: |
          # Black: code formatting (plate presentation)
          black --check .
          
          # isort: import sorting (organize ingredients)
          isort --check-only .
      
      - name: 🔍 Lint Check (Health Inspection)
        run: |
          # Flake8: code linting (safety standards)
          flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
          flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127
      
      - name: 🏷️ Type Check (Recipe Verification)
        run: |
          # MyPy: type checking (ingredient specifications)
          mypy . --ignore-missing-imports

  # ========================================
  # COOKING STATION: Build & Test
  # ========================================
  test:
    name: 🧪 Testing (Taste Test)
    runs-on: ubuntu-latest
    needs: lint  # Wait for inspection to pass
    
    strategy:
      matrix:
        # Test multiple recipes (different Python versions)
        python-version: ['3.10', '3.11', '3.12']
    
    services:
      # Setup ingredients (databases)
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    steps:
      - name: 📥 Checkout Code
        uses: actions/checkout@v3
      
      - name: 🐍 Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'
      
      - name: 📦 Install Dependencies
        run: |
          pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-django
      
      - name: 🗃️ Database Migration (Prep Workspace)
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
        run: |
          python manage.py migrate --no-input
      
      - name: 🧪 Run Tests (Quality Control)
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
          REDIS_URL: redis://localhost:6379/0
        run: |
          # Unit tests (ingredient testing)
          pytest tests/ -v --cov=. --cov-report=xml --cov-report=html
      
      - name: 📊 Upload Coverage (Inspection Report)
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
          flags: unittests
          name: codecov-umbrella

  # ========================================
  # SECURITY STATION: Vulnerability Scan
  # ========================================
  security:
    name: 🔒 Security Scan (Food Safety)
    runs-on: ubuntu-latest
    needs: lint
    
    steps:
      - name: 📥 Checkout Code
        uses: actions/checkout@v3
      
      - name: 🛡️ Dependency Check (Ingredient Safety)
        run: |
          pip install safety bandit
          
          # Safety: check for known vulnerabilities (expired ingredients)
          safety check --json
          
          # Bandit: security linter (food safety inspection)
          bandit -r . -f json -o bandit-report.json
      
      - name: 🔐 Secret Scan (Recipe Protection)
        uses: trufflesecurity/trufflehog@main
        with:
          path: ./
          base: ${{ github.event.repository.default_branch }}
          head: HEAD

  # ========================================
  # BUILD STATION: Docker Image
  # ========================================
  build:
    name: 🐳 Build Docker Image (Package Meal)
    runs-on: ubuntu-latest
    needs: [test, security]
    
    steps:
      - name: 📥 Checkout Code
        uses: actions/checkout@v3
      
      - name: 🏷️ Docker Meta (Menu Description)
        id: meta
        uses: docker/metadata-action@v4
        with:
          images: |
            ghcr.io/${{ github.repository }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=sha
      
      - name: 🔑 Login to GitHub Container Registry
        uses: docker/login-action@v2
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: 🏗️ Build and Push (Cook & Package)
        uses: docker/build-push-action@v4
        with:
          context: .
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # ========================================
  # DEPLOYMENT STATION: Staging
  # ========================================
  deploy-staging:
    name: 🚀 Deploy to Staging (Test Service)
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/develop'
    environment:
      name: staging
      url: https://staging.cnc-copilot.com
    
    steps:
      - name: 📥 Checkout Code
        uses: actions/checkout@v3
      
      - name: 🔧 Configure Kubectl (Setup Service)
        uses: azure/setup-kubectl@v3
      
      - name: 🔑 Setup Kubeconfig (Access Kitchen)
        run: |
          mkdir -p ~/.kube
          echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > ~/.kube/config
      
      - name: 🚀 Deploy to Kubernetes (Serve to Staging)
        run: |
          kubectl set image deployment/cnc-copilot \\
            cnc-copilot=ghcr.io/${{ github.repository }}:sha-${GITHUB_SHA::7} \\
            -n staging
          
          # Wait for rollout (ensure dishes are served)
          kubectl rollout status deployment/cnc-copilot -n staging
      
      - name: 🧪 Smoke Tests (First Bite)
        run: |
          # Wait for service to be ready
          sleep 30
          
          # Health check (taste test)
          curl -f https://staging.cnc-copilot.com/health || exit 1
          
          # API test (quality check)
          curl -f https://staging.cnc-copilot.com/api/ || exit 1

  # ========================================
  # DEPLOYMENT STATION: Production
  # ========================================
  deploy-production:
    name: 🏆 Deploy to Production (Grand Opening)
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/main'
    environment:
      name: production
      url: https://cnc-copilot.com
    
    steps:
      - name: 📥 Checkout Code
        uses: actions/checkout@v3
      
      - name: 🔧 Configure Kubectl
        uses: azure/setup-kubectl@v3
      
      - name: 🔑 Setup Kubeconfig
        run: |
          mkdir -p ~/.kube
          echo "${{ secrets.KUBE_CONFIG_PROD }}" | base64 -d > ~/.kube/config
      
      - name: 🎯 Blue-Green Deployment (Seamless Service)
        run: |
          # Deploy to green environment (prepare new kitchen)
          kubectl apply -f k8s/deployment-green.yaml -n production
          
          # Wait for green to be ready (ensure quality)
          kubectl rollout status deployment/cnc-copilot-green -n production
          
          # Run integration tests on green (final taste test)
          ./scripts/integration-tests.sh https://green.cnc-copilot.com
          
          # Switch traffic to green (open new kitchen)
          kubectl patch service cnc-copilot -n production \\
            -p '{"spec":{"selector":{"version":"green"}}}'
          
          # Wait a bit (observe customer reactions)
          sleep 60
          
          # Scale down blue (close old kitchen)
          kubectl scale deployment/cnc-copilot-blue --replicas=0 -n production
      
      - name: 📢 Notify Team (Announce Opening)
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          text: |
            🎉 Production deployment successful!
            Version: ${{ github.sha }}
            URL: https://cnc-copilot.com
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
        if: always()

  # ========================================
  # CLEANUP STATION: Post-Deployment
  # ========================================
  cleanup:
    name: 🧹 Cleanup (Wash Dishes)
    runs-on: ubuntu-latest
    needs: [deploy-staging, deploy-production]
    if: always()
    
    steps:
      - name: 🗑️ Remove Old Images (Clear Leftovers)
        run: |
          # Keep only last 5 images (don't hoard ingredients)
          gh api --paginate \\
            /orgs/${{ github.repository_owner }}/packages/container/$(echo ${{ github.repository }} | cut -d'/' -f2)/versions \\
            --jq '.[] | select(.metadata.container.tags | length == 0) | .id' \\
            | head -n -5 \\
            | xargs -I{} gh api --method DELETE \\
            /orgs/${{ github.repository_owner }}/packages/container/$(echo ${{ github.repository }} | cut -d'/' -f2)/versions/{}
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
'''

"""
Deployment Strategies Analogy: Restaurant Service

1. Rolling Deployment (Gradual Menu Change):
   - Update one table at a time
   - Customers notice change gradually
   - Can rollback quickly
   - Zero downtime

2. Blue-Green Deployment (Two Kitchens):
   - Prepare entire new kitchen (green)
   - Switch all traffic instantly
   - Quick rollback (switch back)
   - Double resources temporarily

3. Canary Deployment (Test Table):
   - Serve new dish to one table
   - Monitor reactions
   - Gradually expand
   - Safest for major changes

4. Recreate Deployment (Restaurant Closure):
   - Close restaurant
   - Complete renovation
   - Reopen with new setup
   - Downtime required
"""

# Docker Compose for Local Development
DOCKER_COMPOSE = '''
version: '3.8'

# Analogy: Complete Kitchen Setup

services:
  # Main Course: Django Application
  web:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: cnc_web
    command: python manage.py runserver 0.0.0.0:8000
    volumes:
      - .:/app  # Hot reload (like open kitchen where chef adapts)
    ports:
      - "8000:8000"
    env_file:
      - .env
    depends_on:
      - db
      - redis
    networks:
      - cnc_network

  # Side Dish: PostgreSQL Database
  db:
    image: postgres:15
    container_name: cnc_db
    environment:
      POSTGRES_DB: cnc_copilot
      POSTGRES_USER: cnc_user
      POSTGRES_PASSWORD: cnc_password
    volumes:
      - postgres_data:/var/lib/postgresql/data  # Persistent storage (pantry)
    ports:
      - "5432:5432"
    networks:
      - cnc_network

  # Condiments: Redis Cache
  redis:
    image: redis:7-alpine
    container_name: cnc_redis
    ports:
      - "6379:6379"
    networks:
      - cnc_network

  # Beverage Station: NGINX Reverse Proxy
  nginx:
    image: nginx:alpine
    container_name: cnc_nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./static:/static  # Static files (silverware)
    depends_on:
      - web
    networks:
      - cnc_network

volumes:
  postgres_data:  # Cold storage

networks:
  cnc_network:
    driver: bridge  # Kitchen layout
'''

"""
Monitoring Analogy: Kitchen Cameras & Sensors

Metrics to Track:
- Temperature (load/CPU)
- Timers (request duration)
- Counter space (memory)
- Smoke detector (error rate)
- Health inspector (uptime)
- Customer satisfaction (response time)
- Food safety (security)
"""

# Prometheus Configuration
PROMETHEUS_CONFIG = '''
# Prometheus: The Kitchen Dashboard

global:
  scrape_interval: 15s  # Check gauges every 15 seconds
  evaluation_interval: 15s

# Alert Manager (Kitchen Alarm System)
alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093

# Scrape Configurations (What to Monitor)
scrape_configs:
  # Django Application (Main Kitchen)
  - job_name: 'django'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['web:8000']
    
  # PostgreSQL (Pantry Inventory)
  - job_name: 'postgres'
    static_configs:
      - targets: ['db:5432']
    
  # Redis (Condiment Station)
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
    
  # Node Exporter (Kitchen Environment)
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']

# Alert Rules (Kitchen Alarms)
rule_files:
  - 'alerts.yml'
'''

# Alert Rules
ALERT_RULES = '''
groups:
  - name: kitchen_alerts
    interval: 30s
    rules:
      # High Temperature (Overheating)
      - alert: HighCPUUsage
        expr: cpu_usage_percent > 80
        for: 5m
        labels:
          severity: warning
          analogy: "stove_too_hot"
        annotations:
          summary: "CPU usage is high"
          description: "CPU usage has been above 80% for 5 minutes"
      
      # Running Out of Space (Counter Full)
      - alert: HighMemoryUsage
        expr: memory_usage_percent > 90
        for: 5m
        labels:
          severity: critical
          analogy: "counter_full"
        annotations:
          summary: "Memory usage is critical"
          description: "Memory usage has been above 90% for 5 minutes"
      
      # Slow Service (Long Wait Times)
      - alert: SlowResponseTime
        expr: http_response_time_seconds > 2
        for: 10m
        labels:
          severity: warning
          analogy: "slow_service"
        annotations:
          summary: "API response time is slow"
          description: "Response time has been above 2 seconds"
      
      # Kitchen Fire (High Error Rate)
      - alert: HighErrorRate
        expr: error_rate_percent > 5
        for: 5m
        labels:
          severity: critical
          analogy: "kitchen_fire"
        annotations:
          summary: "Error rate is high"
          description: "Error rate > 5% for 5 minutes"
      
      # Restaurant Closed (Service Down)
      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
          analogy: "restaurant_closed"
        annotations:
          summary: "Service is down"
          description: "Service has been down for 1 minute"
'''

"""
Rollback Strategy Analogy: Food Recall

When to Rollback (Recall)
- High error rate (food poisoning)
- Performance degradation (slow service)
- Security breach (contamination)
- Critical bug (wrong ingredients)

How to Rollback:
1. Identify issue (customer complaints)
2. Stop new deployments (halt service)
3. Revert to previous version (old recipe)
4. Verify rollback (taste test)
5. Communicate (inform customers)
6. Post-mortem (what went wrong)
"""

# Rollback Script
ROLLBACK_SCRIPT = '''#!/bin/bash
# Emergency Rollback Script
# Analogy: Emergency Kitchen Shutdown Procedure

set -e  # Exit on error (safety first)

echo "🚨 INITIATING EMERGENCY ROLLBACK"

# Get previous stable version (last good recipe)
PREVIOUS_VERSION=$(kubectl get deployment cnc-copilot -n production \\
                    -o jsonpath='{.metadata.annotations.previous-version}')

echo "📋 Rolling back to version: $PREVIOUS_VERSION"

# Perform rollback (switch back to old kitchen)
kubectl set image deployment/cnc-copilot \\
  cnc-copilot=ghcr.io/cnc-copilot:$PREVIOUS_VERSION \\
  -n production

# Wait for rollback complete (ensure safety)
kubectl rollout status deployment/cnc-copilot -n production

# Verify health (taste test)
curl -f https://cnc-copilot.com/health || {
  echo "❌ Rollback failed health check"
  exit 1
}

echo "✅ Rollback completed successfully"

# Notify team (alert staff)
echo "🔔 Sending alerts..."
./scripts/notify-team.sh "Rollback to $PREVIOUS_VERSION"
'''

print("CI/CD Configuration Complete!")
print("Deployment Kitchen Ready! 🍳")
