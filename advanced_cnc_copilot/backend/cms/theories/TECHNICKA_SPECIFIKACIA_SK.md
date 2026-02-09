# TECHNICKÁ ŠPECIFIKÁCIA: FANUC RISE
## Podrobný prehľad schopností systému

---

## 1. ARCHITEKTÚRA KOMPONENTOV

### 1.1 Základné moduly (Core Intelligence)
| Modul | Úloha | Latencia | Kritickosť |
|-------|-------|----------|------------|
| **Sensory Cortex** | Zber telemetrie z CNC | <1ms | KRITICKÁ |
| **Impact Cortex** | Rozhodovacia logika | <5ms | KRITICKÁ |
| **Dopamine Engine** | Hodnotenie akcií (reward/penalty) | <10ms | VYSOKÁ |
| **Knowledge Graph** | Kauzálna mapa problémov | On-demand | STREDNÁ |
| **Signaling System** | Traffic light (Green/Amber/Red) | <20ms | VYSOKÁ |

### 1.2 API & Interface vrstva
| Modul | Technológia | Výkon |
|-------|-------------|-------|
| **FastAPI Hub** | Python AsyncIO | 10,000 req/sec |
| **WebSocket Stream** | Real-time telemetria | 1kHz update rate |
| **Dashboard** | HTML5 + CSS3 | <100ms load time |

### 1.3 Cloud & Learning vrstva
| Modul | Poskytovateľ | Škálovateľnosť |
|-------|--------------|----------------|
| **LLM Service** | OpenAI/Anthropic | Unlimited (API) |
| **PostgreSQL** | AWS RDS / Azure SQL | 10k TPS |
| **Django ERP** | Self-hosted | 100+ users |

---

## 2. PODPOROVANÉ HARDWAROVÉ KONFIGURÁCIE

### 2.1 CNC kontroléry
✅ **Fanuc**: Série 0i, 16i, 18i, 30i, 31i, 32i (FOCAS Ethernet)  
✅ **Siemens**: 840D sl, 828D (OPC UA)  
✅ **Heidenhain**: TNC 640, TNC 620 (DNC protocol)  
✅ **Haas**: NGC (Ethernet)  
✅ **Mazak**: Mazatrol (Custom adapter)

### 2.2 Edge hardware minimálne požiadavky
- **CPU**: 4-core ARM64 alebo x86_64
- **RAM**: 8GB (16GB odporúčané)
- **Disk**: 128GB SSD
- **Network**: Gigabit Ethernet
- **OS**: Ubuntu 22.04 LTS alebo Windows Server 2022

### 2.3 Senzory (voliteľné rozšírenia)
- Akcelerometer (vibrácie, 0-2g range)
- Teplomer (spindle temp, -20°C až 150°C)
- Mikrofón (audio chatter detection)
- Kamera (vision inspection, 1080p min)

---

## 3. SOFTVÉROVÁ INTEGRÁCIA

### 3.1 CAD/CAM systémy
| Software | Integrácia | Možnosti |
|----------|-----------|----------|
| **Solidworks** | COM API (pywin32) | Read/Write parametre, Run simulation |
| **Fusion 360** | REST API | Download toolpaths, Upload metrics |
| **Mastercam** | File export | Import G-code, Analyze |
| **Siemens NX** | NX Open API | Bidirectional sync |

### 3.2 ERP/MES systémy
| Systém | Metóda | Use case |
|--------|--------|----------|
| **SAP** | REST API | Job scheduling, Material tracking |
| **Oracle NetSuite** | SOAP/REST | Order management |
| **Custom DB** | Direct SQL | Real-time sync |

---

## 4. BEZPEČNOSTNÉ VLASTNOSTI

### 4.1 Autentifikácia
- **Metóda**: JWT tokens (RS256 signing)
- **Expiry**: 15 minút (refresh tokens 7 dní)
- **MFA**: TOTP (Google Authenticator compatible)
- **SSO**: OAuth2 (Google, Microsoft, SAML)

### 4.2 Autorizácia (RBAC)
| Rola | Práva |
|------|-------|
| **Admin** | Všetko (manage users, view finance, override safety) |
| **Engineer** | Upload CAD, Edit macros, Run simulations |
| **Operator** | Execute G-code, Local adjustments (±20%) |
| **Auditor** | View logs, Export reports (read-only) |

### 4.3 Network security
- **TLS 1.3** minimum (cipher suites: ECDHE-RSA-AES256-GCM-SHA384)
- **mTLS** pre machine-to-cloud komunikáciu
- **Firewall**: Whitelist IP ranges, rate limiting (100 req/min)
- **DDoS protection**: Cloudflare alebo AWS Shield

---

## 5. MONITOROVANIE A LOGY

### 5.1 Telemetria (real-time)
| Parameter | Sample rate | Storage |
|-----------|-------------|---------|
| Spindle RPM | 1kHz | Hot: 7 days, Warm: 90 days |
| Servo Load (%) | 1kHz | Hot: 7 days, Warm: 90 days |
| Vibration (g) | 1kHz | Hot: 7 days, Warm: 90 days |
| Tool life (%) | Per block | Indefinite |
| Temperature (°C) | 10Hz | Hot: 30 days |

### 5.2 Log úrovne
- **Developer**: DEBUG level (stack traces, full context)
- **Technical**: INFO level (structured JSON, parseable)
- **Operator**: WARNING/ERROR (human-readable Slovak)

### 5.3 Export formáty
- CSV (Excel-compatible)
- JSON (programmatic access)
- HTML (styled reports)
- PDF (executive summary)

---

## 6. VÝKONNOSTNÉ PARAMETRE

### 6.1 API SLA
| Endpoint | Max latency (p95) | Throughput |
|----------|-------------------|------------|
| `/telemetry` | 50ms | 10k req/sec |
| `/suggest` (LLM) | 500ms | 100 req/sec |
| `/execute` | 100ms | 1k req/sec |

### 6.2 Database performance
- **Write throughput**: 5,000 inserts/sec (PostgreSQL)
- **Read latency**: <10ms (indexed queries)
- **Failover time**: <30s (Multi-AZ deployment)

### 6.3 Uptime SLA
- **Edge services**: 99.9% (8.76 hours downtime/year)
- **Cloud services**: 99.5% (43.8 hours downtime/year)
- **Dashboard**: 99.0% (87.6 hours downtime/year)

---

## 7. ŠKÁLOVATEĽNOSŤ

### 7.1 Vertikálna škála (single factory)
- **1-10 strojov**: Raspberry Pi 5 postačuje
- **11-50 strojov**: Intel NUC (i7, 32GB RAM)
- **51-100 strojov**: Dedicated server (Xeon, 64GB RAM)

### 7.2 Horizontálna škála (multi-factory)
- **Edge**: 1 gateway na závod
- **Cloud**: Auto-scaling (1-100 instances)
- **Database**: Read replicas (up to 15 replicas)

---

## 8. DEPLOYMENT MOŽNOSTI

### 8.1 Docker Compose (Single node)
```yaml
services:
  fastapi:
    image: fanuc-rise:latest
    ports: ["8000:8000"]
  postgres:
    image: postgres:15
  redis:
    image: redis:7
```
**Vhodné pre**: 1-10 strojov, pilot deployment

### 8.2 Kubernetes (Cluster)
- **Master nodes**: 3 (HA setup)
- **Worker nodes**: 5-20 (auto-scale)
- **Ingress**: Nginx Ingress Controller
- **Storage**: Persistent Volumes (EBS/Azure Disk)

**Vhodné pre**: 50+ strojov, production

---

## 9. DÁTOVÁ SUVERENITA & COMPLIANCE

### 9.1 GDPR
- ✅ Right to erasure (DELETE /user/{id})
- ✅ Data portability (Export all data to JSON)
- ✅ Consent management (Opt-in for analytics)

### 9.2 Regiónová separácia
- **EU data**: Frankfurt AWS region (eu-central-1)
- **US data**: Virginia region (us-east-1)
- **Cross-region**: Only anonymized metrics

---

## 10. LICENČNÝ MODEL

### 10.1 Cena podľa strojov
- **1-5 strojov**: €500/stroj/rok
- **6-20 strojov**: €400/stroj/rok
- **21-100 strojov**: €300/stroj/rok
- **101+ strojov**: Custom pricing

### 10.2 Čo je zahrnuté
- ✅ Všetky moduly (Core + Cloud + Dashboard)
- ✅ Updates & patches (rolling release)
- ✅ Email support (response <24h)
- ✅ Community forum access

### 10.3 Prémiové doplnky
- **24/7 Support**: +€5,000/rok
- **Custom LLM training**: +€10,000 (one-time)
- **On-site installation**: €2,000/deň
- **Dedicated account manager**: €15,000/rok

---

## 11. BUDÚCE FUNKCIE (Roadmap)

### Q2 2026
- [ ] Mobilná appka (iOS/Android)
- [ ] Voice commands ("Fanuc, slow down feed")
- [ ] AR vizualizácia (HoloLens integration)

### Q3 2026
- [ ] Generative toolpaths (AI píše G-code from scratch)
- [ ] Swarm optimization (100 virtual ants hľadajú optimum)

### Q4 2026
- [ ] Full autonomy (zero human in loop for standard jobs)
- [ ] Self-healing (auto recovery from stalls)

---

*Technická dokumentácia verzia 1.0 | Január 2026*
