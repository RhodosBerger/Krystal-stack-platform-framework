# HARDWARE PROCUREMENT LIST
## Kompletný Zoznam HW Pre Fanuc Rise Deployment

---

## 📦 DEPLOYMENT SCENÁRE

Tento dokument definuje **3 deployment scenáre** podľa našich MD súborov:
1. **Pilot** (1-5 strojov) - Proof of Concept
2. **Production** (10-50 strojov) - Single Factory
3. **Enterprise** (50-500 strojov) - Multi-Factory

---

## SCENÁR 1: PILOT DEPLOYMENT (1-5 CNC)

### A. Edge Computing Hardware

#### **Option A1: Raspberry Pi 5 (Odporúčané pre štart)**
| Item | Spec | Quantity | Unit Price | Total |
|------|------|----------|------------|-------|
| Raspberry Pi 5 | 8GB RAM | 1x | €80 | **€80** |
| MicroSD Card | 128GB, Class 10 | 1x | €15 | **€15** |
| Power Supply | USB-C, 27W | 1x | €12 | **€12** |
| Case + Cooling | Aluminum case + fan | 1x | €20 | **€20** |
| **Subtotal A1** | | | | **€127** |

**Pros**: Low cost, community support, Debian compatible  
**Cons**: Limited to 5 machines max

#### **Option A2: Intel NUC (Pre vyšší výkon)**
| Item | Spec | Quantity | Unit Price | Total |
|------|------|----------|------------|-------|
| Intel NUC 13 Pro | i5-1340P, 16GB RAM, 512GB SSD | 1x | €650 | **€650** |
| **Subtotal A2** | | | | **€650** |

**Pros**: Do 20 machines, Windows/Linux compatible  
**Cons**: 5x drahší než RPi

---

### B. Network Infrastructure

| Item | Spec | Quantity | Unit Price | Total |
|------|------|----------|------------|-------|
| Ethernet Switch | Gigabit, 8-port | 1x | €35 | **€35** |
| Cat6 Cables | 5m, shielded | 5x | €8 | **€40** |
| Power Strip | Surge protected | 1x | €15 | **€15** |
| **Subtotal B** | | | | **€90** |

---

### C. CNC Connection Hardware (Fanuc Specific)

| Item | Spec | Quantity | Unit Price | Total |
|------|------|----------|------------|-------|
| Fanuc FOCAS License | 1-user perpetual | 1x | €800 | **€800** |
| Ethernet Adapter | For older Fanuc models without Ethernet | 1x | €120 | **€120** |
| **Subtotal C** | | | | **€920** |

**Note**: Siemens/Heidenhain nepotrebujú license (OPC UA free).

---

### D. Optional Sensors (Pre advanced features)

| Item | Spec | Quantity | Unit Price | Total |
|------|------|----------|------------|-------|
| Accelerometer | 3-axis, 0-2g, I2C | 3x | €45 | **€135** |
| Thermal Camera | IR, -20°C to 150°C | 1x | €280 | **€280** |
| USB Camera | 1080p, for chip detection | 1x | €60 | **€60** |
| **Subtotal D (Optional)** | | | | **€475** |

---

### **PILOT TOTAL (Option A1 + Basic)**:
- Edge HW (RPi): €127
- Network: €90
- CNC Connection: €920
- **TOTAL MINIMUM**: **€1,137**
- **TOTAL S OPTIONS**: €1,612 (+ sensors)

### **PILOT TOTAL (Option A2 + Premium)**:
- Edge HW (NUC): €650
- Network: €90
- CNC Connection: €920
- Sensors: €475
- **TOTAL PREMIUM**: **€2,135**

---

## SCENÁR 2: PRODUCTION (10-50 CNC)

### A. Edge Computing (Scaled)

| Item | Spec | Quantity | Unit Price | Total |
|------|------|----------|------------|-------|
| Edge Server | Xeon E-2388G, 32GB RAM, 1TB SSD | 1x | €1,800 | **€1,800** |
| UPS Battery | 1500VA, 900W | 1x | €250 | **€250** |
| Rack Mount | 19" rack, 12U | 1x | €180 | **€180** |
| **Subtotal A** | | | | **€2,230** |

---

### B. Network Infrastructure (Industrial)

| Item | Spec | Quantity | Unit Price | Total |
|------|------|----------|------------|-------|
| Managed Switch | Gigabit, 48-port, Layer 2 | 1x | €450 | **€450** |
| Firewall | pfSense compatible, dual WAN | 1x | €350 | **€350** |
| Cat6 Cables | Pre-terminated, various lengths | 50x | €8 | **€400** |
| Fiber Optic | For long runs (>100m) | 2x | €120 | **€240** |
| **Subtotal B** | | | | **€1,440** |

---

### C. Database & Cache Server

| Item | Spec | Quantity | Unit Price | Total |
|------|------|----------|------------|-------|
| Database Server | Xeon Silver, 64GB RAM, 2x2TB SSD (RAID1) | 1x | €2,800 | **€2,800** |
| Redis Server | (Can run on edge server, or dedicated) | Optional | €0 | **€0** |
| **Subtotal C** | | | | **€2,800** |

---

### D. Monitoring & Displays

| Item | Spec | Quantity | Unit Price | Total |
|------|------|----------|------------|-------|
| Dashboard Display | 50" 4K monitor, wall mount | 1x | €400 | **€400** |
| Mini PC (Display driver) | Intel Celeron, 8GB RAM | 1x | €250 | **€250** |
| **Subtotal D** | | | | **€650** |

---

### E. Licenses (Scaled)

| Item | Spec | Quantity | Unit Price | Total |
|------|------|----------|------------|-------|
| Fanuc FOCAS | Network license (up to 50 concurrent) | 1x | €3,500 | **€3,500** |
| Windows Server Std | 16-core license (if not using Linux) | 1x | €900 | **€900** |
| **Subtotal E** | | | | **€4,400** |

---

### **PRODUCTION TOTAL**:
- Edge: €2,230
- Network: €1,440
- Database: €2,800
- Monitoring: €650
- Licenses: €4,400
- **TOTAL**: **€11,520**

**Per-Machine Cost**: €230 (for 50 machines) - cheaper than individual solutions!

---

## SCENÁR 3: ENTERPRISE (50-500 CNC, Multi-Factory)

### A. On-Premise Cloud Infrastructure (Primary Datacenter)

| Item | Spec | Quantity | Unit Price | Total |
|------|------|----------|------------|-------|
| Server Rack | 42U, climate controlled | 1x | €2,500 | **€2,500** |
| Compute Nodes | Dual Xeon, 128GB RAM, 1TB NVMe (×3 for HA) | 3x | €5,500 | **€16,500** |
| Storage Array | 50TB usable, SSD+HDD tiered | 1x | €12,000 | **€12,000** |
| Network Switch | 10GbE, 48-port | 2x | €3,200 | **€6,400** |
| Load Balancer | Hardware LB, 10Gbps | 1x | €4,500 | **€4,500** |
| Firewall | Enterprise dual WAN, IPS/IDS | 1x | €2,800 | **€2,800** |
| UPS System | 10kVA, 3-phase | 1x | €4,500 | **€4,500** |
| **Subtotal A** | | | | **€49,200** |

---

### B. Edge Gateways (Per Factory Location)

| Item | Spec | Quantity | Unit Price | Total |
|------|------|----------|------------|-------|
| Edge Server | (Same as Production scenario) | 3x | €1,800 | **€5,400** |
| Network Equipment | (Switches, cabling per site) | 3x | €1,500 | **€4,500** |
| **Subtotal B (3 sites)** | | | | **€9,900** |

---

### C. Cloud Services (If Hybrid Model)

| Item | Spec | Annual Cost | Notes |
|------|------|-------------|-------|
| AWS EC2 | 3x m5.xlarge (reserved instances) | €6,500 | API + LLM inference |
| AWS RDS | db.r5.2xlarge Multi-AZ | €8,400 | PostgreSQL |
| AWS S3 + Glacier | 100TB storage | €2,400 | Telemetry archive |
| CloudFront CDN | Data transfer 10TB/mo | €1,200 | Dashboard delivery |
| **Subtotal C (Annual)** | | **€18,500/year** | |

**Or**: Self-hosted (0€ cloud, but higher upfront HW cost)

---

### D. Licenses (Enterprise Scale)

| Item | Spec | Quantity | Unit Price | Total |
|------|------|----------|------------|-------|
| Fanuc FOCAS | Enterprise unlimited license | 1x | €15,000 | **€15,000** |
| OPC UA Suite | For Siemens/Heidenhain | 1x | €5,000 | **€5,000** |
| Windows Server Datacenter | Unlimited VMs | 1x | €6,200 | **€6,200** |
| VMware vSphere Std | 6x hosts | 1x | €8,500 | **€8,500** |
| **Subtotal D** | | | | **€34,700** |

---

### E. Security & Compliance

| Item | Spec | Quantity | Unit Price | Total |
|------|------|----------|------------|-------|
| HSM (Hardware Security Module) | For JWT signing keys | 1x | €3,800 | **€3,800** |
| SIEM System | Security monitoring | 1x | €6,500 | **€6,500** |
| Backup Appliance | Veeam-compatible, 100TB | 1x | €8,200 | **€8,200** |
| **Subtotal E** | | | | **€18,500** |

---

### **ENTERPRISE TOTAL (On-Premise)**:
- Datacenter: €49,200
- Edge (3 sites): €9,900
- Licenses: €34,700
- Security: €18,500
- **TOTAL CAPEX**: **€112,300**
- **OPEX (Cloud)**: €18,500/year (if hybrid)

**Per-Machine Cost**: €225 (for 500 machines) + €37/machine/year cloud

---

## 📊 COMPARISON MATRIX

| Scenario | Machines | CAPEX | OPEX/Year | Per-Machine |
|----------|----------|-------|-----------|-------------|
| **Pilot** | 1-5 | €1,137-2,135 | €500/machine | €227-427 |
| **Production** | 10-50 | €11,520 | €20,000 | €230-1,152 |
| **Enterprise** | 50-500 | €112,300 | €203,500 | €225-2,246 |

**Note**: OPEX includes Fanuc Rise licenses (€500/machine), cloud costs, support.

---

## 🛒 PROCUREMENT RECOMMENDATIONS

### Phase 1: Immediate (Week 1)
- [ ] Edge hardware (RPi alebo NUC)
- [ ] Network switch + cables
- [ ] FOCAS license application (6-8 weeks lead time!)

### Phase 2: Month 1
- [ ] Database server (if >10 machines)
- [ ] UPS system
- [ ] Sensors (if predictive maintenance needed)

### Phase 3: Month 2+
- [ ] Dashboard displays
- [ ] Cloud infrastructure setup
- [ ] Enterprise security (HSM, SIEM)

---

## 💡 COST OPTIMIZATION TIPS

1. **Start Local**: Use existing PC/Server for pilot (€0 HW cost)
2. **BYOL (Bring Your Own License)**: Use Linux → save €900 on Windows
3. **Cloud-First**: Skip on-premise datacenter, use AWS → -€49k CAPEX
4. **Gradual Sensors**: Start without sensors, add later
5. **Open-Source Tools**: Prometheus + Grafana (free) vs paid monitoring

---

## 🔧 MAINTENANCE & SPARES

| Item | Reason | Quantity | Cost |
|------|--------|----------|------|
| Spare RPi/NUC | Edge gateway failure | 1x | €130-650 |
| Replacement SSD | Database disk failure | 2x | €300 |
| Network cables | Physical damage | 10x | €80 |
| UPS batteries | 3-year lifespan | Set | €150 |
| **Annual Spares Budget** | | | **€1,160** |

---

**ZÁVER**: Pre **Pilot start**, minimálna investícia je **€1,137** (RPi + basic network + FOCAS). Pre **Production 50 machines**, kalkuluj **€12k HW + €20k/year OPEX** = total 3-year TCO ~€72k = €1,440/machine (less než half of traditional per-machine monitoring).

*Shopping list by Dusan Berger, based on 43-phase architecture, January 2026*
