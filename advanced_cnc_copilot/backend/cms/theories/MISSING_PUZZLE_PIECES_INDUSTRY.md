# MISSING PUZZLE PIECES - INDUSTRY INSPIRATION
## Features Ktoré Nám Ešte Chýbajú (Založené Na Industry Standards)

---

## 🧩 CURRENT STATE: 50% COMPLETE

**Čo máme (45 fáz hotových)**:
- ✅ Cognitive core (dopamine, impact cortex, signaling)
- ✅ Real-time telemetry monitoring
- ✅ LLM suggestions
- ✅ Dynamic UI panels
- ✅ Multi-level logging
- ✅ Economic tracking (cost/part)

**Čo chýba (druhá polovica puzzle)**:
- ❌ OEE tracking (Overall Equipment Effectiveness)
- ❌ SPC charts (Statistical Process Control)
- ❌ Tool life management system
- ❌ Production scheduling & job queue
- ❌ Quality inspection integration
- ❌ Energy monitoring
- ❌ Inventory/material tracking
- ❌ Collaborative features (multi-user)
- ❌ Mobile app
- ❌ Video/camera feed integration

---

## 1. OEE TRACKING (Industry Standard Metric)

### **Čo je OEE?**
Overall Equipment Effectiveness = Availability × Performance × Quality

```
Availability = (Operating Time / Planned Production Time)
Performance = (Actual Output / Target Output)
Quality = (Good Units / Total Units)

Example:
- Planned: 8 hours (480 min)
- Downtime: 30 min (setup, breaks)
- Operating: 450 min → Availability = 93.75%
- Target: 50 parts → Actual: 45 parts → Performance = 90%
- Good: 43 parts → Quality = 95.6%

OEE = 93.75% × 90% × 95.6% = 80.6% (GOOD, world-class is 85%)
```

### **Implementation**:

```python
# cms/oee_tracker.py
class OEETracker:
    def __init__(self, machine_id):
        self.machine_id = machine_id
        self.shift_start = datetime.now()
        self.planned_time = 8 * 60  # 8 hour shift in minutes
        
    def calculate_availability(self):
        downtime_events = Downtime.objects.filter(
            machine=self.machine_id,
            timestamp__gte=self.shift_start
        )
        total_downtime = sum([e.duration for e in downtime_events])
        operating_time = self.planned_time - total_downtime
        return (operating_time / self.planned_time) * 100
    
    def calculate_performance(self):
        actual_parts = PartCounter.objects.filter(
            machine=self.machine_id,
            timestamp__gte=self.shift_start
        ).count()
        
        target_parts = self.get_target_from_schedule()
        return (actual_parts / target_parts) * 100
    
    def calculate_quality(self):
        total = PartCounter.objects.filter(machine=self.machine_id).count()
        rejected = QualityReject.objects.filter(machine=self.machine_id).count()
        good = total - rejected
        return (good / total) * 100
    
    def get_oee(self):
        avail = self.calculate_availability()
        perf = self.calculate_performance()
        qual = self.calculate_quality()
        
        oee = (avail / 100) * (perf / 100) * (qual / 100) * 100
        
        return {
            "oee": round(oee, 2),
            "availability": round(avail, 2),
            "performance": round(perf, 2),
            "quality": round(qual, 2),
            "classification": "World Class" if oee >= 85 else ("Good" if oee >= 60 else "Needs Improvement")
        }
```

### **Dashboard Visualization**:
```
┌─────────────────────────────────────┐
│ OEE: 80.6% (Good)              🟢  │
├─────────────────────────────────────┤
│ Availability: 93.8% [████████░░]   │
│ Performance:  90.0% [████████░░]   │
│ Quality:      95.6% [█████████░]   │
├─────────────────────────────────────┤
│ Shift: Day (08:00-16:00)           │
│ Good Parts: 43 / 50 target         │
│ Downtime: 30 min (Setup: 20, Break: 10) │
└─────────────────────────────────────┘
```

---

## 2. SPC CHARTS (Statistical Process Control)

### **Čo sú SPC Charts?**
Real-time kontrola či proces je "in control" alebo "out of control".

**Key Charts**:
- X̄-R Chart (Average & Range)
- X̄-S Chart (Average & Standard Deviation)
- p-Chart (Proportion defective)
- c-Chart (Count of defects)

### **Example: X̄-Chart Pre Dimensio Control**

```
Measurement: Diameter of bearing bore (target: 25.00mm, tolerance: ±0.05mm)

UCL (Upper Control Limit) = 25.05mm
Target = 25.00mm
LCL (Lower Control Limit) = 24.95mm

Plot:
25.10 |                                    × (OUT!)
25.05 |─────────────UCL─────────────────────
25.00 |    ×  ×  ×     ×  ×        ×
24.95 |─────────────LCL─────────────────────
24.90 |                    ×                 (OUT!)
      └─────────────────────────────────────
       1  2  3  4  5  6  7  8  9  10 (samples)
```

**Implementation**:
```python
# cms/spc_analyzer.py
import numpy as np

class SPCChart:
    def __init__(self, target, tolerance):
        self.target = target
        self.ucl = target + tolerance
        self.lcl = target - tolerance
        self.measurements = []
    
    def add_measurement(self, value):
        self.measurements.append({
            "value": value,
            "timestamp": datetime.now(),
            "status": self.check_status(value)
        })
    
    def check_status(self, value):
        if value > self.ucl or value < self.lcl:
            return "OUT_OF_CONTROL"
        elif value > self.target + (0.67 * (self.ucl - self.target)):
            return "WARNING"  # 2-sigma
        else:
            return "IN_CONTROL"
    
    def detect_trends(self):
        """Detect Western Electric Rules violations"""
        recent = self.measurements[-9:]  # Last 9 samples
        
        # Rule 1: One point beyond 3σ
        if any(m['status'] == "OUT_OF_CONTROL" for m in recent):
            return {"rule": 1, "alert": "Point beyond control limits!"}
        
        # Rule 2: 9 points in a row on same side of centerline
        if len(recent) == 9:
            all_above = all(m['value'] > self.target for m in recent)
            all_below = all(m['value'] < self.target for m in recent)
            if all_above or all_below:
                return {"rule": 2, "alert": "Process drift detected!"}
        
        return None
```

### **Dashboard Integration**:
```html
<div class="spc-panel">
  <h3>SPC: Bore Diameter (Ø25mm)</h3>
  <canvas id="spc-chart"></canvas>
  
  <div class="spc-status">
    <span class="status-badge green">IN CONTROL</span>
    <p>Last 10 samples within ±2σ</p>
  </div>
  
  <div class="spc-alerts" style="display:none;">
    <span class="alert-badge red">⚠️ RULE 2 VIOLATION</span>
    <p>9 consecutive points above centerline - process drift!</p>
    <button onclick="recalibrate()">Recalibrate Machine</button>
  </div>
</div>
```

---

## 3. TOOL LIFE MANAGEMENT SYSTEM

### **Chýbajúce Features**:
- Tool inventory database
- Automatic tool change alerts
- Tool usage history
- Cost per tool tracking
- Vendor management

### **Implementation**:

```python
# cms/tool_manager.py
class Tool:
    id: str  # T01, T02, T03...
    type: str  # Endmill, Drill, Bore...
    diameter: float  # mm
    length: float  # mm
    material: str  # HSS, Carbide, PCD
    coating: str  # TiN, TiAlN, AlCrN
    vendor: str
    purchase_price: float
    expected_life: int  # minutes
    current_usage: int  # minutes used
    status: str  # NEW, IN_USE, WORN, BROKEN
    
    @property
    def remaining_life(self):
        return max(0, self.expected_life - self.current_usage)
    
    @property
    def life_percentage(self):
        return (self.remaining_life / self.expected_life) * 100
    
    def should_replace(self):
        return self.life_percentage < 10

class ToolLifeManager:
    def update_usage(self, tool_id, cycle_time):
        tool = Tool.objects.get(id=tool_id)
        tool.current_usage += cycle_time
        tool.save()
        
        if tool.should_replace():
            self.create_alert(tool)
    
    def create_alert(self, tool):
        Alert.objects.create(
            type="TOOL_REPLACEMENT",
            message=f"Tool {tool.id} ({tool.type} Ø{tool.diameter}mm) needs replacement!",
            priority="HIGH",
            suggested_action=f"Order from {tool.vendor}"
        )
    
    def recommend_alternative(self, tool_id):
        """LLM-powered tool recommendation"""
        tool = Tool.objects.get(id=tool_id)
        
        from cms.protocol_conductor import ProtocolConductor
        conductor = ProtocolConductor()
        
        recommendation = conductor.suggest_tool_alternative({
            "current_tool": tool.to_dict(),
            "material": "Aluminum 6061",
            "budget": "medium"
        })
        
        return recommendation
```

### **Dashboard**:
```
┌────────────────────────────────────────┐
│ TOOL INVENTORY & LIFE TRACKING        │
├────────────────────────────────────────┤
│ T01: Endmill Ø8mm                     │
│ Life: [████████░░] 82% remaining      │
│ Usage: 18min / 100min expected        │
│ Status: 🟢 IN USE                     │
│                                        │
│ T03: Drill Ø25mm                      │
│ Life: [███░░░░░░░] 28% remaining      │
│ Usage: 32min / 45min expected         │
│ Status: ⚠️ NEEDS REPLACEMENT SOON     │
│ [Order from Sandvik] [Find Alternative]│
│                                        │
│ T05: Boring Bar                        │
│ Life: [█░░░░░░░░░] 8% remaining ❗    │
│ Status: 🔴 CRITICAL - REPLACE NOW!    │
│ [Emergency Order] [Use Backup Tool]   │
└────────────────────────────────────────┘
```

---

## 4. PRODUCTION SCHEDULING & JOB QUEUE

### **Chýba nám**:
- Job priority management
- Gantt chart pre machine loading
- Capacity planning
- Bottleneck detection

### **Implementation**:

```python
# cms/production_scheduler.py
from datetime import datetime, timedelta

class Job:
    id: str
    part_number: str
    quantity: int
    estimated_time_per_part: float  # minutes
    priority: int  # 1=URGENT, 5=LOW
    deadline: datetime
    material_available: bool
    assigned_machine: str
    status: str  # QUEUED, IN_PROGRESS, COMPLETED
    
    @property
    def total_time_estimate(self):
        return self.quantity * self.estimated_time_per_part

class ProductionScheduler:
    def optimize_schedule(self, jobs, machines):
        """
        Genetic algorithm pre optimálne rozloženie jobs.
        Objectives:
        1. Minimize total makespan
        2. Maximize high-priority jobs completion
        3. Balance machine load
        """
        from scipy.optimize import linear_sum_assignment
        
        # Create cost matrix (job × machine)
        cost_matrix = np.zeros((len(jobs), len(machines)))
        
        for i, job in enumerate(jobs):
            for j, machine in enumerate(machines):
                # Cost = Time × Priority Penalty
                cost_matrix[i][j] = job.total_time_estimate * (6 - job.priority)
        
        # Solve assignment problem
        job_indices, machine_indices = linear_sum_assignment(cost_matrix)
        
        # Build schedule
        schedule = []
        for job_idx, machine_idx in zip(job_indices, machine_indices):
            schedule.append({
                "job": jobs[job_idx],
                "machine": machines[machine_idx],
                "start_time": self.calculate_start_time(machine_idx, schedule)
            })
        
        return schedule
    
    def detect_bottleneck(self, schedule):
        """Find machine with highest utilization"""
        utilization = {}
        for entry in schedule:
            machine = entry["machine"].id
            utilization[machine] = utilization.get(machine, 0) + entry["job"].total_time_estimate
        
        bottleneck = max(utilization, key=utilization.get)
        return {
            "machine": bottleneck,
            "utilization": utilization[bottleneck],
            "recommendation": f"Consider load balancing or adding capacity"
        }
```

### **Gantt Chart Visualization**:
```
Machine Timeline (Next 8 hours):

CNC_01: [████JOB_A████][██JOB_C██][    idle    ]
CNC_02: [██████JOB_B██████][████JOB_D████][idle]
CNC_03: [░░░░setup░░░░][████████JOB_E████████]
        ├────────────────────────────────────┤
        08:00         12:00         16:00
        
Legend:
█ Production  ░ Setup/Changeover  [ ] Idle

Alerts:
⚠️ CNC_02 at 98% utilization (bottleneck!)
💡 Consider moving JOB_D to CNC_01 to balance load
```

---

## 5. QUALITY INSPECTION INTEGRATION

### **Missing Features**:
- CMM (Coordinate Measuring Machine) data import
- Automatic first-article inspection
- Non-conformance tracking
- Corrective action workflow

### **Implementation**:

```python
# cms/quality_inspector.py
class InspectionReport:
    part_id: str
    measured_dimensions: Dict[str, float]
    specified_dimensions: Dict[str, Tuple[float, float]]  # (nominal, tolerance)
    inspector: str
    timestamp: datetime
    
    def check_conformance(self):
        results = {}
        for dimension, measured in self.measured_dimensions.items():
            nominal, tolerance = self.specified_dimensions[dimension]
            ucl = nominal + tolerance
            lcl = nominal - tolerance
            
            if lcl <= measured <= ucl:
                results[dimension] = "PASS"
            else:
                results[dimension] = "FAIL"
                deviation = measured - nominal
                self.create_ncr(dimension, deviation)
        
        return results
    
    def create_ncr(self, dimension, deviation):
        """Non-Conformance Report"""
        NCR.objects.create(
            part_id=self.part_id,
            dimension=dimension,
            deviation=deviation,
            root_cause=self.analyze_root_cause(dimension, deviation),
            corrective_action=self.suggest_correction(dimension, deviation)
        )
    
    def analyze_root_cause(self, dimension, deviation):
        """LLM-powered root cause analysis"""
        from cms.protocol_conductor import ProtocolConductor
        conductor = ProtocolConductor()
        
        analysis = conductor.analyze_quality_issue({
            "dimension": dimension,
            "deviation": deviation,
            "recent_telemetry": self.get_recent_telemetry(),
            "tool_condition": self.get_tool_status()
        })
        
        return analysis["root_cause"]
```

---

## 6. ENERGY MONITORING

### **Industry 4.0 Requirement**: Track energy per part for sustainability reporting.

```python
# cms/energy_monitor.py
class EnergyMonitor:
    def track_power(self):
        """Poll smart meter every second"""
        current_power = self.read_power_meter()  # kW
        
        EnergyLog.objects.create(
            timestamp=datetime.now(),
            machine=self.machine_id,
            power_kw=current_power,
            spindle_load=self.get_spindle_load(),
            coolant_pump=self.get_coolant_status()
        )
    
    def calculate_energy_per_part(self, job_id):
        job = Job.objects.get(id=job_id)
        energy_logs = EnergyLog.objects.filter(
            machine=job.assigned_machine,
            timestamp__range=(job.start_time, job.end_time)
        )
        
        total_kwh = sum([log.power_kw * (1/3600) for log in energy_logs])  # kWh
        energy_per_part = total_kwh / job.quantity
        cost_per_part = energy_per_part * 0.12  # €0.12/kWh
        
        return {
            "total_kwh": total_kwh,
            "cost": cost_per_part,
            "co2_kg": total_kwh * 0.5  # 0.5 kg CO2/kWh (EU average)
        }
```

### **Dashboard**:
```
┌────────────────────────────────────┐
│ ENERGY DASHBOARD                   │
├────────────────────────────────────┤
│ Current Power: 12.5 kW 📊         │
│ Today Total: 85.2 kWh              │
│ Cost: €10.22                       │
│ CO2: 42.6 kg 🌱                    │
│                                    │
│ Per Part:                          │
│ • Energy: 0.15 kWh/part           │
│ • Cost: €0.018/part               │
│ • Target: <€0.02 ✅               │
└────────────────────────────────────┘
```

---

## 7. COLLABORATIVE FEATURES (Multi-User)

### **Missing**:
- Real-time collaboration (multiple users viewing same machine)
- Chat/comments on jobs
- Shift handover notes
- Knowledge sharing

```python
# cms/collaboration.py
class ShiftHandover:
    from_shift: str  # DAY, NIGHT
    to_shift: str
    machine: str
    notes: str
    issues: List[str]
    completed_jobs: List[str]
    pending_jobs: List[str]
    
class MachineChat:
    machine_id: str
    messages: List[Dict]
    
    def add_message(self, user, message):
        self.messages.append({
            "user": user,
            "message": message,
            "timestamp": datetime.now()
        })
        
        # WebSocket broadcast to all users watching this machine
        broadcast_to_room(f"machine_{self.machine_id}", {
            "type": "chat_message",
            "data": message
        })
```

---

## 8. MOBILE APP

### **Essential Features**:
- Push notifications (alerts, job completion)
- Quick status check (OEE, current job)
- Approve/reject AI suggestions remotely
- Photo upload (issues, finished parts)

```dart
// Flutter mobile app
class MachineStatusWidget extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Card(
      child: Column(
        children: [
          Text('CNC_VMC_01', style: TextStyle(fontSize: 24)),
          StatusIndicator(color: Colors.green, label: 'RUNNING'),
          Text('Job: BRACKET_ALU_V3'),
          LinearProgressIndicator(value: 0.65),
          Text('65% Complete (32 / 50 parts)'),
          
          Row(
            children: [
              IconButton(
                icon: Icon(Icons.pause),
                onPressed: () => pauseMachine(),
              ),
              IconButton(
                icon: Icon(Icons.notifications),
                onPressed: () => manageAlerts(),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
```

---

## 9. VIDEO/CAMERA INTEGRATION

### **Use Cases**:
- Live camera feed of machining
- Time-lapse recording of jobs
- AI-powered chip detection (bird's nest)
- Remote visual inspection

```python
# cms/video_analyzer.py
import cv2
import torch

class ChipDetector:
    def __init__(self):
        self.model = torch.load('yolov5_chip_detector.pt')
    
    def analyze_frame(self, frame):
        """Detect chip buildup"""
        results = self.model(frame)
        
        chips_detected = results.pandas().xyxy[0]
        if len(chips_detected) > 0:
            return {
                "alert": True,
                "chip_area": sum(chips_detected['area']),
                "recommendation": "Pause and clear chips to avoid bird's nest!"
            }
        return {"alert": False}
```

---

## 10. INVENTORY/MATERIAL TRACKING

### **Missing Piece**:
- Raw material stock levels
- Automatic reorder alerts
- Material traceability (batch/lot numbers)
- Scrap tracking

```python
# cms/inventory_manager.py
class MaterialStock:
    material_type: str  # Aluminum 6061
    form: str  # Bar, Plate, Sheet
    size: str  # Ø50mm × 3000mm
    quantity_on_hand: int
    min_stock_level: int
    supplier: str
    lead_time_days: int
    
    def check_reorder(self):
        if self.quantity_on_hand < self.min_stock_level:
            self.create_purchase_order()
    
    def reserve_for_job(self, job_id, quantity):
        if self.quantity_on_hand >= quantity:
            Reservation.objects.create(
                job=job_id,
                material=self,
                quantity=quantity
            )
            self.quantity_on_hand -= quantity
            return True
        else:
            return {"error": "Insufficient stock!"}
```

---

## PRIORITY ROADMAP (Next 10 Features)

### **Phase 44-50** (Next 4 Weeks):

**Week 1**: OEE Tracking + SPC Charts  
**Week 2**: Tool Life Manager + Job Scheduling  
**Week 3**: Quality Integration + Energy Monitor  
**Week 4**: Collaboration + Mobile App (MVP)

### **Phase 51-60** (Months 2-3):

- Video/camera integration
- Inventory tracking
- Advanced analytics (AI insights dashboard)
- ERP integration (SAP/Oracle connectors)
- Augmented reality (AR) for training
- Voice commands (Alexa/Google Assistant)
- Blockchain for traceability (optional, if customer demands)

---

## ZÁVER: FROM 50% → 100%

**Current Puzzle**: 45 hotových fáz (kognitívny core, UI, LLM)  
**Missing Pieces**: 10+ industry-standard features (OEE, SPC, scheduling...)  
**Target**: 60 fáz = Plnohodnotný MES/Industry 4.0 platform

**Inšpirácia z industry**: DMG MORI CELOS, Siemens MindSphere, Mazak Smooth, Fanuc MT-LINKi - všetci majú OEE, scheduling, tool management.

**Naša výhoda**: Kognitívna architektúra (dopamine) ktorú oni NEMAJÚ.

*Industry Inspiration Analysis by Dusan Berger, January 2026*
