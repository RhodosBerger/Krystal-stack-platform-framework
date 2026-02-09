# FUTURE UI COMPONENTS - VISUAL SPECIFICATIONS
## Detailné Popisy Vizuálnych Častí Pre Features Z Fázy 44-60

---

## 1. OEE DASHBOARD PANEL

### A. Main OEE Widget (Hero Component)

```
┌─────────────────────────────────────────────────────────┐
│  OEE OVERALL EQUIPMENT EFFECTIVENESS                    │
│  ═══════════════════════════════════════════════════   │
│                                                         │
│           ┌──────────────────────┐                     │
│           │                      │                     │
│           │        80.6%         │   🏆 GOOD          │
│           │      ████████░░      │                     │
│           │                      │   Target: 85%       │
│           └──────────────────────┘   World Class       │
│                                                         │
│  ┌─────────────────┬─────────────────┬─────────────┐  │
│  │  AVAILABILITY   │  PERFORMANCE    │  QUALITY    │  │
│  │                 │                 │             │  │
│  │      93.8%      │      90.0%      │    95.6%    │  │
│  │  [████████░░]   │  [████████░░]   │[█████████░] │  │
│  │                 │                 │             │  │
│  │  Operating:     │  Actual: 45     │  Good: 43   │  │
│  │  450/480 min    │  Target: 50     │  Total: 45  │  │
│  └─────────────────┴─────────────────┴─────────────┘  │
│                                                         │
│  📊 TREND (Last 7 Days)                                │
│  [Line chart showing OEE % over time]                  │
│                                                         │
│  ⚠️ LOSS BREAKDOWN:                                    │
│  • Setup Loss: 20 min (67% of downtime)                │
│  • Break: 10 min (33% of downtime)                     │
│  • Performance Loss: 5 parts (10% below target)        │
│  • Quality Loss: 2 rejects (4.4% defect rate)          │
│                                                         │
│  [📥 Export Report] [⚙️ Settings] [🔔 Set Alerts]     │
└─────────────────────────────────────────────────────────┘
```

**Visual Properties**:
- **Size**: Full-width card, 600px height
- **Color Scheme**:
  - OEE 85%+: Green (#10b981)
  - OEE 60-84%: Amber (#f59e0b)
  - OEE <60%: Red (#ef4444)
- **Animation**: Number count-up when page loads
- **Interactive**: Click each sub-metric for drill-down

**User Interaction**:
1. **Hover** on Availability → Tooltip: "Operating time / Planned time"
2. **Click** on Performance bar → Modal with detailed breakdown
3. **Drag** time range selector in trend chart

---

## 2. SPC CHART COMPONENT (Statistical Process Control)

### A. X̄-R Chart Widget

```
┌──────────────────────────────────────────────────────────┐
│  SPC CHART: Bore Diameter (Ø25.00mm ±0.05mm)      [×]  │
│  ════════════════════════════════════════════════════   │
│                                                          │
│  X̄ Chart (Average):                                     │
│  25.10 |                                × ❗ OUT!       │
│  25.05 |─────────────────UCL────────────────────────   │
│  25.02 |        ×     ×                                 │
│  25.00 |  ×  ×     ×        ×  ×     ×                 │
│  24.98 |                          ×                      │
│  24.95 |─────────────────LCL────────────────────────   │
│  24.90 |                                                 │
│        └──────────────────────────────────────────────  │
│         1   5   10  15  20  25  30 (sample #)          │
│                                                          │
│  R Chart (Range):                                        │
│  0.10  |                                                 │
│  0.05  |  ×  ×  ×  ×  ×  ×  ×  ×  ×  ×                 │
│  0.00  └──────────────────────────────────────────────  │
│                                                          │
│  📊 STATUS: ⚠️ WARNING                                  │
│  • Rule 2 Violation: 9 consecutive points above center │
│  • Process drift detected → Recalibration needed!      │
│                                                          │
│  ACTIONS:                                                │
│  [🔧 Recalibrate] [📸 Take Photo] [💬 Add Comment]    │
│  [📤 Export CSV] [🔔 Set Alert Threshold]              │
└──────────────────────────────────────────────────────────┘
```

**Visual Properties**:
- **Chart Type**: SVG line chart (responsive)
- **Point Colors**:
  - In control: Blue (#3b82f6)
  - Warning (2σ): Amber (#f59e0b)
  - Out of control (3σ): Red (#ef4444)
- **Grid**: Dashed lines for UCL/LCL/Center
- **Hover**: Show exact value + timestamp

**Interactive Elements**:
1. **Click point** → Show measurement details (inspector, tool, temperature)
2. **Zoom**: Pinch gesture or scroll to zoom X-axis
3. **Add annotation**: Right-click point → Add note

---

## 3. TOOL LIFE MANAGER

### A. Tool Inventory Grid

```
┌──────────────────────────────────────────────────────────────┐
│  TOOL INVENTORY & LIFE TRACKING                     [+ Add] │
│  ══════════════════════════════════════════════════════════ │
│                                                              │
│  [Search tools...] [Filter: All ▼] [Sort: Life % ▼]        │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ T01  📏 Endmill Ø8mm × 60mm        HSS-Co   TiN       │ │
│  │      ────────────────────────────────────────────────  │ │
│  │      Life: [████████████░░░░░░░░] 82% (18/100 min)   │ │
│  │      Status: 🟢 IN USE  |  Machine: CNC_VMC_01       │ │
│  │      Cost: €45  |  Vendor: Sandvik  |  Stock: 2 pcs │ │
│  │      [📊 History] [🔄 Replace] [🛒 Reorder]          │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ T03  🔩 Drill Ø25mm × 120mm        Carbide  AlCrN    │ │
│  │      ────────────────────────────────────────────────  │ │
│  │      Life: [████░░░░░░░░░░░░░░░░] 28% (32/45 min)   │ │
│  │      Status: ⚠️ REPLACE SOON  |  Machine: IDLE       │ │
│  │      Cost: €120  |  Lead time: 3 days  |  Stock: 0  │ │
│  │      💡 AI Suggestion: "Order now to avoid stockout" │ │
│  │      [⚡ Quick Order] [🔍 Find Alternative]           │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ T05  🛠️ Boring Bar Ø40mm          Carbide Insert     │ │
│  │      ────────────────────────────────────────────────  │ │
│  │      Life: [█░░░░░░░░░░░░░░░░░░░] 8% (46/50 min) ❗  │ │
│  │      Status: 🔴 CRITICAL - REPLACE NOW!              │ │
│  │      Last used: 2 hours ago on JOB_B042              │ │
│  │      [🚨 Emergency Order] [🔄 Use Backup: T06]       │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                              │
│  Total Tools: 24  |  Active: 8  |  Needs Replace: 3        │
│  Total Value: €5,420  |  Monthly consumption: €380          │
└──────────────────────────────────────────────────────────────┘
```

**Visual Properties**:
- **Card Layout**: Stacked, alternating bg colors
- **Life Bar Colors**:
  - 50-100%: Green gradient
  - 20-49%: Amber gradient
  - 0-19%: Red gradient + pulsing animation
- **Icons**: Emoji or FontAwesome for tool types

**Interactive Features**:
1. **Drag & drop** to reorder priority
2. **Quick filter** buttons (In Use, Need Replace, All)
3. **Bulk actions**: Select multiple → Order all

---

## 4. PRODUCTION SCHEDULER (Gantt Chart)

### A. Machine Loading Timeline

```
┌────────────────────────────────────────────────────────────────┐
│  PRODUCTION SCHEDULE - GANTT CHART           [Today ▼] [Week] │
│  ════════════════════════════════════════════════════════════ │
│                                                                │
│  Machine      08:00   10:00   12:00   14:00   16:00   18:00  │
│  ──────────────────────────────────────────────────────────── │
│  CNC_01  [████JOB_A-P1████][██JOB_C-P3██][    idle    ]      │
│          ↑ Setup 15min      ↑ Running     ↑ Available        │
│                                                                │
│  CNC_02  [██████JOB_B-P2██████][████JOB_D-P1████][idle]      │
│          ↑ 98% utilization (bottleneck!)                      │
│                                                                │
│  CNC_03  [░░setup░░][████████JOB_E-P5████████][planned]      │
│          ↑ Tool change                                         │
│  ──────────────────────────────────────────────────────────── │
│                                                                │
│  Legend:                                                       │
│  ████ Production (P1=Urgent, P2=High, P3=Normal, P5=Low)     │
│  ░░░░ Setup / Changeover                                      │
│  [  ] Idle / Available                                        │
│  ┆┆┆┆ Planned (not started)                                  │
│                                                                │
│  ⚠️ ALERTS:                                                   │
│  • CNC_02 overloaded (98% util) → Move JOB_D to CNC_01?      │
│  • JOB_E deadline: 18:00 (risk of delay)                     │
│                                                                │
│  ACTIONS:                                                      │
│  [🔄 Re-optimize] [➕ Add Job] [⚡ Rush Priority]            │
│  [📊 Capacity Report] [📅 Weekly View]                       │
└────────────────────────────────────────────────────────────────┘
```

**Visual Properties**:
- **Color Coding**: Priority-based (P1=Red, P2=Orange, P3=Yellow, P5=Green)
- **Drag-and-drop**: Jobs can be moved between machines
- **Zoom**: Horizontal scroll + zoom controls
- **Hover**: Show job details (parts, est. time, material)

**Interactive Features**:
1. **Click job block** → Edit modal (change priority, split job)
2. **Drag job** → Validates if machine compatible
3. **Right-click** → Context menu (Clone, Cancel, Mark Complete)

---

## 5. QUALITY INSPECTION INTERFACE

### A. First-Article Inspection Form

```
┌──────────────────────────────────────────────────────────┐
│  FIRST ARTICLE INSPECTION                           [×] │
│  ══════════════════════════════════════════════════════ │
│                                                          │
│  Part: BRACKET_ALU_042        Job: JOB_A                │
│  Drawing: DWG-2024-1234       Rev: C                    │
│  Inspector: Dusan B.          Date: 2026-01-23          │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │ DIMENSION MEASUREMENTS                             │ │
│  ├────────────────────────────────────────────────────┤ │
│  │ #   Feature      Nominal  Tolerance  Measured  ✓  │ │
│  ├────────────────────────────────────────────────────┤ │
│  │ 1   Bore Ø       25.00mm   ±0.05    25.02mm   ✅  │ │
│  │ 2   Length       100.00mm  ±0.10    100.08mm  ✅  │ │
│  │ 3   Thickness    10.00mm   ±0.05    10.12mm   ❌  │ │
│  │     └─ Deviation: +0.12mm (OUT OF SPEC!)          │ │
│  │     └─ Root Cause: [AI analyzing...] 🤖           │ │
│  │ 4   Hole spacing 50.00mm   ±0.10    49.98mm   ✅  │ │
│  │ 5   Surface Ra   1.6μm     max       1.2μm    ✅  │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
│  📸 PHOTOS:                                             │
│  [▢ Photo 1] [▢ Photo 2] [+ Add Photo]                 │
│                                                          │
│  RESULT: ⚠️ CONDITIONAL PASS (1 non-conformance)        │
│                                                          │
│  NON-CONFORMANCE #1: Thickness +0.12mm                  │
│  🤖 AI Root Cause Analysis:                             │
│  "Thermal expansion detected. Spindle temp was 58°C     │
│   during machining (8°C above baseline). Recommend:     │
│   - Add 10min cooldown before finish pass               │
│   - Enable coolant earlier in cycle"                    │
│                                                          │
│  CORRECTIVE ACTION:                                      │
│  [✏️ Enter Action] [📧 Notify Engineer] [🔄 Re-make]   │
│  [✅ Accept As-Is] [❌ Reject Part]                     │
└──────────────────────────────────────────────────────────┘
```

**Visual Properties**:
- **Table**: Striped rows, highlight out-of-spec in red
- **Status Icons**: ✅ Green checkmark, ❌ Red X
- **AI Section**: Gradient background to differentiate
- **Photos**: Thumbnail grid, click to enlarge

**Interactive Features**:
1. **Auto-calculate** deviation when measured value entered
2. **AI analysis** triggers on out-of-spec detection
3. **Photo upload**: Drag-and-drop or camera capture (mobile)

---

## 6. ENERGY MONITORING DASHBOARD

### A. Real-Time Power Widget

```
┌──────────────────────────────────────────────────────────┐
│  ⚡ ENERGY MONITORING                                    │
│  ══════════════════════════════════════════════════════ │
│                                                          │
│  ┌────────────────────┐  ┌──────────────────────────┐  │
│  │  CURRENT POWER     │  │  TODAY'S CONSUMPTION     │  │
│  │                    │  │                          │  │
│  │      12.5 kW       │  │       85.2 kWh          │  │
│  │   [████████░░]     │  │    Cost: €10.22         │  │
│  │   Max: 15 kW       │  │    CO₂: 42.6 kg 🌱      │  │
│  └────────────────────┘  └──────────────────────────┘  │
│                                                          │
│  📊 POWER TREND (Last Hour):                            │
│  15 kW |     ╱╲                                         │
│  12 kW |    ╱  ╲    ╱╲                                 │
│  10 kW |   ╱    ╲  ╱  ╲                                │
│   5 kW |  ╱      ╲╱    ╲╱                              │
│   0 kW └────────────────────────────────────────────   │
│        17:00      17:30      18:00                      │
│        ↑ Spike: Setup    ↑ Normal machining            │
│                                                          │
│  💰 COST BREAKDOWN:                                     │
│  • Per Part: €0.018 (Target: <€0.02) ✅                │
│  • Peak rate: €0.15/kWh (14:00-20:00)                  │
│  • Off-peak: €0.09/kWh (22:00-06:00)                   │
│                                                          │
│  💡 RECOMMENDATIONS:                                     │
│  "Schedule batch jobs after 22:00 to save 40% on       │
│   energy costs. Estimated savings: €45/month"          │
│                                                          │
│  [📅 Schedule Optimizer] [🌍 Sustainability Report]    │
└──────────────────────────────────────────────────────────┘
```

**Visual Properties**:
- **Gradient meter**: Green (low) → Red (high)
- **Live update**: Numbers refresh every 1s
- **Sparkline**: Mini chart showing trend
- **CO₂ icon**: Leaf emoji with green color

---

## 7. COLLABORATION PANEL (Chat & Notes)

### A. Machine Chat Interface

```
┌──────────────────────────────────────────────────────────┐
│  💬 MACHINE CHAT: CNC_VMC_01                    [Shift] │
│  ══════════════════════════════════════════════════════ │
│                                                          │
│  👤 Dusan B. (Day Shift)               14:32            │
│  │ "Tool T03 acting weird, vibration higher than normal"│
│  │ [📸 Photo attached]                                  │
│                                                          │
│  👤 Martin K. (Engineer)                14:35           │
│  │ "Checked your photo. Tool looks OK. What RPM?"      │
│                                                          │
│  👤 Dusan B.                            14:37           │
│  │ "3500 RPM, same as always"                          │
│                                                          │
│  🤖 AI Copilot                          14:37           │
│  │ "Analysis: Spindle bearing temp +8°C above baseline.│
│  │  Recommend: Schedule spindle inspection next week.  │
│  │  Short-term: Reduce RPM to 3200."                   │
│  │  [Apply Suggestion ✅]                               │
│                                                          │
│  👤 Martin K.                           14:40           │
│  │ "Good catch, AI! @Dusan please reduce RPM and I'll  │
│  │  schedule that bearing check. Thanks! 👍"           │
│                                                          │
│  [Type message...                          ] [📎] [😊] │
└──────────────────────────────────────────────────────────┘
```

**Visual Properties**:
- **Chat bubbles**: User messages (blue), AI (gradient purple/blue)
- **Timestamps**: Gray, right-aligned
- **Mentions**: @username highlighted
- **Attachments**: Thumbnail preview

**Interactive Features**:
1. **@mention** autocomplete
2. **Emoji picker** 😊
3. **Voice message** (hold to record)
4. **AI suggestion** buttons (Apply, Dismiss)

---

## 8. MOBILE APP SCREENS

### A. Machine Status Card (Mobile)

```
┌─────────────────────────┐
│  🔧 CNC_VMC_01         │
│  ─────────────────────  │
│                         │
│  Status: 🟢 RUNNING    │
│                         │
│  Job: BRACKET_ALU_V3    │
│  [████████████░░░░] 65% │
│  32 / 50 parts          │
│                         │
│  OEE: 80.6% (Good)     │
│  Load: 67%              │
│  Vibration: 0.03g      │
│                         │
│  ┌────────┬────────┐   │
│  │ PAUSE  │ NOTIFY │   │
│  └────────┴────────┘   │
│                         │
│  Last update: 2s ago    │
│                         │
└─────────────────────────┘
```

**Mobile-Specific Features**:
- **Swipe right** → Pause machine (with confirmation)
- **Swipe left** → View detailed metrics
- **Pull down** → Refresh
- **Shake device** → Emergency stop (if enabled)

---

## VISUAL DESIGN SYSTEM

### Color Palette:
```css
:root {
  --primary: #38bdf8;    /* Sky blue */
  --success: #10b981;    /* Green */
  --warning: #f59e0b;    /* Amber */
  --danger: #ef4444;     /* Red */
  --bg-dark: #0f1419;    /* Almost black */
  --bg-panel: #1a1f2e;   /* Dark blue-gray */
  --text-primary: #e4e4e7;
  --text-secondary: #a1a1aa;
}
```

### Typography:
- **Headings**: Inter, 600 weight
- **Body**: Inter, 400 weight
- **Monospace**: JetBrains Mono (for numbers)

### Spacing:
- **Base unit**: 8px
- **Card padding**: 24px
- **Component gap**: 16px

---

**ZÁVER**: Každý component má jasný purpose, interaction pattern, a visual identity. Ready for implementation! 🎨

*UI Specs by Dusan Berger, January 2026*
