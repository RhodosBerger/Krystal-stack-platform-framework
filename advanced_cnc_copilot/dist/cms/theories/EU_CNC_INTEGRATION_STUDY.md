# EU CNC INTEGRATION STUDY: Hardware Landscape
## Strategic Research for "Fanuc Rise" Expansion

> **Objective**: To define the integration criteria for Fanuc (Primary) and major European alternatives (Siemens, Heidenhain).
> **Market Context**: Europe heavily utilizes high-end controllers for precision manufacturing (Automotive, Aerospace).

---

## 1. Primary Target: FANUC (The Global Standard)
*   **Market Share**: Extremely high globally; standard in general machining.
*   **Connectivity Protocol**: **FOCAS 1 / FOCAS 2** (Fanuc Open CNC API Specifications).
    *   **Physical Layer**: HSSB (High-Speed Serial Bus) (Optic fiber, PCI card) OR Ethernet (Standard RJ45).
    *   **Library**: `Fwlib32.dll` (Windows standard).
*   **Criteria for "Rise" Integration**:
    *   **Sampling Rate**: Ethernet (~10-50ms latency), HSSB (<1ms). *Rise requires HSSB for real-time vibration control.*
    *   **Data Availability**: load, speed, macro variables are standard.
    *   **Write Access**: Requires "API Password" or Unlock on some machines.

---

## 2. Alternative Scenario A: SIEMENS SINUMERIK (The German Standard)
*   **Target Machines**: DMG Mori, Grob, Hermle (High-end 5-Axis).
*   **Connectivity Protocol**: **OPC-UA** (Open Platform Communications - Unified Architecture).
    *   **Status**: Built-in to modern controllers (840D sl, One).
    *   **Advantage**: No proprietary DLLs needed. Standard TCP/IP.
*   **Criteria**:
    *   **Authentication**: Certificate-based security.
    *   **Data Structure**: Object-oriented node tree (unlike Fanuc's flat memory map).
    *   **Latency**: ~100ms via OPC-UA (Slower than FOCAS). *Requires "Local Reflex" logic for safety.*

---

## 3. Alternative Scenario B: HEIDENHAIN (The Mold & Die Standard)
*   **Target Machines**: Mikron, Hermle, Alzmetall.
*   **Connectivity Protocol**: **Heidenhain DNC** (RemoTools SDK) or **LookAhead Interface**.
*   **Criteria**:
    *   **LSV-2 Protocol**: Low-level serial/TCP protocol.
    *   **Python Integration**: Heidenhain offers a Python packet for their Option 18 (DNC).
    *   **Philosophy**: Very strict state management. Harder to "hack" than Fanuc.

---

## 4. Hardware Abstraction Layer (HAL) Strategy
To support these diverse systems, the `SensoryCortex` must evolve.

### The "GenericController" Interface
```python
class GenericController(ABC):
    @abstractmethod
    def read_load(self) -> float: pass
    
    @abstractmethod
    def override_feed(self, percent: int): pass
    
    @abstractmethod
    def get_latency_ms(self) -> int: pass
```

### Driver Adapters
1.  **`FanucAdapter`**: Wraps `ctypes` calls to `fwlib32.dll`. Ultra-low latency.
2.  **`SiemensAdapter`**: Wraps `asyncua` (Python OPC-UA client). High latency options.
3.  **`HeidenhainAdapter`**: Wraps TCP socket commands (LSV-2).

## 5. Summary of Expansion Risks
*   **Siemens**: Easy to read (OPC-UA), hard to write real-time control (Safety Integrated locks).
*   **Heidenhain**: Expensive licensing (Option 18 often costs €1000+ per machine).
*   **Fanuc**: Oldest API, but fast and reliable. "It just works."

**Recommendation**: Stick to Fanuc for the "Neuro-Safety" prototype (HSSB speed is required). Use Siemens/OPC-UA only for *monitoring*, not for control, until local edge hardware is verified.
