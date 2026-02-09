# FANUC RISE: DevOps & Operations Handbook

> **Role**: System Reliability Engineer (SRE) / CNC Operator
> **Objective**: Deploy, Monitor, and Maintain the Neuro-Safe CNC System.

---

## 1. Installation & Deployment

### Prerequisites
*   Python 3.8+
*   Fanuc FOCAS Libraries (for real hardware) or `fanuc_rise_bridge.py` (simulated).

### Setup
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/Dev-contitional/advanced_cnc_copilot.git
    cd advanced_cnc_copilot
    ```
2.  **Initialize the Environment**:
    ```bash
    # Create the Artifact Directory (Brain)
    mkdir -p brain/memory
    
    # Verify Dependencies
    python3 -c "import dataclasses; import asyncio; print('Ready')"
    ```

---

## 2. Configuration Management

### The "Seed" Database
*   **File**: `cms/elements_db.py`
*   **Purpose**: Defines the "Axioms" of the machine.
*   **Action**: Edit this file to add new Materials (e.g., `Inconel718`) or adjust `mantinel_limit` (Safety Power Limit).

### The Policy File
*   **File**: `dopamine_policy.json` (Auto-Generated)
*   **Purpose**: Stores the "Personality" of the machine (Preferred Strategies).
*   **DevOps Rule**: Do *not* manually edit this file unless performing a "Lobotomy" (Reset). Let `nightly_training.py` manage it.

---

## 3. Monitoring & Observability

### The Command Center
*   **Command**: `python cms/fanuc_dashboard.py`
*   **Usage**: Real-time visualization of the "Mind" and "Machine".
*   **Key Metrics**:
    *   **Dopamine**: < 20 means "Depression" (Machine is too cautious).
    *   **Cortisol**: > 80 means "Panic" (Stop immediately).
    *   **Scanner**: If the `O` point leaves the `.` zone, the plan is unsafe.

### The Quadratic Scanner (Headless)
*   **Command**: `python cms/quadratic_scanner.py`
*   **Usage**: Quick validation of a specific operating point against the Mantinels.

---

## 4. Maintenance & Learning Service

### The "Sleep Cycle" (Nightly Training)
*   **Frequency**: Run once every 24 hours (Cron Job).
*   **Command**: `python cms/nightly_training.py`
*   **Function**:
    1.  Reads `hippocampus` history.
    2.  Calculates "Net Score" (Reward - Stress).
    3.  Updates `preferred_strategy` for each Material.
    
    *Example Cron:*
    ```cron
    0 2 * * * cd /opt/fanuc_rise && python cms/nightly_training.py >> /var/log/fanuc_sleep.log
    ```

### Data Aggregation Reports
*   **Frequency**: Weekly.
*   **Command**: `python cms/aggregation_demo.py` (Script wrapper around `hippocampus_aggregator.py`).
*   **Purpose**: Identify "Trouble Spots" (e.g., "Why does Titanium always cause Cortisol spikes?").

---

## 5. Troubleshooting (Incident Response)

### Incident: "The Machine is Stuck in 'Cautious Mode'"
*   **Symptom**: `Dopamine` stays low (< 20). Feed rate is capped at 50%.
*   **Cause**: Recent trauma (Tool Breakage) stored in Hippocampus.
*   **Fix**:
    1.  Run `nightly_training.py` to see if it self-corrects based on newer successful runs.
    2.  **Hard Reset**: Delete `dopamine_policy.json` to revert to `elements_db.py` defaults.

### Incident: "Cortisol False Positives"
*   **Symptom**: Machine stops for "Ghost Vibration".
*   **Fix**: Adjust `cortisol_threshold` in `cms/elements_db.py` for that specific material.
