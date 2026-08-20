# ROADMAP: The "Shadow Council" Architecture Implementation

This roadmap defines the path to realizing the **Parallel Interaction System** brainstormed in `INTERACTION_DESIGN_BRAINSTORM.md`.

## Phase 1: The Nervous System (Foundation)
**Goal:** Create the communication backbone that allows independent systems to talk without blocking each other.

- [ ] **Step 1.1: The Message Bus**
    - Create `cms/message_bus.py`.
    - Implement a Publish/Subscribe pattern (`subscribe("USER_INTENT")`, `publish("ALERT")`).
    - Ensure async compatibility (using `asyncio`).

- [ ] **Step 1.2: The Supervisor Stub**
    - Create `cms/interaction_supervisor.py`.
    - Implement the `Supervisor` class that initializes the Bus and holds references to the sub-systems.

## Phase 2: Assembling the Council (Integration)
**Goal:** Connect the existing distinct modules into the Message Bus.

- [ ] **Step 2.1: The Accountant (Economic Engine)**
    - Wrap `manufacturing_economics.py` with a listener.
    - **Trigger**: Listens for `DRAFT_PLAN`.
    - **Action**: Runs calculation.
    - **Output**: Publishes `ECONOMIC_IMPACT` (e.g., "Cost +15%").

- [ ] **Step 2.2: The Auditor (Monitoring System)**
    - Wrap `cms_core.py` (KnowledgeBase) with a listener.
    - **Trigger**: Listens for `DRAFT_PLAN` and `USER_INTENT`.
    - **Action**: Queries `KnowledgeBase.validate_plan()`.
    - **Output**: Publishes `VALIDATION_RESULT` (Pass/Fail + Criticality).

- [ ] **Step 2.3: The Creator (LLM Interface)**
    - Create a mock/real LLM wrapper.
    - **Trigger**: Listens for `USER_INPUT`.
    - **Action**: Generates text.
    - **Constraint**: Must wait for `VALIDATION_RESULT` before publishing `FINAL_OUTPUT` if Criticality is High.

- [ ] **Step 2.4: The Visualizer (The Eye)**
    - Create `cms/solidworks_tf_bridge.py`.
    - **Function**: Uses TensorFlow to analyze "Solidworks Data" (simulated topology).
    - **Trigger**: Listens for `GEOMETRY_LOAD`.
    - **Output**: Publishes `PART_FEATURES` (e.g., "Sharp Edges Detected", "Complexity Score: 0.8").

## Phase 3: The Learning Loop (Adaptability)
**Goal:** Allow the system to rewrite its own rules based on user overrides.

- [ ] **Step 3.1: Override Detection**
    - Implement logic in Supervisor to detect when User says "Ignore that rule" or "Override".
    
- [ ] **Step 3.2: The Proposal Mechanism**
    - If User overrides -> Auditor publishes `RULE_UPDATE_PROPOSAL`.
    - Supervisor stores this as a "Pending Rule" in `cms_core`.

## Phase 4: The Interface (UI)
**Goal:** Visualize the parallel processing for the user.

- [ ] **Step 4.1: CLI Dashboard**
    - Update `copilot_ui_demo.py` to be event-driven.
    - Display distinct logs for [AUDITOR], [ACCOUNTANT], and [CREATOR].

- [ ] **Step 4.2: Web/GUI (Future)**
    - Plan for a React/Next.js frontend that subscribes to these streams.
