# BRAINSTORM: The "Parallel Mind" Architecture
## CMS, User Input, and the Monitoring Cycle

### 1. The Core Concept: "The Shadow Council"
Instead of a single LLM processing user input, we run **Parallel Streams**:
*   **The Creator (User-LLM)**: Focused on helpfulness, code generation, and answering the engineer.
*   **The Auditor (Monitor-LLM)**: Identifies risks, checks against `cms_core`, and verifies "Ground Truth".
*   **The Accountant (Econ-LLM)**: (From our `manufacturing_economics`) Constantly calculates the cost of the *proposed* solution in real-time.

### 2. The "Interruption" Mechanism
How do these parallel systems interact?
*   **Passive Monitoring**: The Auditor watches the chat silently. It only speaks when a **High Criticality** rule is violated.
*   **Active Validation**: Before the Creator acts (e.g., "Generating G-Code..."), it *must* receive a "Signed Key" from the Auditor.
    *   *User*: "Run this at 5000 RPM."
    *   *Creator*: "Okay..." (Paused)
    *   *Auditor*: "STOP. Rule #42: Titanium limit is 2000 RPM."
    *   *Creator*: "I cannot do that. Titanium limit is 2000 RPM."

### 3. Dynamic Rule Learning (The "Adaptability" Phase)
What if the Engineer is right and the Rule is outdated?
*   **The Override Loop**:
    *   *User*: "Override Rule #42. We are using a coated carbide tool."
    *   *Auditor*: "New condition detected. Propose update to CMS?"
    *   *CMS Core*: Records `Rule #42-B`: "Titanium limit 5000 RPM *IF* Tool == Coated Carbide."
    *   **Result**: The system *learns* and adapts its constraints.

### 4. Structure of Interaction
We need a `MessageBus` to handle these signals.
```python
class MessageBus:
    def publish(channel, message): ...
    def subscribe(channel, callback): ...

# Flow
User -> Input -> MessageBus("USER_INTENT")
Creator -> Sub("USER_INTENT") -> Generates Draft -> MessageBus("DRAFT_PLAN")
Auditor -> Sub("DRAFT_PLAN") -> Validates -> MessageBus("VALIDATION_RESULT")
Creator -> Sub("VALIDATION_RESULT") -> Outputs Final Response to User
```

### 5. Integration with CNC Bridge
The `cnc_optimization_engine` becomes a *client* of this CMS.
*   It asks the CMS: "What are the max feeds for this material?"
*   It publishes its results: "I generated a RUSH plan."
*   The Auditor checks: "Does RUSH mode violate the Safety Protocol?"

### 6. User Experience (UI)
The UI should show these "Parallel Minds" working.
*   **Main Chat**: The conversation.
*   **Sidebar**:
    *   🟢 **Auditor**: "Checks Passed." / 🔴 "Safety Alert!"
    *   💰 **Accountant**: "Cost: $150.00 (+5%)" (Updates live as you type)
    *   📚 **CMS**: "Referenced Rule: Titanium Specs v1.2"
