# Krystal-Stack // Deep Research: WordPress & Oxygen Builder MCP Integration

**Date:** 2026-10-03  
**Subject:** Wiring Krystal-Stack MRP Backend to WordPress/Oxygen via REST API and MCP  
**Author:** Krystal-Stack Architecture Team  

---

## 1. Architectural Overview (The Bridge)
To surface the Krystal-Stack accounting system (MRP, OCR, CRUD, RBAC) into a production-ready UI, we use WordPress powered by Oxygen Builder. The integration avoids hardcoding UI in Python and instead leverages the **nabytok47 MCP Server** to programmatically build the interface.

**The Pipeline:**
`Physical Invoice` $\rightarrow$ `Krystal-Stack OpenCV/OCR` $\rightarrow$ `SQLite CRUD` $\rightarrow$ `Krystal REST API` $\rightarrow$ `MCP Agent Orchestrator` $\rightarrow$ `WordPress Custom Post Types (CPT) + Oxygen Templates`

## 2. Step-by-Step Integration Plan

### Phase A: Exposing Krystal-Stack REST API
The local `server.py` will be extended with dedicated MRP endpoints:
*   `GET /api/mrp/invoices` - Returns JSON of all scanned/processed invoices.
*   `POST /api/mrp/invoice` - Webhook receiver for new scans.
*   `GET /api/mrp/audit` - RBAC protected endpoint for auditors.

### Phase B: Data Mapping to WordPress (Custom Post Types)
We map the Krystal backend data to WordPress structures:
1.  **Invoices as Posts:** Each scanned invoice becomes a WordPress post of a custom type `accounting_invoice`.
2.  **Dynamic Metadata (ACF / Meta):** The extracted OCR data (Supplier, Amount, IBAN, Status) is saved as post meta-data.
3.  **MCP Tool Usage:** The AI agent uses `oxygen-create-post` to push new invoices from the SQLite DB directly into the WordPress database.

### Phase C: Programmatic UI via Oxygen Builder MCP Extensions
Instead of manually clicking through the WordPress admin, we use the `nabytok47` Oxygen MCP tools to build the UI programmatically:

1.  **Template Generation:**
    *   Call `oxygen-create-template` to scaffold an "Invoice Detail" template applied to the `accounting_invoice` post type.
2.  **Dynamic Data Binding:**
    *   Call `oxygen-get-dynamic-fields` to map the WordPress post meta (Supplier, Amount) into Oxygen text elements.
3.  **Role-Based Access Control (RBAC) in UI:**
    *   Call `oxygen-set-element-conditions`. We can dynamically hide the "Storno Invoice" button if the logged-in WordPress user does not have the "ADMIN" role (mirroring our Python `RoleManager`).
4.  **Aesthetic Styling (Krystal-Cyan):**
    *   Call `oxygen-insert-css-variables` to inject our specific CSS variables (`--accent-cyan: #66fcf1; --bg-color: #0b0c10`) into Oxygen's Global Settings, instantly unifying the brand identity across the accounting portal.

## 3. The "Mirror" Concept in Practice
The `MirrorConverter` we built previously acts as the exact payload formatter. When an invoice is scanned:
1. OpenCV reads the data.
2. `MirrorConverter.map_to_oxygen_builder_mcp()` structures the JSON.
3. The Agent executes `call_mcp_tool(ToolName="oxygen-create-post", Arguments={...})`.
4. The accountant immediately sees the new invoice rendered perfectly in the Oxygen Builder frontend.

## 4. Next Actions
To implement this live, we would:
1. Ensure the `nabytok47` MCP server is connected to the target WordPress installation.
2. Write a continuous polling loop in `krystal_janet/compositor_terminal.janet` that watches the SQLite DB and triggers the MCP calls automatically when new records appear.
