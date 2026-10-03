// Krystal-Stack // Studio ERP & Agenda Logic

document.addEventListener("DOMContentLoaded", () => {
    console.log("Krystal Studio ERP Initialized.");
    loadRegistryData();
});

function loadRegistryData() {
    // In a real scenario, this fetches from /api/mrp/invoices
    // We will simulate the data returned from our Python SQLite Backend
    
    const mockData = [
        { id: "fefe22a8", supplier: "KRYSTAL-STACK CLOUD SERVICES", amount: "1450.50", date: "2026-10-03 12:45", status: "DIGITIZED" },
        { id: "8492abcd", supplier: "Novak s.r.o. (WooCommerce)", amount: "125.50", date: "2026-10-03 10:15", status: "PAID" },
        { id: "1a2b3c4d", supplier: "Amazon Web Services", amount: "340.00", date: "2026-10-02 09:00", status: "PENDING" },
    ];

    const tbody = document.getElementById("invoice-table-body");
    tbody.innerHTML = "";

    mockData.forEach(inv => {
        let statusClass = inv.status === "PAID" ? "status-paid" : "status-pending";
        if(inv.status === "DIGITIZED") statusClass = "status-paid";

        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td style="font-family: monospace; color: var(--accent-cyan);">${inv.id}</td>
            <td style="font-weight: 500;">${inv.supplier}</td>
            <td style="color: var(--gold); font-weight: 600;">${inv.amount} €</td>
            <td style="color: #888;">${inv.date}</td>
            <td><span class="status-badge ${statusClass}">${inv.status}</span></td>
            <td>
                <button class="btn" style="padding: 0.3rem 0.6rem; font-size: 0.75rem;" onclick="viewDetail('${inv.id}')">Detail</button>
            </td>
        `;
        tbody.appendChild(tr);
    });
}

function triggerOCRScan() {
    alert("[ML Scanner] Initiating OpenCV Region Detection and OCR Extraction. Please wait...");
    // Simulating a backend call to /api/mrp/scan
    setTimeout(() => {
        const tbody = document.getElementById("invoice-table-body");
        const newTr = document.createElement("tr");
        newTr.innerHTML = `
            <td style="font-family: monospace; color: var(--accent-cyan);">99xx88yy</td>
            <td style="font-weight: 500;">Alza.sk a.s.</td>
            <td style="color: var(--gold); font-weight: 600;">890.00 €</td>
            <td style="color: #888;">Práve teraz</td>
            <td><span class="status-badge status-paid">DIGITIZED</span></td>
            <td><button class="btn" style="padding: 0.3rem 0.6rem; font-size: 0.75rem;">Detail</button></td>
        `;
        tbody.insertBefore(newTr, tbody.firstChild);
        alert("OCR Scan Successful. New record inserted into the Evidence.");
    }, 1500);
}

function syncWooCommerce() {
    alert("[WooCommerce Bridge] Synchronizing orders from REST API...");
    setTimeout(() => {
        alert("Synchronization complete. Ledger updated.");
        loadRegistryData(); // Reload
    }, 1000);
}

function exportToOxygen() {
    alert("[MCP Bridge] Triggering 'oxygen-create-post' to generate Headless UI for these records.");
}

function viewDetail(id) {
    alert("Viewing detailed OCR JSON and Ledger Posting for ID: " + id);
}
