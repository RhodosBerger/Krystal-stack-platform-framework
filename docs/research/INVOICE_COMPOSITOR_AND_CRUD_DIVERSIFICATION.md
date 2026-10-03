# Krystal-Stack // Deep Research: Invoice Compositor & Application Layer Diversification

**Date:** 2026-10-03  
**Subject:** Versatile Accounting Architecture, OCR Digitalization, and CRUD Layer Separation  
**Author:** Krystal-Stack Architecture Team  

---

## 1. Paradigm Shift: From Visual Synthesis to Business Logic
We have established that the Krystal-Stack's deep architecture (Cyclic Organism, Thermodynamic Entropic fields, OpenVINO edge detection) is universally applicable. We are now pivoting from pure graphical rendering to **Automated Business Operations**.

The **Invoice Compositor** is the evolution of the Neural Photobank Compositor. Instead of blending textures and geometric landscapes, it blends financial data (Text, QR codes, tables) into formalized document templates (PDFs/Images) or extracts data from them.

## 2. Machine Learning Patterns: OCR & Text Extraction
To digitize physical evidence (scanned invoices), the system requires Optical Character Recognition (OCR). 
*   **Previous ML Pattern:** Edge detection via OpenVINO / OpenCV (Canny).
*   **New ML Pattern:** Region of Interest (ROI) Text Extraction via Tesseract or OpenVINO-based Text Detection Models (e.g., PixelLink).
*   **Execution:** The system will locate text bounding boxes, extract strings, identify key fields (Total Amount, VAT, IBAN, Date), and convert them into structured JSON.

## 3. Application Layer Diversification (CRUD & Database)
Currently, our components act dynamically and statelessly (in memory UUID meshes). To build robust accounting software, we must diversify the application layers:

1.  **Cognitive / Ingestion Layer:** The `MirrorConverter` and `InvoiceCompositor` that handle OpenCV scanning, QR decoding, and OCR text extraction.
2.  **Persistence / Database Layer (CRUD):** A relational or document-based database (SQLite for MVP, later Spanner/PostgreSQL) that stores digitized invoice records. This layer handles **C**reate, **R**ead, **U**pdate, and **D**elete operations securely.
3.  **Presentation / UI Layer (Oxygen Builder):** The frontend assembled dynamically via MCP tools on WordPress/Oxygen, pulling data from the Database Layer via REST APIs.

## 4. Versatility & Generative Capabilities
The Invoice Compositor is bi-directional:
*   **Ingestion:** Reads scanned images $\rightarrow$ OCR $\rightarrow$ JSON $\rightarrow$ Database.
*   **Generation:** Reads JSON $\rightarrow$ Database $\rightarrow$ Injects text into blank invoice templates $\rightarrow$ Generates an outgoing Invoice Image/PDF.

By separating the Database (CRUD) from the UI (Oxygen) and the Engine (OpenVINO), we achieve a decoupled, enterprise-grade architecture capable of scaling massively.

---
*End of Research Phase. Proceeding to implement `invoice_compositor.py` with SQLite CRUD and OCR ML patterns.*
