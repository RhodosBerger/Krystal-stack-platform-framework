import sqlite3
import os
import time
import uuid
import json
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

class InvoiceDatabaseCRUD:
    """
    Diversified Persistence Layer.
    Handles CRUD operations for digitized invoices using SQLite.
    """
    def __init__(self, db_path="invoices_data.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS invoices (
                    uuid TEXT PRIMARY KEY,
                    supplier TEXT,
                    amount REAL,
                    iban TEXT,
                    scan_timestamp REAL,
                    status TEXT,
                    raw_ocr_text TEXT
                )
            ''')
            conn.commit()

    def create(self, invoice_data: dict) -> str:
        inv_uuid = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO invoices (uuid, supplier, amount, iban, scan_timestamp, status, raw_ocr_text)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                inv_uuid,
                invoice_data.get('supplier', 'Unknown'),
                invoice_data.get('amount', 0.0),
                invoice_data.get('iban', ''),
                time.time(),
                invoice_data.get('status', 'DIGITIZED'),
                invoice_data.get('raw_ocr_text', '')
            ))
            conn.commit()
        print(f"[CRUD] Created new invoice record: {inv_uuid}")
        return inv_uuid

    def read(self, inv_uuid: str) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM invoices WHERE uuid = ?', (inv_uuid,))
            row = cursor.fetchone()
            if row:
                return {
                    "uuid": row[0], "supplier": row[1], "amount": row[2],
                    "iban": row[3], "scan_timestamp": row[4], "status": row[5], "raw_ocr_text": row[6]
                }
        return None

    def update_status(self, inv_uuid: str, new_status: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE invoices SET status = ? WHERE uuid = ?', (new_status, inv_uuid))
            conn.commit()
        print(f"[CRUD] Updated invoice {inv_uuid} to {new_status}")

    def get_all(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM invoices ORDER BY scan_timestamp DESC')
            return cursor.fetchall()


class MLInvoiceScanner:
    """
    Cognitive Ingestion Layer.
    Uses Machine Learning patterns (OpenCV/OpenVINO concepts) to digitize scanned images via OCR.
    """
    def __init__(self):
        print("[ML Scanner] Initialized Invoice Text & OCR Region Detection.")
        
    def digitize_invoice(self, image_path: str) -> dict:
        """Simulates OCR extraction of key text fields from a physical scan."""
        print(f"[ML Scanner] Processing optical scan of {image_path}...")
        
        # In a real environment, this would call pytesseract.image_to_string()
        # or an OpenVINO text detection inference model.
        # We simulate the OCR heuristic output here.
        simulated_extracted_data = {
            "supplier": "KRYSTAL-STACK CLOUD SERVICES",
            "amount": 1450.50,
            "iban": "SK12 0200 0000 0000 1234 5678",
            "raw_ocr_text": "INVOICE #2026-001\nSupplier: KRYSTAL-STACK\nTotal: 1450.50 EUR\nIBAN: SK12..."
        }
        print("[ML Scanner] OCR pattern extraction successful.")
        return simulated_extracted_data


class InvoiceCompositor:
    """
    Generative Presentation Layer.
    Can insert text dynamically into blank invoice templates to generate outgoing invoices.
    """
    def __init__(self):
        self.output_dir = "generated_invoices"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_invoice(self, data: dict, output_filename: str):
        """Creates a brand new digital invoice by rendering text over a programmatic canvas."""
        print(f"[Invoice Compositor] Generating new digital invoice: {output_filename}")
        
        # Create a blank white A4-ratio canvas
        width, height = 800, 1130
        img = Image.new('RGB', (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        # Use default font or fallback
        try:
            # Try to load a nice font if available in system
            font_title = ImageFont.truetype("arial.ttf", 36)
            font_text = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font_title = ImageFont.load_default()
            font_text = ImageFont.load_default()
            
        # Draw borders and UI elements mimicking the Arch UI aesthetics (cyan accents)
        draw.rectangle([(20, 20), (width-20, height-20)], outline=(69, 162, 158), width=3)
        
        # Insert Text dynamically
        draw.text((50, 50), "TAX INVOICE", fill=(11, 12, 16), font=font_title)
        draw.line([(50, 100), (width-50, 100)], fill=(69, 162, 158), width=2)
        
        y_offset = 150
        for key, value in data.items():
            draw.text((50, y_offset), f"{str(key).upper()}:", fill=(100, 100, 100), font=font_text)
            draw.text((250, y_offset), str(value), fill=(0, 0, 0), font=font_text)
            y_offset += 40
            
        # Output file
        filepath = os.path.join(self.output_dir, output_filename)
        img.save(filepath)
        print(f"[Invoice Compositor] Output saved to {filepath}")


if __name__ == "__main__":
    # 1. Initialize Diversified Application Layers
    db = InvoiceDatabaseCRUD()
    scanner = MLInvoiceScanner()
    compositor = InvoiceCompositor()
    
    # 2. Ingestion (OCR Scanning)
    extracted_data = scanner.digitize_invoice("dummy_physical_scan.jpg")
    
    # 3. Persistence (CRUD)
    inv_id = db.create(extracted_data)
    
    # 4. Generation (Compositing a new invoice based on DB record)
    db_record = db.read(inv_id)
    if db_record:
        compositor.generate_invoice({
            "Invoice ID": db_record["uuid"][:8],
            "Supplier": db_record["supplier"],
            "Amount due (EUR)": db_record["amount"],
            "IBAN": db_record["iban"],
            "Status": db_record["status"]
        }, f"out_invoice_{db_record['uuid'][:8]}.png")
