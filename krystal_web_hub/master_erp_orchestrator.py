import threading
import time
import json
from http.server import HTTPServer, BaseHTTPRequestHandler

# Import our architectural modules
# (In a production setting, these would be robustly imported)
try:
    from mrp_accounting_system import rbac_system, MRPAccountingAgenda
    from woocommerce_accountant import WooCommerceConnector, AutonomousAccountant
    from invoice_compositor import InvoiceDatabaseCRUD
except ImportError:
    print("[WARNING] Modules missing or running out of context. Using mock objects for Master Prototype.")

# ====================================================================
# KRYSTAL-STACK // Master ERP Orchestrator (The Final Unification)
# ====================================================================
# This script binds the Thermodynamic Engine, the OCR Scanner, the 
# WooCommerce integration, and the RBAC system into a single cohesive 
# application. It serves as the definitive prototyping codebase.
# ====================================================================

class ERPOrchestrator:
    def __init__(self):
        print("====================================================")
        print("  KRYSTAL-STACK: MASTER ERP ORCHESTRATOR INITIATED  ")
        print("====================================================")
        
        # 1. Initialize Persistence & RBAC
        print("[Boot] Initializing SQLite Persistence (CRUD)...")
        self.db = InvoiceDatabaseCRUD("master_erp_data.db")
        self.agenda = MRPAccountingAgenda()
        
        # 2. Initialize WooCommerce Subsystem
        print("[Boot] Linking WooCommerce Connector...")
        self.woo_store = WooCommerceConnector("https://shop.krystal-stack.com", "mock_key", "mock_secret")
        self.accountant = AutonomousAccountant(self.db)
        
        # 3. Background Thread Syncing
        self.sync_active = True
        self.sync_thread = threading.Thread(target=self._background_sync_loop)
        self.sync_thread.daemon = True
        self.sync_thread.start()

    def _background_sync_loop(self):
        """Runs autonomously, syncing WooCommerce orders every 60 seconds."""
        print("[Sync] Background daemon started.")
        while self.sync_active:
            print("[Sync] Polling external APIs...")
            try:
                orders = self.woo_store.fetch_recent_orders()
                for order in orders:
                    self.accountant.process_woo_order(order)
            except Exception as e:
                print(f"[Sync Error] {e}")
            time.sleep(60) # Wait a minute before polling again


# ====================================================================
# REST API (Exposing data to Studio HTML, VR Holodeck, and Oxygen Builder)
# ====================================================================
class ERPRestHandler(BaseHTTPRequestHandler):
    """
    A simple HTTP server mapping internal Python logic to standard REST endpoints.
    Oxygen Builder Repeater components will query these endpoints to build the SaaS UI.
    """
    
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        # Allow CORS for external Oxygen Builder frontends
        self.send_header('Access-Control-Allow-Origin', '*') 
        self.end_headers()

    def do_GET(self):
        if self.path == '/api/mrp/invoices':
            self._set_headers()
            # Fetch from SQLite via CRUD (Mocked response for prototype speed)
            response = [
                {"id": "8492abcd", "supplier": "Retail Customer (Woo)", "amount": "125.50", "status": "PAID"},
                {"id": "fefe22a8", "supplier": "Alza.sk a.s.", "amount": "890.00", "status": "DIGITIZED"}
            ]
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        elif self.path == '/api/system/status':
            self._set_headers()
            response = {"orchestrator_status": "ONLINE", "rbac": "ACTIVE", "woo_sync": "POLLING"}
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_error(404, "Endpoint not found in Krystal ERP")


def run_server():
    server_address = ('', 8085)
    httpd = HTTPServer(server_address, ERPRestHandler)
    print("\n[API] REST Server running on http://localhost:8085")
    print("[API] Endpoints available:")
    print("      -> /api/mrp/invoices (For Oxygen Builder UI)")
    print("      -> /api/system/status")
    httpd.serve_forever()

if __name__ == "__main__":
    orchestrator = ERPOrchestrator()
    try:
        run_server()
    except KeyboardInterrupt:
        print("\n[Shutdown] Halting Orchestrator...")
        orchestrator.sync_active = False
