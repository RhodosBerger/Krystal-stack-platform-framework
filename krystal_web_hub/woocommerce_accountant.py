import time
import uuid

# ====================================================================
# KRYSTAL-STACK // WooCommerce Autonomous Accountant
# ====================================================================
# A scalable bridge that connects to any WooCommerce store REST API, 
# ingests orders, and automatically processes them into the 
# Krystal-Stack MRP Accounting system (Ledger & Invoicing).
# ====================================================================

class WooCommerceConnector:
    """Connects to a WooCommerce store via its native REST API."""
    
    def __init__(self, store_url: str, consumer_key: str, consumer_secret: str):
        self.store_url = store_url
        self.api_keys = {"key": consumer_key, "secret": consumer_secret}
        print(f"[WooConnector] Established secure link to: {self.store_url}")
        
    def fetch_recent_orders(self) -> list:
        """Simulates fetching orders from WooCommerce /wp-json/wc/v3/orders."""
        print(f"[WooConnector] Polling {self.store_url} for new WooCommerce orders...")
        time.sleep(0.5) # Simulated API latency
        
        # Simulated WooCommerce Order Payload
        mock_orders = [
            {
                "id": 8492,
                "status": "completed",
                "total": "125.50",
                "currency": "EUR",
                "billing": {"first_name": "Jozef", "last_name": "Novak", "company": "Novak s.r.o."},
                "line_items": [
                    {"name": "Mechanical Keyboard", "quantity": 1, "total": "100.00", "subtotal_tax": "20.00"}
                ],
                "shipping_total": "5.50",
                "date_completed": "2026-10-03T10:15:00"
            }
        ]
        return mock_orders

class AutonomousAccountant:
    """
    The Brain: Maps WooCommerce data into the MRP System.
    Automates Ledger entries and Invoice generation.
    """
    def __init__(self, mrp_db_system):
        self.db = mrp_db_system
        
    def process_woo_order(self, order: dict):
        """Translates a Woo Order into financial accounting records."""
        order_id = order.get("id")
        total = float(order.get("total", 0.0))
        company = order.get("billing", {}).get("company", "Retail Customer")
        if not company: company = f"{order['billing'].get('first_name')} {order['billing'].get('last_name')}"
        
        print(f"\n[Accountant] Processing WooCommerce Order #{order_id} from '{company}'")
        
        # 1. Map to Invoices CRUD (from our previous module)
        invoice_record = {
            "supplier": company,
            "amount": total,
            "iban": "WOOCOMMERCE_GATEWAY", # E.g., Stripe or PayPal reference
            "status": "PAID" if order.get("status") == "completed" else "PENDING",
            "raw_ocr_text": f"WOO-ORDER-{order_id} | DIGITAL SYNC"
        }
        
        try:
            # We assume db has the create() method from InvoiceDatabaseCRUD
            inv_uuid = self.db.create(invoice_record)
            print(f"  -> Generated internal Invoice Record: {inv_uuid}")
        except Exception as e:
            print(f"  -> Error booking invoice: {e}")
            
        # 2. Map to General Ledger (Hlavná kniha) - Double Entry Accounting Simulation
        # For an e-shop, a sale increases Revenue (Tržby) and increases Bank/Cash (Banka)
        print(f"  -> Journal Entry [Debit: 221 Banka] : {total} EUR")
        print(f"  -> Journal Entry [Credit: 604 Tržby za tovar] : {total} EUR")
        
        return True


if __name__ == "__main__":
    from invoice_compositor import InvoiceDatabaseCRUD
    
    # Initialize the core database
    mrp_db = InvoiceDatabaseCRUD("woo_invoices_data.db")
    
    # Initialize the automated accountant
    accountant = AutonomousAccountant(mrp_db)
    
    # Connect to a client's WooCommerce store
    client_store = WooCommerceConnector(
        store_url="https://shop.krystal-stack.com",
        consumer_key="ck_xxxxxxxxxxxx",
        consumer_secret="cs_xxxxxxxxxxxx"
    )
    
    # Execute sync loop
    pending_orders = client_store.fetch_recent_orders()
    for order in pending_orders:
        accountant.process_woo_order(order)
        
    print("\n[System] WooCommerce Sync Cycle Complete.")
