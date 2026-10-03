import sqlite3
import uuid
import time
from functools import wraps

# ====================================================================
# KRYSTAL-STACK // MRP-Inspired Accounting Agenda & RBAC System
# ====================================================================
# This module brings classical accounting workflows (Invoicing, Ledger, 
# Assets) into the platform, protected by a strict Role-Based Access 
# Control (RBAC) system.
# ====================================================================

class RoleManager:
    """Handles User Roles and Permissions (System rozdelenia rolí)."""
    
    ROLES = {
        "ADMIN": ["create_user", "delete_user", "view_all", "edit_all", "delete_invoice", "post_ledger"],
        "ACCOUNTANT": ["view_all", "create_invoice", "edit_invoice", "post_ledger"],
        "AUDITOR": ["view_all"],
        "GUEST": ["view_public"]
    }

    def __init__(self, db_path="mrp_users.db"):
        self.db_path = db_path
        self._init_db()
        # Create a default admin if none exists
        if not self.get_user("admin"):
            self.create_user("admin", "ADMIN")

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    role TEXT NOT NULL
                )
            ''')
            conn.commit()

    def create_user(self, username: str, role: str):
        if role not in self.ROLES:
            raise ValueError(f"Role {role} does not exist.")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT OR REPLACE INTO users (username, role) VALUES (?, ?)', (username, role))
            conn.commit()
        print(f"[RBAC] Created/Updated user '{username}' with role '{role}'.")

    def get_user(self, username: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT role FROM users WHERE username = ?', (username,))
            row = cursor.fetchone()
            return {"username": username, "role": row[0]} if row else None

    def has_permission(self, username: str, permission: str) -> bool:
        user = self.get_user(username)
        if not user:
            return False
        return permission in self.ROLES.get(user["role"], [])


# Singleton instance for decorators
rbac_system = RoleManager()

def requires_permission(permission: str):
    """Decorator to enforce RBAC on accounting functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, active_user: str, *args, **kwargs):
            if not rbac_system.has_permission(active_user, permission):
                print(f"[ACCESS DENIED] User '{active_user}' lacks permission '{permission}'.")
                raise PermissionError(f"Access Denied: Requires '{permission}'")
            return func(self, active_user, *args, **kwargs)
        return wrapper
    return decorator


class MRPAccountingAgenda:
    """
    MRP-Inspired Accounting Workflows.
    Manages Invoices, Ledger (Hlavná Kniha), and Payroll (Mzdy) securely.
    """
    def __init__(self):
        print("[MRP Agenda] System Initialized. Awaiting authenticated requests.")
        
    @requires_permission("create_invoice")
    def register_incoming_invoice(self, active_user: str, invoice_data: dict):
        """Accountant workflow: Register a new scanned invoice into the system."""
        print(f"[MRP Agenda] {active_user} is registering an invoice...")
        
        # Here we would normally hook into the InvoiceDatabaseCRUD we built earlier
        inv_id = str(uuid.uuid4())[:8]
        print(f"[MRP Agenda] Invoice {inv_id} successfully booked by {active_user}.")
        return inv_id

    @requires_permission("delete_invoice")
    def storno_invoice(self, active_user: str, invoice_id: str):
        """Admin workflow: Storno (Delete/Cancel) an invoice."""
        print(f"[MRP Agenda] {active_user} executed STORNO on invoice {invoice_id}.")
        return True

    @requires_permission("post_ledger")
    def post_to_general_ledger(self, active_user: str, amount: float, account_code: str):
        """Post a transaction to the Hlavná Kniha (General Ledger)."""
        print(f"[MRP Agenda] {active_user} posted {amount} EUR to account {account_code}.")
        return True

    @requires_permission("view_all")
    def generate_audit_report(self, active_user: str):
        """Auditor workflow: Pull all financial data for review."""
        print(f"[MRP Agenda] {active_user} generated the full system Audit Report.")
        return {"report_type": "AUDIT", "status": "CLEAN"}


if __name__ == "__main__":
    agenda = MRPAccountingAgenda()
    
    # 1. Setup roles (Admin creates an accountant and an auditor)
    rbac_system.create_user("jozo_uctovnik", "ACCOUNTANT")
    rbac_system.create_user("peter_auditor", "AUDITOR")
    rbac_system.create_user("fero_brigadnik", "GUEST")
    print("-" * 40)

    # 2. Test Workflows based on Roles
    
    # Accountant creates an invoice (SUCCESS)
    inv_id = agenda.register_incoming_invoice("jozo_uctovnik", {"amount": 1000})
    
    # Accountant tries to delete an invoice (FAIL)
    try:
        agenda.storno_invoice("jozo_uctovnik", inv_id)
    except PermissionError:
        pass
        
    # Admin deletes the invoice (SUCCESS)
    agenda.storno_invoice("admin", inv_id)
    
    # Auditor reads the report (SUCCESS)
    agenda.generate_audit_report("peter_auditor")
    
    # Auditor tries to post to ledger (FAIL)
    try:
        agenda.post_to_general_ledger("peter_auditor", 500, "518.100")
    except PermissionError:
        pass

    print("-" * 40)
    print("[MRP Agenda] RBAC Workflow Demonstration Complete.")
