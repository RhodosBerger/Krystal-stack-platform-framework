#!/usr/bin/env python3
"""
CLOUD AUTH MODEL (Mock Django)
Multilevel User Hierarchy Definition.
"""

from dataclasses import dataclass, field
from typing import List, Set
from enum import Enum, auto

# --- 1. THE HIERARCHY (Roles) ---
class UserRole(Enum):
    GLOBAL_ADMIN = "global_admin"       # The Platform Owner
    ORG_ADMIN = "org_admin"             # Factory Manager (The General)
    ENGINEER = "engineer"               # Designer (The Architect)
    OPERATOR = "operator"               # Machinist (The Pilot)
    AUDITOR = "auditor"                 # Accountant (The Shadow)

# --- 2. THE PERMISSIONS (Capabilities) ---
class Permission(Enum):
    # Strategic
    VIEW_FINANCIALS = auto()
    MANAGE_USERS = auto()
    GLOBAL_OVERRIDE = auto()
    
    # Creative
    UPLOAD_CAD = auto()
    EDIT_MACROS = auto()
    RUN_SIMULATION = auto()
    
    # Operational
    EXECUTE_GCODE = auto()
    VIEW_TELEMETRY = auto()
    LOCAL_OVERRIDE = auto() # Small adjustments only

    # Audit
    VIEW_LOGS = auto()

# --- 3. THE MODELS (Mock ORM) ---

@dataclass
class Organization:
    """A Factory or Company Tenant."""
    id: int
    name: str
    subscription_tier: str = "ENTERPRISE" # Free, Pro, Enterprise

@dataclass
class RiseUser:
    """The User Account."""
    id: int
    username: str
    email: str
    organization: Organization
    role: UserRole
    
    # Computed property based on Role
    @property
    def permissions(self) -> Set[Permission]:
        """
        Multilevel Permission Logic.
        """
        perms = set()
        
        # Base Permissions (Everyone)
        perms.add(Permission.VIEW_TELEMETRY) 

        # LEVEL 3: OPERATOR
        if self.role in [UserRole.OPERATOR, UserRole.ENGINEER, UserRole.ORG_ADMIN]:
            perms.add(Permission.EXECUTE_GCODE)
            perms.add(Permission.LOCAL_OVERRIDE)

        # LEVEL 2: ENGINEER
        if self.role in [UserRole.ENGINEER, UserRole.ORG_ADMIN]:
            perms.add(Permission.UPLOAD_CAD)
            perms.add(Permission.EDIT_MACROS)
            perms.add(Permission.RUN_SIMULATION)

        # LEVEL 1: ADMIN
        if self.role == UserRole.ORG_ADMIN:
            perms.add(Permission.MANAGE_USERS)
            perms.add(Permission.VIEW_FINANCIALS)
            perms.add(Permission.GLOBAL_OVERRIDE)
            
        # LEVEL 4: AUDITOR
        if self.role == UserRole.AUDITOR:
            perms.add(Permission.VIEW_FINANCIALS)
            perms.add(Permission.VIEW_LOGS)
            # Auditors CANNOT execute code
            perms.discard(Permission.EXECUTE_GCODE)
            perms.discard(Permission.LOCAL_OVERRIDE)

        return perms

    def has_perm(self, perm: Permission) -> bool:
        return perm in self.permissions

# --- TEST SCENARIO ---
if __name__ == "__main__":
    # Create Org
    factory_x = Organization(id=1, name="Tesla Gigafactory", subscription_tier="ENTERPRISE")
    
    # Create Users
    general_zod = RiseUser(1, "zod", "zod@tesla.com", factory_x, UserRole.ORG_ADMIN)
    arch_neo = RiseUser(2, "neo", "neo@tesla.com", factory_x, UserRole.ENGINEER)
    pilot_mav = RiseUser(3, "mav", "mav@tesla.com", factory_x, UserRole.OPERATOR)
    shadow_acc = RiseUser(4, "acc", "acc@irs.gov", factory_x, UserRole.AUDITOR)

    # Verify Access to "Financials"
    print(f"General Zod can View Money? {general_zod.has_perm(Permission.VIEW_FINANCIALS)}") # True
    print(f"Pilot Mav can View Money?   {pilot_mav.has_perm(Permission.VIEW_FINANCIALS)}")   # False
    
    # Verify Access to "Upload CAD"
    print(f"Architect Neo can Upload?   {arch_neo.has_perm(Permission.UPLOAD_CAD)}")      # True
    print(f"Pilot Mav can Upload?       {pilot_mav.has_perm(Permission.UPLOAD_CAD)}")      # False
    
    # Verify Safety (Auditor cannot run machine)
    print(f"Shadow Acc can Execute?     {shadow_acc.has_perm(Permission.EXECUTE_GCODE)}")   # False
