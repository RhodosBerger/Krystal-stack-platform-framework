"""
FOCAS Bridge Module
Wraps the proprietary Fanuc FOCAS library (fwlib32.dll or libfwlib32.so) using ctypes.
Allows Python to talk directly to the CNC machine's HSSB or Ethernet port.
"""
import ctypes
import os
import platform
import logging
from typing import Dict, Tuple

logger = logging.getLogger("FOCAS_BRIDGE")

# --- FOCAS Data Structures (C-Types) ---
# These must match the Fanuc FOCAS2 Library Specification exactly.

class ODBST(ctypes.Structure):
    """CNC Status Structure"""
    _fields_ = [
        ("dummy", ctypes.c_short),
        ("type", ctypes.c_short),
        ("const", ctypes.c_short),
        ("ext_no", ctypes.c_short),
        ("axis", ctypes.c_short),
        ("solid", ctypes.c_short),
    ]

class ODBSP(ctypes.Structure):
    """Spindle Data Structure"""
    _fields_ = [
        ("data", ctypes.c_short * 4), # Spindle speed, etc.
    ]

class OD ஆகியவற்றAxis(ctypes.Structure):
    """Axis Position Structure"""
    _fields_ = [
        ("data", ctypes.c_long), # Position
        ("dec", ctypes.c_short), # Decimal places
        ("unit", ctypes.c_short),
        ("disp", ctypes.c_short),
        ("name", ctypes.c_char),
        ("suff", ctypes.c_char),
    ]

class FocasBridge:
    def __init__(self, ip: str, port: int, timeout: int = 10):
        self.ip = ip
        self.port = port
        self.timeout = timeout
        self.lib = None
        self.handle = ctypes.c_ushort(0)
        self.is_connected = False
        
        self._load_library()

    def _load_library(self):
        """Attempts to load the FOCAS DLL/Shared Object."""
        system = platform.system()
        lib_name = "fwlib32.dll" if system == "Windows" else "libfwlib32.so"
        
        # Try common paths or local directory
        possible_paths = [
            lib_name,
            os.path.join(os.getcwd(), "libs", lib_name),
            os.path.join("C:\\Windows\\System32", lib_name)
        ]
        
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    self.lib = ctypes.cdll.LoadLibrary(path)
                    logger.info(f"✅ Loaded FOCAS Library from: {path}")
                    break
            except Exception as e:
                pass
        
        if not self.lib:
            logger.warning(f"⚠️ FOCAS Library ({lib_name}) not found. Running in SIMULATION MODE.")

    def connect(self) -> int:
        """Connects to the CNC via Ethernet."""
        if not self.lib:
            return -1 # Library not found
            
        try:
            ret = self.lib.cnc_allclibhndl3(
                self.ip.encode('utf-8'),
                self.port,
                self.timeout,
                ctypes.byref(self.handle)
            )
            
            if ret == 0:
                self.is_connected = True
                logger.info(f"✅ Connected to CNC at {self.ip}")
            else:
                logger.error(f"❌ Connection Failed. Error Code: {ret}")
                
            return ret
        except Exception as e:
            logger.error(f"FOCAS Call Failed: {e}")
            return -1

    def disconnect(self):
        if self.is_connected and self.lib:
            self.lib.cnc_freelibhndl(self.handle)
            self.is_connected = False
            logger.info("Disconnected from CNC.")

    def read_spindle_speed(self) -> float:
        """Reads actual spindle speed (S-meter)."""
        if not self.is_connected:
            return 0.0
            
        spindle_data = ODBSP()
        # 0 = Main Spindle, 1 = First Spindle Data (Speed)
        ret = self.lib.cnc_rdspeed(self.handle, 0, ctypes.byref(spindle_data))
        
        if ret == 0:
            # Depending on FOCAS version, data[0] or data[1] might be load vs speed
            return float(spindle_data.data[0]) 
        return 0.0

    def read_status(self) -> str:
        """Reads CNC Run Status (STOP, HOLD, START)."""
        if not self.is_connected:
            return "OFFLINE"
            
        status = ODBST()
        ret = self.lib.cnc_statinfo(self.handle, ctypes.byref(status))
        
        if ret == 0:
            # Mapping status.run to text
            # 0: STOP, 1: HOLD, 2: START, 3: MSTR
            modes = {0: "STOP", 1: "HOLD", 2: "RUNNING", 3: "MSTR"}
            return modes.get(status.type, "UNKNOWN")
        return "ERROR"

    def read_axis_load(self) -> float:
        """Reads servo load meter (approximate global load)."""
        # This is a complex FOCAS call requiring sv_readsvmeter
        # For this prototype, we return a mock based on connection
        if self.is_connected:
            return 0.0 # Placeholder for actual implementation
        return 0.0
