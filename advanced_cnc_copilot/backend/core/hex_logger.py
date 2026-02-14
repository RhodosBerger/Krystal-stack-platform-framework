import logging
import numpy as np
import binascii
import struct
import platform
import os

class HexTraceLogger:
    """
    Advanced Log Composer that creates a Hexadecimal Trace of system links.
    Captures Covalent State (Numpy), System Info (Conda/Docker), and timestamps.
    """
    def __init__(self):
        self.logger = logging.getLogger("HexTrace")
        self.system_fingerprint = self._get_system_fingerprint()

    def _get_system_fingerprint(self) -> bytes:
        """
        Captures static system info (Platform, Node) as bytes.
        """
        uname = platform.uname()
        info = f"{uname.system}-{uname.release}-{uname.node}"
        return info.encode('utf-8')[:16].ljust(16, b'\0')

    def compose_trace(self, spectrum: np.ndarray, event_id: int) -> str:
        """
        Composes a Hexadecimal Trace String from the current state.
        Format: [EventID][SystemFP][SpectrumData][CheckSum]
        """
        # 1. Event ID (4 bytes)
        b_id = struct.pack('>I', event_id)
        
        # 2. System Fingerprint (16 bytes)
        b_sys = self.system_fingerprint

        # 3. Spectrum Data (Numpy -> Bytes)
        # We start with the raw bytes of the float array
        b_spectrum = spectrum.tobytes()
        
        # 4. Compose Payload
        payload = b_id + b_sys + b_spectrum
        
        # 5. Checksum (CRC32)
        checksum = binascii.crc32(payload)
        b_checksum = struct.pack('>I', checksum)
        
        # 6. Final Hex String
        final_bytes = payload + b_checksum
        hex_trace = binascii.hexlify(final_bytes).decode('ascii').upper()
        
        return hex_trace

    def log_trace(self, spectrum: np.ndarray, event_id: int):
        trace = self.compose_trace(spectrum, event_id)
        self.logger.info(f"TRACE_HEX: {trace}")
        return trace
