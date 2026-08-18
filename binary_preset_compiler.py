import struct
import json
import base64
from typing import List, Dict

class BinaryPresetCompiler:
    """
    The 'Compiler' that takes the High-Level Python Logic (from the Generator)
    and freezes it into a 'Binary Preset'.
    
    This simulates the 'Light Travel Data Adaptation' - converting 
    slow-to-parse text logic into fast-to-execute binary structures.
    """
    
    OPCODES = {
        "BOOST_THREAD": 0x01,
        "CLEAR_CACHE": 0x02,
        "MIGRATE_TO_VULKAN": 0x03,
        "SLEEP": 0x04,
        "THROTTLE": 0x05
    }
    
    CONDITIONS = {
        "cpu": 0xA0,
        "ram_usage": 0xA1,
        "temp": 0xA2,
        "fps": 0xA3
    }
    
    def __init__(self):
        self.binary_stream = bytearray()

    def compile_logic_gene(self, gene_data: Dict) -> bytes:
        """
        Compiles a single logic dictionary into a binary packet.
        Format: [HEADER 2b] [COND_ID 1b] [VAL_THRESH 4b (float)] [OPCODE 1b] [TAIL 1b]
        """
        # Parse simple formula: "cpu > 0.8"
        parts = gene_data['condition_formula'].split(' ')
        metric = parts[0]
        threshold = float(parts[2])
        action = gene_data['action_type']
        
        # Encode
        cond_id = self.CONDITIONS.get(metric, 0x00)
        opcode = self.OPCODES.get(action, 0x00)
        
        # Pack structure
        # H=unsigned short (Header), B=unsigned char, f=float
        packet = struct.pack('>H B f B B', 
                             0xFFFF,       # Header (Start of Frame)
                             cond_id,      # Condition ID
                             threshold,    # Threshold value
                             opcode,       # Action Opcode
                             0xEE          # Tail (End of Frame)
                             )
        return packet

    def build_preset_library(self, genes: List[Dict], filename="optimized_presets.bin"):
        """
        Takes a list of 'Evolved Genes' and writes them to a .bin file.
        This file acts as the 'Register' for the Coprocessor.
        """
        print(f"Compiling {len(genes)} logic circuits to Binary...")
        
        with open(filename, 'wb') as f:
            # Write File Header (Magic Bytes: 'KRYSTAL')
            f.write(b'KRYSTAL')
            
            for gene in genes:
                bin_data = self.compile_logic_gene(gene)
                f.write(bin_data)
                
        print(f"Success. Wrote binary preset to {filename}")
        print(f"Size: {len(genes) * 9 + 7} bytes") # Approx calculation

    def decode_preset_library(self, filename="optimized_presets.bin"):
        """
        Decodes the binary file back to human readable format (Introspective Debugging).
        """
        print(f"\n--- Decoding {filename} (Reverse Logic) ---")
        with open(filename, 'rb') as f:
            header = f.read(7)
            if header != b'KRYSTAL':
                print("Invalid File Format")
                return

            while True:
                # Read 9 bytes (Size of our packet)
                chunk = f.read(9)
                if not chunk: break
                
                # Unpack
                _, cond_id, threshold, opcode, _ = struct.unpack('>H B f B B', chunk)
                
                # Reverse Map
                cond_str = [k for k, v in self.CONDITIONS.items() if v == cond_id][0]
                op_str = [k for k, v in self.OPCODES.items() if v == opcode][0]
                
                print(f"  [BIN] Rule: IF {cond_str} > {threshold:.2f} THEN {op_str}")

if __name__ == "__main__":
    # Mock Data from the "Inventor"
    mock_genes = [
        {'condition_formula': 'cpu > 0.95', 'action_type': 'THROTTLE'},
        {'condition_formula': 'fps < 30.0', 'action_type': 'BOOST_THREAD'},
        {'condition_formula': 'ram_usage > 0.85', 'action_type': 'CLEAR_CACHE'}
    ]
    
    compiler = BinaryPresetCompiler()
    compiler.build_preset_library(mock_genes)
    
    # Validate
    compiler.decode_preset_library()
