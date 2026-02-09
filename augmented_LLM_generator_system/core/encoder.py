"""
Essential Encoder Module
Implements multi-strategy encoding for the Augmented LLM Generator System.
Based on legacy documentation patterns.
"""

import json
import base64
import zlib
import hashlib
from enum import Enum
from typing import Any, Dict, List, Optional, Union
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

class EncodingType(Enum):
    BINARY = "binary"
    BASE64 = "base64"
    JSON = "json"
    COMPRESSED = "compressed"
    NEURAL = "neural"
    HEX = "hex"

class EncodedResult:
    def __init__(self, data: Any, encoding_type: EncodingType, metadata: Dict[str, Any] = None):
        self.data = data
        self.encoding_type = encoding_type
        self.metadata = metadata or {}
        self.size_encoded = len(str(data)) # Approx size
        self.hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calculates SHA-256 hash of the encoded data for integrity verification."""
        if isinstance(self.data, bytes):
            return hashlib.sha256(self.data).hexdigest()
        return hashlib.sha256(str(self.data).encode('utf-8')).hexdigest()

class EssentialEncoder:
    def __init__(self):
        self.stats = {
            "total_encodings": 0,
            "bytes_processed": 0
        }

    def encode(self, data: Any, encoding_type: EncodingType) -> EncodedResult:
        """
        Main encoding entry point.
        """
        self.stats["total_encodings"] += 1
        
        if encoding_type == EncodingType.JSON:
            return self._encode_json(data)
        elif encoding_type == EncodingType.BASE64:
            return self._encode_base64(data)
        elif encoding_type == EncodingType.COMPRESSED:
            return self._encode_compressed(data)
        elif encoding_type == EncodingType.NEURAL:
            return self._encode_neural(data)
        elif encoding_type == EncodingType.HEX:
            return self._encode_hex(data)
        else:
            return self._encode_binary(data)

    def _encode_json(self, data: Any) -> EncodedResult:
        encoded = json.dumps(data)
        return EncodedResult(encoded, EncodingType.JSON, {"original_type": type(data).__name__})

    def _encode_base64(self, data: Any) -> EncodedResult:
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, bytes):
            data_bytes = data
        else:
            data_bytes = str(data).encode('utf-8')
            
        encoded = base64.b64encode(data_bytes).decode('utf-8')
        return EncodedResult(encoded, EncodingType.BASE64)

    def _encode_compressed(self, data: Any) -> EncodedResult:
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = str(data).encode('utf-8')
            
        compressed = zlib.compress(data_bytes)
        # We return base64 of compressed data to make it transport-safe
        encoded = base64.b64encode(compressed).decode('utf-8')
        
        ratio = len(data_bytes) / len(compressed) if len(compressed) > 0 else 0
        return EncodedResult(encoded, EncodingType.COMPRESSED, {"compression_ratio": ratio})

    def _encode_neural(self, data: Any) -> EncodedResult:
        """
        Optimizes data for neural network input (Normalization).
        Expects numerical data (list or numpy array).
        """
        if not NUMPY_AVAILABLE:
            # Pure Python Fallback
            try:
                if isinstance(data, list):
                    arr = [float(x) for x in data]
                    min_val = min(arr)
                    max_val = max(arr)
                    if max_val > min_val:
                        normalized = [(x - min_val) / (max_val - min_val) for x in arr]
                    else:
                        normalized = arr
                    
                    return EncodedResult(normalized, EncodingType.NEURAL, {
                        "shape": (len(arr),),
                        "min": min_val,
                        "max": max_val,
                        "backend": "pure_python"
                    })
                else:
                    return EncodedResult(str(data), EncodingType.NEURAL, {"error": "Numpy not available and data is not a list", "fallback": True})
            except Exception as e:
                return EncodedResult(str(data), EncodingType.NEURAL, {"error": str(e), "fallback": True})

        try:
            arr = np.array(data, dtype=float)
            # Simple Min-Max Normalization to [0, 1]
            if arr.max() > arr.min():
                normalized = (arr - arr.min()) / (arr.max() - arr.min())
            else:
                normalized = arr # Handle constant values
                
            return EncodedResult(normalized.tolist(), EncodingType.NEURAL, {
                "shape": arr.shape,
                "min": float(arr.min()),
                "max": float(arr.max()),
                "backend": "numpy"
            })
        except Exception as e:
            # Fallback for non-numerical data
            return EncodedResult(str(data), EncodingType.NEURAL, {"error": str(e), "fallback": True})

    def _encode_hex(self, data: Any) -> EncodedResult:
        if isinstance(data, int):
            encoded = hex(data)
        elif isinstance(data, str):
            encoded = data.encode('utf-8').hex()
        else:
            encoded = str(data).encode('utf-8').hex()
            
        return EncodedResult(encoded, EncodingType.HEX)

    def _encode_binary(self, data: Any) -> EncodedResult:
        # Placeholder for raw binary
        return EncodedResult(data, EncodingType.BINARY)
