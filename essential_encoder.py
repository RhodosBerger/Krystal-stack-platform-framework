#!/usr/bin/env python3
"""
Essential Encoder for GAMESA/KrystalStack Framework

This module implements an essential encoder that provides core encoding/decoding
functionality for the GAMESA framework. The encoder supports multiple encoding
formats optimized for different use cases including neural network inputs,
compressed representations, and optimized data structures.
"""

import numpy as np
import struct
import base64
import json
from typing import Union, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
import zlib
import pickle
from datetime import datetime
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EncodingType(Enum):
    """Types of encoding supported by the essential encoder."""
    BINARY = "binary"
    BASE64 = "base64"
    JSON = "json"
    COMPRESSED = "compressed"
    NEURAL = "neural"
    HEX = "hex"
    CUSTOM = "custom"


@dataclass
class EncodingResult:
    """Result of an encoding operation."""
    data: bytes
    encoding_type: EncodingType
    metadata: Dict[str, Any]
    size_original: int
    size_encoded: int
    compression_ratio: float
    encoding_time: float


class EssentialEncoder:
    """
    Essential Encoder for the GAMESA framework.
    
    Provides multiple encoding strategies optimized for different use cases
    including neural network processing, compression, and data transmission.
    """
    
    def __init__(self):
        self.encoding_stats = {}
        self.last_encoding_result: Optional[EncodingResult] = None
        
    def encode(self, data: Any, encoding_type: EncodingType = EncodingType.BINARY, 
               custom_params: Optional[Dict] = None) -> EncodingResult:
        """
        Encode data using the specified encoding type.
        
        Args:
            data: Input data to encode
            encoding_type: Type of encoding to use
            custom_params: Custom parameters for encoding
            
        Returns:
            EncodingResult containing encoded data and metadata
        """
        import time
        start_time = time.time()
        
        if encoding_type == EncodingType.BINARY:
            encoded_data = self._encode_binary(data)
        elif encoding_type == EncodingType.BASE64:
            encoded_data = self._encode_base64(data)
        elif encoding_type == EncodingType.JSON:
            encoded_data = self._encode_json(data)
        elif encoding_type == EncodingType.COMPRESSED:
            encoded_data = self._encode_compressed(data)
        elif encoding_type == EncodingType.NEURAL:
            encoded_data = self._encode_neural(data)
        elif encoding_type == EncodingType.HEX:
            encoded_data = self._encode_hex(data)
        elif encoding_type == EncodingType.CUSTOM:
            encoded_data = self._encode_custom(data, custom_params)
        else:
            raise ValueError(f"Unsupported encoding type: {encoding_type}")
        
        # Calculate statistics
        original_size = len(pickle.dumps(data)) if isinstance(data, (dict, list, tuple)) else len(str(data).encode())
        encoded_size = len(encoded_data)
        compression_ratio = original_size / encoded_size if encoded_size > 0 else 0
        
        result = EncodingResult(
            data=encoded_data,
            encoding_type=encoding_type,
            metadata={
                "original_type": type(data).__name__,
                "encoding_timestamp": datetime.utcnow().isoformat(),
                "encoding_uuid": str(uuid.uuid4()),
                "custom_params": custom_params or {}
            },
            size_original=original_size,
            size_encoded=encoded_size,
            compression_ratio=compression_ratio,
            encoding_time=time.time() - start_time
        )
        
        self.last_encoding_result = result
        self._update_stats(encoding_type, result)
        
        return result
    
    def decode(self, encoded_data: bytes, encoding_type: EncodingType) -> Any:
        """
        Decode data using the specified encoding type.
        
        Args:
            encoded_data: Encoded data to decode
            encoding_type: Type of encoding used
            
        Returns:
            Decoded data
        """
        if encoding_type == EncodingType.BINARY:
            return self._decode_binary(encoded_data)
        elif encoding_type == EncodingType.BASE64:
            return self._decode_base64(encoded_data)
        elif encoding_type == EncodingType.JSON:
            return self._decode_json(encoded_data)
        elif encoding_type == EncodingType.COMPRESSED:
            return self._decode_compressed(encoded_data)
        elif encoding_type == EncodingType.NEURAL:
            return self._decode_neural(encoded_data)
        elif encoding_type == EncodingType.HEX:
            return self._decode_hex(encoded_data)
        elif encoding_type == EncodingType.CUSTOM:
            raise NotImplementedError("Custom decoding not implemented")
        else:
            raise ValueError(f"Unsupported encoding type: {encoding_type}")
    
    def _encode_binary(self, data: Any) -> bytes:
        """Encode data as binary."""
        if isinstance(data, bytes):
            return data
        elif isinstance(data, str):
            return data.encode('utf-8')
        elif isinstance(data, (int, float)):
            if isinstance(data, int):
                return struct.pack('q', data)  # 8-byte integer
            else:
                return struct.pack('d', data)  # 8-byte float
        elif isinstance(data, (list, tuple)):
            return pickle.dumps(data)
        elif isinstance(data, dict):
            return pickle.dumps(data)
        elif isinstance(data, np.ndarray):
            return data.tobytes()
        else:
            return pickle.dumps(data)
    
    def _decode_binary(self, data: bytes) -> Any:
        """Decode binary data."""
        # Try to determine the original type from the data
        try:
            # Check if it's a numpy array by attempting to reconstruct
            # For now, return as-is or attempt pickle load
            if len(data) == 8:
                # Might be a single int or float
                try:
                    return struct.unpack('q', data)[0]  # Try as int
                except:
                    try:
                        return struct.unpack('d', data)[0]  # Try as float
                    except:
                        pass

            # Try pickle load for complex objects
            return pickle.loads(data)
        except:
            # If pickle fails, try to decode as string
            try:
                return data.decode('utf-8')
            except:
                # Return as bytes if nothing else works
                return data
    
    def _encode_base64(self, data: Any) -> bytes:
        """Encode data as base64."""
        binary_data = self._encode_binary(data)
        return base64.b64encode(binary_data)
    
    def _decode_base64(self, data: bytes) -> Any:
        """Decode base64 data."""
        binary_data = base64.b64decode(data)
        return self._decode_binary(binary_data)
    
    def _encode_json(self, data: Any) -> bytes:
        """Encode data as JSON."""
        # Convert numpy types to standard Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            else:
                return obj
        
        json_serializable = convert_numpy_types(data)
        json_str = json.dumps(json_serializable, separators=(',', ':'))
        return json_str.encode('utf-8')
    
    def _decode_json(self, data: bytes) -> Any:
        """Decode JSON data."""
        json_str = data.decode('utf-8')
        return json.loads(json_str)
    
    def _encode_compressed(self, data: Any) -> bytes:
        """Encode and compress data."""
        binary_data = self._encode_binary(data)
        return zlib.compress(binary_data)
    
    def _decode_compressed(self, data: bytes) -> Any:
        """Decode compressed data."""
        decompressed_data = zlib.decompress(data)
        return self._decode_binary(decompressed_data)
    
    def _encode_neural(self, data: Any) -> bytes:
        """
        Encode data optimized for neural network processing.
        
        This encoding is optimized for neural network inputs, normalizing
        values and structuring data for efficient processing.
        """
        if isinstance(data, np.ndarray):
            # Normalize array to 0-1 range if it's not already
            arr = data.astype(np.float32)
            if arr.size > 0:
                min_val, max_val = arr.min(), arr.max()
                if max_val != min_val:
                    arr = (arr - min_val) / (max_val - min_val)
            return arr.tobytes()
        elif isinstance(data, list):
            # Convert to normalized numpy array
            arr = np.array(data, dtype=np.float32)
            if arr.size > 0:
                min_val, max_val = arr.min(), arr.max()
                if max_val != min_val:
                    arr = (arr - min_val) / (max_val - min_val)
            return arr.tobytes()
        else:
            # Convert to numpy array and normalize
            arr = np.array([data], dtype=np.float32)
            if arr.size > 0:
                min_val, max_val = arr.min(), arr.max()
                if max_val != min_val:
                    arr = (arr - min_val) / (max_val - min_val)
            return arr.tobytes()
    
    def _decode_neural(self, data: bytes) -> np.ndarray:
        """Decode neural network optimized data."""
        # Determine array size from data length (assuming float32)
        array_size = len(data) // 4  # 4 bytes per float32
        return np.frombuffer(data, dtype=np.float32, count=array_size)
    
    def _encode_hex(self, data: Any) -> bytes:
        """Encode data as hexadecimal."""
        binary_data = self._encode_binary(data)
        return binary_data.hex().encode('utf-8')
    
    def _decode_hex(self, data: bytes) -> Any:
        """Decode hexadecimal data."""
        hex_str = data.decode('utf-8')
        binary_data = bytes.fromhex(hex_str)
        return self._decode_binary(binary_data)
    
    def _encode_custom(self, data: Any, params: Optional[Dict]) -> bytes:
        """Custom encoding with user-defined parameters."""
        if params is None:
            params = {}
        
        # Example custom encoding: apply a transformation based on parameters
        if isinstance(data, list) and 'multiplier' in params:
            multiplier = params['multiplier']
            if isinstance(data[0], (int, float)):
                data = [x * multiplier for x in data]
        
        return self._encode_binary(data)
    
    def _update_stats(self, encoding_type: EncodingType, result: EncodingResult):
        """Update encoding statistics."""
        if encoding_type.value not in self.encoding_stats:
            self.encoding_stats[encoding_type.value] = {
                'total_encoded': 0,
                'total_size_original': 0,
                'total_size_encoded': 0,
                'avg_compression_ratio': 0,
                'avg_encoding_time': 0
            }
        
        stats = self.encoding_stats[encoding_type.value]
        stats['total_encoded'] += 1
        stats['total_size_original'] += result.size_original
        stats['total_size_encoded'] += result.size_encoded
        stats['avg_compression_ratio'] = (
            (stats['avg_compression_ratio'] * (stats['total_encoded'] - 1) + result.compression_ratio) /
            stats['total_encoded']
        )
        stats['avg_encoding_time'] = (
            (stats['avg_encoding_time'] * (stats['total_encoded'] - 1) + result.encoding_time) /
            stats['total_encoded']
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get encoding statistics."""
        return self.encoding_stats
    
    def encode_with_hash(self, data: Any, encoding_type: EncodingType = EncodingType.BINARY) -> Tuple[EncodingResult, str]:
        """Encode data and return with SHA-256 hash for integrity verification."""
        result = self.encode(data, encoding_type)
        data_hash = hashlib.sha256(result.data).hexdigest()
        return result, data_hash
    
    def verify_integrity(self, encoded_data: bytes, expected_hash: str) -> bool:
        """Verify data integrity using SHA-256 hash."""
        actual_hash = hashlib.sha256(encoded_data).hexdigest()
        return actual_hash == expected_hash


class NeuralEncoder:
    """
    Specialized neural network encoder for the GAMESA framework.
    
    Optimized for neural network inputs, preprocessing, and feature encoding.
    """
    
    def __init__(self):
        self.feature_encoders = {}
        self.normalization_params = {}
        
    def encode_features(self, features: Union[np.ndarray, List, Dict], 
                       normalize: bool = True, encode_categorical: bool = True) -> np.ndarray:
        """
        Encode features for neural network processing.
        
        Args:
            features: Input features to encode
            normalize: Whether to normalize features to [0, 1] range
            encode_categorical: Whether to encode categorical features
            
        Returns:
            Encoded features as numpy array
        """
        if isinstance(features, dict):
            # Convert dict to array
            features = np.array(list(features.values()), dtype=np.float32)
        elif isinstance(features, list):
            features = np.array(features, dtype=np.float32)
        elif not isinstance(features, np.ndarray):
            features = np.array([features], dtype=np.float32)
        
        if normalize and features.size > 0:
            # Store normalization parameters for potential decoding
            min_val, max_val = features.min(), features.max()
            if max_val != min_val:
                features = (features - min_val) / (max_val - min_val)
                self.normalization_params['min'] = min_val
                self.normalization_params['max'] = max_val
        
        return features.astype(np.float32)
    
    def encode_categorical(self, categories: List[str], 
                          target_categories: Optional[List[str]] = None) -> np.ndarray:
        """
        One-hot encode categorical variables.
        
        Args:
            categories: List of categorical values
            target_categories: Predefined list of all possible categories
            
        Returns:
            One-hot encoded array
        """
        if target_categories is None:
            target_categories = list(set(categories))
        
        encoded = np.zeros((len(categories), len(target_categories)), dtype=np.float32)
        for i, category in enumerate(categories):
            if category in target_categories:
                idx = target_categories.index(category)
                encoded[i, idx] = 1.0
        
        return encoded
    
    def encode_sequence(self, sequence: List[Any], max_length: int = 128) -> np.ndarray:
        """
        Encode a sequence for neural network processing.
        
        Args:
            sequence: Input sequence to encode
            max_length: Maximum length of sequence (pad or truncate)
            
        Returns:
            Encoded sequence as numpy array
        """
        # Convert sequence to numerical representation
        if isinstance(sequence[0], str):
            # For string sequences, create a simple numerical encoding
            unique_items = list(set(sequence))
            item_to_id = {item: idx for idx, item in enumerate(unique_items)}
            numerical_seq = [item_to_id[item] for item in sequence]
        else:
            numerical_seq = [float(x) for x in sequence]
        
        # Pad or truncate to max_length
        if len(numerical_seq) < max_length:
            numerical_seq.extend([0] * (max_length - len(numerical_seq)))
        else:
            numerical_seq = numerical_seq[:max_length]
        
        return np.array(numerical_seq, dtype=np.float32)


class QuantizedEncoder:
    """
    Quantized encoder for efficient neural network processing.
    
    Implements quantization techniques to reduce precision while maintaining
    model accuracy, useful for deployment on resource-constrained devices.
    """
    
    def __init__(self, bits: int = 8):
        self.bits = bits
        self.quantization_scale = 2 ** bits - 1  # e.g., 255 for 8-bit
        
    def quantize(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Quantize data to lower precision.
        
        Args:
            data: Input data to quantize
            
        Returns:
            Quantized data and quantization parameters
        """
        # Determine min/max for quantization
        min_val, max_val = data.min(), data.max()
        
        # Quantize to integer range
        if max_val != min_val:
            scale = self.quantization_scale / (max_val - min_val)
            zero_point = -min_val * scale
            quantized = np.clip(np.round(data * scale + zero_point), 0, self.quantization_scale)
        else:
            scale = 1.0
            zero_point = 0.0
            quantized = np.full_like(data, self.quantization_scale // 2, dtype=np.int32)
        
        quantized = quantized.astype(np.uint8 if self.bits == 8 else np.int32)
        
        params = {
            'scale': float(scale),
            'zero_point': float(zero_point),
            'min_val': float(min_val),
            'max_val': float(max_val)
        }
        
        return quantized, params
    
    def dequantize(self, quantized_data: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """
        Dequantize data back to original precision.
        
        Args:
            quantized_data: Quantized data to dequantize
            params: Quantization parameters
            
        Returns:
            Dequantized data
        """
        scale = params['scale']
        zero_point = params['zero_point']
        
        # Convert back to original range
        dequantized = (quantized_data.astype(np.float32) - zero_point) / scale
        
        return dequantized


def demo_essential_encoder():
    """Demonstrate the essential encoder functionality."""
    print("=" * 80)
    print("ESSENTIAL ENCODER DEMONSTRATION")
    print("=" * 80)
    
    # Create encoder instance
    encoder = EssentialEncoder()
    neural_encoder = NeuralEncoder()
    quantized_encoder = QuantizedEncoder(bits=8)
    
    print("[OK] Encoders initialized")
    
    # Test data
    test_data = {
        "telemetry": [1.2, 3.4, 5.6, 7.8],
        "performance": {"fps": 60.5, "temperature": 75.2},
        "config": {"setting1": True, "setting2": 42}
    }
    
    # Test different encoding types
    encoding_types = [
        EncodingType.BINARY,
        EncodingType.BASE64,
        EncodingType.JSON,
        EncodingType.COMPRESSED,
        EncodingType.NEURAL,
        EncodingType.HEX
    ]
    
    print("\n--- Encoding Tests ---")
    for enc_type in encoding_types:
        try:
            result = encoder.encode(test_data, enc_type)
            print(f"  {enc_type.value}: Original={result.size_original}B, "
                  f"Encoded={result.size_encoded}B, "
                  f"Ratio={result.compression_ratio:.2f}x, "
                  f"Time={result.encoding_time:.4f}s")
        except Exception as e:
            print(f"  {enc_type.value}: ERROR - {e}")
    
    # Test neural encoding
    print("\n--- Neural Encoding Tests ---")
    neural_features = [1.0, 2.5, 3.7, 4.2, 5.1]
    encoded_neural = neural_encoder.encode_features(neural_features)
    print(f"  Neural encoding: {len(neural_features)} -> {encoded_neural.shape}")
    print(f"  Data range: [{encoded_neural.min():.3f}, {encoded_neural.max():.3f}]")
    
    # Test categorical encoding
    categories = ["cpu", "gpu", "memory", "cpu", "disk"]
    encoded_categorical = neural_encoder.encode_categorical(categories)
    print(f"  Categorical encoding: {len(categories)} -> {encoded_categorical.shape}")
    
    # Test sequence encoding
    sequence = ["state1", "state2", "state3", "state1"]
    encoded_sequence = neural_encoder.encode_sequence(sequence, max_length=8)
    print(f"  Sequence encoding: {len(sequence)} -> {encoded_sequence.shape}")
    
    # Test quantization
    print("\n--- Quantization Tests ---")
    float_data = np.random.random((10, 5)).astype(np.float32) * 100
    quantized_data, params = quantized_encoder.quantize(float_data)
    dequantized_data = quantized_encoder.dequantize(quantized_data, params)
    
    print(f"  Original shape: {float_data.shape}, dtype: {float_data.dtype}")
    print(f"  Quantized shape: {quantized_data.shape}, dtype: {quantized_data.dtype}")
    print(f"  Quantization error (MSE): {np.mean((float_data - dequantized_data)**2):.6f}")
    
    # Test integrity verification
    print("\n--- Integrity Verification ---")
    result, hash_val = encoder.encode_with_hash(test_data, EncodingType.COMPRESSED)
    is_valid = encoder.verify_integrity(result.data, hash_val)
    print(f"  Integrity check: {'PASS' if is_valid else 'FAIL'}")
    
    # Show encoding statistics
    print("\n--- Encoding Statistics ---")
    stats = encoder.get_stats()
    for enc_type, stat in stats.items():
        print(f"  {enc_type}: {stat['total_encoded']} encodings, "
              f"avg ratio: {stat['avg_compression_ratio']:.2f}x, "
              f"avg time: {stat['avg_encoding_time']:.4f}s")
    
    print("\n" + "=" * 80)
    print("ESSENTIAL ENCODER DEMONSTRATION COMPLETE")
    print("Encoder provides multiple encoding strategies optimized for different use cases")
    print("including neural network processing, compression, and data integrity")
    print("=" * 80)


if __name__ == "__main__":
    demo_essential_encoder()