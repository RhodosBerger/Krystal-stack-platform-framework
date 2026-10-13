"""GENERATED from native/spec/binary_layouts.yaml - DO NOT EDIT."""
import ctypes

class kr_resource_evidence_envelope(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ('magic', ctypes.c_uint8 * 4),
        ('abi_version', ctypes.c_uint16),
        ('flags', ctypes.c_uint16),
        ('state_version', ctypes.c_uint64),
        ('observed_ns', ctypes.c_uint64),
        ('lease_ttl_ns', ctypes.c_uint64),
        ('value', ctypes.c_double),
        ('uncertainty', ctypes.c_double),
        ('domain_id', ctypes.c_uint32),
        ('epistemic_kind', ctypes.c_uint8),
        ('temporal_mode', ctypes.c_uint8),
        ('ownership', ctypes.c_uint8),
        ('reserved0', ctypes.c_uint8),
        ('crc32', ctypes.c_uint32),
        ('reserved1', ctypes.c_uint32),
    ]

SIZE = 64
OFFSETS = {'magic': 0, 'abi_version': 4, 'flags': 6, 'state_version': 8, 'observed_ns': 16, 'lease_ttl_ns': 24, 'value': 32, 'uncertainty': 40, 'domain_id': 48, 'epistemic_kind': 52, 'temporal_mode': 53, 'ownership': 54, 'reserved0': 55, 'crc32': 56, 'reserved1': 60}
