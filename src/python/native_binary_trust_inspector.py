"""
CrystalSec 3.0 Phase 5.3D: Native Binary Trust Inspector.

Audits Windows Authenticode, Code Integrity (UMCI / Device Guard), and
LoadLibraryExW outcomes.

Strictly separates:
    provenance_verified (hash matches manifest)
    from
    execution_trusted (Windows kernel and security policy allow image execution).
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import subprocess
from ctypes import wintypes
from dataclasses import asdict, dataclass
from typing import Any


# WinTrust GUID & Structures
class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_ulong),
        ("Data2", ctypes.c_ushort),
        ("Data3", ctypes.c_ushort),
        ("Data4", ctypes.c_ubyte * 8),
    ]


_WINTRUST_ACTION_GENERIC_VERIFY_V2 = _GUID(
    0x00AAC56B,
    0xCD44,
    0x11D0,
    (ctypes.c_ubyte * 8)(0x8C, 0xC2, 0x00, 0xC0, 0x4F, 0xC2, 0x95, 0xEE),
)


class _WINTRUST_FILE_INFO(ctypes.Structure):
    _fields_ = [
        ("cbStruct", wintypes.DWORD),
        ("pcwszFilePath", wintypes.LPCWSTR),
        ("hFile", wintypes.HANDLE),
        ("pgKnownSubject", ctypes.c_void_p),
    ]


class _WINTRUST_DATA(ctypes.Structure):
    _fields_ = [
        ("cbStruct", wintypes.DWORD),
        ("pPolicyCallbackData", ctypes.c_void_p),
        ("pSIPClientData", ctypes.c_void_p),
        ("dwUIChoice", wintypes.DWORD),
        ("fdwRevocationChecks", wintypes.DWORD),
        ("dwUnionChoice", wintypes.DWORD),
        ("pFile", ctypes.POINTER(_WINTRUST_FILE_INFO)),
        ("dwStateAction", wintypes.DWORD),
        ("hWVTStateData", wintypes.HANDLE),
        ("pwszURLReference", wintypes.LPCWSTR),
        ("dwProvFlags", wintypes.DWORD),
        ("dwUIContext", wintypes.DWORD),
        ("pSignatureSettings", ctypes.c_void_p),
    ]


@dataclass
class BinaryTrustReport:
    file_path: str
    sha256: str
    authenticode_status: str
    certificate_subject: str | None
    certificate_thumbprint: str | None
    signature_validity: bool
    win_verify_trust_result: str
    code_integrity_load_outcome: str
    loader_error_code: int
    policy_classification: str
    provenance_verified: bool
    execution_trusted: bool
    can_execute_native: bool
    diagnostic_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NativeBinaryTrustInspector:
    """Audits file integrity, signature validity, and Windows UMCI loader acceptance."""

    @staticmethod
    def query_umci_status() -> str:
        if platform.system() != "Windows":
            return "NON_WINDOWS"
        try:
            cmd = [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance -ClassName Win32_DeviceGuard -Namespace root\\Microsoft\\Windows\\DeviceGuard).UsermodeCodeIntegrityPolicyEnforcementStatus",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            out_code = res.stdout.strip()
            if out_code == "2":
                return "UMCI_ENFORCED"
            elif out_code == "1":
                return "UMCI_AUDIT"
            elif out_code == "0":
                return "UMCI_DISABLED"
            return f"UNKNOWN_CODE_{out_code}"
        except Exception:
            return "UNKNOWN"

    @staticmethod
    def inspect(file_path: str, manifest_path: str | None = None) -> BinaryTrustReport:
        abs_path = os.path.abspath(file_path) if file_path else ""
        host_policy = NativeBinaryTrustInspector.query_umci_status()

        # Default fallback values
        sha256_val = ""
        auth_status = "FILE_NOT_FOUND"
        cert_subject = None
        cert_thumb = None
        sig_valid = False
        wvt_result_str = "0xFFFFFFFF"
        load_outcome = "FILE_NOT_FOUND"
        loader_err = 126
        prov_verified = False
        exec_trusted = False

        if not os.path.exists(abs_path):
            diag = (
                f"OBSERVED: File not found at '{abs_path}' (WinError 126). "
                f"HYPOTHESIS: Build toolchain not present on host. "
                f"UNKNOWN: Actual Code Integrity load verdict cannot be evaluated on non-existent binary."
            )
            return BinaryTrustReport(
                file_path=abs_path,
                sha256="",
                authenticode_status="FILE_NOT_FOUND",
                certificate_subject=None,
                certificate_thumbprint=None,
                signature_validity=False,
                win_verify_trust_result="0xFFFFFFFF",
                code_integrity_load_outcome="FILE_NOT_FOUND",
                loader_error_code=126,
                policy_classification=host_policy,
                provenance_verified=False,
                execution_trusted=False,
                can_execute_native=False,
                diagnostic_note=diag,
            )

        # 1. Compute SHA-256
        h = hashlib.sha256()
        with open(abs_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        sha256_val = h.hexdigest().lower()

        # 2. Check manifest provenance
        if manifest_path and os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                expected_hash = manifest.get("dll_sha256")
                if expected_hash and expected_hash.lower() == sha256_val:
                    prov_verified = True
            except Exception:
                prov_verified = False

        # 3. Policy classification from host
        policy_class = host_policy

        # 4. Check Authenticode Signature via WinVerifyTrust
        if platform.system() == "Windows":
            try:
                wintrust = ctypes.windll.wintrust
                wintrust.WinVerifyTrust.argtypes = [
                    wintypes.HWND,
                    ctypes.POINTER(_GUID),
                    ctypes.c_void_p,
                ]
                wintrust.WinVerifyTrust.restype = wintypes.LONG

                file_info = _WINTRUST_FILE_INFO(
                    cbStruct=ctypes.sizeof(_WINTRUST_FILE_INFO),
                    pcwszFilePath=abs_path,
                    hFile=None,
                    pgKnownSubject=None,
                )
                trust_data = _WINTRUST_DATA(
                    cbStruct=ctypes.sizeof(_WINTRUST_DATA),
                    pPolicyCallbackData=None,
                    pSIPClientData=None,
                    dwUIChoice=2,  # WTD_UI_NONE
                    fdwRevocationChecks=0,  # WTD_REVOKE_NONE
                    dwUnionChoice=1,  # WTD_CHOICE_FILE
                    pFile=ctypes.pointer(file_info),
                    dwStateAction=0,  # WTD_STATEACTION_IGNORE
                    hWVTStateData=None,
                    pwszURLReference=None,
                    dwProvFlags=0x00000080 | 0x00000010,  # WTD_CACHE_ONLY_URL_RETRIEVAL | WTD_REVOCATION_CHECK_NONE
                    dwUIContext=0,
                    pSignatureSettings=None,
                )
                wvt_res = wintrust.WinVerifyTrust(
                    None,
                    ctypes.byref(_WINTRUST_ACTION_GENERIC_VERIFY_V2),
                    ctypes.byref(trust_data),
                )
                wvt_result_str = hex(wvt_res & 0xFFFFFFFF)

                if wvt_res == 0:
                    auth_status = "SIGNED_VALID"
                    sig_valid = True
                elif (wvt_res & 0xFFFFFFFF) == 0x800B0109:
                    auth_status = "SIGNED_UNTRUSTED_ROOT"
                    sig_valid = False
                elif (wvt_res & 0xFFFFFFFF) == 0x800B0100:
                    auth_status = "UNSIGNED"
                    sig_valid = False
                else:
                    auth_status = f"SIGNATURE_INVALID_{wvt_result_str}"
                    sig_valid = False
            except Exception as e:
                auth_status = f"VERIFY_ERROR_{type(e).__name__}"
                sig_valid = False

            # 5. Controlled Loader Probe via LoadLibraryExW (LOAD_WITH_ALTERED_SEARCH_PATH)
            try:
                kernel32 = ctypes.windll.kernel32
                kernel32.LoadLibraryExW.argtypes = [
                    wintypes.LPCWSTR,
                    wintypes.HANDLE,
                    wintypes.DWORD,
                ]
                kernel32.LoadLibraryExW.restype = ctypes.c_void_p
                kernel32.FreeLibrary.argtypes = [ctypes.c_void_p]
                kernel32.FreeLibrary.restype = wintypes.BOOL

                # Flag 0x00000008 = LOAD_WITH_ALTERED_SEARCH_PATH
                h_mod = kernel32.LoadLibraryExW(abs_path, None, 0x00000008)
                if h_mod:
                    load_outcome = "LOADED_SUCCESS"
                    loader_err = 0
                    exec_trusted = True
                    kernel32.FreeLibrary(h_mod)
                else:
                    loader_err = kernel32.GetLastError()
                    if loader_err == 4551:
                        load_outcome = "BLOCKED_BY_UMCI_POLICY"
                    elif loader_err == 126:
                        load_outcome = "MODULE_NOT_FOUND"
                    elif loader_err == 193:
                        load_outcome = "BAD_EXE_FORMAT"
                    else:
                        load_outcome = f"LOAD_FAILED_ERR_{loader_err}"
                    exec_trusted = False
            except Exception as exc:
                load_outcome = f"LOAD_EXCEPTION_{type(exc).__name__}"
                loader_err = -1
                exec_trusted = False

        can_exec = prov_verified and exec_trusted

        return BinaryTrustReport(
            file_path=abs_path,
            sha256=sha256_val,
            authenticode_status=auth_status,
            certificate_subject=cert_subject,
            certificate_thumbprint=cert_thumb,
            signature_validity=sig_valid,
            win_verify_trust_result=wvt_result_str,
            code_integrity_load_outcome=load_outcome,
            loader_error_code=loader_err,
            policy_classification=policy_class,
            provenance_verified=prov_verified,
            execution_trusted=exec_trusted,
            can_execute_native=can_exec,
        )
