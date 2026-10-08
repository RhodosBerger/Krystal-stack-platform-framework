#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: JANET BYTECODE PROFILER & BINARY ALERT DECODER (BRIDGE)
==============================================================================
Module: krystal_kernel/janet_bytecode_decoder.py
Description: Python bridge and execution engine for Janet bytecode profiling,
             binary instruction decoding, dual-representation hex logging,
             and structured kernel alert ("hlásenie") synthesis.

Key Functions:
  - Binary parsing of 64-bit aligned KSYN instruction streams.
  - Dual-representation: Hexadecimal trace log tied directly to raw binary.
  - Alert synthesis: Decodes binary into alerts with severity levels (:INFO, :WARNING, :OPTIMIZED).
  - SVG vector profile rendering: Visual blueprint of power, domain, and pipeline stages.
  - Terminal ASCII command map rendering.

System Invariant: VITAL_MAX_HP = 6.
Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import sys
import struct
from typing import Dict, Any, List, Union

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

VITAL_MAX_HP: int = 6
KSYN_MAGIC: bytes = b"KSYN"

# Opcode dictionary matching krystal_janet/bytecode_profiler_and_alert_decoder.janet
OPCODES: Dict[int, Dict[str, Any]] = {
    0x00: {"name": "OP_HALT", "desc": "Zastavenie výpočtového vlákna", "domain": "core", "power_w": 0.5},
    0x01: {"name": "OP_VITAL_ASSERT_HP", "desc": "Overenie invariantu VITAL_MAX_HP == 6", "domain": "integrity", "power_w": 1.2},
    0x02: {"name": "OP_TERRAIN_MULTIOCTAVE", "desc": "Procedurálny terénny fraktál (fBm)", "domain": "compute", "power_w": 18.5},
    0x03: {"name": "OP_SDF_CHALICE", "desc": "SDF Raymarching: Alchymistický kalich", "domain": "graphics", "power_w": 14.0},
    0x04: {"name": "OP_SDF_ATHAME", "desc": "SDF Raymarching: Rituálna dýka", "domain": "graphics", "power_w": 14.0},
    0x05: {"name": "OP_URBAN_EXTRUDE_SPIRE", "desc": "Morfologická extrúzia veží z máp", "domain": "geometry", "power_w": 12.0},
    0x06: {"name": "OP_BALLISTICS_MORTAR", "desc": "Disperzná balistika mínometu", "domain": "physics", "power_w": 9.5},
    0x07: {"name": "OP_BULLET_TIME_DILATE", "desc": "Časová dilatácia a Coxeterovo zrkadlo", "domain": "spacetime", "power_w": 15.0},
    0x08: {"name": "OP_BAYER_DITHER_SAMPLE", "desc": "Bayerov poltónový raster a luma", "domain": "raster", "power_w": 6.0},
    0x09: {"name": "OP_COXETER_DIHEDRAL", "desc": "Dihedrálne Coxeterovo zrkadlenie", "domain": "math", "power_w": 11.0},
    0x0A: {"name": "OP_METABOLIC_PULSE", "desc": "Metabolický pulz kmeňa (fáza Beta)", "domain": "governor", "power_w": 3.5},
    0x0B: {"name": "OP_LEDGER_MINT_CREDITS", "desc": "Emisia kreditov suverénneho kmeňa", "domain": "economy", "power_w": 2.0},
    0x0C: {"name": "OP_PREFETCH_L1_STREAM", "desc": "K-ISA Špekulatívny prefetch do L1", "domain": "memory", "power_w": 4.0},
    0x0D: {"name": "OP_BARRIER_L3_SYNC", "desc": "Pamäťová bariéra a synchronizácia L3", "domain": "memory", "power_w": 5.0},
    0xA1: {"name": "K_SPEC_PREFETCH_UMA", "desc": "Odomknutá UMA tenzorová predpríprava", "domain": "kisa", "power_w": 8.0},
    0xA2: {"name": "K_SPEC_INTERPOLATE_FRAME", "desc": "Špekulatívna interpolácia 120Hz medzisnímku", "domain": "kisa", "power_w": 16.0},
    0xA3: {"name": "K_SAFE_VOLT_CLAMP", "desc": "Arrheniusov napäťový strop (<=1.02V)", "domain": "kisa", "power_w": 1.0},
    0xA4: {"name": "K_FALLBACK_REVERT", "desc": "Rollback stavu s nulovou réžiou", "domain": "kisa", "power_w": 1.5},
    0xA5: {"name": "K_FUSE_INT8_DP4A", "desc": "Zlúčený DP4A INT8 dot-product na Iris Xe", "domain": "kisa", "power_w": 22.0},
    0xA6: {"name": "K_VERIFY_INVARIANT_HP", "desc": "Hardvérový zámok integrity HP = 6", "domain": "kisa", "power_w": 0.8},
}


class JanetBytecodeDecoder:
    """Bridge for parsing KSYN binary streams, formatting hex logs, and synthesizing alert reports."""

    @staticmethod
    def encode_ksyn_stream(instructions: List[Dict[str, Any]], version: int = 0x0100, seed: int = 1337) -> bytes:
        """Encodes an instruction list into a raw KSYN binary byte stream."""
        header = struct.pack(">4sHH", KSYN_MAGIC, version, seed)
        words = bytearray(header)
        for inst in instructions:
            opcode = int(inst.get("opcode", 0)) & 0xFF
            flags = int(inst.get("flags", 0)) & 0xFF
            param1 = int(inst.get("param1", 0)) & 0xFFFF
            param2 = int(inst.get("param2", 0)) & 0xFFFFFFFF
            words.extend(struct.pack(">BBHI", opcode, flags, param1, param2))
        return bytes(words)

    @staticmethod
    def decode_binary_word(buf: bytes, offset: int) -> Dict[str, Any]:
        """Decodes an 8-byte instruction word from a binary buffer."""
        if offset + 8 > len(buf):
            raise ValueError(f"Nedostatočný počet bajtov na offsete {offset}")
        opcode, flags, param1, param2 = struct.unpack(">BBHI", buf[offset:offset+8])
        info = OPCODES.get(opcode, {
            "name": f"OP_UNKNOWN_0x{opcode:02X}",
            "desc": "Neznáma inštrukcia",
            "domain": "unknown",
            "power_w": 1.0
        })
        return {
            "offset": offset,
            "opcode": opcode,
            "opcode_hex": f"0x{opcode:02X}",
            "name": info["name"],
            "description": info["desc"],
            "domain": info["domain"],
            "flags": flags,
            "param1": param1,
            "param2": param2,
            "estimated_power_w": info["power_w"],
            "raw_word_hex": buf[offset:offset+8].hex().upper()
        }

    @classmethod
    def parse_ksyn_binary_stream(cls, raw_bytes: bytes) -> Dict[str, Any]:
        """Parses a full KSYN binary stream (magic header, version, seed, instruction words)."""
        if len(raw_bytes) < 8:
            return {
                "status": "error",
                "message": "Binárka je príliš krátka (menej ako 8 bajtov hlavičky)",
                "vital_max_hp": VITAL_MAX_HP
            }

        magic = raw_bytes[:4]
        if magic != KSYN_MAGIC:
            return {
                "status": "error",
                "message": f"Neplatná hlavička: očakávané 'KSYN', nájdené '{magic.decode('latin1', errors='replace')}'",
                "vital_max_hp": VITAL_MAX_HP
            }

        version_num, seed = struct.unpack(">HH", raw_bytes[4:8])
        version_str = f"{(version_num >> 8)}.{version_num & 0xFF}"

        instructions: List[Dict[str, Any]] = []
        cur_off = 8
        while cur_off + 8 <= len(raw_bytes):
            instructions.append(cls.decode_binary_word(raw_bytes, cur_off))
            cur_off += 8

        # Dual-representation: Hexadecimal trace log
        hex_log_lines = []
        for i in range(0, len(raw_bytes), 16):
            chunk = raw_bytes[i:i+16]
            hex_part = " ".join(f"{b:02X}" for b in chunk)
            ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
            hex_log_lines.append(f"{i:04X}: {hex_part:<48} | {ascii_part}")

        return {
            "status": "ok",
            "magic": magic.decode("latin1"),
            "version": version_str,
            "seed": seed,
            "total_bytes": len(raw_bytes),
            "total_words": len(instructions),
            "instructions": instructions,
            "hex_trace_log": "\n".join(hex_log_lines),
            "vital_max_hp": VITAL_MAX_HP
        }

    @classmethod
    def synthesize_alert_report(cls, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decodes binary stream and produces a human/system readable status report ('hlásenie')."""
        if parsed_data.get("status") == "error":
            return {
                "severity": "CRITICAL",
                "title": "CHYBA DEKÓDOVANIA BINÁRNEHO STREAMU",
                "summary": parsed_data.get("message", "Neznáma chyba"),
                "vital_max_hp": VITAL_MAX_HP,
                "alerts": [{
                    "severity": "CRITICAL",
                    "code": "CORRUPT_BINARY_STREAM",
                    "message": parsed_data.get("message", "Binárny stream je poškodený alebo má neplatnú hlavičku.")
                }]
            }

        insts = parsed_data.get("instructions", [])
        total_pwr = sum(inst.get("estimated_power_w", 0.0) for inst in insts)
        has_hp_assert = any(inst.get("opcode") in (0x01, 0xA6) for inst in insts)
        has_kisa_spec = any(inst.get("opcode") == 0xA2 for inst in insts)
        has_volt_clamp = any(inst.get("opcode") == 0xA3 for inst in insts)
        alerts: List[Dict[str, Any]] = []

        # 1. Validácia integrity VITAL_MAX_HP
        if has_hp_assert:
            alerts.append({
                "severity": "INFO",
                "code": "INVARIANT_VERIFIED",
                "message": f"Systémový invariant VITAL_MAX_HP = {VITAL_MAX_HP} úspešne overený v inštrukčnom toku."
            })
        else:
            alerts.append({
                "severity": "WARNING",
                "code": "MISSING_INVARIANT_ASSERT",
                "message": f"Inštrukčný tok neobsahuje explicitnú kontrolu VITAL_MAX_HP = {VITAL_MAX_HP}!"
            })

        # 2. Analýza spotreby a napätia
        if total_pwr > 85.0:
            alerts.append({
                "severity": "WARNING",
                "code": "HIGH_POWER_ENVELOPE",
                "message": f"Kumulatívna spotreba inštrukčného bloku ({total_pwr:.1f} W) prekračuje odporúčaný envelope. Odporúčaný K_SAFE_VOLT_CLAMP."
            })
        else:
            alerts.append({
                "severity": "INFO",
                "code": "POWER_ENVELOPE_SAFE",
                "message": f"Spotreba v bezpečnom pásme: {total_pwr:.1f} W (< 85W ceiling)."
            })

        # 3. Špekulatívna akcelerácia K-ISA
        if has_kisa_spec:
            alerts.append({
                "severity": "OPTIMIZED",
                "code": "KISA_SPECULATIVE_ACTIVE",
                "message": "K-ISA Špekulatívna interpolácia medzisnímku aktívna: plynulých 120 Hz zabezpečených."
            })

        if has_volt_clamp:
            alerts.append({
                "severity": "OPTIMIZED",
                "code": "VOLT_CLAMP_ACTIVE",
                "message": "Arrheniusov napäťový strop aktívny (<= 1.02V) - degradácia kremíka potlačená."
            })

        overall_severity = "WARNING" if any(a["severity"] == "WARNING" for a in alerts) else "NOMINAL"
        if any(a["severity"] == "CRITICAL" for a in alerts):
            overall_severity = "CRITICAL"

        decoded_commands = [
            f"[{inst['opcode_hex']}] {inst['name']:<24} -> {inst['description']} (P1:{inst['param1']}, P2:{inst['param2']}, {inst['estimated_power_w']:.1f}W)"
            for inst in insts
        ]

        return {
            "severity": overall_severity,
            "title": f"HLÁSENIE KERNELU // JANET DEKÓDOVANÁ BINÁRKA (Verzia {parsed_data.get('version')}, Seed {parsed_data.get('seed')})",
            "instruction_count": len(insts),
            "total_power_watts": round(total_pwr, 2),
            "alerts": alerts,
            "decoded_commands": decoded_commands,
            "hex_trace_log": parsed_data.get("hex_trace_log", ""),
            "vital_max_hp": VITAL_MAX_HP
        }

    @staticmethod
    def render_execution_profile_svg(parsed_data: Dict[str, Any], width: int = 900, height: int = 420) -> str:
        """Generates an SVG vector graphic blueprint illustrating the decoded binary command profile."""
        insts = parsed_data.get("instructions", [])
        count = max(1, len(insts))
        bar_w = (width - 120) / count
        bars = []

        domain_colors = {
            "kisa": "#c084fc",
            "compute": "#00f0ff",
            "graphics": "#00ff88",
            "integrity": "#f59e0b",
            "geometry": "#38bdf8",
            "physics": "#fb7185",
            "spacetime": "#e879f9",
            "raster": "#a3e635",
            "math": "#22d3ee",
            "governor": "#f43f5e",
            "economy": "#fbbf24",
            "memory": "#818cf8"
        }

        for idx, inst in enumerate(insts):
            pwr = inst.get("estimated_power_w", 1.0)
            bar_h = (pwr / 25.0) * 180.0
            x = 60 + idx * bar_w
            y = 280 - bar_h
            color = domain_colors.get(inst.get("domain"), "#8892b0")
            bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 4:.1f}" height="{bar_h:.1f}" '
                f'fill="{color}" opacity="0.85" rx="3" stroke="#ffffff" stroke-width="0.5"/>\n'
                f'<text x="{x + bar_w/2:.1f}" y="300" fill="#94a3b8" font-family="monospace" '
                f'font-size="9" transform="rotate(45 {x + bar_w/2:.1f},300)">{inst.get("opcode_hex")}</text>'
            )

        bars_markup = "\n".join(bars)
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" style="background:#07090e; border:1px solid #1e293b; border-radius:8px;">
  <defs>
    <linearGradient id="grid" width="30" height="30" patternUnits="userSpaceOnUse">
      <path d="M 30 0 L 0 0 0 30" fill="none" stroke="rgba(255,255,255,0.03)" stroke-width="1"/>
    </linearGradient>
  </defs>
  <rect width="100%" height="100%" fill="url(#grid)" />
  <text x="30" y="36" fill="#00f0ff" font-family="sans-serif" font-weight="bold" font-size="16">KRYSTAL-STACK // JANET BINARY EXECUTION PROFILE (KSYN)</text>
  <text x="30" y="56" fill="#8892b0" font-family="monospace" font-size="11">Vykreslenie profilu inštrukčných slov zarovnaných na 64-bitov // VITAL_MAX_HP = {VITAL_MAX_HP}</text>
  
  <!-- Y Axis Guidelines -->
  <line x1="50" y1="280" x2="{width - 40}" y2="280" stroke="#334155" stroke-width="1"/>
  <line x1="50" y1="190" x2="{width - 40}" y2="190" stroke="rgba(255,255,255,0.08)" stroke-dasharray="4"/>
  <line x1="50" y1="100" x2="{width - 40}" y2="100" stroke="rgba(255,255,255,0.08)" stroke-dasharray="4"/>
  <text x="15" y="284" fill="#64748b" font-family="monospace" font-size="10">0W</text>
  <text x="15" y="194" fill="#64748b" font-family="monospace" font-size="10">12W</text>
  <text x="15" y="104" fill="#64748b" font-family="monospace" font-size="10">25W</text>
  
  <!-- Instruction Bars -->
  {bars_markup}

  <!-- Legend -->
  <circle cx="60" cy="385" r="4" fill="#c084fc"/><text x="70" y="388" fill="#cbd5e1" font-family="sans-serif" font-size="10">K-ISA Špekulácia</text>
  <circle cx="180" cy="385" r="4" fill="#00f0ff"/><text x="190" y="388" fill="#cbd5e1" font-family="sans-serif" font-size="10">Výpočet (fBm)</text>
  <circle cx="280" cy="385" r="4" fill="#00ff88"/><text x="290" y="388" fill="#cbd5e1" font-family="sans-serif" font-size="10">SDF Raymarching</text>
  <circle cx="410" cy="385" r="4" fill="#f59e0b"/><text x="420" y="388" fill="#cbd5e1" font-family="sans-serif" font-size="10">Integrita (Assert HP)</text>
</svg>"""
        return svg

    @classmethod
    def render_terminal_ascii_profile(cls, parsed_data: Dict[str, Any]) -> str:
        """Renders a high-density ASCII table and opcode map for terminal visualization."""
        insts = parsed_data.get("instructions", [])
        lines = [
            "=" * 80,
            "  KRYSTAL-STACK: JANET BINÁRNE MAPOVANIE & DEKÓDOVANÉ HLÁSENIE (KSYN)",
            "=" * 80,
            f"  Hlavička: {parsed_data.get('magic', 'KSYN')} | Verzia: {parsed_data.get('version', '1.0')} | Seed: {parsed_data.get('seed', 0)} | Počet inštrukcií: {len(insts)}",
            "-" * 80,
            "  IDX | HEX  | NÁZOV PRÍKAZU            | OBLASŤ     | VÝKON  | POPIS",
            "-" * 80,
        ]
        for i, w in enumerate(insts):
            lines.append(
                f"  {i:02d}  | {w['opcode_hex']:<4} | {w['name']:<24} | {w['domain']:<10} | {w['estimated_power_w']:4.1f} W | {w['description']}"
            )
        lines.append("=" * 80)
        return "\n".join(lines)

    @classmethod
    def process_hex_or_binary_input(cls, data: Union[str, bytes]) -> Dict[str, Any]:
        """Convenience dispatcher accepting either raw bytes or hex-string."""
        if isinstance(data, str):
            clean_hex = "".join(c for c in data if c in "0123456789abcdefABCDEF")
            raw_bytes = bytes.fromhex(clean_hex)
        elif isinstance(data, (bytes, bytearray)):
            raw_bytes = bytes(data)
        else:
            raise TypeError("Vstup musí byť hexadecimálny reťazec alebo raw bytes")

        parsed = cls.parse_ksyn_binary_stream(raw_bytes)
        report = cls.synthesize_alert_report(parsed)
        return {
            "parsed": parsed,
            "report": report
        }


def generate_canonical_demo_stream() -> bytes:
    """Generates a reference KSYN binary stream exercising key engine and K-ISA opcodes."""
    instructions = [
        {"opcode": 0x01, "flags": 0x00, "param1": 6, "param2": 0},          # OP_VITAL_ASSERT_HP
        {"opcode": 0x0A, "flags": 0x01, "param1": 2, "param2": 100},        # OP_METABOLIC_PULSE
        {"opcode": 0x0C, "flags": 0x00, "param1": 1024, "param2": 4096},    # OP_PREFETCH_L1_STREAM
        {"opcode": 0x02, "flags": 0x03, "param1": 8, "param2": 256},        # OP_TERRAIN_MULTIOCTAVE
        {"opcode": 0x03, "flags": 0x00, "param1": 512, "param2": 512},      # OP_SDF_CHALICE
        {"opcode": 0x09, "flags": 0x00, "param1": 16, "param2": 64},        # OP_COXETER_DIHEDRAL
        {"opcode": 0xA1, "flags": 0x01, "param1": 2048, "param2": 16384},   # K_SPEC_PREFETCH_UMA
        {"opcode": 0xA2, "flags": 0x02, "param1": 120, "param2": 1},        # K_SPEC_INTERPOLATE_FRAME
        {"opcode": 0xA3, "flags": 0x00, "param1": 1020, "param2": 0},       # K_SAFE_VOLT_CLAMP (1.02V)
        {"opcode": 0xA5, "flags": 0x04, "param1": 256, "param2": 512},      # K_FUSE_INT8_DP4A
        {"opcode": 0xA6, "flags": 0x00, "param1": 6, "param2": 0},          # K_VERIFY_INVARIANT_HP
        {"opcode": 0x0D, "flags": 0x00, "param1": 0, "param2": 0},          # OP_BARRIER_L3_SYNC
        {"opcode": 0x00, "flags": 0x00, "param1": 0, "param2": 0},          # OP_HALT
    ]
    return JanetBytecodeDecoder.encode_ksyn_stream(instructions, version=0x0100, seed=1337)


if __name__ == "__main__":
    demo_stream = generate_canonical_demo_stream()
    print(f"Generated KSYN demo stream: {len(demo_stream)} bytes")
    res = JanetBytecodeDecoder.process_hex_or_binary_input(demo_stream)
    print("\n" + JanetBytecodeDecoder.render_terminal_ascii_profile(res["parsed"]))
    print(f"\nReport Title: {res['report']['title']}")
    print(f"Overall Severity: {res['report']['severity']}")
    print(f"Total Power: {res['report']['total_power_watts']} W")
    print(f"Alerts ({len(res['report']['alerts'])}):")
    for a in res['report']['alerts']:
        print(f"  [{a['severity']}] {a['code']}: {a['message']}")
    print("\nHexadecimal Trace Log:")
    print(res["report"]["hex_trace_log"])
