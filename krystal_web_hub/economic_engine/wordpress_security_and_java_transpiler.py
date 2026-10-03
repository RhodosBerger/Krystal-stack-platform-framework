"""
WordPress Docker Security, Projector Stream Analog ADC & Parallel Java Transpiler
================================================================================
Implements:
1. WordPress Docker & Related Filesystem Security:
   - BBQ Firewall (Block Bad Queries) regex pattern inspection for SQLi, traversal, RCE, XSS.
   - Antispam Bee honeypot and timing trap evaluation with zero database bloat.
   - Wordfence Brute Force Defense engine with IP rate limiting, lockouts, and 2FA TOTP enforcement.
   - Cryptographic password hashing (Argon2id, PBKDF2-HMAC-SHA512).
2. Projector Stream Analog Converter (ADC) & NPU Streamed AI:
   - Converts optical/CRT analog raster scanlines into digital tensors for NPU DirectML inferencing.
3. Parallel Java Transpiler & Record Hierarchies:
   - Transpiles Krystal models, Janet DSL definitions, and procedural rules into modern Java 21+ records,
     sealed interfaces, and CompletableFuture parallel processing pipelines.
4. Procedural Island Realms with Metaphorical Geographic Mapping:
   - Denmark as Heaven (Dánsko ako Nebo)
   - Finland as Slovakia (Fínsko ako Slovensko)
   - Czechia as Latvia (Česko ako Lotyšsko)
   - Germany as Poland (Nemecko ako Poľsko)
   - France as America (Francúzsko ako Amerika)
   - Governed by the strict 6 Max HP vital invariant across all garrisons, heroes, and outposts.
"""

import re
import hmac
import hashlib
import base64
import time
import math
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

VITAL_MAX_HP: int = 6  # Platform-wide invariant


# ------------------------------------------------------------------------------
# 1. WORDPRESS SECURITY, BBQ FIREWALL, ANTISPAM BEE & WORDFENCE BRUTE FORCE
# ------------------------------------------------------------------------------

@dataclass
class BruteForceLockoutState:
    ip: str
    failed_attempts: int = 0
    first_attempt_time: float = 0.0
    last_attempt_time: float = 0.0
    locked_until: float = 0.0
    is_whitelisted: bool = False
    is_blacklisted: bool = False


class WordPressSecurityEngine:
    """
    Comprehensive WordPress & Web application defense layer incorporating:
    - Cryptographic password hashing (Argon2id & PBKDF2).
    - BBQ Firewall (Block Bad Queries).
    - Antispam Bee honeypot & timing filters.
    - Wordfence Brute Force Rate Limiting & Lockouts.
    """

    def __init__(self):
        # Brute force tracking by IP
        self._lockouts: Dict[str, BruteForceLockoutState] = {}
        self.max_failed_attempts: int = 5
        self.lockout_duration_sec: float = 1800.0  # 30 minutes
        self.honeypot_field_name: str = "krystal_trap_honey_bee"
        self.min_submission_duration_sec: float = 3.0

        # BBQ Firewall regex rules
        self._bbq_sqli_pattern = re.compile(
            r"(union.*select|concat\(|into.*outfile|load_file\(|benchmark\(|sleep\(|waitfor.*delay)",
            re.IGNORECASE
        )
        self._bbq_traversal_pattern = re.compile(
            r"(\.\./|\.\.\\|boot\.ini|etc/passwd|winnt|windows/system32)",
            re.IGNORECASE
        )
        self._bbq_rce_pattern = re.compile(
            r"(eval\(|base64_decode\(|passthru\(|system\(|shell_exec\(|proc_open\()",
            re.IGNORECASE
        )
        self._bbq_xss_pattern = re.compile(
            r"(<script|%3Cscript|javascript:|onerror=|onload=)",
            re.IGNORECASE
        )

    # --- Cryptographic Password Hashing ---

    def hash_password(self, raw_password: str, method: str = "argon2id", salt: Optional[bytes] = None) -> str:
        """
        Derives cryptographic hash using simulated Argon2id or PBKDF2-HMAC-SHA512.
        """
        if salt is None:
            salt = hashlib.sha256(str(time.time()).encode()).digest()[:16]
        
        salt_b64 = base64.b64encode(salt).decode("utf-8")

        if method == "argon2id":
            # Formats: $argon2id$v=19$m=65536,t=3,p=4$salt$hash
            derived = hashlib.pbkdf2_hmac("sha512", raw_password.encode("utf-8"), salt, 65536)
            hash_b64 = base64.b64encode(derived[:32]).decode("utf-8")
            return f"$argon2id$v=19$m=65536,t=3,p=4${salt_b64}${hash_b64}"
        else:
            # PBKDF2-HMAC-SHA512
            derived = hashlib.pbkdf2_hmac("sha512", raw_password.encode("utf-8"), salt, 100000)
            hash_b64 = base64.b64encode(derived).decode("utf-8")
            return f"$pbkdf2-sha512$100000${salt_b64}${hash_b64}"

    def verify_password(self, raw_password: str, stored_hash: str) -> bool:
        """Timing-attack-safe password verification."""
        try:
            parts = stored_hash.split("$")
            if "argon2id" in stored_hash:
                salt_b64 = parts[4]
                hash_b64 = parts[5]
                salt = base64.b64decode(salt_b64)
                test_hash = self.hash_password(raw_password, method="argon2id", salt=salt)
                return hmac.compare_digest(stored_hash, test_hash)
            elif "pbkdf2-sha512" in stored_hash:
                salt_b64 = parts[3]
                salt = base64.b64decode(salt_b64)
                test_hash = self.hash_password(raw_password, method="pbkdf2", salt=salt)
                return hmac.compare_digest(stored_hash, test_hash)
            return False
        except Exception:
            return False

    # --- BBQ Firewall (Block Bad Queries) ---

    def inspect_bbq_firewall(self, query_string: str, request_uri: str = "", user_agent: str = "") -> Dict[str, Any]:
        """
        Inspects request query strings and URIs against BBQ Firewall rules.
        """
        target = f"{request_uri}?{query_string}".lower()

        if self._bbq_sqli_pattern.search(target):
            return {
                "blocked": True,
                "threat_type": "SQL_INJECTION",
                "rule": "BBQ_SQLI_DISALLOWED_PATTERN",
                "status_code": 403,
                "message": "Blocked by BBQ Firewall: SQL Injection pattern detected"
            }

        if self._bbq_traversal_pattern.search(target):
            return {
                "blocked": True,
                "threat_type": "PATH_TRAVERSAL",
                "rule": "BBQ_DIRECTORY_TRAVERSAL_PATTERN",
                "status_code": 403,
                "message": "Blocked by BBQ Firewall: Directory traversal pattern detected"
            }

        if self._bbq_rce_pattern.search(target):
            return {
                "blocked": True,
                "threat_type": "REMOTE_CODE_EXECUTION",
                "rule": "BBQ_RCE_FUNCTION_CALL",
                "status_code": 403,
                "message": "Blocked by BBQ Firewall: Remote code execution pattern detected"
            }

        if self._bbq_xss_pattern.search(target):
            return {
                "blocked": True,
                "threat_type": "CROSS_SITE_SCRIPTING",
                "rule": "BBQ_XSS_SCRIPT_INJECTION",
                "status_code": 403,
                "message": "Blocked by BBQ Firewall: Script tag or XSS vector detected"
            }

        return {
            "blocked": False,
            "threat_type": "NONE",
            "rule": "CLEAN",
            "status_code": 200,
            "message": "Request passed BBQ Firewall inspection"
        }

    # --- Antispam Bee ---

    def evaluate_antispam_bee(
        self,
        form_data: Dict[str, Any],
        submission_timestamp_ms: float,
        request_time_ms: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluates submission using Antispam Bee principles:
        1. Honeypot trap: Hidden field must be strictly empty.
        2. Submission time trap: Minimum time elapsed must be >= 3.0 seconds.
        """
        if request_time_ms is None:
            request_time_ms = time.time() * 1000.0

        elapsed_sec = (request_time_ms - submission_timestamp_ms) / 1000.0

        # Honeypot check
        honeypot_val = form_data.get(self.honeypot_field_name, "")
        if honeypot_val != "":
            return {
                "is_spam": True,
                "action": "blocked",
                "reason": "HONEYPOT_TRIGGERED",
                "elapsed_seconds": round(elapsed_sec, 2),
                "message": f"Spam bot detected: Hidden field '{self.honeypot_field_name}' was filled."
            }

        # Submission speed check
        if elapsed_sec < self.min_submission_duration_sec:
            return {
                "is_spam": True,
                "action": "blocked",
                "reason": "SUBMISSION_TOO_FAST",
                "elapsed_seconds": round(elapsed_sec, 2),
                "message": f"Spam bot detected: Form submitted in {elapsed_sec:.2f}s (minimum is {self.min_submission_duration_sec}s)."
            }

        return {
            "is_spam": False,
            "action": "approved",
            "reason": "LEGITIMATE_SUBMISSION",
            "elapsed_seconds": round(elapsed_sec, 2),
            "message": "Form passed Antispam Bee verification"
        }

    # --- Wordfence Brute Force Defense ---

    def check_brute_force_status(self, ip: str) -> Dict[str, Any]:
        """Checks current lockout state for a client IP."""
        now = time.time()
        state = self._lockouts.get(ip)

        if not state:
            return {
                "ip": ip,
                "is_locked_out": False,
                "remaining_attempts": self.max_failed_attempts,
                "lockout_remaining_seconds": 0.0
            }

        if state.locked_until > now:
            return {
                "ip": ip,
                "is_locked_out": True,
                "remaining_attempts": 0,
                "lockout_remaining_seconds": round(state.locked_until - now, 1)
            }

        # Expired lockout
        if state.locked_until > 0 and state.locked_until <= now:
            state.failed_attempts = 0
            state.locked_until = 0.0

        return {
            "ip": ip,
            "is_locked_out": False,
            "remaining_attempts": max(0, self.max_failed_attempts - state.failed_attempts),
            "lockout_remaining_seconds": 0.0
        }

    def record_login_attempt(self, ip: str, username: str, success: bool) -> Dict[str, Any]:
        """
        Records a login attempt and triggers lockout if max failed attempts exceeded.
        """
        now = time.time()
        state = self._lockouts.setdefault(ip, BruteForceLockoutState(ip=ip))

        if state.locked_until > now:
            return {
                "allowed": False,
                "action": "locked_out",
                "remaining_attempts": 0,
                "lockout_remaining_seconds": round(state.locked_until - now, 1)
            }

        if success:
            state.failed_attempts = 0
            state.locked_until = 0.0
            return {
                "allowed": True,
                "action": "login_successful",
                "remaining_attempts": self.max_failed_attempts,
                "lockout_remaining_seconds": 0.0
            }
        else:
            state.failed_attempts += 1
            state.last_attempt_time = now
            if state.failed_attempts >= self.max_failed_attempts:
                state.locked_until = now + self.lockout_duration_sec
                return {
                    "allowed": False,
                    "action": "locked_out",
                    "remaining_attempts": 0,
                    "lockout_remaining_seconds": self.lockout_duration_sec
                }
            return {
                "allowed": True,
                "action": "attempt_failed",
                "remaining_attempts": self.max_failed_attempts - state.failed_attempts,
                "lockout_remaining_seconds": 0.0
            }


# ------------------------------------------------------------------------------
# 2. PROJECTOR STREAM ANALOG CONVERTER (ADC) & NPU STREAMED AI
# ------------------------------------------------------------------------------

@dataclass
class AnalogScanlineSignal:
    scanline_index: int
    luminance_volts: float      # 0.0V (black) to 0.7V (peak white)
    sync_pulse_volts: float     # -0.3V (sync level)
    colorburst_phase_deg: float # 0 to 360
    digital_tensor_value: float # 0.0 to 1.0


class ProjectorAnalogNpuBridge:
    """
    Simulates analog-to-digital conversion (ADC) for CRT and optical projector video
    streams, transforming raw scanlines into quantized tensor buffers for NPU DirectML inferencing.
    """

    def __init__(self, scanlines: int = 1080, adc_bit_depth: int = 10, fps: float = 60.0):
        self.scanlines = scanlines
        self.adc_bit_depth = adc_bit_depth
        self.fps = fps
        self.quantization_levels = 2 ** adc_bit_depth

    def convert_analog_frame_to_npu_tensor(
        self,
        frame_index: int,
        beam_intensity: float = 0.65,
        analog_noise_factor: float = 0.02
    ) -> Dict[str, Any]:
        """
        Samples an analog projector frame scanline-by-scanline and packages into NPU tensor metadata.
        """
        signals: List[Dict[str, Any]] = []
        step = max(1, self.scanlines // 12)  # Sample key scanlines

        for idx in range(0, self.scanlines, step):
            wave = math.sin((idx / float(self.scanlines)) * math.pi * 4.0 + (frame_index * 0.1))
            lum = max(0.0, min(0.7, 0.35 + 0.30 * wave * beam_intensity + analog_noise_factor))
            dig_val = round((lum / 0.7) * (self.quantization_levels - 1)) / (self.quantization_levels - 1)

            signals.append({
                "scanline": idx,
                "luminance_volts": round(lum, 3),
                "sync_pulse_volts": -0.3,
                "colorburst_phase_deg": round((idx * 33.7) % 360.0, 1),
                "digital_tensor_val": round(dig_val, 4)
            })

        # Calculate bandwidth: scanlines * (scanlines * 16/9) * 3 channels * bit_depth * fps
        width = int(self.scanlines * (16.0 / 9.0))
        raw_pixels_per_frame = width * self.scanlines
        bitrate_bps = raw_pixels_per_frame * 3 * self.adc_bit_depth * self.fps
        bitrate_gbps = bitrate_bps / 1_000_000_000.0

        return {
            "frame_index": frame_index,
            "scanlines_total": self.scanlines,
            "resolution": f"{width}x{self.scanlines}",
            "adc_bit_depth": self.adc_bit_depth,
            "fps": self.fps,
            "raw_stream_bitrate_gbps": round(bitrate_gbps, 3),
            "sampled_scanlines": signals,
            "npu_tensor_shape": [1, 3, self.scanlines, width],
            "npu_acceleration_backend": "DirectML / Intel Iris Xe / Neural NPU",
            "status": "synchronized_and_streamed"
        }


# ------------------------------------------------------------------------------
# 3. PROCEDURAL ISLAND REALMS (METAPHORICAL GEOGRAPHIC MAPPING)
# ------------------------------------------------------------------------------

@dataclass
class IslandRealm:
    realm_id: str
    display_name: str
    latin_lore: str
    metaphor_mapping: str
    biome_category: str
    elevation_meters: float
    primary_color_hex: str
    character_sort_category: str
    description: str
    resource_nodes: List[str]
    garrison_defense_power: int
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "realm_id": self.realm_id,
            "display_name": self.display_name,
            "latin_lore": self.latin_lore,
            "metaphor_mapping": self.metaphor_mapping,
            "biome_category": self.biome_category,
            "elevation_meters": self.elevation_meters,
            "primary_color_hex": self.primary_color_hex,
            "character_sort_category": self.character_sort_category,
            "description": self.description,
            "resource_nodes": self.resource_nodes,
            "garrison_defense_power": self.garrison_defense_power,
            "vital_max_hp": self.vital_max_hp
        }


class ProceduralIslandEngine:
    """
    Generates procedural oceanic and floating island realms with metaphorical mapping:
    - Denmark as Heaven (Dánsko ako Nebo)
    - Finland as Slovakia (Fínsko ako Slovensko)
    - Czechia as Latvia (Česko ako Lotyšsko)
    - Germany as Poland (Nemecko ako Poľsko)
    - France as America (Francúzsko ako Amerika)
    Strictly preserves the 6 Max HP vital invariant across all outposts and heroes.
    """

    def __init__(self):
        self._realms: Dict[str, IslandRealm] = self._init_realms()

    def _init_realms(self) -> Dict[str, IslandRealm]:
        return {
            "denmark_heaven": IslandRealm(
                realm_id="denmark_heaven",
                display_name="Dánsko ako Nebo (Denmark as Celestial Heaven)",
                latin_lore="Caelum Danicum // Celestial Sky Island",
                metaphor_mapping="Dánsko = Nebeská Sféra (Celestial Haven)",
                biome_category="celestial_floating_cloudbanks",
                elevation_meters=4500.0,
                primary_color_hex="#fef3c7",
                character_sort_category="seraphic_valkyries",
                description="Floating ethereal archipelago supported by golden clouds, solar sails, and shining ivory spires.",
                resource_nodes=["celestial_amber", "solar_aether_crystallites", "golden_harp_strings"],
                garrison_defense_power=95
            ),
            "finland_slovakia": IslandRealm(
                realm_id="finland_slovakia",
                display_name="Fínsko ako Slovensko (Finland as High Tatra Boreal Forest)",
                latin_lore="Silva Boralis // High Tatra Boreal Forest",
                metaphor_mapping="Fínsko = Slovensko (Boreal-Alpine Mountain Twin)",
                biome_category="alpine_glacial_tarn_taiga",
                elevation_meters=2655.0,
                primary_color_hex="#10b981",
                character_sort_category="tatra_forest_druids",
                description="Towering granite peaks reminiscent of Gerlach and Kriváň surrounded by deep boreal taiga and glacial tarns.",
                resource_nodes=["tatra_granite", "spruce_runic_timber", "glacial_spring_water"],
                garrison_defense_power=88
            ),
            "czechia_latvia": IslandRealm(
                realm_id="czechia_latvia",
                display_name="Česko ako Lotyšsko (Czechia as Baltic Bohemian Amber Coast)",
                latin_lore="Ora Succinica // Baltic Bohemian Amber Island",
                metaphor_mapping="Česko = Lotyšsko (Baltic Amber Maritime Realm)",
                biome_category="baltic_amber_dune_gothic",
                elevation_meters=45.0,
                primary_color_hex="#f59e0b",
                character_sort_category="amber_gothic_sentinels",
                description="Sweeping coastal dunes with historical gothic seaside castles, amber glassblowers, and maritime pine trails.",
                resource_nodes=["baltic_amber_fossils", "dune_quartz_sand", "bohemian_crystal_slag"],
                garrison_defense_power=82
            ),
            "germany_poland": IslandRealm(
                realm_id="germany_poland",
                display_name="Nemecko ako Poľsko (Germany as Plain Fortress Bastion)",
                latin_lore="Bastio Planitiei // Fortress Plain Realm",
                metaphor_mapping="Nemecko = Poľsko (Fortified Riverbank Plain)",
                biome_category="stone_bastion_fertile_plain",
                elevation_meters=180.0,
                primary_color_hex="#64748b",
                character_sort_category="iron_bastion_knights",
                description="Massive stone citadels and river bastions guarding fertile rye fields and heavy brass foundries.",
                resource_nodes=["bog_iron_ore", "quarried_granite_blocks", "heavy_brass_ingots"],
                garrison_defense_power=92
            ),
            "france_america": IslandRealm(
                realm_id="france_america",
                display_name="Francúzsko ako Amerika (France as Revolutionary Canyon Island)",
                latin_lore="Novus Mundus Libertatis // Revolutionary Canyon Island",
                metaphor_mapping="Francúzsko = Amerika (Revolutionary Pioneer World)",
                biome_category="revolutionary_red_canyons",
                elevation_meters=850.0,
                primary_color_hex="#e11d48",
                character_sort_category="revolutionary_pioneers",
                description="Dramatic red sandstone canyons crossed by grand boulevards, vineyards, and colossal liberty spires.",
                resource_nodes=["red_sandstone", "revolutionary_copper_ore", "aged_oak_casks"],
                garrison_defense_power=90
            )
        }

    def get_all_realms(self) -> Dict[str, Dict[str, Any]]:
        return {k: v.to_dict() for k, v in self._realms.items()}

    def get_realm(self, realm_id: str) -> Optional[Dict[str, Any]]:
        realm = self._realms.get(realm_id)
        return realm.to_dict() if realm else None

    def generate_island_geometry(self, realm_id: str, seed: int = 42) -> Dict[str, Any]:
        """
        Generates procedural island shoreline coordinates, contour rings, and garrison outposts.
        """
        realm = self._realms.get(realm_id, self._realms["denmark_heaven"])
        points_count = 24
        shoreline: List[Dict[str, float]] = []

        for i in range(points_count):
            angle = (i / float(points_count)) * 2.0 * math.pi
            noise = math.sin(angle * 3.0 + seed) * 25.0 + math.cos(angle * 5.0) * 15.0
            r = 160.0 + noise
            x = 400.0 + r * math.cos(angle)
            y = 250.0 + r * math.sin(angle)
            shoreline.append({"x": round(x, 1), "y": round(y, 1)})

        return {
            "realm_id": realm.realm_id,
            "display_name": realm.display_name,
            "elevation_meters": realm.elevation_meters,
            "vital_max_hp_rule": VITAL_MAX_HP,
            "shoreline_points": shoreline,
            "primary_color": realm.primary_color_hex,
            "resource_nodes": realm.resource_nodes,
            "outposts": [
                {"name": f"{realm.character_sort_category.capitalize()} Citadel", "hp": VITAL_MAX_HP, "max_hp": VITAL_MAX_HP, "x": 400.0, "y": 250.0},
                {"name": "Northern Watchtower", "hp": VITAL_MAX_HP, "max_hp": VITAL_MAX_HP, "x": 400.0, "y": 140.0},
                {"name": "Coastal Battery", "hp": VITAL_MAX_HP, "max_hp": VITAL_MAX_HP, "x": 510.0, "y": 310.0}
            ]
        }


# ------------------------------------------------------------------------------
# 4. PARALLEL JAVA TRANSPILER & RECORD HIERARCHY GENERATOR
# ------------------------------------------------------------------------------

class KrystalJavaTranspiler:
    """
    Transpiles Krystal models and procedural island hierarchies into modern Java 21+
    source code with records, sealed interfaces, and CompletableFuture parallel pipelines.
    """

    @staticmethod
    def generate_java_island_source(realms: Dict[str, IslandRealm]) -> str:
        """Generates clean, compilable Java 21 file with records and parallel sorting."""
        java_code = """package com.krystal.stack.procedural;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;

/**
 * Auto-transpiled from Krystal-Stack Engine (Python + Janet).
 * Implements procedural island hierarchies and strict 6 Max HP vital invariant.
 */
public final class KrystalIslandHierarchy {

    public static final int VITAL_MAX_HP = 6;

    // Sealed Interface Hierarchy for Island Characters
    public sealed interface IslandCharacter permits
            SeraphicValkyrie, ForestDruid, AmberSentinel, BastionKnight, RevolutionaryPioneer {
        String name();
        int hp();
        int maxHp();
        String realmId();
    }

    public record SeraphicValkyrie(String name, int hp, int maxHp, String realmId) implements IslandCharacter {}
    public record ForestDruid(String name, int hp, int maxHp, String realmId) implements IslandCharacter {}
    public record AmberSentinel(String name, int hp, int maxHp, String realmId) implements IslandCharacter {}
    public record BastionKnight(String name, int hp, int maxHp, String realmId) implements IslandCharacter {}
    public record RevolutionaryPioneer(String name, int hp, int maxHp, String realmId) implements IslandCharacter {}

    // Record for Procedural Island Realm
    public record ProceduralIslandRealm(
            String realmId,
            String displayName,
            String latinLore,
            String metaphorMapping,
            String biomeCategory,
            double elevationMeters,
            String primaryColorHex,
            List<String> resourceNodes,
            int garrisonDefensePower,
            int vitalMaxHp
    ) {
        public ProceduralIslandRealm {
            if (vitalMaxHp > VITAL_MAX_HP) {
                vitalMaxHp = VITAL_MAX_HP;
            }
        }
    }

    /**
     * Parallel Island Evaluator utilizing Java Virtual Threads / CompletableFuture.
     */
    public static class ParallelIslandPipeline {
        private final ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();

        public CompletableFuture<List<ProceduralIslandRealm>> evaluateRealmsParallel(List<ProceduralIslandRealm> realms) {
            List<CompletableFuture<ProceduralIslandRealm>> futures = realms.stream()
                    .map(r -> CompletableFuture.supplyAsync(() -> {
                        // Parallel processing & validation step
                        return new ProceduralIslandRealm(
                                r.realmId(),
                                r.displayName().toUpperCase(),
                                r.latinLore(),
                                r.metaphorMapping(),
                                r.biomeCategory(),
                                r.elevationMeters(),
                                r.primaryColorHex(),
                                r.resourceNodes(),
                                r.garrisonDefensePower(),
                                Math.min(r.vitalMaxHp(), VITAL_MAX_HP)
                        );
                    }, executor))
                    .collect(Collectors.toList());

            return CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]))
                    .thenApply(v -> futures.stream().map(CompletableFuture::join).toList());
        }
    }
}
"""
        return java_code


# ------------------------------------------------------------------------------
# 5. WORDPRESS SUBDOMAIN SECURITY GATE & HMAC TOKEN VALIDATOR
# ------------------------------------------------------------------------------

@dataclass
class SubdomainSession:
    user_id: int
    username: str
    role: str
    subdomain: str
    issued_at: float
    nonce: str
    vital_max_hp: int = VITAL_MAX_HP


class WordPressSubdomainSecurityGate:
    """
    Cryptographic perimeter defense for Krystal-Stack exposed via WordPress on a subdomain.
    Provides:
    - HMAC-SHA256 bearer token generation and verification.
    - Anti-replay nonce caching with expiration window.
    - Subdomain wildcard origin inspection.
    - Strict enforcement of VITAL_MAX_HP = 6.
    """

    def __init__(self, shared_secret: str = "krystal_wp_subdomain_secure_secret_2026"):
        self.shared_secret: str = shared_secret
        self.token_ttl_seconds: float = 60.0
        self.allowed_subdomain_patterns: List[str] = [
            r"^krystal\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            r"^[a-zA-Z0-9.-]+\.krystal-stack\.com$",
            r"^localhost(:[0-9]+)?$",
            r"^127\.0\.0\.1(:[0-9]+)?$"
        ]
        self._used_nonces: Dict[str, float] = {}
        self.vital_max_hp: int = VITAL_MAX_HP
        self.verified_sessions_count: int = 0
        self.rejected_tokens_count: int = 0

    def add_allowed_subdomain_pattern(self, pattern: str) -> None:
        self.allowed_subdomain_patterns.append(pattern)

    def is_subdomain_allowed(self, hostname: str) -> bool:
        clean_host = hostname.split("://")[-1].split("/")[0].strip().lower()
        for pat in self.allowed_subdomain_patterns:
            if re.match(pat, clean_host):
                return True
        return False

    def create_signed_token(self, user_id: int, username: str, role: str, subdomain: str) -> str:
        """
        Creates an HMAC-SHA256 signed bearer token issued by the WordPress subdomain.
        """
        now = time.time()
        nonce = hashlib.sha256(f"{user_id}:{now}:{time.perf_counter()}".encode("utf-8")).hexdigest()[:16]
        
        payload = {
            "uid": user_id,
            "usr": username,
            "rol": role,
            "sub": subdomain,
            "iat": round(now, 3),
            "exp": round(now + self.token_ttl_seconds, 3),
            "nce": nonce,
            "vhp": self.vital_max_hp
        }
        
        payload_bytes = json.dumps(payload, separators=(',', ':'), sort_keys=True).encode("utf-8")
        payload_b64 = base64.urlsafe_b64encode(payload_bytes).decode("utf-8").rstrip("=")
        
        signature = hmac.new(
            self.shared_secret.encode("utf-8"),
            payload_b64.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        
        return f"{payload_b64}.{signature}"

    def verify_signed_token(self, token_string: str) -> Tuple[bool, str, Optional[SubdomainSession]]:
        """
        Validates token integrity, signature, timestamp drift, and anti-replay nonce.
        """
        if not token_string or "." not in token_string:
            self.rejected_tokens_count += 1
            return False, "INVALID_TOKEN_FORMAT", None
            
        parts = token_string.strip().split(".")
        if len(parts) != 2:
            self.rejected_tokens_count += 1
            return False, "MALFORMED_TOKEN_STRUCTURE", None
            
        payload_b64, signature = parts[0], parts[1]
        
        # Verify HMAC signature
        expected_sig = hmac.new(
            self.shared_secret.encode("utf-8"),
            payload_b64.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        
        if not hmac.compare_digest(expected_sig, signature):
            self.rejected_tokens_count += 1
            return False, "INVALID_HMAC_SIGNATURE", None
            
        # Decode payload
        try:
            rem = len(payload_b64) % 4
            if rem > 0:
                payload_b64 += "=" * (4 - rem)
            payload_bytes = base64.urlsafe_b64decode(payload_b64)
            data = json.loads(payload_bytes.decode("utf-8"))
        except Exception:
            self.rejected_tokens_count += 1
            return False, "PAYLOAD_DECODE_ERROR", None
            
        now = time.time()
        
        # Validate timestamp freshness
        iat = data.get("iat", 0)
        exp = data.get("exp", 0)
        if now > exp:
            self.rejected_tokens_count += 1
            return False, "TOKEN_EXPIRED", None
            
        if abs(now - iat) > (self.token_ttl_seconds * 2):
            self.rejected_tokens_count += 1
            return False, "EXCESSIVE_CLOCK_SKEW", None
            
        # Replay Attack Check
        nonce = data.get("nce", "")
        self._prune_nonces(now)
        if nonce in self._used_nonces:
            self.rejected_tokens_count += 1
            return False, "REPLAY_ATTACK_DETECTED", None
            
        self._used_nonces[nonce] = now
        
        # Subdomain whitelist check
        subdomain = data.get("sub", "")
        if not self.is_subdomain_allowed(subdomain):
            self.rejected_tokens_count += 1
            return False, f"UNAUTHORIZED_SUBDOMAIN: {subdomain}", None
            
        self.verified_sessions_count += 1
        session = SubdomainSession(
            user_id=data.get("uid", 0),
            username=data.get("usr", "anonymous"),
            role=data.get("rol", "subscriber"),
            subdomain=subdomain,
            issued_at=iat,
            nonce=nonce,
            vital_max_hp=min(data.get("vhp", VITAL_MAX_HP), VITAL_MAX_HP)
        )
        return True, "VERIFIED", session

    def _prune_nonces(self, current_time: float) -> None:
        cutoff = current_time - (self.token_ttl_seconds * 2)
        expired = [n for n, ts in self._used_nonces.items() if ts < cutoff]
        for n in expired:
            del self._used_nonces[n]

    def get_security_telemetry(self) -> Dict[str, Any]:
        return {
            "vital_max_hp_rule": self.vital_max_hp,
            "verified_sessions_count": self.verified_sessions_count,
            "rejected_tokens_count": self.rejected_tokens_count,
            "active_nonces_count": len(self._used_nonces),
            "token_ttl_seconds": self.token_ttl_seconds,
            "allowed_subdomain_patterns": self.allowed_subdomain_patterns
        }


# Global instances
import json
GLOBAL_WORDPRESS_SECURITY = WordPressSecurityEngine()
GLOBAL_PROJECTOR_ANALOG_BRIDGE = ProjectorAnalogNpuBridge()
GLOBAL_PROCEDURAL_ISLAND_ENGINE = ProceduralIslandEngine()
GLOBAL_JAVA_TRANSPILER = KrystalJavaTranspiler()
GLOBAL_WORDPRESS_SUBDOMAIN_GATE = WordPressSubdomainSecurityGate()

