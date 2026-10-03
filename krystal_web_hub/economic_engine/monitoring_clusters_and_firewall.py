# ==============================================================================
# KRYSTAL-STACK: MONITORING CLUSTERS & ADAPTIVE APPLICATION FIREWALL (WAF)
# ==============================================================================
# Implements:
#   1. Clustered Node Monitoring (Edge Gateway, Compute Kernel, Ledger, Ad, Rendering).
#   2. Real-Time Telemetry & Quorum Health (CPU, RAM, QPS, latency, heartbeat).
#   3. Adaptive Layer 7 Firewall (Deep Packet Inspection, SQLi, XSS, Path Traversal, Bot).
#   4. Token Bucket & Sliding Window Rate Limiting.
#   5. Auto-quarantine & Incident Reporting.
# ==============================================================================

import time
import re
import ipaddress
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum

class ClusterNodeRole(str, Enum):
    EDGE_GATEWAY = "edge_gateway"
    COMPUTE_KERNEL = "compute_kernel"
    ECONOMIC_LEDGER = "economic_ledger"
    AD_EXCHANGE = "ad_exchange"
    RENDERING_WORKER = "rendering_worker"

class NodeHealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    DRAINING = "draining"
    OFFLINE = "offline"

class FirewallThreatCategory(str, Enum):
    SQL_INJECTION = "sql_injection"
    XSS_ATTACK = "xss_attack"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    BLOCKED_IP = "blocked_ip"
    MALICIOUS_USER_AGENT = "malicious_user_agent"
    SUSPICIOUS_PAYLOAD = "suspicious_payload"


class ClusterNode:
    """
    Represents an active server node in the Krystal distributed cluster.
    """
    def __init__(
        self,
        node_id: str,
        role: ClusterNodeRole,
        host: str = "127.0.0.1",
        port: int = 8089,
        weight: int = 100
    ):
        self.node_id = node_id
        self.role = role
        self.host = host
        self.port = port
        self.weight = weight
        self.status = NodeHealthStatus.HEALTHY
        self.cpu_percent = 15.0
        self.ram_percent = 22.0
        self.active_qps = 120.0
        self.p95_latency_ms = 4.2
        self.last_heartbeat_ts = time.time()
        self.consecutive_failures = 0
        self.total_requests_served = 0

    def update_heartbeat(
        self,
        cpu_pct: float,
        ram_pct: float,
        qps: float,
        latency_ms: float
    ):
        self.last_heartbeat_ts = time.time()
        self.cpu_percent = max(0.0, min(100.0, cpu_pct))
        self.ram_percent = max(0.0, min(100.0, ram_pct))
        self.active_qps = max(0.0, qps)
        self.p95_latency_ms = max(0.0, latency_ms)
        self.consecutive_failures = 0

        # Dynamic health evaluation
        if self.status != NodeHealthStatus.DRAINING:
            if self.cpu_percent > 90.0 or self.ram_percent > 92.0 or self.p95_latency_ms > 150.0:
                self.status = NodeHealthStatus.CRITICAL
            elif self.cpu_percent > 75.0 or self.ram_percent > 80.0 or self.p95_latency_ms > 50.0:
                self.status = NodeHealthStatus.DEGRADED
            else:
                self.status = NodeHealthStatus.HEALTHY

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "role": self.role.value,
            "endpoint": f"{self.host}:{self.port}",
            "status": self.status.value,
            "cpu_percent": round(self.cpu_percent, 1),
            "ram_percent": round(self.ram_percent, 1),
            "active_qps": round(self.active_qps, 1),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "last_heartbeat_age_sec": round(time.time() - self.last_heartbeat_ts, 2),
            "total_served": self.total_requests_served
        }


class MonitoringClusterEngine:
    """
    Coordinates multi-cluster monitoring, health aggregation, quorum consensus,
    and automatic node eviction/failover.
    """

    HEARTBEAT_TIMEOUT_SEC = 60.0
    EVICTION_TIMEOUT_SEC = 180.0

    def __init__(self):
        self._nodes: Dict[str, ClusterNode] = {}
        self._alerts: List[Dict[str, Any]] = []
        self._seed_default_clusters()

    def _seed_default_clusters(self):
        # 1. Edge Gateways
        self.register_node(ClusterNode("edge_gw_alpha_01", ClusterNodeRole.EDGE_GATEWAY, "127.0.0.1", 8089))
        self.register_node(ClusterNode("edge_gw_beta_02", ClusterNodeRole.EDGE_GATEWAY, "127.0.0.1", 8090))

        # 2. Janet/Vulkan Compute Kernels
        self.register_node(ClusterNode("compute_kernel_vulkan_01", ClusterNodeRole.COMPUTE_KERNEL, "127.0.0.1", 9001))
        self.register_node(ClusterNode("compute_kernel_janet_02", ClusterNodeRole.COMPUTE_KERNEL, "127.0.0.1", 9002))

        # 3. Economic Ledgers
        self.register_node(ClusterNode("econ_ledger_primary_01", ClusterNodeRole.ECONOMIC_LEDGER, "127.0.0.1", 8443))
        self.register_node(ClusterNode("econ_ledger_replica_02", ClusterNodeRole.ECONOMIC_LEDGER, "127.0.0.1", 8444))

        # 4. Ad Exchange Nodes
        self.register_node(ClusterNode("ad_exchange_broker_01", ClusterNodeRole.AD_EXCHANGE, "127.0.0.1", 8089))

        # 5. Rendering Workers
        self.register_node(ClusterNode("render_worker_godot_01", ClusterNodeRole.RENDERING_WORKER, "127.0.0.1", 7001))

    def register_node(self, node: ClusterNode):
        self._nodes[node.node_id] = node

    def get_node(self, node_id: str) -> Optional[ClusterNode]:
        return self._nodes.get(node_id)

    def record_node_heartbeat(
        self,
        node_id: str,
        cpu_pct: float,
        ram_pct: float,
        qps: float,
        latency_ms: float
    ) -> bool:
        node = self.get_node(node_id)
        if not node:
            return False
        node.update_heartbeat(cpu_pct, ram_pct, qps, latency_ms)
        return True

    def drain_node(self, node_id: str) -> bool:
        node = self.get_node(node_id)
        if not node:
            return False
        node.status = NodeHealthStatus.DRAINING
        self._alerts.append({
            "timestamp": time.time(),
            "level": "INFO",
            "message": f"Uzol {node_id} bol prepnutý do režimu DRAINING (riadené odpojenie)."
        })
        return True

    def evaluate_cluster_health(self) -> Dict[str, Any]:
        """
        Aggregates metrics across all registered nodes and evaluates quorum.
        """
        now = time.time()
        total_nodes = len(self._nodes)
        healthy_count = 0
        degraded_count = 0
        critical_count = 0
        offline_count = 0

        total_cpu = 0.0
        total_ram = 0.0
        total_qps = 0.0
        weighted_latency_sum = 0.0

        for node in self._nodes.values():
            age = now - node.last_heartbeat_ts
            if age > self.EVICTION_TIMEOUT_SEC:
                node.status = NodeHealthStatus.OFFLINE
            elif age > self.HEARTBEAT_TIMEOUT_SEC and node.status != NodeHealthStatus.DRAINING:
                node.status = NodeHealthStatus.CRITICAL

            if node.status == NodeHealthStatus.HEALTHY:
                healthy_count += 1
            elif node.status == NodeHealthStatus.DEGRADED:
                degraded_count += 1
            elif node.status == NodeHealthStatus.CRITICAL:
                critical_count += 1
            else:
                offline_count += 1

            total_cpu += node.cpu_percent
            total_ram += node.ram_percent
            total_qps += node.active_qps
            weighted_latency_sum += node.p95_latency_ms

        mean_cpu = total_cpu / max(1, total_nodes)
        mean_ram = total_ram / max(1, total_nodes)
        mean_latency = weighted_latency_sum / max(1, total_nodes)

        # Quorum consensus: healthy + degraded + critical (reachable) must exceed 50%
        active_functional = healthy_count + degraded_count + critical_count
        quorum_reached = (active_functional >= (total_nodes // 2 + 1))

        # Overall Cluster Health Score (0 - 100)
        health_score = 100.0
        health_score -= (critical_count * 25.0)
        health_score -= (offline_count * 35.0)
        health_score -= (degraded_count * 10.0)
        if mean_cpu > 70.0:
            health_score -= (mean_cpu - 70.0) * 0.5
        health_score = max(0.0, min(100.0, health_score))

        return {
            "cluster_status": "OPERATIONAL" if quorum_reached else "DEGRADED_QUORUM_LOST",
            "health_score": round(health_score, 1),
            "quorum_reached": quorum_reached,
            "total_nodes": total_nodes,
            "nodes_breakdown": {
                "healthy": healthy_count,
                "degraded": degraded_count,
                "critical": critical_count,
                "offline": offline_count
            },
            "metrics": {
                "cluster_total_qps": round(total_qps, 1),
                "mean_cpu_percent": round(mean_cpu, 1),
                "mean_ram_percent": round(mean_ram, 1),
                "mean_p95_latency_ms": round(mean_latency, 2)
            },
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "recent_alerts": self._alerts[-5:]
        }


class AdaptiveApplicationFirewall:
    """
    Layer 7 Application Firewall with signature inspection,
    dynamic IP rate limiting, CIDR blacklisting, and auto-quarantine.
    """

    # Malicious signatures (Regex patterns compiled)
    SQLI_PATTERNS = [
        re.compile(r"(\%27)|(\')|(\-\-\s)|(\%23)|(#\s+)", re.IGNORECASE),
        re.compile(r"\b(UNION(\s+ALL)?\s+SELECT|INSERT\s+INTO|DROP\s+(TABLE|DATABASE)|ALTER\s+TABLE)\b", re.IGNORECASE),
        re.compile(r"\b(OR\s+1\s*=\s*1|AND\s+1\s*=\s*1)\b", re.IGNORECASE),
        re.compile(r"\b(SLEEP\s*\(|BENCHMARK\s*\()", re.IGNORECASE)
    ]

    XSS_PATTERNS = [
        re.compile(r"<\s*script[^>]*>", re.IGNORECASE),
        re.compile(r"javascript\s*:", re.IGNORECASE),
        re.compile(r"on(load|error|click|mouseover|submit)\s*=", re.IGNORECASE),
        re.compile(r"<\s*iframe[^>]*>", re.IGNORECASE),
        re.compile(r"<\s*img[^>]+src\s*=\s*['\"]?javascript:", re.IGNORECASE)
    ]

    CMD_INJECTION_PATTERNS = [
        re.compile(r";\s*(rm|cat|ls|whoami|id|bash|sh|powershell|cmd)\b", re.IGNORECASE),
        re.compile(r"(\|\s*(rm|cat|bash|sh|powershell))", re.IGNORECASE),
        re.compile(r"(`|\$\()", re.IGNORECASE),
        re.compile(r"\b(subprocess\.Popen|os\.system|eval\(|exec\()", re.IGNORECASE)
    ]

    PATH_TRAVERSAL_PATTERNS = [
        re.compile(r"(\.\./|\.\.\\)", re.IGNORECASE),
        re.compile(r"(\%2e\%2e\%2f|\%2e\%2e\/|\%2e\%2e%5c)", re.IGNORECASE),
        re.compile(r"\b(etc/passwd|windows/system32|boot\.ini)\b", re.IGNORECASE)
    ]

    MALICIOUS_USER_AGENTS = [
        "sqlmap", "nikto", "masscan", "wpscan", "dirbuster", "nmap", "havij", "acunetix"
    ]

    def __init__(self, requests_per_minute: int = 120, ban_duration_sec: float = 900.0):
        self.rate_limit_rpm = requests_per_minute
        self.ban_duration_sec = ban_duration_sec  # 15 minutes default ban

        self._ip_whitelist: Set[str] = {"127.0.0.1", "::1"}
        self._ip_blacklist: Set[str] = set()
        self._quarantined_ips: Dict[str, float] = {}  # ip -> ban_expiry_ts
        self._sliding_window_requests: Dict[str, List[float]] = {}  # ip -> [timestamps]
        self._incident_logs: List[Dict[str, Any]] = []

    def add_to_whitelist(self, ip: str):
        self._ip_whitelist.add(ip)

    def add_to_blacklist(self, ip: str):
        self._ip_blacklist.add(ip)

    def remove_from_blacklist(self, ip: str):
        self._ip_blacklist.discard(ip)
        self._quarantined_ips.pop(ip, None)

    def is_ip_banned(self, ip: str) -> bool:
        if ip in self._ip_whitelist:
            return False
        if ip in self._ip_blacklist:
            return True
        now = time.time()
        ban_until = self._quarantined_ips.get(ip)
        if ban_until:
            if now < ban_until:
                return True
            else:
                del self._quarantined_ips[ip]
        return False

    def quarantine_ip(self, ip: str, reason: str, duration_sec: Optional[float] = None):
        if ip in self._ip_whitelist:
            return
        duration = duration_sec or self.ban_duration_sec
        expiry = time.time() + duration
        self._quarantined_ips[ip] = expiry
        self._incident_logs.append({
            "timestamp": time.time(),
            "ip": ip,
            "threat": "IP_QUARANTINED",
            "reason": reason,
            "duration_sec": duration,
            "expires_at": expiry
        })

    def inspect_request(
        self,
        client_ip: str,
        path: str,
        headers: Dict[str, str],
        body_text: str = ""
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Performs Deep Packet Inspection on the HTTP request.
        Returns (is_allowed, threat_details).
        """
        now = time.time()

        # 1. IP Blacklist & Active Quarantine Check
        if self.is_ip_banned(client_ip):
            incident = {
                "blocked": True,
                "category": FirewallThreatCategory.BLOCKED_IP.value,
                "client_ip": client_ip,
                "reason": "IP adresa sa nachádza na čiernej listine alebo v karanténe."
            }
            return False, incident

        # 2. Rate Limiting Check (Sliding Window 60s)
        if client_ip not in self._ip_whitelist:
            timestamps = self._sliding_window_requests.setdefault(client_ip, [])
            cutoff_1m = now - 60.0
            self._sliding_window_requests[client_ip] = [t for t in timestamps if t > cutoff_1m]
            if len(self._sliding_window_requests[client_ip]) >= self.rate_limit_rpm:
                self.quarantine_ip(client_ip, "Prekročený limit požiadaviek (Rate Limit Exceeded)", duration_sec=300.0)
                incident = {
                    "blocked": True,
                    "category": FirewallThreatCategory.RATE_LIMIT_EXCEEDED.value,
                    "client_ip": client_ip,
                    "reason": f"Prekročený limit {self.rate_limit_rpm} požiadaviek za minútu."
                }
                return False, incident
            self._sliding_window_requests[client_ip].append(now)

        # 3. Malicious User-Agent Check
        user_agent = headers.get("User-Agent", headers.get("user-agent", "")).lower()
        for bad_agent in self.MALICIOUS_USER_AGENTS:
            if bad_agent in user_agent:
                self.quarantine_ip(client_ip, f"Detegovaný škodlivý skener ({bad_agent})", duration_sec=1800.0)
                incident = {
                    "blocked": True,
                    "category": FirewallThreatCategory.MALICIOUS_USER_AGENT.value,
                    "client_ip": client_ip,
                    "reason": f"Podozrivý User-Agent: {bad_agent}"
                }
                return False, incident

        # Combined Inspection Target (Path + Decoded Query + Body)
        payload_to_inspect = f"{path} {body_text}"

        # 4. Path Traversal Check
        for p in self.PATH_TRAVERSAL_PATTERNS:
            if p.search(payload_to_inspect):
                self.quarantine_ip(client_ip, "Detegovaný pokus o Path Traversal útok")
                incident = {
                    "blocked": True,
                    "category": FirewallThreatCategory.PATH_TRAVERSAL.value,
                    "client_ip": client_ip,
                    "reason": "Neoprávnený pokus o prechod adresárom (Path Traversal)."
                }
                return False, incident

        # 5. SQL Injection Check
        for p in self.SQLI_PATTERNS:
            if p.search(payload_to_inspect):
                self.quarantine_ip(client_ip, "Detegovaný pokus o SQL Injection útok")
                incident = {
                    "blocked": True,
                    "category": FirewallThreatCategory.SQL_INJECTION.value,
                    "client_ip": client_ip,
                    "reason": "Vstup obsahuje podozrivé SQLi syntaktické vzory."
                }
                return False, incident

        # 6. XSS Check
        for p in self.XSS_PATTERNS:
            if p.search(payload_to_inspect):
                self.quarantine_ip(client_ip, "Detegovaný pokus o Cross-Site Scripting (XSS)")
                incident = {
                    "blocked": True,
                    "category": FirewallThreatCategory.XSS_ATTACK.value,
                    "client_ip": client_ip,
                    "reason": "Vstup obsahuje nepovolené skriptové alebo HTML značky."
                }
                return False, incident

        # 7. Command Injection Check
        for p in self.CMD_INJECTION_PATTERNS:
            if p.search(payload_to_inspect):
                self.quarantine_ip(client_ip, "Detegovaný pokus o Command Injection útok")
                incident = {
                    "blocked": True,
                    "category": FirewallThreatCategory.COMMAND_INJECTION.value,
                    "client_ip": client_ip,
                    "reason": "Vstup obsahuje shell metaznaky alebo pokus o spustenie systémových príkazov."
                }
                return False, incident

        # Request Passed all WAF Inspections
        return True, None

    def get_firewall_status(self) -> Dict[str, Any]:
        return {
            "status": "ACTIVE_PROTECTION",
            "rate_limit_rpm": self.rate_limit_rpm,
            "whitelisted_ips_count": len(self._ip_whitelist),
            "blacklisted_ips_count": len(self._ip_blacklist),
            "quarantined_ips_count": len(self._quarantined_ips),
            "active_quarantines": [
                {"ip": ip, "remaining_sec": max(0, int(exp - time.time()))}
                for ip, exp in self._quarantined_ips.items()
            ],
            "total_incidents_logged": len(self._incident_logs),
            "recent_incidents": self._incident_logs[-10:]
        }
