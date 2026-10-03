import unittest
import time
from krystal_web_hub.economic_engine import (
    AdFormat, AdCampaignStatus, AdCampaign, SecureAdExchangeProtocol,
    ClusterNodeRole, NodeHealthStatus, FirewallThreatCategory,
    ClusterNode, MonitoringClusterEngine, AdaptiveApplicationFirewall
)

class TestAdProtocolClustersAndFirewall(unittest.TestCase):

    def setUp(self):
        self.ad_protocol = SecureAdExchangeProtocol(server_secret_key="test_super_secret_ad_key")
        self.cluster_engine = MonitoringClusterEngine()
        self.firewall = AdaptiveApplicationFirewall(requests_per_minute=10, ban_duration_sec=60.0)

    # -------------------------------------------------------------------------
    # 1. AD PROTOCOL & SECURITY TESTS
    # -------------------------------------------------------------------------
    def test_ad_campaign_creation_and_budget(self):
        camp = AdCampaign(
            campaign_id="camp_test_01",
            advertiser_id="adv_test",
            name="Test Campaign",
            ad_format=AdFormat.REWARDED_VIDEO,
            bid_cpm_nuggets=1000,
            total_budget_nuggets=100,
            min_view_seconds=10.0
        )
        self.assertEqual(camp.cost_per_verified_impression, 10) # 1000 / 100
        self.assertEqual(camp.remaining_budget, 100)
        payout = camp.record_verified_payout()
        self.assertEqual(payout, 10)
        self.assertEqual(camp.remaining_budget, 90)

    def test_ad_auction_and_signed_ticket(self):
        res = self.ad_protocol.run_ad_auction("player_hero_1", "crystal", AdFormat.REWARDED_VIDEO)
        self.assertTrue(res["success"])
        ticket = res["ticket"]
        self.assertIn("signature", ticket)
        self.assertIn("nonce", ticket)
        self.assertGreater(ticket["reward_nuggets"], 0)
        self.assertGreater(ticket["expires_at"], time.time())

    def test_proof_of_viewing_verification_success(self):
        auction = self.ad_protocol.run_ad_auction("player_hero_1", "crystal", AdFormat.REWARDED_VIDEO)
        ticket = auction["ticket"]

        pov_data = {
            "view_duration_seconds": 15.0,
            "mouse_event_count": 8,
            "viewport_focus_percent": 98.0,
            "is_headless_bot": False
        }
        verify_res = self.ad_protocol.verify_proof_of_viewing_and_settle(ticket, pov_data)
        self.assertTrue(verify_res["success"])
        self.assertIn("settlement", verify_res)
        self.assertEqual(verify_res["settlement"]["player_id"], "player_hero_1")
        self.assertGreater(verify_res["settlement"]["payout_nuggets"], 0)

    def test_proof_of_viewing_insufficient_time_rejected(self):
        auction = self.ad_protocol.run_ad_auction("player_hero_1", "crystal", AdFormat.REWARDED_VIDEO)
        ticket = auction["ticket"]

        pov_data = {
            "view_duration_seconds": 2.5,  # Too short!
            "mouse_event_count": 1,
            "viewport_focus_percent": 100.0,
            "is_headless_bot": False
        }
        verify_res = self.ad_protocol.verify_proof_of_viewing_and_settle(ticket, pov_data)
        self.assertFalse(verify_res["success"])
        self.assertIn("INSUFFICIENT_VIEW_DURATION", verify_res["error"])

    def test_tampered_signature_rejected(self):
        auction = self.ad_protocol.run_ad_auction("player_hero_1", "crystal", AdFormat.REWARDED_VIDEO)
        ticket = auction["ticket"]
        ticket["reward_nuggets"] = 99999  # Tampering attempt
        ticket["player_id"] = "hacker_account"

        pov_data = {
            "view_duration_seconds": 15.0,
            "mouse_event_count": 5,
            "viewport_focus_percent": 100.0
        }
        verify_res = self.ad_protocol.verify_proof_of_viewing_and_settle(ticket, pov_data)
        self.assertFalse(verify_res["success"])
        self.assertEqual(verify_res["error"], "CRYPTOGRAPHIC_SIGNATURE_MISMATCH")

    def test_nonce_replay_attack_rejected(self):
        auction = self.ad_protocol.run_ad_auction("player_hero_1", "crystal", AdFormat.REWARDED_VIDEO)
        ticket = auction["ticket"]
        pov_data = {
            "view_duration_seconds": 15.0,
            "mouse_event_count": 5,
            "viewport_focus_percent": 100.0
        }

        # First settlement succeeds
        first_res = self.ad_protocol.verify_proof_of_viewing_and_settle(ticket, pov_data)
        self.assertTrue(first_res["success"])

        # Second settlement with identical ticket MUST be rejected (anti-replay)
        replay_res = self.ad_protocol.verify_proof_of_viewing_and_settle(ticket, pov_data)
        self.assertFalse(replay_res["success"])
        self.assertIn("NONCE_REPLAY", replay_res["error"])

    def test_headless_bot_detection(self):
        auction = self.ad_protocol.run_ad_auction("bot_01", "toxic", AdFormat.REWARDED_VIDEO)
        ticket = auction["ticket"]
        pov_data = {
            "view_duration_seconds": 20.0,
            "mouse_event_count": 0,
            "viewport_focus_percent": 100.0,
            "is_headless_bot": True
        }
        verify_res = self.ad_protocol.verify_proof_of_viewing_and_settle(ticket, pov_data)
        self.assertFalse(verify_res["success"])
        self.assertIn("HEADLESS_BOT_FLAGGED", verify_res["error"])

    # -------------------------------------------------------------------------
    # 2. CLUSTER MONITORING TESTS
    # -------------------------------------------------------------------------
    def test_cluster_monitoring_health_and_quorum(self):
        health = self.cluster_engine.evaluate_cluster_health()
        self.assertEqual(health["cluster_status"], "OPERATIONAL")
        self.assertTrue(health["quorum_reached"])
        self.assertGreaterEqual(health["health_score"], 80.0)
        self.assertGreater(health["total_nodes"], 4)

    def test_cluster_node_heartbeat_and_degradation(self):
        node = self.cluster_engine.get_node("compute_kernel_vulkan_01")
        self.assertIsNotNone(node)
        self.assertEqual(node.status, NodeHealthStatus.HEALTHY)

        # Overheat CPU to 95% -> Node should degrade to CRITICAL
        self.cluster_engine.record_node_heartbeat("compute_kernel_vulkan_01", 95.0, 40.0, 500.0, 120.0)
        self.assertEqual(node.status, NodeHealthStatus.CRITICAL)

        # Cool down -> recovers to HEALTHY
        self.cluster_engine.record_node_heartbeat("compute_kernel_vulkan_01", 30.0, 35.0, 200.0, 5.0)
        self.assertEqual(node.status, NodeHealthStatus.HEALTHY)

    def test_cluster_node_drain(self):
        drained = self.cluster_engine.drain_node("render_worker_godot_01")
        self.assertTrue(drained)
        node = self.cluster_engine.get_node("render_worker_godot_01")
        self.assertEqual(node.status, NodeHealthStatus.DRAINING)

    # -------------------------------------------------------------------------
    # 3. FIREWALL TESTS
    # -------------------------------------------------------------------------
    def test_firewall_clean_request_allowed(self):
        allowed, incident = self.firewall.inspect_request(
            client_ip="192.168.1.100",
            path="/api/roster/characters",
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            body_text='{"race": "crystal", "tier": 3}'
        )
        self.assertTrue(allowed)
        self.assertIsNone(incident)

    def test_firewall_sqli_blocked(self):
        allowed, incident = self.firewall.inspect_request(
            client_ip="192.168.1.101",
            path="/api/cards?query=1' OR 1=1 --",
            headers={"User-Agent": "Mozilla/5.0"},
            body_text=""
        )
        self.assertFalse(allowed)
        self.assertEqual(incident["category"], FirewallThreatCategory.SQL_INJECTION.value)
        self.assertTrue(self.firewall.is_ip_banned("192.168.1.101"))

    def test_firewall_xss_blocked(self):
        allowed, incident = self.firewall.inspect_request(
            client_ip="192.168.1.102",
            path="/api/match/chat",
            headers={"User-Agent": "Mozilla/5.0"},
            body_text='{"message": "<script>alert(1)</script>"}'
        )
        self.assertFalse(allowed)
        self.assertEqual(incident["category"], FirewallThreatCategory.XSS_ATTACK.value)
        self.assertTrue(self.firewall.is_ip_banned("192.168.1.102"))

    def test_firewall_path_traversal_blocked(self):
        allowed, incident = self.firewall.inspect_request(
            client_ip="192.168.1.103",
            path="/api/assets/../../etc/passwd",
            headers={"User-Agent": "Mozilla/5.0"},
            body_text=""
        )
        self.assertFalse(allowed)
        self.assertEqual(incident["category"], FirewallThreatCategory.PATH_TRAVERSAL.value)
        self.assertTrue(self.firewall.is_ip_banned("192.168.1.103"))

    def test_firewall_malicious_user_agent_blocked(self):
        allowed, incident = self.firewall.inspect_request(
            client_ip="192.168.1.104",
            path="/api/status",
            headers={"User-Agent": "sqlmap/1.5.2#stable"},
            body_text=""
        )
        self.assertFalse(allowed)
        self.assertEqual(incident["category"], FirewallThreatCategory.MALICIOUS_USER_AGENT.value)
        self.assertTrue(self.firewall.is_ip_banned("192.168.1.104"))

    def test_firewall_rate_limiting(self):
        ip = "192.168.1.105"
        # We configured rate_limit_rpm=10
        for _ in range(10):
            allowed, _ = self.firewall.inspect_request(ip, "/api/status", {"User-Agent": "Mozilla/5.0"})
            self.assertTrue(allowed)

        # 11th request in same minute should be rejected and IP quarantined
        blocked, incident = self.firewall.inspect_request(ip, "/api/status", {"User-Agent": "Mozilla/5.0"})
        self.assertFalse(blocked)
        self.assertEqual(incident["category"], FirewallThreatCategory.RATE_LIMIT_EXCEEDED.value)
        self.assertTrue(self.firewall.is_ip_banned(ip))

if __name__ == '__main__':
    unittest.main()
