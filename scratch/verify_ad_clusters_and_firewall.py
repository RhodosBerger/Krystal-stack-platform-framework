import urllib.request
import urllib.parse
import urllib.error
import json
import sys

def test_get(endpoint, headers=None):
    url = f"http://127.0.0.1:8089{endpoint}"
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        print(f"[GET] {endpoint} => Status: {resp.status}, Success: {data.get('success', False)}")
        return data

def test_post(endpoint, payload, headers=None):
    url = f"http://127.0.0.1:8089{endpoint}"
    body = json.dumps(payload).encode('utf-8')
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=body, headers=h)
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        print(f"[POST] {endpoint} => Status: {resp.status}, Success: {data.get('success', False)}")
        return data

try:
    print("--- 1. Testing GET /api/ad_protocol/campaigns ---")
    camps = test_get("/api/ad_protocol/campaigns")
    assert camps["success"] is True
    assert len(camps["campaigns"]) >= 4

    print("--- 2. Testing POST /api/ad_protocol/auction_bid ---")
    auction_res = test_post("/api/ad_protocol/auction_bid", {
        "player_id": "hero_live_test",
        "player_tribe": "crystal",
        "ad_format": "rewarded_video"
    })
    assert auction_res["success"] is True
    ticket = auction_res["ticket"]
    assert "signature" in ticket
    assert "nonce" in ticket

    print("--- 3. Testing POST /api/ad_protocol/verify_impression ---")
    settle_res = test_post("/api/ad_protocol/verify_impression", {
        "ticket": ticket,
        "proof_of_viewing": {
            "view_duration_seconds": 15.0,
            "mouse_event_count": 6,
            "viewport_focus_percent": 95.0,
            "is_headless_bot": False
        }
    })
    assert settle_res["success"] is True
    assert "settlement" in settle_res

    print("--- 4. Testing GET /api/ad_protocol/ledger ---")
    ledger = test_get("/api/ad_protocol/ledger")
    assert ledger["success"] is True
    assert ledger["ledger"]["total_settlements"] >= 1

    print("--- 5. Testing GET /api/cluster/monitoring/health ---")
    cluster_health = test_get("/api/cluster/monitoring/health")
    assert cluster_health["success"] is True
    assert cluster_health["cluster_health"]["quorum_reached"] is True

    print("--- 6. Testing POST /api/cluster/monitoring/heartbeat ---")
    hb_res = test_post("/api/cluster/monitoring/heartbeat", {
        "node_id": "compute_kernel_vulkan_01",
        "cpu_percent": 25.5,
        "ram_percent": 30.0,
        "qps": 250.0,
        "latency_ms": 3.8
    })
    assert hb_res["success"] is True

    print("--- 7. Testing GET /api/security/firewall/status ---")
    fw_status = test_get("/api/security/firewall/status")
    assert fw_status["success"] is True
    assert fw_status["firewall"]["status"] == "ACTIVE_PROTECTION"

    print("--- 8. Testing Firewall Deep Packet Inspection: SQLi Blocking ---")
    sqli_blocked = False
    try:
        url = "http://127.0.0.1:8089/api/cards?search=" + urllib.parse.quote("' OR 1=1 --")
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            pass
    except urllib.error.HTTPError as e:
        if e.code == 403:
            sqli_blocked = True
            err_data = json.loads(e.read().decode('utf-8'))
            print(f"[FIREWALL INTERCEPT] SQLi blocked with 403 Forbidden: {err_data['error']}")

    assert sqli_blocked is True, "Firewall should have blocked SQL injection!"

    print("--- 9. Testing Firewall Deep Packet Inspection: Malicious User-Agent Blocking ---")
    agent_blocked = False
    try:
        url = "http://127.0.0.1:8089/api/status"
        req = urllib.request.Request(url, headers={"User-Agent": "sqlmap/1.4.1"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            pass
    except urllib.error.HTTPError as e:
        if e.code == 403:
            agent_blocked = True
            err_data = json.loads(e.read().decode('utf-8'))
            print(f"[FIREWALL INTERCEPT] Malicious agent blocked with 403 Forbidden: {err_data['error']}")

    assert agent_blocked is True, "Firewall should have blocked malicious user-agent!"

    print("\nALL 9 LIVE AD PROTOCOL, CLUSTER & FIREWALL TESTS PASSED PERFECTLY!")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
