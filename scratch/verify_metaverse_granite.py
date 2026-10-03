import urllib.request
import urllib.parse
import json
import sys

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_URL = "http://127.0.0.1:8089"

def req_get(path):
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def req_post(path, data):
    url = f"{BASE_URL}{path}"
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def verify_all():
    print("==================================================================")
    print(" [VERIFICATION] METAVERSE MARKET & 2B / 4B IBM GRANITE LLM ENGINE")
    print("==================================================================")

    # 1. GET /api/metaverse/market/catalog
    cat = req_get("/api/metaverse/market/catalog")
    assert cat["success"] is True, "Catalog failed"
    assert cat["count"] >= 5, f"Expected at least 5 assets, got {cat['count']}"
    print(f" [+] 1. GET /api/metaverse/market/catalog: OK ({cat['count']} assets, First: '{cat['catalog'][0]['name']}')")

    # 2. GET /api/metaverse/market/orderbook
    ob = req_get("/api/metaverse/market/orderbook?pair=LAND_042/AET")
    assert ob["success"] is True, "Orderbook failed"
    order_book = ob["order_book"]
    assert order_book["best_bid"] > 0 and order_book["best_ask"] > 0
    print(f" [+] 2. GET /api/metaverse/market/orderbook: OK (Midpoint: {order_book['midpoint_price']} AET, Spread: {order_book['spread_percentage']}%)")

    # 3. POST /api/metaverse/market/order (Place Limit Buy)
    order_res = req_post("/api/metaverse/market/order", {
        "pair": "LAND_042/AET",
        "order_type": "limit_buy",
        "price": order_book["best_ask"],
        "amount": 1.0,
        "trader": "collector_metaverse"
    })
    assert order_res["success"] is True, "Place order failed"
    print(f" [+] 3. POST /api/metaverse/market/order: OK (Filled: {order_res['order']['filled']} / {order_res['order']['amount']}, Trades: {order_res['executed_trades_count']})")

    # 4. GET /api/metaverse/market/amm_pools
    pools = req_get("/api/metaverse/market/amm_pools")
    assert pools["success"] is True, "AMM pools failed"
    print(f" [+] 4. GET /api/metaverse/market/amm_pools: OK ({pools['count']} pools, First: {pools['pools'][0]['pair']})")

    # 5. POST /api/metaverse/market/amm_swap
    swap_res = req_post("/api/metaverse/market/amm_swap", {
        "pool_id": "pool_aet_gold",
        "token_in": "AET",
        "amount_in": 15.0,
        "slippage_tolerance": 0.05
    })
    assert swap_res["success"] is True, "Swap failed"
    print(f" [+] 5. POST /api/metaverse/market/amm_swap: OK (Swapped 15.0 AET -> {swap_res['amount_out']} GOLD, Impact: {swap_res['price_impact_pct']}%)")

    # 6. GET /api/metaverse/llm/models (2B & 4B Granite)
    models = req_get("/api/metaverse/llm/models")
    assert models["success"] is True, "Models failed"
    assert models["count"] >= 4
    model_names = [m["name"] for m in models["models"]]
    print(f" [+] 6. GET /api/metaverse/llm/models: OK ({models['count']} models: {', '.join(model_names[:2])})")

    # 7. GET /api/metaverse/llm/budget (Hardware & VRAM Budget Verification)
    budget = req_get("/api/metaverse/llm/budget?model_key=ibm-granite-3.1-4b-instruct-2026")
    assert budget["success"] is True, "Budget verification failed"
    b_data = budget["budget"]
    assert b_data["fits_in_edge_vram"] is True
    print(f" [+] 7. GET /api/metaverse/llm/budget: OK (Verdict: {b_data['hardware_verdict']}, VRAM: {b_data['vram_mb']}MB, Context: {b_data['context_window']})")

    # 8. POST /api/metaverse/llm/market_eval (IBM Granite 4B 2026 Market Analysis)
    eval_res = req_post("/api/metaverse/llm/market_eval", {
        "pair": "LAND_042/AET",
        "model_key": "ibm-granite-3.1-4b-instruct-2026"
    })
    assert eval_res["success"] is True, "Market eval failed"
    ev = eval_res["evaluation"]
    print(f" [+] 8. POST /api/metaverse/llm/market_eval: OK (Sentiment: {ev['sentiment']}, Confidence: {ev['confidence_score_pct']}%)")

    # 9. POST /api/metaverse/llm/merchant_barter (Granite 2B NPC Merchant Barter)
    barter_res = req_post("/api/metaverse/llm/merchant_barter", {
        "offered_item": "Aéterový Plášť",
        "offered_nominal_val": 130.0,
        "requested_item": "Aéterová Batéria",
        "requested_nominal_val": 100.0,
        "merchant_archetype": "Kováč z Citadely",
        "greed_factor": 0.15,
        "model_key": "ibm-granite-3.0-2b-instruct"
    })
    assert barter_res["success"] is True, "Barter failed"
    b = barter_res["barter"]
    print(f" [+] 9. POST /api/metaverse/llm/merchant_barter: OK (Verdict: {b['verdict']}, Dialogue: {b['merchant_dialogue'][:45]}...)")

    # 10. POST /api/metaverse/llm/manipulation_audit
    audit_res = req_post("/api/metaverse/llm/manipulation_audit", {
        "model_key": "ibm-granite-3.1-4b-instruct-2026",
        "order_events": [
            {"action": "TRADE", "buyer": "trader_1", "seller": "trader_2", "amount": 5, "price": 450}
        ]
    })
    assert audit_res["success"] is True, "Audit failed"
    print(f" [+] 10. POST /api/metaverse/llm/manipulation_audit: OK (Verdict: {audit_res['audit']['audit_verdict']}, Risk: {audit_res['audit']['risk_score_pct']}%)")

    # 11. GET Static Studio Page
    req = urllib.request.Request(f"{BASE_URL}/static/metaverse_market_studio.html")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200, "Studio HTML page failed to load"
        print(f" [+] 11. GET /static/metaverse_market_studio.html: OK (Status 200)")

    print("==================================================================")
    print(" ALL 11 METAVERSE & GRANITE LLM ENDPOINTS VERIFIED 100% OK!")
    print("==================================================================")

if __name__ == '__main__':
    verify_all()
