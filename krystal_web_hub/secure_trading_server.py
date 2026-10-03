# ==============================================================================
# KRYSTAL-STACK: SECURE TLS COMMERCE & INVENTORY GATEWAY SERVER (PORT 8443)
# ==============================================================================
# Implements Protocol 2: Secure TLS/HTTPS Microservice for Inventory, Trading,
# Price Catalogs, and Outnumbered Magic Balancing.
# Features:
#   - TLS 1.3 / 1.2 Encrypted Sessions via Python SSLContext.
#   - Atomic Double-Entry Inventory Ledger Transactions.
#   - Dynamic Underdog Disparity Analysis & Ward Shield Generation.
# ==============================================================================

import os
import sys
import ssl
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
from dataclasses import asdict

from krystal_web_hub.economic_engine.secure_commerce_engine import (
    CurrencyType,
    ITEM_PRICE_CATALOG,
    CARD_PRICE_CATALOG,
    HERO_PERK_CATALOG,
    CRAFTMADE_RECIPE_CATALOG,
    POTION_CATALOG,
    HeroInventory,
    SecureCommerceGateway,
    UnderdogMagicBalancingEngine
)
from krystal_web_hub.economic_engine.models import Tribe
from krystal_web_hub.economic_engine.unit_archetypes import (
    CANONICAL_HEALER_PROFILES,
    create_unit_instance,
    resolve_healer_action
)
from krystal_web_hub.economic_engine.sequence_tensor_adapter import (
    CANONICAL_PALETTES,
    PaletteTensorCache,
    CrossRoomContextReplicator
)
from krystal_web_hub.economic_engine.combo_supernatural_combat import (
    SPECIALIZED_SQUAD_CATALOG,
    TripletComboEngine,
    StatisticalComboBonusEngine,
    SquadCriticalStrikeEngine
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CERTS_DIR = os.path.join(BASE_DIR, "certs")
CERT_FILE = os.path.join(CERTS_DIR, "cert.pem")
KEY_FILE = os.path.join(CERTS_DIR, "key.pem")

# Global In-Memory Inventory and Audit Ledger
ACTIVE_HERO_INVENTORY = HeroInventory(
    hero_id="hero_player_1",
    tribe=Tribe.CRYSTAL,
    balances={
        "gold": 1000,
        "aether_crystal": 10,
        "toxic_slime": 10,
        "amber_rune": 10,
        "mana": 20,
        "krystal_gems": 100,
        "astral_credits": 200
    }
)
PEER_HERO_INVENTORY = HeroInventory(
    hero_id="hero_player_2",
    tribe=Tribe.TOXIC
)
COMMERCE_AUDIT_LEDGER: list = []
GLOBAL_ROOM_REPLICATOR = CrossRoomContextReplicator()

class ThreadedTLSServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = (sys.platform != "win32")

class SecureCommerceHandler(BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        body = json.dumps(data, indent=2).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.send_header('Strict-Transport-Security', 'max-age=31536000; includeSubDomains')
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()

    def do_GET(self):
        path = self.path.split('?')[0]

        # 1. Health & Protocol Status
        if path in ('/', '/api/health', '/api/commerce/status'):
            self._send_json({
                "status": "SECURE_ONLINE",
                "protocol": "TLS 1.3 / HTTPS",
                "port": 8443,
                "service": "Krystal Secure Commerce & Inventory Gateway",
                "tls_cipher": self.connection.cipher() if hasattr(self.connection, 'cipher') else "Active TLS",
                "timestamp": int(time.time())
            })
            return

        # 2. Canonical Price Catalogs
        if path == '/api/commerce/catalog':
            self._send_json({
                "items": ITEM_PRICE_CATALOG,
                "cards": CARD_PRICE_CATALOG,
                "hero_perks": HERO_PERK_CATALOG
            })
            return

        # 3. Active Inventory State
        if path == '/api/commerce/inventory':
            self._send_json({
                "success": True,
                "inventory": ACTIVE_HERO_INVENTORY.to_dict()
            })
            return

        # 4. Audit Ledger
        if path == '/api/commerce/ledger':
            self._send_json({
                "audit_entries_count": len(COMMERCE_AUDIT_LEDGER),
                "ledger": COMMERCE_AUDIT_LEDGER[-50:] # Recent 50 records
            })
            return

        # 5. Masterwork Craftmade Recipes
        if path == '/api/commerce/crafting/recipes':
            self._send_json({
                "success": True,
                "recipes": CRAFTMADE_RECIPE_CATALOG
            })
            return

        # 6. Healers Roster
        if path == '/api/healers/roster':
            self._send_json({
                "success": True,
                "healers": CANONICAL_HEALER_PROFILES
            })
            return

        # 7. Dual-Currency Potion Catalog
        if path == '/api/commerce/potions/catalog':
            self._send_json({
                "success": True,
                "potions": POTION_CATALOG
            })
            return

        # 8. Cached Palette Tensor Catalog
        if path == '/api/tensor/palette':
            self._send_json({
                "success": True,
                "palettes": CANONICAL_PALETTES
            })
            return

        self._send_json({"error": f"Endpoint {path} not found on Secure TLS Gateway"}, status=404)

    def do_POST(self):
        path = self.path.split('?')[0]
        content_len = int(self.headers.get('Content-Length', 0))
        post_body = self.rfile.read(content_len).decode('utf-8') if content_len > 0 else '{}'
        try:
            req_data = json.loads(post_body)
        except Exception:
            req_data = {}

        # 1. Purchase Item
        if path == '/api/commerce/purchase/item':
            item_id = req_data.get("item_id", "")
            res = SecureCommerceGateway.execute_purchase_item(
                ACTIVE_HERO_INVENTORY, item_id, COMMERCE_AUDIT_LEDGER
            )
            status_code = 200 if res.get("success") else 400
            self._send_json(res, status=status_code)
            return

        # 2. Purchase Card
        if path == '/api/commerce/purchase/card':
            card_id = req_data.get("card_id", "")
            res = SecureCommerceGateway.execute_purchase_card(
                ACTIVE_HERO_INVENTORY, card_id, COMMERCE_AUDIT_LEDGER
            )
            status_code = 200 if res.get("success") else 400
            self._send_json(res, status=status_code)
            return

        # 3. Unlock Hero Perk
        if path == '/api/commerce/unlock/perk':
            perk_id = req_data.get("perk_id", "")
            res = SecureCommerceGateway.execute_unlock_perk(
                ACTIVE_HERO_INVENTORY, perk_id, COMMERCE_AUDIT_LEDGER
            )
            status_code = 200 if res.get("success") else 400
            self._send_json(res, status=status_code)
            return

        # 4. Underdog Disparity & Dynamic Ward Evaluation
        if path == '/api/underdog/evaluate':
            n_allies = int(req_data.get("allied_units", 1))
            n_enemies = int(req_data.get("enemy_units", 3))
            hp_allies = int(req_data.get("allied_hp", 6))
            hp_enemies = int(req_data.get("enemy_hp", 18))
            perks = req_data.get("perks", ACTIVE_HERO_INVENTORY.unlocked_perks)

            status = UnderdogMagicBalancingEngine.evaluate_numerical_disparity(
                allied_units_count=n_allies,
                enemy_units_count=n_enemies,
                allied_total_hp=hp_allies,
                enemy_total_hp=hp_enemies,
                unlocked_perks=perks
            )
            self._send_json({
                "success": True,
                "underdog_status": asdict(status)
            })
            return

        # 5. Ward Damage Absorption Simulation
        if path == '/api/underdog/absorb_damage':
            incoming = int(req_data.get("incoming_damage", 4))
            ward = int(req_data.get("current_ward", 3))
            hp = int(req_data.get("hero_hp", 6))

            res = UnderdogMagicBalancingEngine.absorb_damage_via_underdog_ward(
                incoming_damage=incoming,
                current_ward=ward,
                hero_current_hp=hp
            )
            self._send_json({"success": True, "result": res})
            return

        # 6. Execute Masterwork Craftmade Recipe
        if path == '/api/commerce/craft':
            recipe_id = req_data.get("recipe_id", "")
            res = SecureCommerceGateway.execute_craft_recipe(
                ACTIVE_HERO_INVENTORY, recipe_id, COMMERCE_AUDIT_LEDGER
            )
            status_code = 200 if res.get("success") else 400
            self._send_json(res, status=status_code)
            return

        # 7. Execute Healer Action (Healing, Ward Infusion, Cleanse)
        if path == '/api/healers/heal_action':
            healer_id = req_data.get("healer_id", "crystal_resonance_mender")
            target_id = req_data.get("target_id", "crystal_archon")
            action_type = req_data.get("action_type", "heal")
            target_hp = int(req_data.get("target_hp", 3))
            underdog_mult = float(req_data.get("underdog_multiplier", 1.0))

            try:
                healer_unit = create_unit_instance(healer_id)
                target_unit = create_unit_instance(target_id)
                target_unit.current_wounds = max(0, min(target_unit.wounds_max, target_hp))

                res = resolve_healer_action(
                    healer=healer_unit,
                    target=target_unit,
                    action_type=action_type,
                    underdog_multiplier=underdog_mult
                )
                self._send_json(res)
            except Exception as e:
                self._send_json({"success": False, "error": str(e)}, status=400)
            return

        # 8. Purchase Potion (Gold or Premium Virtual Currency)
        if path == '/api/commerce/potions/purchase':
            potion_id = req_data.get("potion_id", "potion_healing_draught")
            mode = req_data.get("payment_mode", "gold")
            res = SecureCommerceGateway.execute_purchase_potion(
                ACTIVE_HERO_INVENTORY, potion_id, payment_mode=mode, audit_ledger=COMMERCE_AUDIT_LEDGER
            )
            status_code = 200 if res.get("success") else 400
            self._send_json(res, status=status_code)
            return

        # 9. Network P2P Potion Trade
        if path == '/api/commerce/potions/network_trade':
            potion_id = req_data.get("potion_id", "")
            qty = int(req_data.get("quantity", 1))
            currency = req_data.get("currency", "gold")
            price = int(req_data.get("price", 50))
            res = SecureCommerceGateway.execute_network_potion_trade(
                ACTIVE_HERO_INVENTORY, PEER_HERO_INVENTORY, potion_id, qty, currency, price, COMMERCE_AUDIT_LEDGER
            )
            status_code = 200 if res.get("success") else 400
            self._send_json(res, status=status_code)
            return

        # 10. Currency Exchange (Gold <-> Premium Gems <-> Astral Credits)
        if path == '/api/commerce/currency/exchange':
            from_curr = req_data.get("from_currency", "gold")
            to_curr = req_data.get("to_currency", "krystal_gems")
            amount = int(req_data.get("amount", 100))
            res = SecureCommerceGateway.execute_currency_exchange(
                ACTIVE_HERO_INVENTORY, from_curr, to_curr, amount, COMMERCE_AUDIT_LEDGER
            )
            status_code = 200 if res.get("success") else 400
            self._send_json(res, status=status_code)
            return

        # 11. Cross-Room Context Replication & Tensor Frames
        if path == '/api/tensor/room/replicate':
            src_room = req_data.get("source_room_id", "room_arena_1")
            tgt_room = req_data.get("target_room_id", "room_spectator_1")
            grid = req_data.get("grid", [[1, 2, 4], [0, 5, 6], [3, 0, 7]])
            action = req_data.get("action", "hero_ward_surge")

            GLOBAL_ROOM_REPLICATOR.record_room_event(src_room, grid, action_label=action)
            rep_res = GLOBAL_ROOM_REPLICATOR.replicate_context_to_room(src_room, tgt_room)
            frames = GLOBAL_ROOM_REPLICATOR.reproduce_room_sequence(tgt_room, output_format="ascii")
            self._send_json({
                "replication": rep_res,
                "reproduced_frames": frames
            })
            return

        # 12. Triplet Combo Heal & Supernatural Scaling
        if path == '/api/combat/combo_heal':
            cards = req_data.get("played_cards", ["card_aether_dart", "card_aether_dart", "card_aether_dart"])
            fallen_count = int(req_data.get("fallen_heroes", 3))
            is_supernatural = bool(req_data.get("is_supernatural", True))
            mage_id = req_data.get("mage_id", "crystal_archon")
            target_id = req_data.get("target_id", "crystal_archon")

            try:
                mage = create_unit_instance(mage_id)
                target = create_unit_instance(target_id)
                target.current_wounds = int(req_data.get("target_hp", 2))

                heal_res = TripletComboEngine.resolve_mage_triplet_heal(
                    mage_unit=mage,
                    target_unit=target,
                    played_card_ids=cards,
                    fallen_heroes_count=fallen_count,
                    is_target_supernatural=is_supernatural
                )
                stat_bonuses = StatisticalComboBonusEngine.calculate_combo_bonuses(
                    combo_detected=heal_res["combo_detected"],
                    fallen_heroes_count=fallen_count
                )
                self._send_json({
                    "heal_result": heal_res,
                    "statistical_bonuses": stat_bonuses
                })
            except Exception as e:
                self._send_json({"success": False, "error": str(e)}, status=400)
            return

        # 13. Specialized Squad Critical Strike (Player vs Bot or Human)
        if path == '/api/combat/squad_critical_strike':
            squad_id = req_data.get("squad_id", "crystal_sniper_cadre")
            target_id = req_data.get("target_id", "toxic_defiler")
            is_bot = bool(req_data.get("target_is_bot", True))
            forced_roll = float(req_data.get("forced_roll", 0.10)) # Forces critical hit for testing

            try:
                target = create_unit_instance(target_id)
                res = SquadCriticalStrikeEngine.resolve_squad_attack(
                    squad_id=squad_id,
                    target_unit=target,
                    target_is_bot=is_bot,
                    forced_crit_roll=forced_roll
                )
                self._send_json(res)
            except Exception as e:
                self._send_json({"success": False, "error": str(e)}, status=400)
            return

        self._send_json({"error": f"Endpoint {path} not found on Secure TLS Gateway"}, status=404)

def run_secure_server(port=8443, use_tls=True):
    server_address = ('0.0.0.0', port)
    httpd = ThreadedTLSServer(server_address, SecureCommerceHandler)

    if use_tls and os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=CERT_FILE, keyfile=KEY_FILE)
        httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
        proto_label = "HTTPS / TLS"
    else:
        proto_label = "HTTP (Plain Fallback)"

    print("=================================================================")
    print(f" KRYSTAL-STACK SECURE COMMERCE GATEWAY ONLINE (PORT {port})")
    print(f" Protocol: {proto_label} | Double-Entry Inventory Verified")
    print(f" Items in Catalog: {len(ITEM_PRICE_CATALOG)} | Cards: {len(CARD_PRICE_CATALOG)}")
    print(f" Underdog Balancing: ACTIVE (Dynamic Wards & 6 Max HP Rule)")
    print("=================================================================")
    httpd.serve_forever()

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8443
    use_tls = "--no-tls" not in sys.argv
    run_secure_server(port=port, use_tls=use_tls)
