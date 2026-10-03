import sys
import time

# ====================================================================
# KRYSTAL-STACK // THE LAST TRIBE (POSLEDNÍ KMEN) - CLI MASTER GAME
# ====================================================================
# Unifies:
# 1. State Machine (PosledniKmenState - 6 HP, Rules)
# 2. MRP Accounting Logic (Facade translated to Mana/Energy Ledger)
# 3. Autonomous AI Bot (ASCII Art Generation for Scenes/Axioms)
# ====================================================================

# --- 1. ASCII ART GENERATOR (AI BOT FACADE) ---
def generate_ascii_scene(theme):
    print(f"\n[AI Bot] Synthesizing Scene Axiom: {theme}...")
    time.sleep(1)
    if "ICE" in theme or "KRYSTAL" in theme:
        print("""
            .  *  .  . *       *    .
         *   .  *      *  .   *   .
           .      /\\       .  *
          *      /  \\  *         .
            .   /____\\   .    *
        """)
    elif "TOXIC" in theme or "POISON" in theme:
        print("""
               (  )   (   )  )
                ) (   )  (  (
                ( )  (    ) )
                =============
               |   POISON    |
               |    SWAMP    |
                =============
        """)
    elif "DRUID" in theme or "NATURE" in theme:
        print("""
                 oxoxoo    ooxoo
                ooxoxo oo  oxoxooo
               oooo xxoxoo ooo ooox
               oxo o oxoxo  xoxxoxo
                oxo  \\|/    /o
                      |    /
                      |   /
                      |  |
                      |  |
        """)
    else:
        print("""
               [ AXIOMATIC BURST ]
                 \\ | /
                -  *  -
                 / | \\
        """)

# --- 2. THE GAME CLASSES ---
class TribePlayer:
    def __init__(self, name: str, tribe: str, element: str):
        self.name = name
        self.tribe = tribe
        self.element = element
        self.hp = 6
        self.mana = 10  # This represents our MRP "Ledger Balance"
        self.deck_size = 43
        
    def show_status(self):
        print(f"\n>>> HRÁČ: {self.name} [{self.tribe}]")
        print(f"    HP: {'♥' * self.hp} ({self.hp}/6) | MANA (Ledger): {'♦' * self.mana} ({self.mana}) | Deck: {self.deck_size}")

class Card:
    def __init__(self, id_num, name, cost, damage, theme):
        self.id = id_num
        self.name = name
        self.cost = cost
        self.damage = damage
        self.theme = theme

# --- 3. THE CLI GAME LOOP ---
def run_game():
    print("====================================================")
    print("      POSLEDNÍ KMEN // KRYSTAL-STACK TERMINAL       ")
    print("====================================================")
    print("Initializing State Machine... OK (6 HP max)")
    print("Initializing Ledger/Accounting... OK (Mapped to Mana)")
    
    # Init Players
    p1 = TribePlayer("Hráč 1", "KRYSTALOVÝ KMEN", "KRYSTAL/ICE")
    p2 = TribePlayer("AI Protivník", "JEDOVATÝ KMEN", "TOXIC/POISON")
    
    # Hand/Cards
    hand = [
        Card(1, "Mráz a Zpomalení", cost=3, damage=2, theme="ICE"),
        Card(2, "Severní Meteory", cost=5, damage=4, theme="ICE_BURST"),
        Card(3, "Obranný Štít", cost=2, damage=0, theme="KRYSTAL_SHIELD")
    ]
    
    while p1.hp > 0 and p2.hp > 0:
        p1.show_status()
        p2.show_status()
        
        print("\n--- TVOJ ŤAH ---")
        print("Tvoje karty na ruke:")
        for c in hand:
            print(f"[{c.id}] {c.name} (Cena: {c.cost} Many, Zranenie: {c.damage})")
            
        print("[0] Ukončiť hru")
        
        choice = input("\nVyber číslo karty, ktorú chceš zahrať: ")
        
        if choice == '0':
            print("Hra bola ukončená.")
            break
            
        try:
            selected = next(c for c in hand if str(c.id) == choice)
        except StopIteration:
            print("[CHYBA] Neplatná voľba.")
            continue
            
        # 1. ACCOUNTING RULESET CHECK
        if p1.mana < selected.cost:
            print(f"\n[Axiom Engine] ZAMIETNUTÉ: Nedostatok Many (Ledger Balance). Potrebuješ {selected.cost}, máš {p1.mana}.")
            continue
            
        # 2. EXECUTE CARD (Deduct cost)
        p1.mana -= selected.cost
        print(f"\n[Axiom Engine] SCHVÁLENÉ. Z Ledgeru odpočítané {selected.cost} Many.")
        
        # 3. AUTONOMOUS AI SCENE GENERATOR
        generate_ascii_scene(selected.theme)
        
        # 4. DAMAGE CALCULATION
        if selected.damage > 0:
            print(f"> Zasiahol si protivníka! Udelil si {selected.damage} poškodenie.")
            p2.hp -= selected.damage
            
        if p2.hp <= 0:
            print("\n====================================================")
            print("  VÍŤAZSTVO! JEDOVATÝ KMEN BOL PORAZENÝ.")
            print("====================================================")
            break
            
        # 5. AI TURN (Simple simulation)
        print("\n--- ŤAH PROTIVNÍKA ---")
        time.sleep(1)
        print("[AI] Súper zahral: Hnijící Slatiny (Jed)")
        generate_ascii_scene("TOXIC")
        p1.hp -= 2
        print("> Utrpel si 2 poškodenie jedom!")
        
        if p1.hp <= 0:
            print("\n====================================================")
            print("  PREHRA. TVOJ KMEN VYHYNUL.")
            print("====================================================")
            break

        # Regain mana (Accounting income)
        p1.mana += 2
        p2.mana += 2
        print("\n[Axiom Engine] Kolo ukončené. Obom hráčom boli pripísané 2 Many (Income).")

if __name__ == "__main__":
    try:
        run_game()
    except KeyboardInterrupt:
        print("\n[Ukončené]")
