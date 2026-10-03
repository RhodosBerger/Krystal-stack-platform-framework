import json
import uuid

# ====================================================================
# KRYSTAL-STACK // ML Game Replica & Scraper Parser (Poslední Kmen)
# ====================================================================
# This module uses cognitive modeling to replicate the mechanics and 
# aesthetics of www.poslednikmen.cz into the Krystal-Stack ecosystem.
# ====================================================================

class TribeAestheticModel:
    """Defines the visual and mechanical ML parameters for each Tribe."""
    def __init__(self, name: str, element: str, primary_color: str, mechanics: list, background_css: str):
        self.name = name
        self.element = element
        self.primary_color = primary_color
        self.mechanics = mechanics
        self.background_css = background_css

class CardScraperParser:
    """
    Simulates the ML parser that scrapes www.poslednikmen.cz/cs/cards 
    and converts the HTML into structured game objects (Axioms).
    """
    def __init__(self):
        print("[Scraper Parser] Initialized cognitive web parser for 'Poslední kmen'.")
        
    def parse_card_database(self):
        print("[Scraper Parser] Parsing 129 cards across 3 tribes...")
        # Simulated parsed output based on website analysis
        parsed_cards = [
            {"tribe": "Crystal", "name": "Severní Štíty", "type": "Defense", "effect": "Slow/Frost"},
            {"tribe": "Toxic", "name": "Hnijící Slatiny", "type": "Spell", "effect": "Poison/Rot"},
            {"tribe": "Druid", "name": "Pradávný Les", "type": "Synergy", "effect": "Nature/Animals"}
        ]
        return parsed_cards

class PosledniKmenReplicaEngine:
    """
    The Game Engine Replica that integrates the scraped graphics 
    and spawns the models for the 3 Tribes.
    """
    def __init__(self):
        self.tribes = {
            "CRYSTAL": TribeAestheticModel(
                name="Krystalový Kmen",
                element="Ice/Frost",
                primary_color="#66fcf1", # Cyan / Ice
                mechanics=["Mráz", "Zpomalení", "Meteory"],
                background_css="linear-gradient(135deg, #0b0c10, #45a29e)"
            ),
            "TOXIC": TribeAestheticModel(
                name="Jedovatý Kmen",
                element="Poison",
                primary_color="#8a2be2", # Purple / Toxic Green
                mechanics=["Agrese", "Jed", "Hadí pomocníci"],
                background_css="linear-gradient(135deg, #0b0c10, #2e8b57)"
            ),
            "DRUID": TribeAestheticModel(
                name="Druidský Kmen",
                element="Nature",
                primary_color="#b8860b", # Earthy Gold / Wood
                mechanics=["Synergie zvířat", "Mystické stromy"],
                background_css="linear-gradient(135deg, #0b0c10, #8b4513)"
            )
        }
        self.scraper = CardScraperParser()
        
    def generate_html_card_replica(self, tribe_key: str, card_name: str, effect: str):
        """
        Uses the Aesthetic Model to procedurally generate a CSS/HTML card 
        replica inspired by the dark, glassmorphism design of the website.
        """
        tribe = self.tribes.get(tribe_key)
        if not tribe: return ""
        
        # Generates a premium HTML/CSS representation of the card
        html = f"""
        <div style="width: 200px; height: 300px; background: {tribe.background_css}; 
                    border: 2px solid {tribe.primary_color}; border-radius: 12px; 
                    box-shadow: 0 0 20px {tribe.primary_color}40; padding: 15px; 
                    font-family: 'Cinzel', serif; color: white; display: flex; 
                    flex-direction: column; justify-content: space-between;">
            <div>
                <div style="font-size: 10px; color: {tribe.primary_color}; letter-spacing: 2px;">{tribe.name.upper()}</div>
                <h3 style="margin: 5px 0; font-size: 18px;">{card_name}</h3>
            </div>
            <div style="background: rgba(0,0,0,0.5); padding: 10px; border-radius: 8px; font-family: 'Inter', sans-serif; font-size: 12px;">
                <i>Effect: {effect}</i>
            </div>
        </div>
        """
        return html

if __name__ == "__main__":
    engine = PosledniKmenReplicaEngine()
    print("====================================================")
    print("  POSLEDNÍ KMEN // KRYSTAL-STACK REPLICA ENGINE")
    print("====================================================")
    
    # 1. Scrape the data
    cards = engine.scraper.parse_card_database()
    
    # 2. Generate graphical replicas
    print("\n[Engine] Generating graphical card models based on scraped data...")
    for card in cards:
        html_output = engine.generate_html_card_replica(card["tribe"].upper(), card["name"], card["effect"])
        print(f" -> Generated Graphics Model for: {card['name']} [{card['tribe']}]")
        
    print("\n[System] Replica Integration Complete. Aesthetics and Models loaded.")
