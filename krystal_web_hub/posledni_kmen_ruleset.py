# ====================================================================
# KRYSTAL-STACK // Poslední Kmen: Strict Parameter Ruleset
# ====================================================================
# Fulfills the exact mechanical parameters and axioms scraped from 
# https://www.poslednikmen.cz/cs
# ====================================================================

class GameMode:
    DEATHMATCH = "Deathmatch"
    FLAGS = "Vlajky"
    CLASSIC = "Klasický mód"

class PosledniKmenState:
    """State Machine enforcing the strict parameters of the game."""
    
    # --- GLOBAL GAME PARAMETERS ---
    MIN_PLAYERS = 2
    MAX_PLAYERS = 3
    TOTAL_CARDS = 129
    CARDS_PER_DECK = 43
    ARENA_TILES = 9
    HERO_STARTING_HP = 6
    REACH_CARDS = 6
    
    def __init__(self, mode: str, players: list):
        self.mode = mode
        if len(players) < self.MIN_PLAYERS or len(players) > self.MAX_PLAYERS:
            raise ValueError(f"[ERROR] Hra musí mať {self.MIN_PLAYERS} až {self.MAX_PLAYERS} hráčov.")
            
        self.players = players
        self.arena = [None] * self.ARENA_TILES
        self.flags_collected = {p.name: 0 for p in self.players}
        self.round_number = 1
        
        print(f"[Ruleset] Zapas spusteny. Mod: {self.mode}. Pocet hracov: {len(self.players)}.")
        self._verify_parameters()
        
    def _verify_parameters(self):
        """Strict validation of deck sizes and health points."""
        for player in self.players:
            if len(player.deck) != self.CARDS_PER_DECK:
                print(f"[Warning] {player.name} nemá presne {self.CARDS_PER_DECK} kariet!")
            if player.hp != self.HERO_STARTING_HP:
                print(f"[Warning] {player.name} nemá začiatočných {self.HERO_STARTING_HP} životov!")
                
    def check_victory_condition(self):
        """Evaluates win conditions based on the selected game mode."""
        alive_players = [p for p in self.players if p.hp > 0]
        
        if self.mode == GameMode.DEATHMATCH:
            # Posledný preživší vyhráva
            if len(alive_players) == 1:
                return alive_players[0]
                
        elif self.mode == GameMode.FLAGS:
            # Pevný koniec po 11 kolách (z webu)
            if self.round_number > 11:
                # Vyhráva ten s najviac vlajkami
                winner = max(self.flags_collected, key=self.flags_collected.get)
                return winner
                
        elif self.mode == GameMode.CLASSIC:
            # Súťažné hranie (Kombinácia prežitia a skóre)
            if len(alive_players) == 1:
                return alive_players[0]

        return None


class TribePlayer:
    """Represents a player piloting one of the 3 Tribes."""
    def __init__(self, name: str, tribe_type: str):
        self.name = name
        self.tribe_type = tribe_type
        self.hp = PosledniKmenState.HERO_STARTING_HP
        # Vytvoríme presne 43 kariet pre balíček
        self.deck = [f"Karta {tribe_type} {i}" for i in range(1, PosledniKmenState.CARDS_PER_DECK + 1)]
        self.reach_cards = PosledniKmenState.REACH_CARDS
        print(f"[Player] {self.name} si vybral {self.tribe_type}. HP: {self.hp}, Balicek: {len(self.deck)} kariet.")

    def take_damage(self, amount: int):
        self.hp -= amount
        print(f"[Boj] {self.name} ({self.tribe_type}) utrpel {amount} zranenie. Zostava HP: {self.hp}")


if __name__ == "__main__":
    print("====================================================")
    print("  POSLEDNÍ KMEN // STRICT PARAMETER VALIDATION")
    print("====================================================\n")
    
    # Inicializácia hráčov podľa pravidiel z webu (Každý kmeň má presne stanovený balík)
    p1 = TribePlayer("Hrac 1", "KRYSTALOVY KMEN")
    p2 = TribePlayer("Hrac 2", "JEDOVATY KMEN")
    
    # Spustenie Deathmatch módu
    game = PosledniKmenState(mode=GameMode.DEATHMATCH, players=[p1, p2])
    
    print("\n[Simulacia] Hrac 1 utoci. Super nema reakciu.")
    p2.take_damage(6) # Fatal hit
    
    winner = game.check_victory_condition()
    if winner:
        print(f"\n[Koniec Hry] Vitaz zapasu: {winner.name} ({winner.tribe_type})!")
