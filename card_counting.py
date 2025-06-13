import numpy as np

class AdvancedCardCounting:
    def __init__(self, infinite_deck=True, num_decks=6):
        self.infinite_deck = infinite_deck
        self.num_decks = num_decks
        self.cards_seen = 0
        self.systems = {
            'hi_lo': {
                'values': {'2': 1, '3': 1, '4': 1, '5': 1, '6': 1,
                          '7': 0, '8': 0, '9': 0,
                          '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1},
                'running': 0,
                'true': 0
            },
            'omega_ii': {
                'values': {'2': 1, '3': 1, '4': 2, '5': 2, '6': 2,
                          '7': 1, '8': 0, '9': -1,
                          '10': -2, 'J': -2, 'Q': -2, 'K': -2, 'A': 0},
                'running': 0,
                'true': 0
            },
            'wong_halves': {
                'values': {'2': 0.5, '3': 1, '4': 1, '5': 1.5, '6': 1,
                          '7': 0.5, '8': 0, '9': -0.5,
                          '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1},
                'running': 0.0,
                'true': 0.0
            }
        }
        
    def _convert_card_value(self, card):
        """Convert card to consistent string format"""
        if isinstance(card, str):
            card = card.upper()
            if card in ['J','Q','K']:
                return '10'
            elif card == 'A':
                return 'A'
            return card
        else:
            card = int(card)
            if card == 11:
                return 'J'
            elif card == 12:
                return 'Q'
            elif card == 13:
                return 'K'
            elif card == 1:
                return 'A'
            return str(card)

    def update_count(self, card):
        """Update all counting systems with a single card"""
        card_value = self._convert_card_value(card)
        
        for system in self.systems:
            count_value = self.systems[system]['values'].get(card_value, 0)
            if system == 'wong_halves':
                self.systems[system]['running'] += float(count_value)
            else:
                self.systems[system]['running'] += count_value
        
        # Update cards seen and calculate true count
        self.cards_seen += 1
        if not self.infinite_deck:
            cards_per_deck = 52
            total_cards = self.num_decks * cards_per_deck
            remaining_cards = max(1, total_cards - self.cards_seen)
            decks_remaining = remaining_cards / cards_per_deck
            
            for system in self.systems:
                self.systems[system]['true'] = self.systems[system]['running'] / max(1, decks_remaining)
        else:
            for system in self.systems:
                self.systems[system]['true'] = self.systems[system]['running']
        
        return self.get_counts()

    def get_counts(self):
        """Return current counts from all systems with rounded values"""
        counts = {}
        for system in self.systems:
            counts[system] = {
                'running': round(self.systems[system]['running'], 2),
                'true': round(self.systems[system]['true'], 2),
                'decks_remaining': 1.0 if self.infinite_deck else max(0.1, (self.num_decks * 52 - self.cards_seen) / 52)
            }
        return counts

    def get_system_confidence(self, system_name):
        """Get confidence level for a specific counting system (0-1)"""
        if system_name not in self.systems:
            return 0.5
        
        true_count = abs(self.systems[system_name]['true'])
        confidence = min(1.0, max(0.5, 0.5 + true_count * 0.05))  # More gradual confidence increase
        return confidence

    def reset_all_counts(self):
        """Reset all counting systems"""
        self.cards_seen = 0
        for system in self.systems:
            self.systems[system]['running'] = 0
            self.systems[system]['true'] = 0
        return self.get_counts()