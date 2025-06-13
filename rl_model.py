import gym
from gym import spaces
import numpy as np
from card_counting import AdvancedCardCounting
import random
import math

class BlackjackENV(gym.Env):
    def __init__(self, infinite_deck=True, num_decks=6):
        self.card_counter = AdvancedCardCounting(infinite_deck=infinite_deck, num_decks=num_decks)
        super(BlackjackENV, self).__init__()
        
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0, high=32, shape=(6,), dtype=np.float32)
        
        self.player_hands = []
        self.current_hand_index = 0
        self.dealer_cards = []
        self.done = False
        self.insurance_offered = False
        self.insurance_taken = False
        self.last_outcome = None
        self.last_outcome_info = None
        self.dealer_hits = []  # Track dealer hits explicitly
        
        # Enhanced betting parameters
        self.min_bet = 2
        self.max_bet = 20
        self.base_bet = 5
        self.current_bet = self.base_bet
        self.bankroll = 2000

    def calculate_bet(self, confidence):
        """Conservative betting strategy with count awareness"""
        counts = self.card_counter.get_counts()
        hi_lo = counts.get('hi_lo', {})
        true_count = hi_lo.get('running', 0) / max(1.0, hi_lo.get('decks_remaining', 1))
        
        # Base bet as 1% of bankroll or min bet, whichever is higher
        base_bet = max(self.min_bet, min(5, int(self.bankroll * 0.01)))
        
        # Adjust based on true count
        if true_count <= 1:
            return base_bet
        elif true_count <= 2:
            return min(base_bet * 1.5, self.max_bet)
        elif true_count <= 3:
            return min(base_bet * 2, self.max_bet)
        elif true_count <= 4:
            return min(base_bet * 2.5, self.max_bet)
        else:
            return min(base_bet * 3, self.max_bet)

    def _sum_hand(self, hand):
        total = 0
        aces = 0
        
        for card in hand:
            if card == 1:  # Ace
                total += 1
                aces += 1
            else:
                total += min(card, 10)
        
        while total <= 11 and aces > 0:
            total += 10
            aces -= 1
        
        return total

    def _usable_ace(self, hand):
        return 1 in hand and self._sum_hand(hand) + 10 <= 21

    def _should_stand_early(self, player_total, dealer_upcard):
        """Enhanced conservative standing rules"""
        if player_total >= 17:
            return True
        if player_total >= 15 and dealer_upcard in [2,3,4,5,6]:
            return True
        if player_total >= 13 and dealer_upcard in [4,5,6]:
            return True
        return False

    def get_valid_actions(self):
        """Returns valid actions with count-based insurance and optimal doubling"""
        if self.current_hand_index >= len(self.player_hands):
            return [False] * 6
        
        current_hand = self.player_hands[self.current_hand_index]
        counts = self.card_counter.get_counts()
        hi_lo = counts.get('hi_lo', {})
        decks_remaining = hi_lo.get('decks_remaining', 1.0)
        true_count = hi_lo.get('running', 0) / max(1.0, decks_remaining)
        
        valid_actions = [False] * 6
        valid_actions[0] = True  # Hit always allowed
        valid_actions[1] = True  # Stand always allowed
        
        # Optimal doubling rules
        player_total = self._sum_hand(current_hand)
        is_soft = self._usable_ace(current_hand)
        valid_actions[2] = (len(current_hand) == 2 and 
                          ((9 <= player_total <= 11) or 
                           (is_soft and (player_total >= 13 and player_total <= 18))))
        
        # More flexible splitting rules
        valid_actions[3] = (len(current_hand) == 2 and 
                          current_hand[0] == current_hand[1] and
                          current_hand[0] in [1, 2, 3, 4, 6, 7, 8, 9])
        
        # Insurance only at true count >= +3
        valid_actions[4] = (self.insurance_offered and true_count >= 3)
        valid_actions[5] = self.insurance_offered
        
        return valid_actions

    def draw_card(self):
        card = random.choice([1,2,3,4,5,6,7,8,9,10,10,10,10])
        self.card_counter.update_count(card)
        return card

    def reset(self):
        self.card_counter.reset_all_counts()
        self.player_hands = [[self.draw_card(), self.draw_card()]]
        self.current_hand_index = 0
        self.dealer_cards = [self.draw_card(), self.draw_card()]
        self.done = False
        self.insurance_offered = self.dealer_cards[0] == 1
        self.insurance_taken = False
        self.last_outcome = None
        self.last_outcome_info = None
        self.dealer_hits = []  # Reset dealer hits tracking
        self.current_bet = self.base_bet
        return self._get_obs()

    def _get_obs(self):
        if self.current_hand_index >= len(self.player_hands):
            return np.zeros(6, dtype=np.float32)

        current_hand = self.player_hands[self.current_hand_index]
        counts = self.card_counter.get_counts()
        hi_lo = counts.get('hi_lo', {})
        decks_remaining = hi_lo.get('decks_remaining', 1.0)
        true_count = hi_lo.get('running', 0) / max(1.0, decks_remaining)

        return np.array([
            self._sum_hand(current_hand) / 32.0,
            self.dealer_cards[0] / 11.0 if self.dealer_cards else 0,
            float(self._usable_ace(current_hand)),
            float(len(current_hand) == 2),
            float(len(current_hand) == 2 and current_hand[0] == current_hand[1]),
            true_count
        ], dtype=np.float32)

    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, {'dealer_hits': self.dealer_hits.copy()}
        
        current_hand = self.player_hands[self.current_hand_index]
        reward = 0.0
        info = {'dealer_hits': self.dealer_hits.copy()}

        if action == 0:  # hit
            drawn_card = self.draw_card()
            current_hand.append(drawn_card)
            info['drawn_card'] = drawn_card
            
            if self._sum_hand(current_hand) > 21:
                self.done = True
                result = self._resolve_hand()
                # Merge info dictionaries
                final_info = info.copy()
                if len(result) > 3:
                    final_info.
