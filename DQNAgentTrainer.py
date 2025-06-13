import random
import numpy as np
from collections import deque
import os
import torch
import torch.nn as nn
import torch.optim as optim
from rl_model import BlackjackENV
import time
import math
from torch.utils.tensorboard import SummaryWriter

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_size)
        self.dropout = nn.Dropout(0.2)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        x = self.fc1(x)
        if x.size(0) > 1:
            x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        if x.size(0) > 1:
            x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=200000)
        self.gamma = 0.97
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-5)
        self.criterion = nn.SmoothL1Loss()
        self.wins = 0
        self.losses = 0
        self.episode_rewards = []
        self.last_save_winrate = 0
        self.start_time = time.time()
        self.update_target_every = 1000
        self.steps = 0
        self.max_episodes_below_threshold = 30
        self.episodes_below_threshold = 0
        self.best_val_winrate = 0
        self.writer = SummaryWriter('runs/blackjack_dqn')
        self.best_bankroll = 2000
        self.adaptive_epsilon = True
        self.warmup_episodes = 500
        self.min_bet_multiplier = 0.5
        self.conservative_threshold = 17

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_actions_mask):
        player_total = int(state[0] * 32)  # Extract player total from state
        
        # Enhanced conservative play - force stand on high totals
        if player_total >= 21:
            return 1  # Always stand on 21+
        elif player_total >= 20:
            return 1  # Always stand on 20
        elif player_total >= 19:
            return 1  # Always stand on 19
        elif player_total >= 18:
            # Stand on 18 unless we're in exploration mode
            if np.random.rand() > self.epsilon:
                return 1
        elif player_total >= self.conservative_threshold:
            # For 17, be very conservative during exploitation
            if (np.random.rand() > self.epsilon and 
                len(self.episode_rewards) >= self.warmup_episodes):
                return 1  # Stand
        
        # Normal action selection for lower totals
        if np.random.rand() <= self.epsilon or len(self.episode_rewards) < self.warmup_episodes:
            valid_actions = [i for i, valid in enumerate(valid_actions_mask) if valid]
            # Even in exploration, avoid hitting on dangerous totals
            if player_total >= 17 and 0 in valid_actions and 1 in valid_actions:
                # Heavily bias toward standing on 17+
                if np.random.rand() < 0.8:  # 80% chance to stand even in exploration
                    return 1
            return random.choice(valid_actions)
        
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        valid_indices = [i for i in range(self.action_size) if valid_actions_mask[i]]
        best_action = valid_indices[torch.argmax(q_values[0, valid_indices]).item()]
        
        # Final safety check - don't hit on very high totals
        if best_action == 0 and player_total >= 19:
            return 1  # Force stand
        
        return best_action

    def get_confidence(self, state, action):
        self.model.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state)
            max_q = torch.max(q_values).item()
            min_q = torch.min(q_values).item()
            if max_q == min_q:
                return 0.5
            confidence = (q_values[0, action].item() - min_q) / (max_q - min_q)
        self.model.train()
        return confidence

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return 0.0
        
        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([t[0] for t in minibatch]))
        actions = torch.LongTensor(np.array([t[1] for t in minibatch]))
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch]))
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch]))
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch]))

        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            next_actions = self.model(next_states).max(1)[1]
            next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1))
            targets = rewards + (1 - dones) * self.gamma * next_q_values.squeeze()

        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        loss = self.criterion(current_q, targets.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        if self.adaptive_epsilon:
            win_rate = self.win_rate()
            if len(self.episode_rewards) < self.warmup_episodes:
                self.epsilon = max(self.epsilon_min, 0.8)
            elif win_rate < 0.45 and self.epsilon < 0.8:
                self.epsilon = min(0.8, self.epsilon * 1.01)
            elif win_rate > 0.55 and self.epsilon > self.epsilon_min:
                self.epsilon = max(self.epsilon_min, self.epsilon * 0.995)
        elif self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()

    def should_terminate_session(self, env):
        if len(self.episode_rewards) < 100:
            return False
            
        bankroll_ratio = env.bankroll / 2000
        current_win_rate = self.win_rate()
        
        if bankroll_ratio < 0.7:
            return True
        if current_win_rate < 0.35 and bankroll_ratio < 0.8:
            return True
            
        return False

    def save(self, filename):
        torch.save({
            'model_state': self.model.state_dict(),
            'target_model_state': self.target_model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'wins': self.wins,
            'losses': self.losses,
            'memory': list(self.memory)[-10000:],
            'last_save_winrate': self.win_rate(),
            'episode_rewards': self.episode_rewards,
            'best_val_winrate': self.best_val_winrate,
            'best_bankroll': self.best_bankroll,
            'warmup_episodes': self.warmup_episodes
        }, filename)
        self.last_save_winrate = self.win_rate()

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state'])
        self.target_model.load_state_dict(checkpoint['target_model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon = checkpoint['epsilon']
        self.wins = checkpoint['wins']
        self.losses = checkpoint['losses']
        self.memory = deque(checkpoint.get('memory', []), maxlen=200000)
        self.last_save_winrate = checkpoint['last_save_winrate']
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.best_val_winrate = checkpoint.get('best_val_winrate', 0)
        self.best_bankroll = checkpoint.get('best_bankroll', 2000)
        self.warmup_episodes = checkpoint.get('warmup_episodes', 500)

    def win_rate(self):
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0

def card_to_str(card):
    if card == 1: return 'A'
    if card == 11: return 'J'
    if card == 12: return 'Q'
    if card == 13: return 'K'
    return str(card)

def print_game_stats(episode, agent, env, action, reward, confidence, action_history, game_info):
    """Enhanced game stats with guaranteed dealer hit display"""
    player_hands = []
    for hand in env.player_hands:
        player_hands.append(' '.join([card_to_str(card) for card in hand]))
    dealer_hand = ' '.join([card_to_str(card) for card in env.dealer_cards])

    action_names = ['HIT', 'STAND', 'DOUBLE', 'SPLIT', 'INSURANCE', 'NO INSURANCE']
    action_str = action_names[action]

    action_sequence = ""
    for act, card in action_history:
        if act in [0, 2]:
            action_sequence += f" → {action_names[act]}: {card_to_str(card)}"

    if action == 4:
        action_sequence = " → INSURANCE"
        if hasattr(env, 'last_outcome') and env.last_outcome is not None:
            if "INSURANCE_WIN" in env.last_outcome:
                action_sequence += " → INSURANCE_WIN"
            elif "INSURANCE_LOSE" in env.last_outcome:
                action_sequence += " → INSURANCE_LOSE"

    if not action_sequence or action != 4:
        outcome = env.last_outcome if hasattr(env, 'last_outcome') and env.last_outcome is not None else None
        if outcome is None:
            player_sum = env._sum_hand(env.player_hands[0]) if env.player_hands else 0
            dealer_sum = env._sum_hand(env.dealer_cards)
            if player_sum > 21:
                outcome = "BUST"
            elif dealer_sum > 21:
                outcome = "WIN"
            elif player_sum == dealer_sum:
                outcome = "DRAW"
            elif player_sum > dealer_sum:
                outcome = "WIN"
            else:
                outcome = "LOSE"
        action_sequence += f" → {outcome}"

    counts = env.card_counter.get_counts()
    hi_lo = counts.get('hi_lo', {})
    true_count = hi_lo.get('running', 0) / max(1, hi_lo.get('decks_remaining', 1))

    print(f"\n=== Episode {episode + 1} ===")
    print(f"Player hand: {player_hands} (Total: {env._sum_hand(env.player_hands[0]) if env.player_hands else 0})")
    print(f"Dealer hand: {dealer_hand} (Total: {env._sum_hand(env.dealer_cards)})")

    # Display dealer hits from game_info
    dealer_hits = game_info.get('dealer_hits', [])
    if dealer_hits:
        print(f"Dealer hits: {' '.join([card_to_str(card) for card in dealer_hits])}")
    else:
        print("Dealer did not hit.")

    print(f"Action: {action_str}{action_sequence}")
    print(f"Current bet: {env.current_bet} (Bankroll: {env.bankroll})")
    print(f"True count: {true_count:.2f}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Current win rate: {agent.win_rate()*100:.2f}% ({agent.wins}W/{agent.losses}L)")
    print(f"Last save win rate: {agent.last_save_winrate*100:.2f}%")
    print(f"Epsilon: {agent.epsilon:.4f}")
    print(f"Time elapsed: {time.strftime('%H:%M:%S', time.gmtime(time.time() - agent.start_time))}")
    print(f"Warmup episodes remaining: {max(0, agent.warmup_episodes - len(agent.episode_rewards))}")

def run_validation(agent, env, episodes=200):
    val_rewards = []
    val_wins = 0
    val_losses = 0
    
    for _ in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            valid_actions_mask = env.get_valid_actions()
            action = agent.act(state, valid_actions_mask)
            step_return = env.step(action)
            
            if isinstance(step_return, tuple):
                next_state, reward, done, _ = step_return
            else:
                next_state, reward, done = step_return[0], step_return[1], step_return[2]
            
            episode_reward += reward
            state = next_state
            
        if episode_reward > 0:
            val_wins += 1
        elif episode_reward < 0:
            val_losses += 1
            
    val_win_rate = val_wins / (val_wins + val_losses) if (val_wins + val_losses) > 0 else 0
    return val_win_rate

def train_agent():
    env = BlackjackENV(infinite_deck=False, num_decks=6)
    state_size = 6
    action_size = 6
    agent = DQNAgent(state_size, action_size)
    episodes = 10000
    batch_size = 256
    save_interval = 250
    print_interval = 50

    os.makedirs("models", exist_ok=True)

    for e in range(episodes):
        if agent.should_terminate_session(env):
            current_win_rate = agent.win_rate()
            counts = env.card_counter.get_counts()
            hi_lo = counts.get('hi_lo', {})
            true_count = hi_lo.get('running', 0) / max(1.0, hi_lo.get('decks_remaining', 1))
            print(f"\nSession stopped to protect bankroll after {e} episodes")
            print(f"Final win rate: {current_win_rate*100:.2f}%")
            print(f"Bankroll: {env.bankroll}")
            print(f"True count: {true_count:.2f}")
            break
            
        state = env.reset()
        total_reward = 0.0
        done = False
        action_history = []
        game_info = {}
        
        while not done:
            if env.current_hand_index >= len(env.player_hands):
                break
                
            valid_actions_mask = env.get_valid_actions()
            valid_actions = [i for i, valid in enumerate(valid_actions_mask) if valid]
            
            if not valid_actions:
                break
                
            action = agent.act(state, valid_actions_mask)
            confidence = agent.get_confidence(state, action)
            
            if len(agent.episode_rewards) < agent.warmup_episodes:
                bet_multiplier = agent.min_bet_multiplier
            else:
                bet_multiplier = 1.0
                
            true_count = state[5] * 10
            env.current_bet = min(
                env.max_bet,
                max(
                    env.min_bet,
                    int(env.bankroll * 0.01 * bet_multiplier * (1 + true_count/10))
                )
            )
            
            pre_action_hands = [hand.copy() for hand in env.player_hands]
            current_hand_index = env.current_hand_index
            
            step_return = env.step(action)
            if isinstance(step_return, tuple):
                next_state, reward, done, step_info = step_return
                game_info.update(step_info)
            else:
                next_state, reward, done = step_return[0], step_return[1], step_return[2]
            
            try:
                reward = float(reward)
            except (TypeError, ValueError):
                reward = 0.0
            
            shaped_reward = reward
            true_count = state[5] * 10
            
            if action == 2:
                shaped_reward *= 1.5 if true_count > 1 else 0.8
            elif action == 3:
                shaped_reward *= 1.3 if true_count > 2 else 0.9
            elif action == 4:
                shaped_reward = -3 if true_count < 3 else shaped_reward*3
            
            total_reward += shaped_reward
            
            if action in [0, 2] and current_hand_index < len(env.player_hands):
                current_hand = env.player_hands[current_hand_index]
                if len(current_hand) > len(pre_action_hands[current_hand_index]):
                    drawn_card = current_hand[-1]
                    action_history.append((action, drawn_card))
            
            agent.remember(state, action, shaped_reward, next_state, done)
            state = next_state

            if reward > 0:
                agent.wins += 1
            elif reward < 0:
                agent.losses += 1

            if env.current_hand_index < len(env.player_hands) and (e % print_interval == 0):
                print_game_stats(e, agent, env, action, reward, confidence, action_history, game_info)
            
            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)
                if e % print_interval == 0:
                    agent.writer.add_scalar('Loss/train', loss, e)
        
        agent.episode_rewards.append(total_reward)
        agent.writer.add_scalar('Reward/train', total_reward, e)
        agent.writer.add_scalar('WinRate/train', agent.win_rate(), e)
        
        if env.bankroll > agent.best_bankroll:
            agent.best_bankroll = env.bankroll
        
        if (e + 1) % 100 == 0 and len(agent.episode_rewards) < agent.warmup_episodes:
            val_win_rate = run_validation(agent, env, episodes=50)
            agent.writer.add_scalar('WinRate/val', val_win_rate, e)
            print(f"\nValidation Win Rate: {val_win_rate:.2f}")
            if val_win_rate > agent.best_val_winrate:
                agent.best_val_winrate = val_win_rate
                agent.save(f"models/dqn_best_val.pth")
        elif (e + 1) % 500 == 0:
            val_win_rate = run_validation(agent, env)
            agent.writer.add_scalar('WinRate/val', val_win_rate, e)
            print(f"\nValidation Win Rate: {val_win_rate:.2f}")
            if val_win_rate > agent.best_val_winrate:
                agent.best_val_winrate = val_win_rate
                agent.save(f"models/dqn_best_val.pth")
        
        if (e + 1) % save_interval == 0:
            agent.save(f"models/dqn_model_{e+1}.pth")
            print(f"\nSaved model at episode {e+1} with win rate {agent.win_rate()*100:.2f}%")

    agent.save("models/dqn_final.pth")

if __name__ == "__main__":
    train_agent()
