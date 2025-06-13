import asyncio
import websockets
import json
import logging
import time
import numpy as np
from rl_model import BlackjackENV
from DQNAgentTrainer import DQNAgent
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RLDecisionMaker:
    def __init__(self):
        self.env = BlackjackENV()
        state_size = 6
        action_size = 6
        self.agent = DQNAgent(state_size, action_size)
        self.load_model("models/dqn_final.pth")
        self.aggression = 1.0
        self.min_confidence = 0.4
        self.warmup_episodes = 10
        self.min_bet_multiplier = 0.5
        self.conservative_threshold = 17  # New threshold

    def load_model(self, filename):
        try:
            import torch.serialization
            import numpy.core.multiarray
            torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])
            
            checkpoint = torch.load(filename, weights_only=False)
            self.agent.model.load_state_dict(checkpoint['model_state'])
            self.agent.target_model.load_state_dict(checkpoint['target_model_state'])
            self.agent.optimizer.load_state_dict(checkpoint['optimizer_state'])
            
            # Restore agent state
            self.agent.epsilon = checkpoint.get('epsilon', 1.0)
            self.agent.wins = checkpoint.get('wins', 0)
            self.agent.losses = checkpoint.get('losses', 0)
            self.agent.memory = deque(checkpoint.get('memory', []), maxlen=200000)
            self.agent.last_save_winrate = checkpoint.get('last_save_winrate', 0)
            self.agent.episode_rewards = checkpoint.get('episode_rewards', [])
            self.agent.best_val_winrate = checkpoint.get('best_val_winrate', 0)
            self.agent.best_bankroll = checkpoint.get('best_bankroll', 2000)
            self.agent.warmup_episodes = checkpoint.get('warmup_episodes', 10)  # Changed to 10
            
            logger.info(f"Successfully loaded model from {filename}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.agent = DQNAgent(6, 6)

    def set_aggression(self, level):
        self.aggression = max(0.5, min(2.0, level))

    def get_counts(self):
        counts = self.env.card_counter.get_counts()
        return counts if counts else {"hi_lo": {"running": 0, "decks_remaining": 1}}

    def make_decision(self, game_state):
        try:
            player_total = game_state["player_total"]
            
            # Force stand on threshold+
            # Rest of existing decision logic...
            dealer_upcard = game_state["dealer_upcard_value"]
            cards = game_state["player_cards"]
            is_soft = game_state.get("is_soft", False)
            
            counts = self.get_counts()
            hi_lo = counts.get('hi_lo', {})
            decks_remaining = hi_lo.get('decks_remaining', 1.0)
            true_count = hi_lo.get('running', 0) / max(1.0, decks_remaining)
            
            obs = np.array([
                player_total / 32.0,
                dealer_upcard / 11.0,
                float(is_soft),
                float(len(cards) == 2),
                float(len(cards) == 2 and cards[0] == cards[1]),
                true_count
            ], dtype=np.float32)

            self.env.player_hands = [cards]
            self.env.dealer_cards = [dealer_upcard, 0]
            self.env.insurance_offered = game_state.get("insurance_offered", False)
            valid_actions = self.env.get_valid_actions()
            if not game_state.get("can_double", True):
                valid_actions[2] = False
            if not game_state.get("can_split", True):
                valid_actions[3] = False

            action = self.agent.act(obs, valid_actions)
            confidence = self.agent.get_confidence(obs, action)

            if not valid_actions[action]:
                logger.warning(f"Model selected invalid action {action}, forcing HIT")
                action = 1 if valid_actions[1] else 0  # fallback to stand or hit
            
            if len(self.agent.episode_rewards) < self.warmup_episodes:
                bet_multiplier = self.min_bet_multiplier
            else:
                bet_multiplier = 1.0
                
            base_bet = min(
                self.env.max_bet,
                max(
                    self.env.min_bet,
                    int(self.env.bankroll * 0.01 * bet_multiplier * (1 + true_count/10))
                )
            )
            
            if confidence < self.min_confidence or (player_total <= 11 and confidence < 0.5):
                return {
                    "action": "stand",
                    "confidence": confidence,
                    "bet": 0,
                    "count_data": counts
                }
                
            bet = min(
                self.env.max_bet,
                base_bet * min(3.0, max(0.5, confidence * self.aggression))
            )
            
            if (game_state.get("insurance_offered", False) and true_count < 3):
                return {
                    "action": "no_insurance",
                    "confidence": 1.0,
                    "bet": bet,
                    "count_data": counts
                }

            action_map = {
                0: "hit",
                1: "stand",
                2: "double",
                3: "split",
                4: "insurance",
                5: "no_insurance"
            }

            return {
                "action": action_map[action],
                "confidence": confidence,
                "bet": bet,
                "count_data": counts
            }

        except Exception as e:
            logger.error(f"Decision error: {e}")
            return {
                "action": "stand",
                "confidence": 0.5,
                "bet": 0,
                "count_data": {"hi_lo": {"running": 0, "decks_remaining": 1}}
            }

async def handler(websocket):
    logging.info("New WebSocket connection established")
    decision_maker = RLDecisionMaker()
    decision_maker.set_aggression(1.5)  # Moderately aggressive
    reconnect_attempts = 0
    max_reconnect_attempts = 5
    last_message_time = time.time()
    keepalive_interval = 30  # seconds

    while reconnect_attempts < max_reconnect_attempts:
        try:
            async for message in websocket:
                try:
                    last_message_time = time.time()
                    data = json.loads(message)
                    logging.info(f"Received message: {data}")

                    # Handle connection closure gracefully
                    if data.get("action") == "connection_close":
                        await websocket.close(code=1000)
                        logging.info("Connection closed gracefully")
                        return

                    if data.get("action") == "ping":
                        await websocket.send(json.dumps({"action": "pong"}))
                        continue

                    elif data.get("action") == "handshake":
                        await websocket.send(json.dumps({
                            "action": "handshake_response",
                            "status": "connected",
                            "version": "2.0",
                            "aggression": decision_maker.aggression
                        }))
                        continue

                    elif data.get("action") == "game_state":
                        game_state = data.get("payload", {})
                        
                        required_fields = ['player_cards', 'player_total', 'dealer_upcard_value']
                        if not all(field in game_state for field in required_fields):
                            logging.error(f"Missing required fields in game state: {game_state}")
                            await websocket.send(json.dumps({
                                "action": "error",
                                "message": "Missing required game state fields"
                            }))
                            continue

                        game_state.setdefault('is_soft', False)
                        game_state.setdefault('can_double', False)
                        game_state.setdefault('can_split', False)
                        game_state.setdefault('insurance_offered', False)

                        try:
                            decision = decision_maker.make_decision(game_state)
                            response = {
                                "action": "decision",
                                "decision": decision["action"],
                                "confidence": float(decision["confidence"]),
                                "bet": float(decision["bet"]),
                                "count_data": decision["count_data"],
                                "timestamp": data.get("timestamp", time.time()),
                                "player_total": game_state["player_total"],
                                "dealer_upcard": game_state["dealer_upcard_value"],
                                "aggression": decision_maker.aggression
                            }
                            await websocket.send(json.dumps(response))
                        except Exception as e:
                            logging.error(f"Error making decision: {e}")
                            await websocket.send(json.dumps({
                                "action": "error",
                                "message": str(e),
                                "timestamp": data.get("timestamp", time.time())
                            }))

                except json.JSONDecodeError as e:
                    logging.error(f"Invalid JSON received: {e}")
                    await websocket.send(json.dumps({
                        "action": "error",
                        "message": "Invalid JSON format",
                        "timestamp": time.time()
                    }))
                except Exception as e:
                    logging.error(f"Error processing message: {e}")
                    await websocket.send(json.dumps({
                        "action": "error",
                        "message": str(e),
                        "timestamp": time.time()
                    }))
                
                # Check if we need to send a keepalive
                if time.time() - last_message_time > keepalive_interval:
                    try:
                        await websocket.send(json.dumps({"action": "ping"}))
                        last_message_time = time.time()
                    except:
                        break

        except websockets.exceptions.ConnectionClosed as e:
            reconnect_attempts += 1
            if reconnect_attempts < max_reconnect_attempts:
                wait_time = min(30, 2 ** reconnect_attempts)
                logging.warning(f"Connection closed (code {e.code}), reconnecting in {wait_time}s... Attempt {reconnect_attempts}/{max_reconnect_attempts}")
                await asyncio.sleep(wait_time)
                continue
            else:
                logging.error(f"Max reconnection attempts reached. Last error: {e}")
                break
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            break

    logging.info("WebSocket handler terminating")

async def main():
    server = await websockets.serve(
        handler,
        "127.0.0.1",
        6789,
        ping_interval=20,
        ping_timeout=60,
        compression="deflate",
        max_size=2**20
    )
    
    logging.info(f"Server started on ws://127.0.0.1:6789")
    
    try:
        await server.wait_closed()
    except KeyboardInterrupt:
        logging.info("Server shutting down due to keyboard interrupt")
    finally:
        logging.info("Server stopped")

if __name__ == "__main__":
    asyncio.run(main())