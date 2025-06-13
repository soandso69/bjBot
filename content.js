let decisionInProgress = false;

window.addEventListener("load", () => {
  console.log("✅ Page fully loaded. Enabling AutoPlay.");
});

// ---------------------------
// Config Loader
// ---------------------------
let config = {
  betting: {
    base_bet: 0.01,
    max_bet: 0.03,
    bet_percentage: 0.01
  },
  websocket_url: "ws://127.0.0.1:6789"
};
const MIN_BET = config.betting.base_bet;
const SCENARIO_GAME = "none";
const SCENARIO_WIN = "win";
const SCENARIO_LOSE = "lose";

// ---------------------------
// DOM Selectors
// ---------------------------
const PLAYER_HAND_GAME_START = "#main-content > div.parent.svelte-1ydxan2 > div > div > div > div > div.content.svelte-1ku0r3 > div.game-content > div > div.hands > div.player > div > div > div.value.none";
const PLAYER_HAND_WIN = "#main-content > div.parent.svelte-1ydxan2 > div > div > div > div > div.content.svelte-1ku0r3 > div.game-content > div > div.hands > div.player > div > div > div.value.win";
const PLAYER_HAND_LOSE = "#main-content > div.parent.svelte-1ydxan2 > div > div > div > div > div.content.svelte-1ku0r3 > div.game-content > div > div.hands > div.player > div > div > div.value.lose";
const BALANCE_SELECTOR = "span[class*='balance'], div[class*='balance'] span, #svelte > div.wrap.svelte-2gw7o8 > div.main-content.svelte-2gw7o8 > div.navigation.svelte-1nt2705 > div > div > div > div.balance-toggle.svelte-1o8ossz > div > div > div > button > div > div > span.content.svelte-didcjq > span";
const BET_AMOUNT_SELECTOR = "input[type='number'], input[class*='bet'], input[name*='amount'], [data-testid='bet-input']";

// ---------------------------
// Game Constants
// ---------------------------
const HIT_BUTTON = "hit";
const DOUBLE_BUTTON = "double";
const SPLIT_BUTTON = "split";
const STAND_BUTTON = "stand";
const NO_INSURANCE_BUTTON = "noInsurance";
const INSURANCE_BUTTON = "insurance";
const HALF_BET_BUTTON = "amount-halve";
const DOUBLE_BET_BUTTON = "amount-double";
const PLAY_BUTTON = "bet-button";

let selectors = {};
let BASE_BET = config.betting.base_bet;
let MAX_BET = config.betting.max_bet;
let BET_PERCENTAGE = config.betting.bet_percentage;

// ---------------------------
// Helper Functions
// ---------------------------
function waitForDOM() {
  return new Promise(resolve => {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", resolve);
    } else {
      resolve();
    }
  });
}

async function waitForButton(selector, timeout = 15000) {
  console.log(`🔍 Waiting for button: ${selector}`);
  const start = Date.now();
  while (Date.now() - start < timeout) {
    const button = document.querySelector(selector);
    if (button && button.offsetWidth > 0 && !button.disabled) return button;
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  return null;
}

async function calculateBaseBet() {
  let balanceElement = document.querySelector(BALANCE_SELECTOR);
  if (!balanceElement) {
    console.warn("Balance element not found. Using minimum bet.");
    return MIN_BET;
  }
  const balanceText = balanceElement.textContent;
  const balance = parseFloat(balanceText.replace(/[^0-9.]/g, ""));
  return isNaN(balance) ? MIN_BET : Math.max(balance * BET_PERCENTAGE, MIN_BET);
}

function calculateWinProbability(playerTotal) {
  if (playerTotal >= 19) return 0.85;
  if (playerTotal >= 17) return 0.70;
  if (playerTotal >= 13) return 0.50;
  if (playerTotal == 12) return 0.40;
  if (playerTotal == 11) return 0.65;
  if (playerTotal == 10) return 0.60;
  if (playerTotal == 9) return 0.40;
  return 0.30;
}

function createOrUpdateHUD() {
  let hud = document.getElementById('blackjack-hud');
  if (!hud) {
    hud = document.createElement('div');
    hud.id = 'blackjack-hud';
    hud.style.cssText = `
      position: fixed;
      bottom: 50px;
      right: 10px;
      background-color: rgba(0, 0, 0, 0.85);
      color: white;
      padding: 15px;
      border-radius: 5px;
      z-index: 9999;
      width: 300px;
      font-family: Arial, sans-serif;
    `;
    hud.innerHTML = `
      <div class="hud-section" style="margin-bottom: 10px;">
        <h3 style="margin: 0 0 5px 0; border-bottom: 1px solid #444;">Game Stats</h3>
        <div>Player Total: <span id="player-total">0</span></div>
        <div>Current Balance: <span id="current-balance">0</span></div>
      </div>
      <div class="hud-section" style="margin-bottom: 10px;">
        <h3 style="margin: 0 0 5px 0; border-bottom: 1px solid #444;">AI Decision</h3>
        <div>Last Action: <span id="last-action">None</span></div>
        <div>Confidence: <span id="action-confidence">0%</span></div>
        <div>Win Probability: <span id="win-probability">0%</span></div>
      </div>
      <div class="hud-section">
        <h3 style="margin: 0 0 5px 0; border-bottom: 1px solid #444;">Session Stats</h3>
        <div>Profit/Loss: <span id="profit-loss">0</span></div>
        <div>Hands Played: <span id="hands-played">0</span></div>
        <div>Wins: <span id="wins">0</span></div>
        <div>Losses: <span id="losses">0</span></div>
        <div>Win Rate: <span id="win-rate">0%</span></div>
        <div>Connection Status: <span id="connection-status">Disconnected</span></div>
      </div>
    `;
    document.body.appendChild(hud);
  }
  return hud;
}

// ---------------------------
// Selector Management
// ---------------------------
async function updateSelectors() {
  let newSelectors = {
    player_hand: [PLAYER_HAND_GAME_START, PLAYER_HAND_WIN, PLAYER_HAND_LOSE],
    [HIT_BUTTON]: "[data-test-action='hit']",
    [DOUBLE_BUTTON]: "[data-test-action='double']",
    [SPLIT_BUTTON]: "[data-test-action='split']",
    [STAND_BUTTON]: "[data-test-action='stand']",
    [NO_INSURANCE_BUTTON]: "[data-test-action='noInsurance']",
    [INSURANCE_BUTTON]: "[data-test-action='insurance']",
    [HALF_BET_BUTTON]: "[data-testid='amount-halve']",
    [DOUBLE_BET_BUTTON]: "[data-testid='amount-double']",
    [PLAY_BUTTON]: "[data-testid='bet-button']"
  };

  if (location.hostname.includes("sharkoin.com")) {
    newSelectors = {
      [HIT_BUTTON]: "#blackjack-hit",
      [DOUBLE_BUTTON]: "#blackjack-double",
      [SPLIT_BUTTON]: "#blackjack-split",
      [STAND_BUTTON]: "#blackjack-stand",
      [NO_INSURANCE_BUTTON]: "#insurance-question > div > div > div.insurance-question__buttons > button.insurance-question__buttons--no",
      [INSURANCE_BUTTON]: "#insurance-question > div > div > div.insurance-question__buttons > button:nth-child(1)",
      [HALF_BET_BUTTON]: "#blackjack-span.half-bet",
      [DOUBLE_BET_BUTTON]: "#blackjack-span.double-bet",
      [PLAY_BUTTON]: "#blackjack-deal",
      player_hand: [".bj-player__hands .bj-player__summary span"]
    };
  } else if (location.hostname.includes("dingdingding.com")) {
    newSelectors = {
      [HIT_BUTTON]: ".dd-hit",
      [DOUBLE_BUTTON]: ".dd-double",
      [SPLIT_BUTTON]: ".dd-split",
      [STAND_BUTTON]: ".dd-stand",
      [NO_INSURANCE_BUTTON]: ".dd-noinsurance",
      [INSURANCE_BUTTON]: ".dd-insurance",
      [HALF_BET_BUTTON]: ".dd-halfbet",
      [DOUBLE_BET_BUTTON]: ".dd-doublebet",
      [PLAY_BUTTON]: ".dd-play",
      player_hand: [".dd-player .hand"]
    };
  } else if (location.hostname.includes("cafecasino.lv")) {
    newSelectors = {
      [HIT_BUTTON]: "#cafe-hit",
      [DOUBLE_BUTTON]: "#cafe-double",
      [SPLIT_BUTTON]: "#cafe-split",
      [STAND_BUTTON]: "#cafe-stand",
      [NO_INSURANCE_BUTTON]: "#cafe-noinsurance",
      [INSURANCE_BUTTON]: "#cafe-insurance",
      [HALF_BET_BUTTON]: "#cafe-halfbet",
      [DOUBLE_BET_BUTTON]: "#cafe-doublebet",
      [PLAY_BUTTON]: "#cafe-play",
      player_hand: ["#cafe-player .hand"]
    };
  } else if (location.hostname.includes("bovada.lv")) {
    newSelectors = {
      [HIT_BUTTON]: ".bov-hit",
      [DOUBLE_BUTTON]: ".bov-double",
      [SPLIT_BUTTON]: ".bov-split",
      [STAND_BUTTON]: ".bov-stand",
      [NO_INSURANCE_BUTTON]: ".bov-noinsurance",
      [INSURANCE_BUTTON]: ".bov-insurance",
      [HALF_BET_BUTTON]: ".bov-halfbet",
      [DOUBLE_BET_BUTTON]: ".bov-doublebet",
      [PLAY_BUTTON]: ".bov-play",
      player_hand: [".bov-player .hand"]
    };
  } else if (location.hostname.includes("luckybird.io")) {
    newSelectors = {
      [HIT_BUTTON]: "#lb-hit",
      [DOUBLE_BUTTON]: "#lb-double",
      [SPLIT_BUTTON]: "#lb-split",
      [STAND_BUTTON]: "#lb-stand",
      [NO_INSURANCE_BUTTON]: "#lb-noinsurance",
      [INSURANCE_BUTTON]: "#lb-insurance",
      [HALF_BET_BUTTON]: "#lb-halfbet",
      [DOUBLE_BET_BUTTON]: "#lb-doublebet",
      [PLAY_BUTTON]: "#lb-play",
      player_hand: ["#lb-player .hand"]
    };
  }

  Object.assign(selectors, newSelectors);
}

async function updateSelectorsFallback() {
  for (const key in selectors) {
    let element;
    if (Array.isArray(selectors[key])) {
      element = selectors[key].map(sel => document.querySelector(sel)).find(el => el);
    } else {
      element = document.querySelector(selectors[key]);
    }
    if (!element && key !== NO_INSURANCE_BUTTON && key !== INSURANCE_BUTTON) {
      element = Array.from(document.querySelectorAll("button")).find(
        btn => btn.textContent.trim().toLowerCase().includes(key.toLowerCase())
      );
      if (element) {
        const idOrClass = element.id ? `#${element.id}` : element.className ? `.${element.className.split(" ").join(".")}` : null;
        if (idOrClass) {
          selectors[key] = idOrClass;
          chrome.storage.local.set({ [key]: selectors[key] });
        }
      }
    }
  }
}

// ---------------------------
// GameObserver Class
// ---------------------------
class GameObserver {
  constructor() {
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 10;
    this.lastDecision = null;
    this.autoPlayEnabled = false;
    this.isInGame = false;
    this.isReconnecting = false;
    this.websocket = null;
    this.splitInProgress = false;
    this.waitingForAction = false;
    this.remainingHands = 0;
    this.heartbeatInterval = null;
    this.stateSent = false;
    this.handsPlayed = 0;
    this.handsWon = 0;
    this.startingBalance = null;
    this.profit = 0;
    this.lastBetAmount = 0;
    this.sessionStartTime = Date.now();
    this.trainingActive = false;
    this.actionsContainerSelector = '[data-test="actions-insurance"], .actions';

    this.init();
  }

  init() {
    createOrUpdateHUD();
    this.connectWebSocket();
    this.loadStats();

    // Initialize targeted MutationObserver
    this.observer = new MutationObserver((mutations) => {
      const relevantMutations = mutations.filter(mutation => {
        // Check if mutation is in our actions container or its children
        const isRelevant = mutation.target.matches(this.actionsContainerSelector) || 
               Array.from(mutation.addedNodes).some(node => 
                 node.matches && node.matches(this.actionsContainerSelector + ' *'));
        
        return isRelevant;
      });
      
      if (relevantMutations.length > 0) {
        console.log("DOM change detected in actions container");
        this.checkGameState();
      }
    });

    // Try to find and observe the actions container
    const actionsContainer = document.querySelector(this.actionsContainerSelector);
    if (actionsContainer) {
      console.log("Observing actions container:", actionsContainer);
      this.observer.observe(actionsContainer, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['disabled', 'style', 'class']
      });
    } else {
      console.warn("Actions container not found, falling back to document.body");
      this.observer.observe(document.body, {
        childList: true,
        subtree: true
      });
    }

    // Initial check
    this.checkGameState();
  }

  connectWebSocket() {
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    this.splitInProgress = false;
    this.waitingForAction = false;
    this.remainingHands = 0;
    }

    console.log(`🔌 Connecting to WebSocket at ${config.websocket_url}`);
    this.updateConnectionStatus("Connecting...");

    try {
      this.websocket = new WebSocket(config.websocket_url);
      
      this.websocket.onopen = () => {
        console.log("✅ WebSocket connected successfully");
        this.updateConnectionStatus("Connected");
        this.reconnectAttempts = 0;
        this.isReconnecting = false;
        
        // Send handshake
        this.sendMessage({
          action: "handshake",
          type: "blackjack_client",
          version: "1.0",
          site: window.location.hostname
        });
        
        if (this.autoPlayEnabled) {
          this.startAutoPlayLoop();
        }
      };
      
      this.websocket.onerror = (error) => {
        console.error("WebSocket error:", error);
        this.updateConnectionStatus(`Error: ${error.message}`);
        if (!this.isReconnecting) {
          this.reconnect();
        }
      };
      
      this.websocket.onclose = (event) => {
        console.warn(`WebSocket closed (code: ${event.code}, reason: ${event.reason})`);
        this.updateConnectionStatus("Disconnected");
        if (!this.isReconnecting && this.autoPlayEnabled) {
          this.reconnect();
        }
      };
      
      this.websocket.onmessage = (event) => {
    try {
        const message = JSON.parse(event.data);
        console.log("Received message:", message);
        
        if (message.action === "decision") {
            console.log("🔄 Received decision:", message.decision);
            this.updateLastAction(message.decision);
            this.lastDecision = message;
            
            // Update HUD with confidence
            const confidenceElem = document.getElementById("action-confidence");
            if (confidenceElem) {
                confidenceElem.textContent = `${(message.confidence * 100).toFixed(0)}%`;
                confidenceElem.style.color = message.confidence > 0.7 ? "green" : 
                                          message.confidence < 0.4 ? "red" : "yellow";
            }

            this.waitingForAction = true;
            this.executeDecision();

        }
        else if (message.action === "handshake_response") {
            console.log("🤝 Handshake complete with server");
        }
        else if (message.action === "error") {
            console.error("Server error:", message.message);
        }
    } catch (e) {
        console.error("Error parsing message:", e, "Raw data:", event.data);
    }
};
    } catch (error) {
      console.error("WebSocket initialization error:", error);
      this.reconnect();
    }
  }

  sendMessage(message) {
    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
        
      try {
        console.log("Sending:", message.action);
        const messageStr = JSON.stringify(message);
        this.websocket.send(messageStr);
        console.log("Sent message:", message);
      } catch (error) {
        console.error("Error sending message:", error);
      }
    } else {
      console.warn("Cannot send message - WebSocket not connected");
    }
  }

  updateConnectionStatus(status) {
    const statusElem = document.getElementById("connection-status");
    if (statusElem) {
      statusElem.textContent = status;
      statusElem.style.color = status === "Connected" ? "green" : "red";
    }
  }

  updateLastAction(action) {
    const actionElem = document.getElementById("last-action");
    if (actionElem) {
      actionElem.textContent = action;
      actionElem.style.color = "yellow";
    }
  }

  loadStats() {
    chrome.storage.local.get(['blackjackStats'], (result) => {
      if (result.blackjackStats) {
        this.handsPlayed = result.blackjackStats.handsPlayed || 0;
        this.handsWon = result.blackjackStats.handsWon || 0;
        this.profit = result.blackjackStats.profit || 0;
        this.sessionStartTime = result.blackjackStats.sessionStartTime || Date.now();
      }
    });
  }

  saveStats() {
    const stats = {
      handsPlayed: this.handsPlayed,
      handsWon: this.handsWon,
      profit: this.profit,
      sessionStartTime: this.sessionStartTime
    };
    chrome.storage.local.set({ 'blackjackStats': stats });
  }

  reconnect() {
    if (this.isReconnecting || this.reconnectAttempts >= this.maxReconnectAttempts) return;
    this.isReconnecting = true;
    const delay = Math.min(3000, 500 * Math.pow(2, this.reconnectAttempts));
    setTimeout(() => {
      this.reconnectAttempts++;
      this.connectWebSocket();
      this.isReconnecting = false;
    }, delay);
  }

  async checkGameState() {
    const hitButton = document.querySelector(selectors[HIT_BUTTON]);
    const standButton = document.querySelector(selectors[STAND_BUTTON]);
    
    // More rigorous check for in-game state
    const isInGameNow = hitButton && standButton && 
                       hitButton.offsetWidth > 0 && 
                       standButton.offsetWidth > 0 &&
                       !hitButton.disabled && 
                       !standButton.disabled &&
                       hitButton.getAttribute('data-test-action-enabled') === 'true' &&
                       standButton.getAttribute('data-test-action-enabled') === 'true';
    
    if (isInGameNow && !this.isInGame) {
      console.log("🎲 Transition to in-game state detected");
      this.handleInGame();
    } else if (!isInGameNow && this.isInGame) {
      console.log("Transition out of in-game state detected");
      
    }
  }

  handleInGame() {
    if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
      console.error("WebSocket not ready, cannot send game state");
      this.reconnect();
      return;
    }

    const gameState = this.getGameState();
    if (!gameState) {
      console.error("Could not get valid game state");
      return;
    }

    console.log("📤 Sending game state to server");
    this.sendMessage({
      action: "game_state",
      payload: gameState,
      timestamp: Date.now()
    });

    
    this.isInGame = true;
  }

  async startAutoPlayLoop() {
    if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
      console.error("Cannot start autoplay - WebSocket not connected");
      return;
    }

    console.log("🔄 Starting autoplay loop");
    
    const runLoop = async () => {
    if (this.waitingForAction) {
      console.log("⏳ Waiting for previous action to complete");
      setTimeout(runLoop, 500);
      return;
    }
      if (!this.autoPlayEnabled) {
        console.log("Autoplay disabled, stopping loop");
        return;
      }
      
      try {
        // Handle insurance if present
        const insuranceBtn = document.querySelector(selectors[INSURANCE_BUTTON]);
        const noInsuranceBtn = document.querySelector(selectors[NO_INSURANCE_BUTTON]);
        if (insuranceBtn || noInsuranceBtn) {
          console.log("🛡️ Insurance prompt detected");
          (noInsuranceBtn || insuranceBtn).click();
          await this.delay(2000);
          return runLoop();
        }

        // Start new round if not in game
        const hitBtn = document.querySelector(selectors[HIT_BUTTON]);
const standBtn = document.querySelector(selectors[STAND_BUTTON]);

const actionButtonsVisible =
  hitBtn && standBtn &&
  hitBtn.offsetParent !== null && !hitBtn.disabled &&
  standBtn.offsetParent !== null && !standBtn.disabled;

if (!actionButtonsVisible) {
  console.log("🎲 Starting new round (no action buttons found)");

  const balance = await this.getBalance();
  const betAmount = Math.min(
    Math.max(balance * BET_PERCENTAGE, BASE_BET), 
    MAX_BET
  );

  await this.adjustBet(betAmount);
  await this.clickPlay();
  await this.delay(3000);
  return runLoop();
}



        // Check if we need to send game state (if not already sent)
        if (!this.stateSent) {
          const gameState = await this.getGameState();
          if (gameState) {
            console.log("📤 Sending game state to server");
            this.sendMessage({
              action: "game_state",
              payload: gameState,
              timestamp: Date.now()
            });
            this.stateSent = true;
          }
        }


const actionStillPossible =
  hitBtn && standBtn &&
  hitBtn.offsetParent !== null &&
  !hitBtn.disabled &&
  !standBtn.disabled;

if (!actionStillPossible) {
  const outcome = await this.checkRoundOutcome();
  if (outcome) {
    console.log(`🏁 Round outcome: ${outcome}`);
    await this.handleGameOutcome(outcome);
    //await this.delay(2000);
    this.stateSent = false;
  }
}
// Outcome is now calculated via balance change — do not force win manually

      } catch (error) {
        console.error("Autoplay error:", error);
      }
      
      // Schedule next iteration
      setTimeout(runLoop, 1000);
    };

    // Start the loop
    runLoop();
  }

  async executeDecision() {
    if (!this.lastDecision?.decision || this.lastDecision.decision === "stand" && this.getPlayerHand?.total >= 21) {
    console.log("✅ Auto-stand triggered due to player total ≥ 21. Skipping manual click.");
    this.waitingForAction = false;
    
    setTimeout(() => this.clickPlay(), 2000); // start new round after 2s
    return;
}


    const buttonMap = {
        hit: selectors[HIT_BUTTON],
        stand: selectors[STAND_BUTTON],
        double: selectors[DOUBLE_BUTTON],
        split: selectors[SPLIT_BUTTON],
        insurance: selectors[INSURANCE_BUTTON],
        noInsurance: selectors[NO_INSURANCE_BUTTON]
    };

    const selector = buttonMap[this.lastDecision.decision];
    console.log("Using selector:", selector);
    if (!selector) {
        console.error("Invalid action:", this.lastDecision.decision);
        this.waitingForAction = false;
        return;
    }

    console.log(`🖱️ Attempting to click: ${this.lastDecision.decision}`);
    const button = await waitForButton(selector, 5000);
    
    if (button) {
        try {
            // First try standard click
            button.click();
            console.log(`✅ Clicked: ${this.lastDecision.decision}`);
            
            // Verify click worked
            await this.delay(1000);
            const stillVisible = document.querySelector(selector)?.offsetParent !== null;
            if (stillVisible) {
                console.log("⚠️ Button still visible, trying alternative click");
                button.dispatchEvent(new MouseEvent("click", {
                    bubbles: true,
                    cancelable: true,
                    view: window
                }));
            }
        } catch (e) {
            console.error("Click failed:", e);
        }
    } else {
        console.error(`❌ Button not found: ${this.lastDecision.decision}`);
    }
    
   this.waitingForAction = false;

setTimeout(() => {
  const hitBtn = document.querySelector(selectors[HIT_BUTTON]);
  const standBtn = document.querySelector(selectors[STAND_BUTTON]);

  const stillInGame = hitBtn && standBtn &&
    hitBtn.offsetParent !== null && !hitBtn.disabled &&
    standBtn.offsetParent !== null && !standBtn.disabled;

  this.isInGame = stillInGame;
  this.stateSent = false; // 🔥 Allow autoplay to send updated state
  console.log(`🧩 After executing ${this.lastDecision.decision}, still in game? ${this.isInGame}`);
}, 1000);



}

  async getPlayerHand() {
    const cardElements = document.querySelectorAll(".hands .player .card-value-badge, .hands .player .card .value, .hands .player .card-face span");
    if (!cardElements.length) return null;

    const cards = Array.from(cardElements).map(el => el.textContent.trim());
    let total = 0;
    let aces = 0;

    cards.forEach(card => {
      if (card === "A") {
        total += 11;
        aces++;
      } else if (["K", "Q", "J"].includes(card)) {
        total += 10;
      } else {
        total += parseInt(card) || 0;
      }
    });

    while (total > 21 && aces > 0) {
      total -= 10;
      aces--;
    }

    return {
      total,
      cards,
      isSoft: aces > 0
    };
  }

  async getDealerUpcard() {
    const dealerCard = document.querySelector(".hands .dealer .face-content span");
    if (!dealerCard) return null;

    const value = dealerCard.textContent.trim();
    let numericValue = 0;

    if (value === "A") {
      numericValue = 11;
    } else if (["K", "Q", "J"].includes(value)) {
      numericValue = 10;
    } else {
      numericValue = parseInt(value) || 0;
    }

    return {
      card: value,
      value: numericValue
    };
  }

  async getGameState() {
    try {
      const playerHand = await this.getPlayerHand();
      const dealerCard = await this.getDealerUpcard();
      const balance = await this.getBalance();

      const state = {
        player_total: playerHand?.total || 0,
        player_cards: playerHand?.cards || [],
        can_double: !!document.querySelector(selectors[DOUBLE_BUTTON]),
        can_split: playerHand?.cards?.length === 2 &&
                   playerHand.cards[0] === playerHand.cards[1],
        is_soft: playerHand?.isSoft || false,
        dealer_upcard: dealerCard?.card || null,
        dealer_upcard_value: dealerCard?.value || 0,
        balance: balance,
        max_bet: MAX_BET,
        base_bet: BASE_BET,
        timestamp: Date.now()
      };

      return state;
    } catch (error) {
      console.error("Error getting game state:", error);
      return null;
    }
  }

  async adjustBet(amount) {
    try {
      const balance = await this.getBalance();
      const validatedAmount = Math.min(
        Math.max(amount, BASE_BET),
        Math.min(MAX_BET, balance * BET_PERCENTAGE)
      );
      
      const finalAmount = isNaN(validatedAmount) ? BASE_BET : validatedAmount;
      
      const betInput = document.querySelector(BET_AMOUNT_SELECTOR);
      if (betInput) {
        betInput.value = finalAmount.toFixed(2);
        betInput.dispatchEvent(new Event("input", { bubbles: true }));
        this.lastBetAmount = finalAmount;
        await this.delay(500);
      }
    } catch (error) {
      console.error("Error adjusting bet:", error);
    }
  }

  async clickPlay() {
    const button = await waitForButton(selectors[PLAY_BUTTON]);
    if (button) {
      button.click();
      this.isInGame = true;
      await this.delay(2000);
    }
  }

  async checkRoundOutcome() {
    const winElement = document.querySelector(PLAYER_HAND_WIN);
    if (winElement && winElement.offsetParent !== null) {
      return "win";
    }

    const loseElement = document.querySelector(PLAYER_HAND_LOSE);
    if (loseElement && loseElement.offsetParent !== null) {
      return "lose";
    }

    const gameStartElement = document.querySelector(PLAYER_HAND_GAME_START);
    if (gameStartElement && gameStartElement.offsetParent !== null) {
      return null;
    }

    return null;
  }

  async handleGameOutcome(outcome) {
    this.isInGame = false;
    this.handsPlayed++;
    
    if (outcome === "win") {
      this.handsWon++;
      this.profit += this.lastBetAmount;
    } else {
      this.profit -= this.lastBetAmount;
    }

    this.saveStats();
    
    updateHUDData({
      player_total: 0,
      balance: await this.getBalance(),
      profit: this.profit,
      hands_played: this.handsPlayed,
      wins: this.handsWon
    });

    this.sendMessage({
      action: "game_outcome",
      outcome: outcome,
      bet_amount: this.lastBetAmount,
      balance: await this.getBalance(),
      hands_played: this.handsPlayed,
      win_rate: this.handsPlayed > 0 ? (this.handsWon / this.handsPlayed) : 0
    });

    await this.delay(2000);
    await this.clickPlay();
  }

  async getBalance() {
    const balanceElement = document.querySelector(BALANCE_SELECTOR);
    if (!balanceElement) return MIN_BET;
    
    const balanceText = balanceElement.textContent;
    const balance = parseFloat(balanceText.replace(/[^0-9.]/g, ""));
    return isNaN(balance) ? MIN_BET : balance;
  }

  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// ---------------------------
// Initialization and Listeners
// ---------------------------
function observeInsurancePrompt() {
  const observer = new MutationObserver(() => {
    const insuranceBtn = document.querySelector(selectors[INSURANCE_BUTTON]);
    const noInsuranceBtn = document.querySelector(selectors[NO_INSURANCE_BUTTON]);
    if (insuranceBtn || noInsuranceBtn) {
      (noInsuranceBtn || insuranceBtn).click();
      observer.disconnect();
    }
  });
  observer.observe(document.body, { childList: true, subtree: true });
}

function updateHUDData(data) {
  const elements = {
    'player-total': data.player_total,
    'current-balance': data.balance.toFixed(2),
    'profit-loss': data.profit.toFixed(2),
    'hands-played': data.hands_played,
    'wins': data.wins,
    'losses': data.hands_played - data.wins,
    'win-rate': data.hands_played > 0 
      ? `${((data.wins / data.hands_played) * 100).toFixed(1)}%`
      : "0%"
  };

  for (const [id, value] of Object.entries(elements)) {
    const elem = document.getElementById(id);
    if (elem) elem.textContent = value;
  }

  const winProbElem = document.getElementById('win-probability');
  if (winProbElem && data.player_total) {
    const prob = calculateWinProbability(data.player_total);
    winProbElem.textContent = `${(prob * 100).toFixed(0)}%`;
  }
}

let gameObserver;

waitForDOM().then(async () => {
  await updateSelectors();
  await updateSelectorsFallback();
  BASE_BET = Math.max(await calculateBaseBet(), MIN_BET);
  observeInsurancePrompt();
  gameObserver = new GameObserver();
  console.log("✅ Extension fully initialized");
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "toggleAutoPlay") {
    gameObserver.autoPlayEnabled = !gameObserver.autoPlayEnabled;
    console.log("Autoplay:", gameObserver.autoPlayEnabled ? "ON" : "OFF");
    
    if (gameObserver.autoPlayEnabled) {
      gameObserver.startAutoPlayLoop();
    }
    
    sendResponse({ autoPlayEnabled: gameObserver.autoPlayEnabled });
  }
});


function insertLearningHUD() {
  if (!document.querySelector('#learning-hud')) {
    const hud = document.createElement('div');
    hud.id = 'learning-hud';
    hud.style.position = 'fixed';
    hud.style.bottom = '10px';
    hud.style.left = '10px';
    hud.style.background = 'rgba(0,0,0,0.75)';
    hud.style.color = 'lime';
    hud.style.padding = '8px 12px';
    hud.style.fontFamily = 'monospace';
    hud.style.zIndex = '999999';
    hud.innerHTML = '🧠 Agent HUD<br>⏳ Waiting...';
    document.body.appendChild(hud);
  }
}

function updateLearningHUD(data) {
  insertLearningHUD();
  const hud = document.querySelector('#learning-hud');
  hud.innerHTML = `
    📈 Q-values: [${data.qvalues.map(q => q.toFixed(2)).join(', ')}]
  `;
}

async function waitForButton(selector, timeout = 3000) {
  console.log(`🔍 Waiting for button: ${selector}`);
  const start = Date.now();
  while (Date.now() - start < timeout) {
    const button = document.querySelector(selector);
    if (button && getComputedStyle(button).display !== 'none' && !button.disabled) {
      return button;
    }
    await new Promise(resolve => setTimeout(resolve, 300));
  }
  console.warn(`Timeout waiting for button: ${selector}`);
  return null;
}

async function executeDecision() {
  if (!this.lastDecision?.decision) {
    console.warn("No decision to execute");
    this.waitingForAction = false;
    return;
  }
  if (this.lastDecision.decision === "no_action") {
    console.log("⏭️ Skipping no_action decision");
    this.waitingForAction = false;
    return;
  }
  const buttonMap = {
    hit: selectors[HIT_BUTTON],
    stand: selectors[STAND_BUTTON],
    double: selectors[DOUBLE_BUTTON],
    split: selectors[SPLIT_BUTTON],
    insurance: selectors[INSURANCE_BUTTON],
    noInsurance: selectors[NO_INSURANCE_BUTTON]
  };
  const selector = buttonMap[this.lastDecision.decision];
  if (!selector) {
    console.error("Invalid action:", this.lastDecision.decision);
    this.waitingForAction = false;
    return;
  }
  console.log(`🖱️ Attempting to click: ${this.lastDecision.decision}`);
  const button = await waitForButton(selector);
  if (button) {
    try {
      button.click();
      console.log(`✅ Clicked: ${this.lastDecision.decision}`);
    } catch (e) {
      console.error("Click failed:", e);
    }
  } else {
    console.error(`❌ Button not found: ${this.lastDecision.decision}`);
  }
  this.waitingForAction = false;
}