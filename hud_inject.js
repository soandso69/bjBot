// Add this function to create or update the HUD
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
                <div>Training Status: <span id="training-status">Inactive</span></div>
            </div>
        `;

        document.body.appendChild(hud);
        console.log("HUD created and added to the page");
    }
    
    return hud;
}
