let autoPlayEnabled = false;  // Store autoplay state

// Update the extension icon based on the autoplay state
function updateIcon(state) {
    if (!chrome.action) {
        console.error("❌ Error: chrome.action is undefined. Ensure this runs in background.js.");
        return;
    }

    const iconPath = state ? "icon/iconon.png" : "icon/iconoff.png";
    console.log("🔄 Updating icon to:", iconPath);

    chrome.action.setIcon({ path: iconPath }, () => {
        if (chrome.runtime.lastError) {
            console.error("⚠️ Failed to update icon:", chrome.runtime.lastError);
        }
    });
}

// Listen for messages from content scripts
chrome.runtime.onMessage.addListener(async (message, sender, sendResponse) => {
    if (message.action === "toggleAutoPlay") {
        let newState = !gameObserver.autoPlayEnabled; 
        gameObserver.autoPlayEnabled = newState;
        console.log("🔄 Toggling autoplay. New state:", newState);

        // ✅ Update the icon correctly
        updateIcon(newState);

        if (newState) {
            console.log("▶️ Autoplay enabled! Starting game loop...");
            startAutoPlayLoop();  // ✅ Ensure this function exists in gameObserver
        } else {
            console.log("⏸️ Autoplay disabled. Stopping game.");
        }

        let responseReceived = false; 

        if (gameObserver.websocket && gameObserver.websocket.readyState === WebSocket.OPEN) {
            console.log("📡 Sending toggle_training to server...");
            gameObserver.websocket.send(JSON.stringify({
                action: "toggle_training",
                autoplay_enabled: newState
            }));

            gameObserver.websocket.onmessage = (event) => {
                try {
                    const response = JSON.parse(event.data);
                    console.log("✅ Server response received:", response);
                    responseReceived = true;
                    sendResponse(response); 
                } catch (e) {
                    console.error("⚠️ Error parsing server response:", e);
                }
            };
        } else {
            console.warn("⚠️ WebSocket not connected. Cannot send toggle_training message.");
        }

        // Timeout if no response
        setTimeout(() => {
            if (!responseReceived) {
                console.warn("⏳ No response received from server. Closing message channel.");
                sendResponse({ error: "No response from server" });
            }
        }, 5000);

        return true; 
    }
});

// Handle when the extension icon is clicked
chrome.action.onClicked.addListener(() => {
    // Toggle and persist the autoplay state
    chrome.storage.local.get("autoPlayEnabled", (data) => {
        const newState = !data.autoPlayEnabled;
        chrome.storage.local.set({ autoPlayEnabled: newState });

        // Update the icon
        const iconPath = newState ? "icon/iconon.png" : "icon/iconoff.png";
        chrome.action.setIcon({ path: iconPath });

        // Broadcast the new state to all tabs
        chrome.tabs.query({}, (tabs) => {
            tabs.forEach((tab) => {
                chrome.tabs.sendMessage(tab.id, {
                    action: "toggleAutoPlay",
                    autoPlayEnabled: newState
                }).catch((error) => {
                    console.warn(`Could not send message to tab ${tab.id}:`, error);
                });
            });
        });

        console.log(`AutoPlay ${newState ? 'enabled' : 'disabled'}`);
    });
});


// Ensure state is restored on startup or installation
chrome.runtime.onStartup.addListener(() => {
    chrome.storage.local.get("autoPlayEnabled", (data) => {
        autoPlayEnabled = data.autoPlayEnabled || false;
        updateIcon();  // Update the icon based on the saved state
    });
});

chrome.runtime.onInstalled.addListener(() => {
    chrome.storage.local.get("autoPlayEnabled", (data) => {
        autoPlayEnabled = data.autoPlayEnabled || false;
        updateIcon();  // Update the icon based on the saved state
    });
});

console.log("✅ Service worker registered and running.");
