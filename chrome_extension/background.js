// background.js - Chrome MV3 Service Worker for Multilingual Spelling Assistant
// Handles API calls bypassing website Content Security Policy (CSP) and CORS restrictions.

const BACKEND_BASE_URL = 'http://127.0.0.1:8000';

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'SUGGEST_WORD') {
    const payload = message.payload || {};
    fetch(`${BACKEND_BASE_URL}/api/v1/suggest`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: payload.query,
        language: payload.language || null,
        top_k: payload.top_k || 5,
        user_id: payload.user_id || 'chrome_inline_assistant'
      })
    })
      .then(response => {
        if (!response.ok) {
          throw new Error(`Backend error: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        sendResponse({ success: true, data });
      })
      .catch(error => {
        sendResponse({ success: false, error: error.message });
      });

    // Return true to indicate we wish to send a response asynchronously
    return true;
  }

  if (message.type === 'SEND_FEEDBACK') {
    const payload = message.payload || {};
    fetch(`${BACKEND_BASE_URL}/api/v1/feedback`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: payload.query,
        chosen_word: payload.chosen_word,
        user_id: payload.user_id || 'chrome_inline_assistant'
      })
    })
      .then(response => response.json())
      .then(data => sendResponse({ success: true, data }))
      .catch(error => sendResponse({ success: false, error: error.message }));

    return true;
  }

  if (message.type === 'HEALTH_CHECK') {
    fetch(`${BACKEND_BASE_URL}/api/v1/health`)
      .then(response => response.json())
      .then(data => sendResponse({ success: true, data }))
      .catch(error => sendResponse({ success: false, error: error.message }));

    return true;
  }
});
