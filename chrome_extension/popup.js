// Multilingual Search Intelligence - Extension Logic

const API_BASE = "http://127.0.0.1:8000/api/v1";

const searchInput = document.getElementById("search-input");
const clearBtn = document.getElementById("clear-btn");
const langSelect = document.getElementById("lang-select");
const userSelect = document.getElementById("user-select");
const statusIndicator = document.getElementById("status-indicator");
const statusText = document.getElementById("status-text");
const metaBar = document.getElementById("meta-bar");
const metaLang = document.getElementById("meta-lang");
const metaIntent = document.getElementById("meta-intent");
const metaLatency = document.getElementById("meta-latency");
const metaConf = document.getElementById("meta-conf");
const suggestionsList = document.getElementById("suggestions-list");
const resultsCount = document.getElementById("results-count");
const toast = document.getElementById("toast");

let debounceTimer = null;
let currentSuggestions = [];

// ============================================================================
// 1. HEALTH CHECK & INITIALIZATION
// ============================================================================

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { method: "GET" });
    if (res.ok) {
      statusIndicator.className = "status-pill status-connected";
      statusText.textContent = "API Live :8000";
      return true;
    } else {
      throw new Error(`HTTP ${res.status}`);
    }
  } catch (err) {
    statusIndicator.className = "status-pill status-disconnected";
    statusText.textContent = "Offline (Run uvicorn)";
    return false;
  }
}

// ============================================================================
// 2. QUERY EXECUTION & PREDICTIVE SEARCH
// ============================================================================

async function executeSearch(query) {
  const trimmed = query.trim();
  if (!trimmed) {
    clearResults();
    return;
  }

  const selectedLang = langSelect.value;
  const selectedUser = userSelect.value;

  const payload = {
    query: trimmed,
    language: selectedLang === "auto" ? null : selectedLang,
    top_k: 5,
    user_id: selectedUser,
  };

  const t0 = performance.now();

  try {
    const res = await fetch(`${API_BASE}/suggest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      throw new Error(`API returned HTTP ${res.status}`);
    }

    const data = await res.json();
    const t1 = performance.now();
    const roundtripLatency = (t1 - t0).toFixed(1);

    renderResults(data, roundtripLatency);
  } catch (err) {
    console.error("Search failed:", err);
    statusIndicator.className = "status-pill status-disconnected";
    statusText.textContent = "API Error";
    renderError(err.message);
  }
}

// ============================================================================
// 3. UI RENDERING
// ============================================================================

function renderResults(data, latencyMs) {
  currentSuggestions = data.ranked_suggestions || [];

  // Update Meta Bar
  metaBar.classList.remove("hidden");
  metaLang.textContent = (data.detected_language || "unknown").toUpperCase();
  metaIntent.textContent = (data.detected_intent || "general").replace("_", " ");
  metaLatency.textContent = `${data.processing_latency_ms || latencyMs} ms`;

  // Confidence
  const confLevel = (data.confidence_level || "medium").toUpperCase();
  metaConf.textContent = `${confLevel} (${Math.round((data.confidence_score || 0.8) * 100)}%)`;
  metaConf.className = `meta-val ${confLevel === "HIGH" ? "badge-green" : confLevel === "MEDIUM" ? "badge-amber" : "badge-red"}`;

  // Results count
  const count = currentSuggestions.length;
  resultsCount.textContent = `${count} suggestion${count === 1 ? "" : "s"}`;

  if (count === 0) {
    suggestionsList.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">🔍</div>
        <div class="empty-title">No candidate matches found</div>
        <div class="empty-sub">Try adjusting the language selector or typing a prefix</div>
      </div>
    `;
    return;
  }

  const details = data.candidate_details || [];
  suggestionsList.innerHTML = "";

  currentSuggestions.forEach((word, idx) => {
    const cand = details[idx] || {};
    const score = (cand.final_score !== undefined ? cand.final_score : 0.85).toFixed(4);
    const isTop = idx === 0;

    const item = document.createElement("div");
    item.className = "suggestion-item";
    item.innerHTML = `
      <div class="suggestion-left">
        <div class="rank-badge ${isTop ? "rank-1" : ""}">${idx + 1}</div>
        <div class="word-info">
          <div class="suggestion-word">
            ${escapeHtml(word)}
            ${cand.is_personalized ? '<span class="meta-val badge-purple" style="font-size:9px; margin-left:4px;">Personalized</span>' : ''}
            ${cand.is_bursting ? '<span class="meta-val badge-amber" style="font-size:9px; margin-left:4px;">🔥 Trending</span>' : ''}
          </div>
          <div class="suggestion-sub">${escapeHtml(cand.personalization_reason || cand.temporal_reason || cand.source || "Hybrid dynamic ranking")}</div>
        </div>
      </div>
      <div class="suggestion-right">
        <span class="score-tag">${score}</span>
        <button class="action-btn copy-btn" title="Copy suggestion">Copy</button>
        <button class="action-btn search-btn" title="Search on Google">🌐</button>
      </div>
    `;

    // Click on row to copy & feedback
    item.addEventListener("click", (e) => {
      if (e.target.classList.contains("search-btn")) {
        e.stopPropagation();
        openGoogleSearch(word);
        return;
      }
      selectSuggestion(word, searchInput.value.trim());
    });

    suggestionsList.appendChild(item);
  });
}

function clearResults() {
  currentSuggestions = [];
  resultsCount.textContent = "0 suggestions";
  metaBar.classList.add("hidden");
  suggestionsList.innerHTML = `
    <div class="empty-state">
      <div class="empty-icon">⌨️</div>
      <div class="empty-title">Ready for real-time keystroke prediction</div>
      <div class="empty-sub">Type a prefix, misspelling, or Tanglish expression above</div>
    </div>
  `;
}

function renderError(msg) {
  suggestionsList.innerHTML = `
    <div class="empty-state">
      <div class="empty-icon">⚠️</div>
      <div class="empty-title">FastAPI connection error</div>
      <div class="empty-sub">Ensure the backend is running via <code>uvicorn api:app --port 8000</code></div>
    </div>
  `;
}

// ============================================================================
// 4. USER FEEDBACK & ACTIONS
// ============================================================================

async function selectSuggestion(word, originalQuery) {
  // Copy to clipboard
  navigator.clipboard.writeText(word).then(() => {
    showToast(`Copied "${word}"!`);
  }).catch(() => {
    showToast(`Selected "${word}"`);
  });

  // Reinforce online feedback learner in real-time
  try {
    await fetch(`${API_BASE}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: originalQuery,
        selected_word: word,
        user_id: userSelect.value,
        language: langSelect.value === "auto" ? null : langSelect.value,
      }),
    });
  } catch (e) {
    console.warn("Feedback recording offline:", e);
  }
}

function openGoogleSearch(word) {
  const url = `https://www.google.com/search?q=${encodeURIComponent(word)}`;
  chrome.tabs.create({ url });
}

function showToast(text) {
  toast.textContent = text;
  toast.classList.remove("hidden");
  setTimeout(() => {
    toast.classList.add("hidden");
  }, 1600);
}

function escapeHtml(str) {
  return String(str || "").replace(/[&<>"']/g, (m) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
  }[m]));
}

// ============================================================================
// 5. EVENT LISTENERS
// ============================================================================

searchInput.addEventListener("input", (e) => {
  const val = e.target.value;
  clearBtn.style.display = val ? "block" : "none";

  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    executeSearch(val);
  }, 110);
});

searchInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && currentSuggestions.length > 0) {
    e.preventDefault();
    selectSuggestion(currentSuggestions[0], searchInput.value.trim());
  } else if (e.key === "Escape") {
    searchInput.value = "";
    clearBtn.style.display = "none";
    clearResults();
  }
});

clearBtn.addEventListener("click", () => {
  searchInput.value = "";
  clearBtn.style.display = "none";
  searchInput.focus();
  clearResults();
});

langSelect.addEventListener("change", () => {
  if (searchInput.value.trim()) {
    executeSearch(searchInput.value.trim());
  }
});

userSelect.addEventListener("change", () => {
  if (searchInput.value.trim()) {
    executeSearch(searchInput.value.trim());
  }
});

// Preset Chips
document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    const q = chip.getAttribute("data-query");
    searchInput.value = q;
    clearBtn.style.display = "block";
    searchInput.focus();
    executeSearch(q);
  });
});

// Initial boot
checkHealth();
searchInput.focus();
