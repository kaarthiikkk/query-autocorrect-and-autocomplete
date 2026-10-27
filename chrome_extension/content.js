// content.js - Multilingual Spelling Assistant Content Script
// Automatically detects spelling mistakes in supported websites, displays red wavy underlines,
// and provides a sleek Grammarly-style floating correction card.

(function () {
  'use strict';

  // Prevent multiple injections
  if (window.__multilingualSpellAssistantInjected) return;
  window.__multilingualSpellAssistantInjected = true;

  // Configuration
  const CONFIG = {
    DEBOUNCE_MS: 380,
    API_URL: 'http://127.0.0.1:8000/api/v1/suggest',
    FEEDBACK_URL: 'http://127.0.0.1:8000/api/v1/feedback',
    USER_ID: 'chrome_inline_assistant'
  };

  // State Management
  const wordCache = new Map(); // word.toLowerCase() -> { isMisspelled, bestCorrection, alternatives, lang }
  const sessionIgnoredWords = new Set();
  const elementVersions = new WeakMap();
  const mirrorOverlays = new WeakMap();
  let activePopup = null;
  let activeInputContext = null;

  // Common stop words to pre-seed the cache (0ms latency, 0 network requests)
  const PRESEEDED_WORDS = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
    'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but',
    'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
    'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
    'should', 'now'
  ];
  PRESEEDED_WORDS.forEach(w => wordCache.set(w, { isMisspelled: false }));

  // Helper: Match case of replacement word to original word
  function matchCase(original, replacement) {
    if (!original || !replacement) return replacement;
    if (original === original.toUpperCase() && original.length > 1) {
      return replacement.toUpperCase();
    }
    if (original[0] === original[0].toUpperCase() && original.slice(1) === original.slice(1).toLowerCase()) {
      return replacement.charAt(0).toUpperCase() + replacement.slice(1).toLowerCase();
    }
    return replacement;
  }

  // Check if an element should be excluded
  function isExcludedElement(el) {
    if (!el || !el.tagName) return true;
    if (el.tagName === 'INPUT') {
      const type = (el.type || 'text').toLowerCase();
      if (!['text', 'search', 'email', 'url', ''].includes(type)) return true;
    }
    if (el.isContentEditable === false && el.tagName !== 'INPUT' && el.tagName !== 'TEXTAREA') {
      return true;
    }
    if (el.getAttribute('data-multilingual-disabled') === 'true') return true;
    // Exclude password and code editors
    if (el.type === 'password' || el.getAttribute('autocomplete') === 'current-password') return true;
    if (el.closest('.monaco-editor, .cm-editor, .ace_editor, pre, code')) return true;
    if (el.readOnly || el.disabled) return true;
    return false;
  }

  // Backend API Communications (Uses background proxy if in Chrome extension, else direct fetch)
  async function requestSuggest(queryWord) {
    const wordKey = queryWord.toLowerCase();
    if (wordCache.has(wordKey)) {
      return wordCache.get(wordKey);
    }
    if (sessionIgnoredWords.has(wordKey)) {
      return { isMisspelled: false };
    }

    try {
      let data = null;
      if (typeof chrome !== 'undefined' && chrome.runtime && chrome.runtime.sendMessage) {
        // Use background service worker to bypass page CSP
        const resp = await new Promise((resolve) => {
          chrome.runtime.sendMessage(
            { type: 'SUGGEST_WORD', payload: { query: queryWord, top_k: 5, user_id: CONFIG.USER_ID } },
            (response) => {
              if (chrome.runtime.lastError || !response || !response.success) {
                resolve(null);
              } else {
                resolve(response.data);
              }
            }
          );
        });
        if (resp) data = resp;
      }

      // Direct fallback if not in extension context or message failed
      if (!data) {
        const fetchRes = await fetch(CONFIG.API_URL, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: queryWord, top_k: 5, user_id: CONFIG.USER_ID })
        });
        if (fetchRes.ok) {
          data = await fetchRes.json();
        }
      }

      if (!data) {
        return { isMisspelled: false };
      }

      // Determine if misspelled based on actual engine results
      const isTypo = data.error_classification === 'typo';
      const spellingCorrections = data.spelling_corrections || [];
      const rankedSuggestions = data.ranked_suggestions || [];

      // Combine suggestions and filter out self-matches
      const allCandidates = Array.from(new Set([...spellingCorrections, ...rankedSuggestions]))
        .filter(c => c && c.toLowerCase() !== wordKey && c.trim().length > 0);

      if (isTypo && allCandidates.length > 0) {
        const result = {
          isMisspelled: true,
          original: queryWord,
          bestCorrection: allCandidates[0],
          alternatives: allCandidates.slice(1, 4),
          lang: data.detected_language || 'english',
          intent: data.detected_intent || 'general'
        };
        wordCache.set(wordKey, result);
        return result;
      } else {
        const result = { isMisspelled: false };
        wordCache.set(wordKey, result);
        return result;
      }
    } catch (err) {
      console.warn('[Multilingual Assistant] API query error for:', queryWord, err);
      return { isMisspelled: false };
    }
  }

  // Send feedback to reinforce personalization and hybrid weights
  function sendFeedback(originalWord, chosenWord) {
    if (typeof chrome !== 'undefined' && chrome.runtime && chrome.runtime.sendMessage) {
      chrome.runtime.sendMessage({
        type: 'SEND_FEEDBACK',
        payload: { query: originalWord, chosen_word: chosenWord, user_id: CONFIG.USER_ID }
      });
    } else {
      fetch(CONFIG.FEEDBACK_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: originalWord, chosen_word: chosenWord, user_id: CONFIG.USER_ID })
      }).catch(() => {});
    }
  }

  // Extract all letter words and their offsets from a string
  function extractWordTokens(text) {
    const tokens = [];
    // Unicode-aware word matching (covers English, Latin accents, Tamil unicode, etc.)
    const regex = /[\p{L}\p{M}]+/gu;
    let match;
    while ((match = regex.exec(text)) !== null) {
      const word = match[0];
      const start = match.index;
      const end = start + word.length;

      tokens.push({
        word,
        start,
        end
      });
    }
    return tokens;
  }

  // Dismiss currently active correction popup
  function dismissPopup() {
    if (activePopup) {
      activePopup.remove();
      activePopup = null;
      activeInputContext = null;
    }
  }

  // Show the Grammarly-style dark floating correction card
  function showCorrectionPopup(word, anchorRect, onSelect, onIgnore) {
    dismissPopup();

    const wordInfo = wordCache.get(word.toLowerCase());
    if (!wordInfo || !wordInfo.isMisspelled) return;

    const popupContainer = document.createElement('div');
    popupContainer.className = 'msg-popup-container';

    const card = document.createElement('div');
    card.className = 'msg-popup-card';

    // Header
    const header = document.createElement('div');
    header.className = 'msg-popup-header';

    const headerLeft = document.createElement('div');
    headerLeft.className = 'msg-header-left';

    const detectedWordEl = document.createElement('span');
    detectedWordEl.className = 'msg-detected-word';
    detectedWordEl.textContent = word;

    const langBadge = document.createElement('span');
    langBadge.className = 'msg-pill-badge';
    langBadge.textContent = wordInfo.lang || 'multilingual';

    headerLeft.appendChild(detectedWordEl);
    headerLeft.appendChild(langBadge);

    const closeBtn = document.createElement('button');
    closeBtn.className = 'msg-dismiss-btn';
    closeBtn.innerHTML = '&times;';
    closeBtn.title = 'Dismiss';
    closeBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      dismissPopup();
    });

    header.appendChild(headerLeft);
    header.appendChild(closeBtn);
    card.appendChild(header);

    // Primary Best Suggestion Button
    const bestReplacement = matchCase(word, wordInfo.bestCorrection);
    const bestBtn = document.createElement('button');
    bestBtn.type = 'button';
    bestBtn.className = 'msg-best-suggestion';

    const bestLeft = document.createElement('div');
    bestLeft.className = 'msg-best-left';

    const checkIcon = document.createElement('span');
    checkIcon.className = 'msg-check-icon';
    checkIcon.innerHTML = '&#10003;';

    const bestWordEl = document.createElement('span');
    bestWordEl.className = 'msg-best-word';
    bestWordEl.textContent = bestReplacement;

    bestLeft.appendChild(checkIcon);
    bestLeft.appendChild(bestWordEl);

    const bestRight = document.createElement('div');
    bestRight.className = 'msg-best-right';

    const replaceTag = document.createElement('span');
    replaceTag.className = 'msg-replace-btn-label';
    replaceTag.textContent = 'Replace';

    bestRight.appendChild(replaceTag);
    bestBtn.appendChild(bestLeft);
    bestBtn.appendChild(bestRight);

    bestBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      onSelect(bestReplacement);
      dismissPopup();
    });

    card.appendChild(bestBtn);

    // Alternatives List
    if (wordInfo.alternatives && wordInfo.alternatives.length > 0) {
      const altSection = document.createElement('div');
      altSection.className = 'msg-alternatives-section';

      const altLabel = document.createElement('div');
      altLabel.className = 'msg-section-label';
      altLabel.textContent = 'Alternatives';
      altSection.appendChild(altLabel);

      const altList = document.createElement('div');
      altList.className = 'msg-alt-list';

      wordInfo.alternatives.forEach((alt, idx) => {
        const altReplacement = matchCase(word, alt);
        const altItem = document.createElement('button');
        altItem.type = 'button';
        altItem.className = 'msg-alt-item';

        const altWordSpan = document.createElement('span');
        altWordSpan.className = 'msg-alt-word';
        altWordSpan.textContent = altReplacement;

        const altRankSpan = document.createElement('span');
        altRankSpan.className = 'msg-alt-rank';
        altRankSpan.textContent = `#${idx + 2}`;

        altItem.appendChild(altWordSpan);
        altItem.appendChild(altRankSpan);

        altItem.addEventListener('click', (e) => {
          e.stopPropagation();
          onSelect(altReplacement);
          dismissPopup();
        });

        altList.appendChild(altItem);
      });

      altSection.appendChild(altList);
      card.appendChild(altSection);
    }

    // Footer with Session Ignore Option
    const footer = document.createElement('div');
    footer.className = 'msg-popup-footer';

    const ignoreBtn = document.createElement('button');
    ignoreBtn.type = 'button';
    ignoreBtn.className = 'msg-ignore-btn';
    ignoreBtn.textContent = 'Ignore for session';
    ignoreBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      sessionIgnoredWords.add(word.toLowerCase());
      if (onIgnore) onIgnore(word);
      dismissPopup();
    });

    const engineBrand = document.createElement('div');
    engineBrand.className = 'msg-engine-brand';
    const dot = document.createElement('span');
    dot.className = 'msg-engine-dot';
    const brandText = document.createElement('span');
    brandText.textContent = 'Multilingual Assistant';
    engineBrand.appendChild(dot);
    engineBrand.appendChild(brandText);

    footer.appendChild(ignoreBtn);
    footer.appendChild(engineBrand);
    card.appendChild(footer);

    popupContainer.appendChild(card);
    document.body.appendChild(popupContainer);

    // Positioning: place smoothly right below or above the word in viewport coordinates
    const cardRect = card.getBoundingClientRect();
    let posX = anchorRect.left;
    let posY = anchorRect.bottom + 6;

    // Viewport overflow bounds protection
    if (posX + 280 > window.innerWidth) {
      posX = Math.max(10, window.innerWidth - 290);
    }
    if (posY + cardRect.height > window.innerHeight && anchorRect.top > cardRect.height + 10) {
      posY = anchorRect.top - cardRect.height - 6;
    }

    popupContainer.style.setProperty('position', 'fixed', 'important');
    popupContainer.style.setProperty('left', `${posX}px`, 'important');
    popupContainer.style.setProperty('top', `${posY}px`, 'important');
    popupContainer.style.setProperty('z-index', '2147483647', 'important');

    activePopup = popupContainer;
  }

  // ==========================================================================
  // INPUT & TEXTAREA HANDLING VIA SYNCHRONIZED MIRROR OVERLAY
  // ==========================================================================
  function getOrCreateMirror(el) {
    let mirror = mirrorOverlays.get(el);
    if (!mirror || !document.body.contains(mirror)) {
      mirror = document.createElement('div');
      mirror.className = 'msg-mirror-overlay';
      document.body.appendChild(mirror);
      mirrorOverlays.set(el, mirror);

      // Keep scrolling in continuous sync
      el.addEventListener('scroll', () => {
        mirror.scrollTop = el.scrollTop;
        mirror.scrollLeft = el.scrollLeft;
      }, { passive: true });
    }
    return mirror;
  }

  function syncMirrorStyles(el, mirror) {
    const cs = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();

    // If element is hidden or scrolled off-screen, hide mirror
    if (rect.width === 0 || rect.height === 0 || rect.bottom < 0 || rect.top > window.innerHeight) {
      mirror.style.setProperty('display', 'none', 'important');
      return;
    }
    mirror.style.setProperty('display', 'block', 'important');

    // Fixed positioning matches getBoundingClientRect directly
    mirror.style.setProperty('position', 'fixed', 'important');
    mirror.style.setProperty('left', `${rect.left}px`, 'important');
    mirror.style.setProperty('top', `${rect.top}px`, 'important');
    mirror.style.setProperty('width', `${rect.width}px`, 'important');
    mirror.style.setProperty('height', `${rect.height}px`, 'important');
    mirror.style.setProperty('z-index', '2147483640', 'important');

    // Clone typography and layout properties for exact character alignment
    mirror.style.setProperty('font-family', cs.fontFamily, 'important');
    mirror.style.setProperty('font-size', cs.fontSize, 'important');
    mirror.style.setProperty('font-weight', cs.fontWeight, 'important');
    mirror.style.setProperty('letter-spacing', cs.letterSpacing, 'important');
    mirror.style.setProperty('text-align', cs.textAlign, 'important');
    mirror.style.setProperty('padding-top', cs.paddingTop, 'important');
    mirror.style.setProperty('padding-right', cs.paddingRight, 'important');
    mirror.style.setProperty('padding-bottom', cs.paddingBottom, 'important');
    mirror.style.setProperty('padding-left', cs.paddingLeft, 'important');
    mirror.style.setProperty('border-top-width', cs.borderTopWidth, 'important');
    mirror.style.setProperty('border-right-width', cs.borderRightWidth, 'important');
    mirror.style.setProperty('border-bottom-width', cs.borderBottomWidth, 'important');
    mirror.style.setProperty('border-left-width', cs.borderLeftWidth, 'important');
    mirror.style.setProperty('border-style', 'solid', 'important');
    mirror.style.setProperty('border-color', 'transparent', 'important');
    mirror.style.setProperty('box-sizing', cs.boxSizing, 'important');

    // Single-line inputs do not wrap; textareas do
    if (el.tagName === 'INPUT') {
      mirror.style.setProperty('white-space', 'pre', 'important');
      mirror.style.setProperty('overflow', 'hidden', 'important');
      const innerHeight = rect.height - parseFloat(cs.paddingTop || 0) - parseFloat(cs.paddingBottom || 0) - parseFloat(cs.borderTopWidth || 0) - parseFloat(cs.borderBottomWidth || 0);
      if (cs.lineHeight === 'normal' || !cs.lineHeight) {
        mirror.style.setProperty('line-height', `${innerHeight}px`, 'important');
      } else {
        mirror.style.setProperty('line-height', cs.lineHeight, 'important');
      }
    } else {
      mirror.style.setProperty('white-space', cs.whiteSpace || 'pre-wrap', 'important');
      mirror.style.setProperty('line-height', cs.lineHeight, 'important');
      mirror.style.setProperty('overflow', 'hidden', 'important');
    }

    mirror.scrollTop = el.scrollTop;
    mirror.scrollLeft = el.scrollLeft;
  }

  function renderMirrorContent(el, mirror, tokens, errorMap) {
    const text = el.value || '';
    mirror.innerHTML = '';

    let lastIdx = 0;
    const fragment = document.createDocumentFragment();

    tokens.forEach((token) => {
      // Add text preceding this token
      if (token.start > lastIdx) {
        const textNode = document.createTextNode(text.substring(lastIdx, token.start));
        fragment.appendChild(textNode);
      }

      const word = token.word;
      const isError = errorMap.has(word.toLowerCase());

      if (isError) {
        const span = document.createElement('span');
        span.className = 'msg-mirror-wavy';
        span.textContent = word;
        span.setAttribute('data-word', word);
        span.setAttribute('data-start', token.start);
        span.setAttribute('data-end', token.end);

        // Clicking the wavy red line in mirror underlay triggers popup
        span.addEventListener('mousedown', (e) => {
          e.preventDefault();
          e.stopPropagation();
          const spanRect = span.getBoundingClientRect();
          activeInputContext = { el, start: token.start, end: token.end, originalWord: word };
          showCorrectionPopup(
            word,
            spanRect,
            (chosenReplacement) => {
              applyInputCorrection(el, token.start, token.end, word, chosenReplacement);
            },
            () => {
              updateInputErrors(el);
            }
          );
        });

        fragment.appendChild(span);
      } else {
        const plainNode = document.createTextNode(word);
        fragment.appendChild(plainNode);
      }

      lastIdx = token.end;
    });

    if (lastIdx < text.length) {
      fragment.appendChild(document.createTextNode(text.substring(lastIdx)));
    }

    mirror.appendChild(fragment);
  }

  // Replace misspelled word in Input/Textarea preserving surrounding text and cursor
  function applyInputCorrection(el, start, end, originalWord, replacementWord) {
    const text = el.value || '';
    let actualStart = start;
    let actualEnd = end;
    if (text.substring(start, end) !== originalWord) {
      const idx = text.indexOf(originalWord);
      if (idx !== -1) {
        actualStart = idx;
        actualEnd = idx + originalWord.length;
      }
    }

    const beforeText = text.substring(0, actualStart);
    const afterText = text.substring(actualEnd);
    const updatedValue = beforeText + replacementWord + afterText;

    if (typeof el.setRangeText === 'function') {
      el.setRangeText(replacementWord, actualStart, actualEnd, 'end');
    } else {
      el.value = updatedValue;
      el.selectionStart = actualStart + replacementWord.length;
      el.selectionEnd = actualStart + replacementWord.length;
    }

    // Trigger synthetic input and change events for web applications (React, Angular, Vue)
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));

    sendFeedback(originalWord, replacementWord);
    updateInputErrors(el);
  }

  // Check and underline misspelled words in an input/textarea
  async function updateInputErrors(el) {
    if (isExcludedElement(el)) return;
    const currentVersion = (elementVersions.get(el) || 0) + 1;
    elementVersions.set(el, currentVersion);

    const text = el.value || '';
    const mirror = getOrCreateMirror(el);
    syncMirrorStyles(el, mirror);

    if (!text.trim()) {
      mirror.innerHTML = '';
      return;
    }

    const tokens = extractWordTokens(text);
    const wordsToCheck = [];

    tokens.forEach((t) => {
      if (t.word.length >= 3 && !sessionIgnoredWords.has(t.word.toLowerCase())) {
        wordsToCheck.push(t.word);
      }
    });

    // Check distinct unseen words against engine
    const distinctWords = Array.from(new Set(wordsToCheck));
    const errorMap = new Map();

    for (const w of distinctWords) {
      const analysis = await requestSuggest(w);
      // Abort if text has changed in the meantime
      if (elementVersions.get(el) !== currentVersion) return;
      if (analysis && analysis.isMisspelled) {
        errorMap.set(w.toLowerCase(), analysis);
      }
    }

    // Render underlines in mirror layer
    renderMirrorContent(el, mirror, tokens, errorMap);
  }

  // ==========================================================================
  // CONTENTEDITABLE HANDLING (Gmail, LinkedIn, Twitter/X, Web Editors)
  // ==========================================================================
  function getAllTextNodes(root) {
    const textNodes = [];
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: (node) => {
        if (!node.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
        if (node.parentElement && node.parentElement.closest('.msg-popup-container, script, style')) {
          return NodeFilter.FILTER_REJECT;
        }
        return NodeFilter.FILTER_ACCEPT;
      }
    });
    let currentNode;
    while ((currentNode = walker.nextNode())) {
      textNodes.push(currentNode);
    }
    return textNodes;
  }

  async function updateContentEditableErrors(el) {
    if (isExcludedElement(el)) return;
    const currentVersion = (elementVersions.get(el) || 0) + 1;
    elementVersions.set(el, currentVersion);

    const text = el.innerText || el.textContent || '';
    if (!text.trim()) return;

    // Check current selection cursor to avoid interrupting active typing
    const sel = window.getSelection();
    let activeAnchorNode = null;
    if (sel && sel.anchorNode) {
      activeAnchorNode = sel.anchorNode;
    }

    const tokens = extractWordTokens(text);
    const distinctWords = Array.from(new Set(tokens.map(t => t.word).filter(w => w.length >= 3 && !sessionIgnoredWords.has(w.toLowerCase()))));
    const errorMap = new Map();

    for (const w of distinctWords) {
      const analysis = await requestSuggest(w);
      if (elementVersions.get(el) !== currentVersion) return;
      if (analysis && analysis.isMisspelled) {
        errorMap.set(w.toLowerCase(), analysis);
      }
    }

    // Mark errors in text nodes without disrupting cursor
    const textNodes = getAllTextNodes(el);
    for (const node of textNodes) {
      if (node === activeAnchorNode) continue; // Don't disrupt the currently active text node
      const parent = node.parentNode;
      if (parent && parent.classList && parent.classList.contains('msg-wavy-error')) continue;

      const nodeVal = node.nodeValue;
      const nodeTokens = extractWordTokens(nodeVal);
      let hasError = false;

      for (const t of nodeTokens) {
        if (errorMap.has(t.word.toLowerCase())) {
          hasError = true;
          break;
        }
      }

      if (!hasError) continue;

      const fragment = document.createDocumentFragment();
      let lastPos = 0;

      nodeTokens.forEach((t) => {
        if (t.start > lastPos) {
          fragment.appendChild(document.createTextNode(nodeVal.substring(lastPos, t.start)));
        }

        if (errorMap.has(t.word.toLowerCase())) {
          const span = document.createElement('span');
          span.className = 'msg-wavy-error';
          span.textContent = t.word;
          span.setAttribute('data-word', t.word);

          span.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            const rect = span.getBoundingClientRect();
            showCorrectionPopup(
              t.word,
              rect,
              (chosenReplacement) => {
                applyContentEditableCorrection(span, chosenReplacement);
              },
              () => {
                span.classList.remove('msg-wavy-error');
              }
            );
          });

          fragment.appendChild(span);
        } else {
          fragment.appendChild(document.createTextNode(t.word));
        }
        lastPos = t.end;
      });

      if (lastPos < nodeVal.length) {
        fragment.appendChild(document.createTextNode(nodeVal.substring(lastPos)));
      }

      if (parent) {
        parent.replaceChild(fragment, node);
      }
    }
  }

  function applyContentEditableCorrection(spanElement, chosenReplacement) {
    const originalWord = spanElement.getAttribute('data-word') || spanElement.textContent;
    const parent = spanElement.parentNode;
    if (!parent) return;

    const replacementNode = document.createTextNode(chosenReplacement);
    parent.replaceChild(replacementNode, spanElement);

    // Place cursor right after the replacement word
    const sel = window.getSelection();
    if (sel) {
      const range = document.createRange();
      range.setStartAfter(replacementNode);
      range.collapse(true);
      sel.removeAllRanges();
      sel.addRange(range);
    }

    // Trigger synthetic input events on contenteditable root
    const rootEditable = replacementNode.parentElement ? replacementNode.parentElement.closest('[contenteditable="true"]') : null;
    if (rootEditable) {
      rootEditable.dispatchEvent(new Event('input', { bubbles: true }));
    }

    sendFeedback(originalWord, chosenReplacement);
  }

  // ==========================================================================
  // EVENT LISTENERS & ATTACHMENT
  // ==========================================================================
  const debounceTimers = new WeakMap();

  function scheduleCheck(el) {
    if (isExcludedElement(el)) return;
    clearTimeout(debounceTimers.get(el));
    const timer = setTimeout(() => {
      if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
        updateInputErrors(el);
      } else if (el.isContentEditable) {
        updateContentEditableErrors(el);
      }
    }, CONFIG.DEBOUNCE_MS);
    debounceTimers.set(el, timer);
  }

  // Detect click inside input/textarea to see if user clicked a misspelled word
  function handleInputClick(e) {
    const el = e.target;
    if (!el || (el.tagName !== 'INPUT' && el.tagName !== 'TEXTAREA')) return;
    if (isExcludedElement(el)) return;

    const text = el.value || '';
    const clickPos = el.selectionStart;
    const tokens = extractWordTokens(text);

    for (const t of tokens) {
      if (clickPos >= t.start && clickPos <= t.end) {
        const wordKey = t.word.toLowerCase();
        const wordInfo = wordCache.get(wordKey);
        if (wordInfo && wordInfo.isMisspelled && !sessionIgnoredWords.has(wordKey)) {
          const mirror = mirrorOverlays.get(el);
          if (mirror) {
            const mirrorSpan = mirror.querySelector(`[data-start="${t.start}"]`);
            const rect = mirrorSpan ? mirrorSpan.getBoundingClientRect() : el.getBoundingClientRect();
            activeInputContext = { el, start: t.start, end: t.end, originalWord: t.word };
            showCorrectionPopup(
              t.word,
              rect,
              (chosenReplacement) => {
                applyInputCorrection(el, t.start, t.end, t.word, chosenReplacement);
              },
              () => {
                updateInputErrors(el);
              }
            );
            return;
          }
        }
        break;
      }
    }
  }

  // Global event delegates
  document.addEventListener('input', (e) => {
    scheduleCheck(e.target);
  }, true);

  document.addEventListener('keyup', (e) => {
    if (['Space', 'Enter', 'Period', 'Comma'].includes(e.code)) {
      scheduleCheck(e.target);
    }
  }, true);

  document.addEventListener('mouseup', handleInputClick, true);

  document.addEventListener('scroll', () => {
    if (activePopup) dismissPopup();
    document.querySelectorAll('input, textarea').forEach((el) => {
      const mirror = mirrorOverlays.get(el);
      if (mirror) syncMirrorStyles(el, mirror);
    });
  }, true);

  window.addEventListener('resize', () => {
    if (activePopup) dismissPopup();
    document.querySelectorAll('input, textarea').forEach((el) => {
      const mirror = mirrorOverlays.get(el);
      if (mirror) syncMirrorStyles(el, mirror);
    });
  });

  // Dismiss on clicking outside
  document.addEventListener('click', (e) => {
    if (activePopup && !activePopup.contains(e.target) && !e.target.closest('.msg-mirror-wavy, .msg-wavy-error')) {
      dismissPopup();
    }
  }, true);

  // Dismiss on Escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && activePopup) {
      dismissPopup();
    }
  }, true);

  // MutationObserver to auto-attach to dynamically created text fields
  const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      for (const node of mutation.addedNodes) {
        if (node.nodeType === Node.ELEMENT_NODE) {
          if (node.matches && (node.matches('input, textarea, [contenteditable="true"]'))) {
            scheduleCheck(node);
          } else if (node.querySelectorAll) {
            const fields = node.querySelectorAll('input, textarea, [contenteditable="true"]');
            fields.forEach(f => scheduleCheck(f));
          }
        }
      }
    }
  });

  observer.observe(document.body || document.documentElement, {
    childList: true,
    subtree: true
  });

  console.log('[Multilingual Assistant] Content script active with Grammarly-style inline spelling correction.');
})();
