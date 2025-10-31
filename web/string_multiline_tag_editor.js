/**
 * 🏷️ Multiline TTS Tag Editor
 * Advanced multiline text editor with TTS tag support, undo/redo, and full persistence
 */

import { app } from "/scripts/app.js";

// State management class
class EditorState {
    constructor() {
        this.text = "";
        this.history = [];
        this.historyIndex = -1;
        this.presets = {};
        this.lastCharacter = "";
        this.lastLanguage = "";
        this.lastSeed = 0;
        this.lastTemperature = 0.7;
        this.lastPauseDuration = "1s";
        this.lastParameterType = ""; // Persist selected parameter type
        this.sidebarExpanded = true;
        this.lastCursorPosition = 0;
        this.discoveredCharacters = {};
        this.fontSize = 14; // Default font size in pixels
        this.sidebarWidth = 220; // Default sidebar width in pixels
        this.uiScale = 1.0; // UI scaling factor for sidebar elements
    }

    addToHistory(text, caretPos = 0) {
        // Remove any future history if we've undone something
        if (this.historyIndex < this.history.length - 1) {
            this.history = this.history.slice(0, this.historyIndex + 1);
        }

        this.history.push({ text, caretPos });
        this.historyIndex = this.history.length - 1;

        // Keep history size reasonable (limit to 100 states)
        if (this.history.length > 100) {
            this.history = this.history.slice(-100);
            this.historyIndex = this.history.length - 1;
        }

        this.text = text;
    }

    undo() {
        if (this.historyIndex > 0) {
            this.historyIndex--;
            const entry = this.history[this.historyIndex];
            this.text = entry.text;
            return entry;
        }
        const current = this.history[this.historyIndex] || { text: this.text, caretPos: 0 };
        return current;
    }

    redo() {
        if (this.historyIndex < this.history.length - 1) {
            this.historyIndex++;
            const entry = this.history[this.historyIndex];
            this.text = entry.text;
            return entry;
        }
        const current = this.history[this.historyIndex] || { text: this.text, caretPos: 0 };
        return current;
    }

    getHistoryStatus() {
        return `${this.historyIndex + 1}/${this.history.length}`;
    }

    serialize() {
        return JSON.stringify(this);
    }

    static deserialize(json) {
        try {
            const data = JSON.parse(json);
            const state = new EditorState();
            Object.assign(state, data);
            return state;
        } catch (e) {
            console.warn("Failed to deserialize state:", e);
            return new EditorState();
        }
    }

    saveToLocalStorage(key) {
        localStorage.setItem(key, this.serialize());
    }

    static loadFromLocalStorage(key) {
        const json = localStorage.getItem(key);
        if (json) {
            return EditorState.deserialize(json);
        }
        return new EditorState();
    }
}

// Utility functions for tag operations
class TagUtilities {
    static parseExistingTags(text) {
        const tags = [];
        const tagPattern = /\[([^\]]+)\]/g;
        let match;

        while ((match = tagPattern.exec(text)) !== null) {
            const tagContent = match[1];
            const tag = {
                full: `[${tagContent}]`,
                position: match.index,
                character: "",
                language: "",
                parameters: {}
            };

            // Parse character and parameters
            const parts = tagContent.split("|");

            // First part: character or language:character
            const firstPart = parts[0].trim();
            if (firstPart.includes(":") && !firstPart.includes(".")) {
                const [lang, char] = firstPart.split(":", 2);
                tag.language = lang.trim();
                tag.character = char.trim();
            } else {
                tag.character = firstPart;
            }

            // Remaining parts: parameters
            for (let i = 1; i < parts.length; i++) {
                const part = parts[i];
                if (part.includes(":")) {
                    const [paramName, paramValue] = part.split(":", 2);
                    tag.parameters[paramName.trim().toLowerCase()] = paramValue.trim();
                }
            }

            tags.push(tag);
        }

        return tags;
    }

    static validateTagSyntax(text) {
        const tagPattern = /\[([^\]]+)\]/g;
        let match;

        while ((match = tagPattern.exec(text)) !== null) {
            const tagContent = match[1];

            // Check for mismatched brackets
            if ((tagContent.match(/\[/g) || []).length !== (tagContent.match(/\]/g) || []).length) {
                return { valid: false, error: `Mismatched brackets in tag: [${tagContent}]` };
            }

            // Validate parameter syntax if present
            if (tagContent.includes("|")) {
                const parts = tagContent.split("|");
                for (let i = 1; i < parts.length; i++) {
                    if (!parts[i].includes(":")) {
                        return { valid: false, error: `Invalid parameter syntax: ${parts[i]} (expected format: param:value)` };
                    }

                    const [paramName] = parts[i].split(":", 2);
                    if (!paramName.trim()) {
                        return { valid: false, error: `Empty parameter name in ${parts[i]}` };
                    }
                }
            }
        }

        return { valid: true };
    }

    static insertTagAtPosition(text, tag, position, wrapSelection = false, selectionStart = -1, selectionEnd = -1) {
        if (wrapSelection && selectionStart >= 0 && selectionEnd >= 0) {
            const before = text.substring(0, selectionStart);
            const selected = text.substring(selectionStart, selectionEnd);
            const after = text.substring(selectionEnd);
            return `${before}${tag} ${selected}${after}`;
        } else {
            return text.substring(0, position) + tag + " " + text.substring(position);
        }
    }
}

// Create the widget
function addStringMultilineTagEditorWidget(node) {
    const storageKey = `string_multiline_tag_editor_${node.id}`;

    // Load persisted state
    const state = EditorState.loadFromLocalStorage(storageKey);

    // Create main editor container (this will be THE widget)
    const editorContainer = document.createElement("div");
    editorContainer.className = "string-multiline-tag-editor-main";
    editorContainer.style.display = "flex";
    editorContainer.style.gap = "0";
    editorContainer.style.width = "100%";
    editorContainer.style.height = "100%";
    editorContainer.style.background = "#1a1a1a";
    editorContainer.style.borderRadius = "4px";
    editorContainer.style.overflow = "hidden";
    editorContainer.style.flexDirection = "row";
    editorContainer.style.position = "relative";

    // Create notification toast at bottom
    const notificationToast = document.createElement("div");
    notificationToast.style.position = "absolute";
    notificationToast.style.bottom = "10px";
    notificationToast.style.left = "50%";
    notificationToast.style.transform = "translateX(-50%)";
    notificationToast.style.background = "rgba(0, 0, 0, 0.8)";
    notificationToast.style.color = "#0f0";
    notificationToast.style.padding = "8px 12px";
    notificationToast.style.borderRadius = "3px";
    notificationToast.style.fontSize = "11px";
    notificationToast.style.opacity = "0";
    notificationToast.style.pointerEvents = "none";
    notificationToast.style.transition = "opacity 0.3s ease";
    notificationToast.style.zIndex = "100";
    notificationToast.style.maxWidth = "300px";
    notificationToast.style.textAlign = "center";
    notificationToast.style.whiteSpace = "nowrap";
    notificationToast.style.overflow = "hidden";
    notificationToast.style.textOverflow = "ellipsis";

    editorContainer.appendChild(notificationToast);

    // Helper function to show notification
    const showNotification = (message, duration = 2000) => {
        notificationToast.textContent = message;
        notificationToast.style.opacity = "1";

        setTimeout(() => {
            notificationToast.style.opacity = "0";
        }, duration);
    };

    // Create sidebar with resizable width and UI scaling
    const sidebar = document.createElement("div");
    sidebar.className = "string-multiline-tag-editor-sidebar";
    sidebar.style.width = state.sidebarWidth + "px";
    sidebar.style.minWidth = "150px";
    sidebar.style.maxWidth = "400px";
    sidebar.style.height = "100%";
    sidebar.style.background = "#222";
    sidebar.style.borderRight = "1px solid #444";
    sidebar.style.padding = "10px";
    sidebar.style.overflowY = "auto";
    sidebar.style.overflowX = "hidden";
    sidebar.style.fontSize = (11 * state.uiScale) + "px";
    sidebar.style.flexShrink = "0";
    sidebar.style.display = "flex";
    sidebar.style.flexDirection = "column";
    sidebar.style.position = "relative";

    // Function to update sidebar width and persist
    const setSidebarWidth = (newWidth) => {
        newWidth = Math.max(150, Math.min(400, newWidth)); // Clamp between 150px and 400px
        state.sidebarWidth = newWidth;
        sidebar.style.width = newWidth + "px";
        state.saveToLocalStorage(storageKey);
    };

    // Function to update UI scale
    const setUIScale = (factor) => {
        factor = Math.max(0.7, Math.min(1.5, factor)); // Clamp between 0.7 and 1.5
        state.uiScale = factor;
        sidebar.style.fontSize = (11 * factor) + "px";

        // Update all button and input sizes
        const buttons = sidebar.querySelectorAll("button, input[type='text'], input[type='number'], select");
        buttons.forEach(btn => {
            const baseFontSize = 10;
            btn.style.fontSize = (baseFontSize * factor) + "px";
            btn.style.padding = (4 * factor) + "px " + (6 * factor) + "px";
        });

        state.saveToLocalStorage(storageKey);
    };

    // Create editor wrapper for contenteditable
    const textareaWrapper = document.createElement("div");
    textareaWrapper.style.flex = "1 1 auto";
    textareaWrapper.style.display = "flex";
    textareaWrapper.style.minHeight = "0";
    textareaWrapper.style.width = "100%";

    // Create contenteditable div - replaces both textarea and overlay
    const editor = document.createElement("div");
    editor.contentEditable = "true";
    editor.className = "comfy-multiline-input";
    editor.style.flex = "1 1 auto";
    editor.style.fontFamily = "monospace";
    editor.style.fontSize = state.fontSize + "px";
    editor.style.padding = "10px";
    editor.style.border = "none";
    editor.style.background = "#1a1a1a";
    editor.style.color = "#eee";
    editor.style.outline = "none";
    editor.style.margin = "0";
    editor.style.minHeight = "0";
    editor.style.width = "100%";
    editor.style.lineHeight = "1.4";
    editor.style.letterSpacing = "0";
    editor.style.wordSpacing = "0";
    editor.style.whiteSpace = "pre-wrap";
    editor.style.wordWrap = "break-word";
    editor.style.overflowWrap = "break-word";
    editor.style.tabSize = "4";
    editor.style.MozTabSize = "4";
    editor.style.caretColor = "#eee";
    editor.spellcheck = false;

    // Function to update font size and persist it
    const setFontSize = (newSize) => {
        newSize = Math.max(8, Math.min(32, newSize)); // Clamp between 8px and 32px
        state.fontSize = newSize;
        editor.style.fontSize = newSize + "px";
        state.saveToLocalStorage(storageKey);
    };

    // Initialize with text
    editor.textContent = state.text;

    // Helper to get plain text (strip HTML for state management)
    const getPlainText = () => {
        const clone = editor.cloneNode(true);
        // Remove all spans, keeping just the text
        const spans = clone.querySelectorAll('span');
        spans.forEach(span => {
            while (span.firstChild) {
                span.parentNode.insertBefore(span.firstChild, span);
            }
            span.parentNode.removeChild(span);
        });
        return clone.textContent;
    };

    // Save caret position before update
    const getCaretPos = () => {
        const selection = window.getSelection();
        if (selection.rangeCount === 0) return 0;

        const range = selection.getRangeAt(0);
        const preRange = range.cloneRange();
        preRange.selectNodeContents(editor);
        preRange.setEnd(range.endContainer, range.endOffset);

        // Count plain text characters (without HTML spans)
        const tempDiv = document.createElement('div');
        tempDiv.appendChild(preRange.cloneContents());

        // Remove all span elements for counting
        const spans = tempDiv.querySelectorAll('span');
        spans.forEach(span => {
            while (span.firstChild) {
                span.parentNode.insertBefore(span.firstChild, span);
            }
            span.parentNode.removeChild(span);
        });

        return tempDiv.textContent.length;
    };

    // Restore caret position after update
    const setCaretPos = (pos) => {
        const selection = window.getSelection();
        const range = document.createRange();
        let charCount = 0;
        let nodeStack = [editor];
        let node;
        let foundStart = false;

        while (!foundStart && (node = nodeStack.pop())) {
            if (node.nodeType === Node.TEXT_NODE) {
                const nextCharCount = charCount + node.length;
                if (pos <= nextCharCount) {
                    range.setStart(node, pos - charCount);
                    foundStart = true;
                }
                charCount = nextCharCount;
            } else {
                let i = node.childNodes.length;
                while (i--) {
                    nodeStack.push(node.childNodes[i]);
                }
            }
        }

        if (foundStart) {
            range.collapse(true);
            selection.removeAllRanges();
            selection.addRange(range);
        }
    };

    // Function to highlight syntax in contenteditable
    const updateHighlights = () => {
        const plainText = getPlainText();
        const caretPos = getCaretPos();
        let html = plainText;

        // Highlight SRT sequence numbers - bright red
        html = html.replace(
            /^(\d+)\s*\n(\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3})/gm,
            '\x00NUM_START\x00$1\x00NUM_END\x00\n$2'
        );

        // Highlight SRT timings - bright orange
        html = html.replace(
            /\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}/g,
            '\x00SRT_START\x00$&\x00SRT_END\x00'
        );

        // Highlight tags - bright cyan
        html = html.replace(
            /(\[[^\]]+\])/g,
            '\x00TAG_START\x00$1\x00TAG_END\x00'
        );

        // Highlight commas - green
        html = html.replace(/,/g, '\x00COMMA_START\x00,\x00COMMA_END\x00');

        // Highlight periods - golden yellow
        html = html.replace(/\./g, '\x00PERIOD_START\x00.\x00PERIOD_END\x00');

        // Highlight punctuation - light salmon
        html = html.replace(/[?!;]/g, '\x00PUNCT_START\x00$&\x00PUNCT_END\x00');

        // Highlight multiple spaces (2 or more) - subtle background
        html = html.replace(/  +/g, '\x00SPACE_START\x00$&\x00SPACE_END\x00');

        // Escape HTML
        html = html
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");

        // Replace placeholders with spans
        html = html
            .replace(/\x00NUM_START\x00(.*?)\x00NUM_END\x00/g, '<span style="color: #ff5555; font-weight: bold;">$1</span>')
            .replace(/\x00SRT_START\x00(.*?)\x00SRT_END\x00/g, '<span style="color: #ffaa00; font-weight: bold;">$1</span>')
            .replace(/\x00TAG_START\x00(.*?)\x00TAG_END\x00/g, '<span style="color: #00ffff; font-weight: bold;">$1</span>')
            .replace(/\x00COMMA_START\x00(.*?)\x00COMMA_END\x00/g, '<span style="color: #66ff66; font-weight: bold;">$1</span>')
            .replace(/\x00PERIOD_START\x00(.*?)\x00PERIOD_END\x00/g, '<span style="color: #ffcc33; font-weight: bold;">$1</span>')
            .replace(/\x00PUNCT_START\x00(.*?)\x00PUNCT_END\x00/g, '<span style="color: #ff9999;">$1</span>')
            .replace(/\x00SPACE_START\x00(.*?)\x00SPACE_END\x00/g, '<span style="background: #2a2a2a; color: #eee;">$1</span>');

        // Update only if changed to avoid flicker
        if (editor.innerHTML !== html) {
            editor.innerHTML = html;
            setCaretPos(caretPos);
        }
    };

    // Update on input
    editor.addEventListener("input", () => {
        updateHighlights();
    });

    // Initial highlight
    updateHighlights();

    textareaWrapper.appendChild(editor);

    // Make sidebar border-right resizable (invisible drag handle)
    let isResizing = false;
    let lastClickTime = 0;
    let lastClickX = 0;

    sidebar.addEventListener("mousedown", (e) => {
        // Only trigger resize if click is on the very right edge of sidebar (within 8px)
        const rect = sidebar.getBoundingClientRect();
        if (e.clientX > rect.right - 8) {
            // Check for double-click
            const currentTime = Date.now();
            if (currentTime - lastClickTime < 300 && Math.abs(e.clientX - lastClickX) < 5) {
                // Double-click detected - reset to defaults
                setSidebarWidth(220); // Default width
                setUIScale(1.0); // Default UI scale
                setFontSize(14); // Default font size
                showNotification("🔄 Reset: Sidebar width, UI scale, and font size to defaults");
                lastClickTime = 0; // Reset to prevent triple-click
                return;
            }
            lastClickTime = currentTime;
            lastClickX = e.clientX;

            isResizing = true;
            e.preventDefault();
        }
    });

    document.addEventListener("mousemove", (e) => {
        if (!isResizing) return;
        const editorContainerRect = editorContainer.getBoundingClientRect();
        const newWidth = e.clientX - editorContainerRect.left;
        setSidebarWidth(newWidth);
    });

    document.addEventListener("mouseup", () => {
        isResizing = false;
    });

    // Change cursor to col-resize when hovering near the right edge of sidebar
    sidebar.addEventListener("mousemove", (e) => {
        const rect = sidebar.getBoundingClientRect();
        if (e.clientX > rect.right - 8) {
            sidebar.style.cursor = "col-resize";
        } else {
            sidebar.style.cursor = "default";
        }
    });

    sidebar.addEventListener("mouseleave", () => {
        sidebar.style.cursor = "default";
    });

    // Ctrl+wheel on sidebar to change UI scale
    sidebar.addEventListener("wheel", (e) => {
        if ((e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            const delta = e.deltaY > 0 ? -0.1 : 0.1; // Negative scroll = zoom in, positive = zoom out
            const newScale = state.uiScale + delta;
            setUIScale(newScale);
        }
    });

    editorContainer.appendChild(sidebar);
    editorContainer.appendChild(textareaWrapper);

    // Initial highlight
    updateHighlights();

    // Helper to set editor text (updates display and state)
    const setEditorText = (newText) => {
        editor.textContent = newText;
        state.text = newText;
        updateHighlights();
    };

    // Create the widget - this provides the "text" input for the node
    const widget = node.addDOMWidget("text", "customtext", editorContainer, {
        getValue() {
            return getPlainText();
        },
        setValue(v) {
            setEditorText(v);
        }
    });

    widget.inputEl = editor;
    widget.options.minNodeSize = [900, 600];
    widget.options.maxWidth = 1400;

    // Set initial node size on creation
    setTimeout(() => {
        node.setSize([900, 600]);
    }, 0);

    // ==================== SIDEBAR CONTROLS ====================

    // History controls
    const historySection = document.createElement("div");
    historySection.style.marginBottom = "8px";
    historySection.style.paddingBottom = "8px";
    historySection.style.borderBottom = "1px solid #444";

    const historyLabel = document.createElement("div");
    historyLabel.textContent = "History";
    historyLabel.style.fontWeight = "bold";
    historyLabel.style.marginBottom = "5px";
    historyLabel.style.fontSize = "11px";

    const historyControls = document.createElement("div");
    historyControls.style.display = "flex";
    historyControls.style.gap = "5px";
    historyControls.style.marginBottom = "5px";

    const undoBtn = document.createElement("button");
    undoBtn.textContent = "↶";
    undoBtn.title = "Undo (Ctrl+Z)";
    undoBtn.style.flex = "1";
    undoBtn.style.padding = "4px";
    undoBtn.style.cursor = "pointer";
    undoBtn.style.fontSize = "12px";
    undoBtn.style.background = "#3a3a3a";
    undoBtn.style.color = "#eee";
    undoBtn.style.border = "1px solid #555";
    undoBtn.style.borderRadius = "2px";

    const redoBtn = document.createElement("button");
    redoBtn.textContent = "↷";
    redoBtn.title = "Redo (Ctrl+Shift+Z)";
    redoBtn.style.flex = "1";
    redoBtn.style.padding = "4px";
    redoBtn.style.cursor = "pointer";
    redoBtn.style.fontSize = "12px";
    redoBtn.style.background = "#3a3a3a";
    redoBtn.style.color = "#eee";
    redoBtn.style.border = "1px solid #555";
    redoBtn.style.borderRadius = "2px";

    const historyStatus = document.createElement("div");
    historyStatus.style.fontSize = "10px";
    historyStatus.style.textAlign = "center";
    historyStatus.style.color = "#999";

    historyControls.appendChild(undoBtn);
    historyControls.appendChild(redoBtn);

    historySection.appendChild(historyLabel);
    historySection.appendChild(historyControls);
    historySection.appendChild(historyStatus);

    // Character controls
    const charSection = document.createElement("div");
    charSection.style.marginBottom = "8px";
    charSection.style.paddingBottom = "8px";
    charSection.style.borderBottom = "1px solid #444";

    const charLabel = document.createElement("div");
    charLabel.textContent = "Character";
    charLabel.style.fontWeight = "bold";
    charLabel.style.marginBottom = "5px";
    charLabel.style.fontSize = "11px";

    const charSelect = document.createElement("select");
    charSelect.style.width = "100%";
    charSelect.style.marginBottom = "4px";
    charSelect.style.padding = "3px";
    charSelect.style.fontSize = "10px";
    charSelect.style.background = "#2a2a2a";
    charSelect.style.color = "#eee";
    charSelect.style.border = "1px solid #444";
    charSelect.innerHTML = "<option value=''>Select...</option>";

    // Populate characters with default character names
    const populateCharacters = () => {
        try {
            // Show common default character names that most TTS engines have
            const defaultCharacters = ["Alice", "Bob", "Charlie", "Diana", "Emma", "Frank", "Grace", "Henry"];
            defaultCharacters.forEach(char => {
                const option = document.createElement("option");
                option.value = char;
                option.textContent = char;
                charSelect.appendChild(option);
            });
            console.log(`✅ Loaded ${defaultCharacters.length} default character voices`);
        } catch (err) {
            console.warn("Could not populate characters:", err);
        }
    };

    // Load characters immediately
    populateCharacters();

    const charInput = document.createElement("input");
    charInput.type = "text";
    charInput.placeholder = "Custom";
    charInput.style.width = "100%";
    charInput.style.marginBottom = "4px";
    charInput.style.padding = "3px";
    charInput.style.fontSize = "10px";
    charInput.style.background = "#2a2a2a";
    charInput.style.color = "#eee";
    charInput.style.border = "1px solid #444";
    charInput.value = state.lastCharacter;

    charInput.addEventListener("change", () => {
        state.lastCharacter = charInput.value;
        state.saveToLocalStorage(storageKey);
    });

    const addCharBtn = document.createElement("button");
    addCharBtn.textContent = "Add Char";
    addCharBtn.title = "Insert selected character tag at cursor position";
    addCharBtn.style.width = "100%";
    addCharBtn.style.padding = "4px";
    addCharBtn.style.cursor = "pointer";
    addCharBtn.style.fontSize = "10px";
    addCharBtn.style.background = "#3a3a3a";
    addCharBtn.style.color = "#eee";
    addCharBtn.style.border = "1px solid #555";
    addCharBtn.style.borderRadius = "2px";

    charSection.appendChild(charLabel);
    charSection.appendChild(charSelect);
    charSection.appendChild(charInput);
    charSection.appendChild(addCharBtn);

    // Language controls
    const langSection = document.createElement("div");
    langSection.style.marginBottom = "8px";
    langSection.style.paddingBottom = "8px";
    langSection.style.borderBottom = "1px solid #444";

    const langLabel = document.createElement("div");
    langLabel.textContent = "Language";
    langLabel.style.fontWeight = "bold";
    langLabel.style.marginBottom = "5px";
    langLabel.style.fontSize = "11px";

    const langSelect = document.createElement("select");
    langSelect.style.width = "100%";
    langSelect.style.padding = "3px";
    langSelect.style.fontSize = "10px";
    langSelect.style.background = "#2a2a2a";
    langSelect.style.color = "#eee";
    langSelect.style.border = "1px solid #444";
    const languages = ["en", "de", "fr", "ja", "es", "it", "pt", "th", "no"];
    langSelect.innerHTML = "<option value=''>No language</option>";
    languages.forEach(lang => {
        const option = document.createElement("option");
        option.value = lang;
        option.textContent = lang.toUpperCase();
        langSelect.appendChild(option);
    });
    langSelect.value = state.lastLanguage;

    langSelect.addEventListener("change", () => {
        state.lastLanguage = langSelect.value;
        state.saveToLocalStorage(storageKey);
    });

    langSection.appendChild(langLabel);
    langSection.appendChild(langSelect);

    // Parameter controls - dynamic parameter selector
    const paramSection = document.createElement("div");
    paramSection.style.marginBottom = "8px";
    paramSection.style.paddingBottom = "8px";
    paramSection.style.borderBottom = "1px solid #444";

    const paramLabel = document.createElement("div");
    paramLabel.textContent = "Add Parameter";
    paramLabel.style.fontWeight = "bold";
    paramLabel.style.marginBottom = "5px";
    paramLabel.style.fontSize = "11px";

    // Parameter type selector with all supported parameters
    const paramTypeSelect = document.createElement("select");
    paramTypeSelect.style.width = "100%";
    paramTypeSelect.style.marginBottom = "5px";
    paramTypeSelect.style.padding = "3px";
    paramTypeSelect.style.fontSize = "10px";
    paramTypeSelect.style.background = "#2a2a2a";
    paramTypeSelect.style.color = "#eee";
    paramTypeSelect.style.border = "1px solid #444";
    paramTypeSelect.innerHTML = `
        <option value="">Select parameter...</option>
        <optgroup label="Universal">
            <option value="seed">Seed</option>
            <option value="temperature">Temperature</option>
        </optgroup>
        <optgroup label="ChatterBox">
            <option value="cfg">CFG Weight</option>
            <option value="exaggeration">Exaggeration</option>
        </optgroup>
        <optgroup label="F5-TTS / Higgs">
            <option value="speed">Speed</option>
        </optgroup>
        <optgroup label="Higgs / VibeVoice / IndexTTS">
            <option value="top_p">Top P</option>
            <option value="top_k">Top K</option>
        </optgroup>
        <optgroup label="VibeVoice / IndexTTS">
            <option value="steps">Inference Steps</option>
        </optgroup>
        <optgroup label="IndexTTS">
            <option value="emotion_alpha">Emotion Alpha</option>
        </optgroup>
    `;

    // Input container that changes based on parameter type
    const paramInputWrapper = document.createElement("div");
    paramInputWrapper.style.marginBottom = "5px";

    // Helper to create input for a parameter type
    const paramConfig = {
        seed: { type: "number", min: 0, max: 4294967295, step: 1, default: 0 },
        temperature: { type: "slider", min: 0.1, max: 2.0, step: 0.1, default: 0.7, label: "Temp" },
        cfg: { type: "slider", min: 0.0, max: 20.0, step: 0.1, default: 7.0, label: "CFG" },
        speed: { type: "slider", min: 0.5, max: 2.0, step: 0.1, default: 1.0, label: "Speed" },
        exaggeration: { type: "slider", min: 0.0, max: 2.0, step: 0.1, default: 1.0, label: "Exag" },
        top_p: { type: "slider", min: 0.0, max: 1.0, step: 0.01, default: 0.95, label: "Top P" },
        top_k: { type: "number", min: 1, max: 100, step: 1, default: 50 },
        steps: { type: "number", min: 1, max: 100, step: 1, default: 30 },
        emotion_alpha: { type: "slider", min: 0.0, max: 1.0, step: 0.05, default: 0.5, label: "Emotion" }
    };

    const createParamInput = (type) => {
        const wrapper = document.createElement("div");
        const config = paramConfig[type] || { type: "text" };

        if (config.type === "number") {
            const input = document.createElement("input");
            input.type = "number";
            input.min = config.min;
            input.max = config.max;
            input.step = config.step;
            input.placeholder = config.default.toString();
            input.value = config.default;
            input.style.width = "100%";
            input.style.padding = "3px";
            input.style.fontSize = "10px";
            input.style.background = "#2a2a2a";
            input.style.color = "#eee";
            input.style.border = "1px solid #444";
            input.addEventListener("change", () => {
                state[`last${type.charAt(0).toUpperCase() + type.slice(1)}`] = input.value;
                state.saveToLocalStorage(storageKey);
            });
            wrapper.appendChild(input);
            wrapper.getValue = () => input.value;
            return wrapper;
        } else if (config.type === "slider") {
            const label = document.createElement("div");
            label.style.fontSize = "9px";
            label.style.marginBottom = "2px";
            label.style.color = "#999";
            label.textContent = `${config.label}: ${config.default}`;

            const slider = document.createElement("input");
            slider.type = "range";
            slider.min = config.min;
            slider.max = config.max;
            slider.step = config.step;
            slider.value = config.default;
            slider.style.width = "100%";
            slider.addEventListener("input", () => {
                label.textContent = `${config.label}: ${slider.value}`;
                state[`last${type.charAt(0).toUpperCase() + type.slice(1)}`] = slider.value;
                state.saveToLocalStorage(storageKey);
            });

            wrapper.appendChild(label);
            wrapper.appendChild(slider);
            wrapper.getValue = () => slider.value;
            return wrapper;
        } else {
            const input = document.createElement("input");
            input.type = "text";
            input.placeholder = `${type} value`;
            input.style.width = "100%";
            input.style.padding = "3px";
            input.style.fontSize = "10px";
            input.style.background = "#2a2a2a";
            input.style.color = "#eee";
            input.style.border = "1px solid #444";
            wrapper.appendChild(input);
            wrapper.getValue = () => input.value;
            return wrapper;
        }
    };

    let currentParamInput = null;

    // When parameter type changes, update the input
    paramTypeSelect.addEventListener("change", () => {
        paramInputWrapper.innerHTML = "";
        if (paramTypeSelect.value) {
            currentParamInput = createParamInput(paramTypeSelect.value);
            paramInputWrapper.appendChild(currentParamInput);
            // Persist the selected parameter type
            state.lastParameterType = paramTypeSelect.value;
            state.saveToLocalStorage(storageKey);
        } else {
            currentParamInput = null;
        }
    });

    // Restore previously selected parameter type
    if (state.lastParameterType) {
        paramTypeSelect.value = state.lastParameterType;
        const changeEvent = new Event("change");
        paramTypeSelect.dispatchEvent(changeEvent);
    }

    // Add parameter button
    const addParamBtn = document.createElement("button");
    addParamBtn.textContent = "Add to Tag";
    addParamBtn.title = "Add selected parameter to tag at cursor or create new parameter tag";
    addParamBtn.style.width = "100%";
    addParamBtn.style.padding = "4px";
    addParamBtn.style.cursor = "pointer";
    addParamBtn.style.fontSize = "10px";
    addParamBtn.style.background = "#3a3a3a";
    addParamBtn.style.color = "#eee";
    addParamBtn.style.border = "1px solid #555";
    addParamBtn.style.borderRadius = "2px";

    addParamBtn.addEventListener("click", () => {
        if (!paramTypeSelect.value || !currentParamInput) {
            return;
        }

        const paramValue = currentParamInput.getValue();
        if (!paramValue) {
            return;
        }

        const paramStr = `${paramTypeSelect.value}:${paramValue}`;
        const text = getPlainText();
        const caretPos = getCaretPos();
        const selectionStart = caretPos;
        const selectionEnd = caretPos;

        // Check if caret is right after the closing bracket OR inside a tag
        const isRightAfterTag = selectionStart > 0 && text[selectionStart - 1] === "]";

        // Check if caret is INSIDE a tag (between [ and ])
        let isInsideTag = false;
        let tagStartInside = -1;
        let tagEndInside = -1;

        if (!isRightAfterTag) {
            // Look for the nearest tag that contains this position
            let bracketDepth = 0;
            for (let i = selectionStart - 1; i >= 0; i--) {
                if (text[i] === "]") {
                    bracketDepth++;
                } else if (text[i] === "[") {
                    if (bracketDepth === 0) {
                        // Found the opening bracket
                        tagStartInside = i;
                        // Now find the closing bracket
                        let innerDepth = 1;
                        for (let j = i + 1; j < text.length; j++) {
                            if (text[j] === "[") {
                                innerDepth++;
                            } else if (text[j] === "]") {
                                innerDepth--;
                                if (innerDepth === 0) {
                                    tagEndInside = j;
                                    if (tagEndInside > selectionStart) {
                                        isInsideTag = true;
                                    }
                                    break;
                                }
                            }
                        }
                        break;
                    } else {
                        bracketDepth--;
                    }
                }
            }
        }

        if (isRightAfterTag || isInsideTag) {
            let tagStart, tagEnd;

            if (isRightAfterTag) {
                // Find the matching opening bracket for this closing bracket
                tagEnd = selectionStart - 1;

                // Scan backwards to find the matching opening bracket, counting bracket pairs
                let bracketDepth = 1;
                tagStart = -1;
                for (let i = tagEnd - 1; i >= 0; i--) {
                    if (text[i] === "]") {
                        bracketDepth++;
                    } else if (text[i] === "[") {
                        bracketDepth--;
                        if (bracketDepth === 0) {
                            tagStart = i;
                            break;
                        }
                    }
                }
            } else {
                // Use the tag positions we found when checking if inside tag
                tagStart = tagStartInside;
                tagEnd = tagEndInside;
            }

            // Verify the tag is valid (opening bracket found before closing)
            if (tagStart !== -1 && tagStart < tagEnd) {
                // Inside a valid tag - add or replace parameter
                let tagContent = text.substring(tagStart + 1, tagEnd);
                const paramType = paramTypeSelect.value;

                // Check if parameter already exists and replace it
                const paramRegex = new RegExp(`${paramType}:[^|\\]]+`);
                if (paramRegex.test(tagContent)) {
                    // Replace existing parameter
                    tagContent = tagContent.replace(paramRegex, paramStr);
                } else {
                    // Add new parameter
                    tagContent = `${tagContent}|${paramStr}`;
                }

                const newText = text.substring(0, tagStart + 1) + tagContent + "]" + text.substring(tagEnd + 1);

                setEditorText(newText);
                // Move caret to right after the closing bracket
                const newCaretPos = tagStart + 1 + tagContent.length + 1; // +1 for [, +1 for ]
                setTimeout(() => {
                    setCaretPos(newCaretPos);
                    state.addToHistory(newText, newCaretPos);
                    state.saveToLocalStorage(storageKey);
                }, 0);
                widget.callback?.(widget.value);
                historyStatus.textContent = state.getHistoryStatus();
            } else {
                // Invalid tag structure, create new tag instead
                const paramTag = `[${paramStr}]`;
                const newText = text.substring(0, selectionStart) + paramTag + " " + text.substring(selectionStart);
                setEditorText(newText);
                // Move caret to right after the new tag
                const newCaretPos = selectionStart + paramTag.length;
                setTimeout(() => {
                    setCaretPos(newCaretPos);
                    state.addToHistory(newText, newCaretPos);
                    state.saveToLocalStorage(storageKey);
                }, 0);
                widget.callback?.(widget.value);
                historyStatus.textContent = state.getHistoryStatus();
            }
        } else {
            // Caret is NOT right after a tag - create new parameter tag at caret position
            const paramTag = `[${paramStr}]`;
            const newText = text.substring(0, selectionStart) + paramTag + " " + text.substring(selectionStart);

            setEditorText(newText);
            // Move caret to right after the new tag
            const newCaretPos = selectionStart + paramTag.length;
            setTimeout(() => {
                setCaretPos(newCaretPos);
                state.addToHistory(newText, newCaretPos);
                state.saveToLocalStorage(storageKey);
            }, 0);
            widget.callback?.(widget.value);
            historyStatus.textContent = state.getHistoryStatus();
        }

        // Keep the value so user doesn't have to re-enter it
    });

    paramSection.appendChild(paramLabel);
    paramSection.appendChild(paramTypeSelect);
    paramSection.appendChild(paramInputWrapper);
    paramSection.appendChild(addParamBtn);

    // Preset controls
    const presetSection = document.createElement("div");
    presetSection.style.marginBottom = "8px";
    presetSection.style.paddingBottom = "8px";
    presetSection.style.borderBottom = "1px solid #444";

    const presetLabel = document.createElement("div");
    presetLabel.textContent = "Presets";
    presetLabel.style.fontWeight = "bold";
    presetLabel.style.marginBottom = "5px";
    presetLabel.style.fontSize = "11px";

    presetSection.appendChild(presetLabel);

    const presetButtons = {};

    for (let i = 1; i <= 3; i++) {
        const presetKey = `preset_${i}`;
        const presetContainer = document.createElement("div");
        presetContainer.style.marginBottom = "5px";

        const presetTitle = document.createElement("div");
        presetTitle.style.fontSize = "10px";
        presetTitle.style.fontWeight = "bold";
        presetTitle.style.marginBottom = "3px";
        presetTitle.style.color = "#bbb";
        presetTitle.textContent = `P${i}`;

        const presetControls = document.createElement("div");
        presetControls.style.display = "flex";
        presetControls.style.gap = "2px";

        presetButtons[presetKey] = {};

        ["Save", "Load", "Del"].forEach(btnText => {
            const btn = document.createElement("button");
            btn.textContent = btnText;
            if (btnText === "Save") {
                btn.title = "Save current tag or text as preset (glows green when data exists)";
            } else if (btnText === "Load") {
                btn.title = "Load preset into text at cursor position";
            } else {
                btn.title = "Delete this preset";
            }
            btn.style.flex = "1";
            btn.style.padding = "3px";
            btn.style.fontSize = "9px";
            btn.style.cursor = "pointer";
            btn.style.background = "#3a3a3a";
            btn.style.color = "#eee";
            btn.style.border = "1px solid #555";
            btn.style.borderRadius = "2px";
            presetButtons[presetKey][btnText.toLowerCase()] = btn;
            presetControls.appendChild(btn);
        });

        presetContainer.appendChild(presetTitle);
        presetContainer.appendChild(presetControls);
        presetSection.appendChild(presetContainer);
    }

    // Restore preset button glow states based on saved presets
    const updatePresetGlows = () => {
        Object.entries(presetButtons).forEach(([presetKey, buttons]) => {
            if (presetKey in state.presets && state.presets[presetKey]) {
                // Preset exists - keep load button glowing green to show it has data
                buttons.load.style.background = "#00cc00";
                buttons.load.style.boxShadow = "0 0 8px #00cc00";
            } else {
                // Preset empty - normal style
                buttons.load.style.background = "#3a3a3a";
                buttons.load.style.boxShadow = "none";
            }
        });
    };

    // Update glow on load
    updatePresetGlows();

    // Validation controls
    const validSection = document.createElement("div");

    const formatBtn = document.createElement("button");
    formatBtn.textContent = "Format";
    formatBtn.title = "Normalize spacing and structure of tags and text";
    formatBtn.style.width = "100%";
    formatBtn.style.padding = "4px";
    formatBtn.style.marginBottom = "4px";
    formatBtn.style.cursor = "pointer";
    formatBtn.style.fontSize = "10px";
    formatBtn.style.background = "#3a3a3a";
    formatBtn.style.color = "#eee";
    formatBtn.style.border = "1px solid #555";
    formatBtn.style.borderRadius = "2px";

    const validateBtn = document.createElement("button");
    validateBtn.textContent = "Validate";
    validateBtn.title = "Check tag syntax for errors and issues";
    validateBtn.style.width = "100%";
    validateBtn.style.padding = "4px";
    validateBtn.style.cursor = "pointer";
    validateBtn.style.fontSize = "10px";
    validateBtn.style.background = "#3a3a3a";
    validateBtn.style.color = "#eee";
    validateBtn.style.border = "1px solid #555";
    validateBtn.style.borderRadius = "2px";

    validSection.appendChild(formatBtn);
    validSection.appendChild(validateBtn);

    // Assemble sidebar
    sidebar.appendChild(historySection);
    sidebar.appendChild(charSection);
    sidebar.appendChild(langSection);
    sidebar.appendChild(paramSection);
    sidebar.appendChild(presetSection);
    sidebar.appendChild(validSection);

    // ==================== EVENT HANDLERS ====================

    // Editor input - add to history with smart debouncing
    let historyDebounceTimer = null;
    let lastHistoryText = "";

    const flushHistory = () => {
        const plainText = getPlainText();
        if (plainText !== lastHistoryText) {
            const caretPos = getCaretPos();
            state.addToHistory(plainText, caretPos);
            lastHistoryText = plainText;
        }
    };

    editor.addEventListener("input", (e) => {
        const plainText = getPlainText();

        // Update display immediately
        state.saveToLocalStorage(storageKey);
        widget.callback?.(widget.value);

        // Clear existing debounce timer
        if (historyDebounceTimer !== null) {
            clearTimeout(historyDebounceTimer);
        }

        // Set new debounce timer - adds to history after user stops typing for 500ms
        historyDebounceTimer = setTimeout(() => {
            flushHistory();
            historyStatus.textContent = state.getHistoryStatus();
            historyDebounceTimer = null;
        }, 500);
    });

    // Also flush history on specific actions (paste, cut, etc)
    editor.addEventListener("paste", (e) => {
        setTimeout(() => {
            flushHistory();
            historyStatus.textContent = state.getHistoryStatus();
        }, 0);
    });

    editor.addEventListener("cut", (e) => {
        setTimeout(() => {
            flushHistory();
            historyStatus.textContent = state.getHistoryStatus();
        }, 0);
    });

    // Undo/Redo buttons - restore text and caret position from history
    undoBtn.addEventListener("click", () => {
        const entry = state.undo();
        setEditorText(entry.text);
        setTimeout(() => setCaretPos(entry.caretPos || 0), 0);
        state.saveToLocalStorage(storageKey);
        widget.callback?.(widget.value);
        historyStatus.textContent = state.getHistoryStatus();
    });

    redoBtn.addEventListener("click", () => {
        const entry = state.redo();
        setEditorText(entry.text);
        setTimeout(() => setCaretPos(entry.caretPos || 0), 0);
        state.saveToLocalStorage(storageKey);
        widget.callback?.(widget.value);
        historyStatus.textContent = state.getHistoryStatus();
    });

    // Keyboard shortcuts for undo/redo
    editor.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "z") {
            e.preventDefault();
            let entry;
            if (e.shiftKey) {
                entry = state.redo();
            } else {
                entry = state.undo();
            }
            setEditorText(entry.text);
            setTimeout(() => setCaretPos(entry.caretPos || 0), 0);
            state.saveToLocalStorage(storageKey);
            widget.callback?.(widget.value);
            historyStatus.textContent = state.getHistoryStatus();
        }
    });

    // Ctrl+scroll to change font size
    editor.addEventListener("wheel", (e) => {
        if ((e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            const delta = e.deltaY > 0 ? -1 : 1; // Negative scroll = zoom out, positive = zoom in
            const newSize = state.fontSize + delta;
            setFontSize(newSize);
        }
    });

    // Character select dropdown
    charSelect.addEventListener("change", () => {
        if (charSelect.value) {
            charInput.value = charSelect.value;
            state.lastCharacter = charSelect.value;
            state.saveToLocalStorage(storageKey);
        }
    });

    // Add character button
    addCharBtn.addEventListener("click", () => {
        const char = charInput.value.trim() || charSelect.value;
        if (char) {
            state.lastCharacter = char;
            const selectionStart = caretPos;
            const selectionEnd = caretPos;

            const tag = `[${char}]`;
            const newText = TagUtilities.insertTagAtPosition(
                getPlainText(),
                tag,
                selectionStart,
                selectionStart !== selectionEnd,
                selectionStart,
                selectionEnd
            );

            setEditorText(newText);
            // Move cursor after tag
            const newCaretPos = selectionStart + tag.length + 1;
            setTimeout(() => {
                setCaretPos(newCaretPos);
                state.addToHistory(newText, newCaretPos);
                state.saveToLocalStorage(storageKey);
                editor.focus();
            }, 0);
            widget.callback?.(widget.value);
            historyStatus.textContent = state.getHistoryStatus();
        }
    });

    // Format button - normalize spacing and structure
    formatBtn.addEventListener("click", () => {
        let text = getPlainText();
        // Normalize spacing around tags (but preserve newlines before tags)
        text = text.replace(/ *\[ */g, "[").replace(/ *\]/g, "]");
        // Ensure space after closing bracket (unless followed by newline)
        text = text.replace(/\]([^\s\n])/g, "] $1");
        // Remove trailing spaces from each line
        text = text.split("\n").map(line => line.trimEnd()).join("\n");

        setEditorText(text);
        // Restore caret to start after formatting
        setTimeout(() => {
            state.addToHistory(text, 0);
            state.saveToLocalStorage(storageKey);
            setCaretPos(0);
        }, 0);
        widget.callback?.(widget.value);
        historyStatus.textContent = state.getHistoryStatus();
    });

    // Validation
    validateBtn.addEventListener("click", () => {
        const validation = TagUtilities.validateTagSyntax(getPlainText());
        if (validation.valid) {
            showNotification("✅ Tag syntax is valid!");
        } else {
            showNotification("❌ " + validation.error, 3000);
        }
    });

    // Preset button handlers
    Object.entries(presetButtons).forEach(([presetKey, buttons]) => {
        const presetNum = presetKey.split("_")[1];

        buttons.save.addEventListener("click", () => {
            // First check if user selected text in textarea (like [de:Alice|seed:42|temp:0.8])
            const selectionStart = caretPos;
            const selectionEnd = caretPos;
            let selectedText = "";

            if (selectionStart !== selectionEnd) {
                selectedText = getPlainText().substring(selectionStart, selectionEnd);
                // Store the selected text as the preset
                state.presets[presetKey] = {
                    tag: selectedText,
                    isComplexTag: true
                };
                state.saveToLocalStorage(storageKey);
                showNotification(`✅ Preset ${presetNum} saved from selection`);
                updatePresetGlows();
                return;
            }

            // Otherwise save character + language preset
            const currentTag = charInput.value.trim() || charSelect.value;
            if (currentTag) {
                state.presets[presetKey] = {
                    tag: currentTag,
                    parameters: {
                        language: langSelect.value,
                        lastSeed: state.lastSeed,
                        lastTemperature: state.lastTemperature
                    }
                };
                state.saveToLocalStorage(storageKey);
                showNotification(`✅ Preset ${presetNum} saved: ${currentTag}`);
                updatePresetGlows();
            } else {
                showNotification("⚠️ Select text or enter character", 2500);
            }
        });

        buttons.load.addEventListener("click", () => {
            if (presetKey in state.presets) {
                const preset = state.presets[presetKey];

                // If it's a complex tag saved from selection, insert it at cursor
                if (preset.isComplexTag) {
                    const selectionStart = caretPos;
                    const newText = getPlainText().substring(0, selectionStart) + preset.tag + " " + getPlainText().substring(selectionStart);
                    setEditorText(newText);
                    // Move caret to after the inserted text
                    const newCaretPos = selectionStart + preset.tag.length + 1; // +1 for the space after tag
                    setTimeout(() => {
                        setCaretPos(newCaretPos);
                        state.addToHistory(newText, newCaretPos);
                        state.saveToLocalStorage(storageKey);
                        editor.focus();
                    }, 0);
                    widget.callback?.(widget.value);
                    historyStatus.textContent = state.getHistoryStatus();

                    showNotification(`✅ Preset ${presetNum} inserted`);
                } else {
                    // Otherwise load character + parameters
                    charInput.value = preset.tag;
                    state.lastCharacter = preset.tag;

                    if (preset.parameters) {
                        if (preset.parameters.language) langSelect.value = preset.parameters.language;
                        if (preset.parameters.lastSeed) state.lastSeed = preset.parameters.lastSeed;
                        if (preset.parameters.lastTemperature) state.lastTemperature = preset.parameters.lastTemperature;
                    }

                    state.saveToLocalStorage(storageKey);
                    showNotification(`✅ Preset ${presetNum} loaded: ${preset.tag}`);
                }
            } else {
                showNotification("⚠️ Preset is empty", 2000);
            }
        });

        buttons.del.addEventListener("click", () => {
            if (presetKey in state.presets) {
                delete state.presets[presetKey];
                state.saveToLocalStorage(storageKey);
                showNotification(`✅ Preset ${presetNum} deleted`);
                updatePresetGlows();
            } else {
                showNotification("⚠️ Preset already empty", 2000);
            }
        });
    });

    // Middle mouse button panning support
    editor.addEventListener("pointerdown", (e) => {
        if (e.button === 1) {
            app.canvas.processMouseDown(e);
        }
    });

    editor.addEventListener("pointermove", (e) => {
        if ((e.buttons & 4) === 4) {
            app.canvas.processMouseMove(e);
        }
    });

    editor.addEventListener("pointerup", (e) => {
        if (e.button === 1) {
            app.canvas.processMouseUp(e);
        }
    });

    // Store state when node is removed
    widget.onRemove = () => {
        state.saveToLocalStorage(storageKey);
    };

    // Initialize history display
    historyStatus.textContent = state.getHistoryStatus();

    return widget;
}

// Register the widget
app.registerExtension({
    name: "StringMultilineTagEditor",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "StringMultilineTagEditor") {
            // Override onNodeCreated to create our custom widget
            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                // Call original to set up the node
                if (originalOnNodeCreated) {
                    originalOnNodeCreated.call(this);
                }

                // Remove any existing widgets (there shouldn't be any since we have no inputs)
                this.widgets = [];

                // Create our custom widget
                addStringMultilineTagEditorWidget(this);
            };
        }
    }
});
