/**
 * 🏷️ String Multiline Tag Editor
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
        this.sidebarExpanded = true;
        this.lastCursorPosition = 0;
        this.discoveredCharacters = {};
    }

    addToHistory(text) {
        // Remove any future history if we've undone something
        if (this.historyIndex < this.history.length - 1) {
            this.history = this.history.slice(0, this.historyIndex + 1);
        }

        this.history.push(text);
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
            this.text = this.history[this.historyIndex];
            return this.text;
        }
        return this.text;
    }

    redo() {
        if (this.historyIndex < this.history.length - 1) {
            this.historyIndex++;
            this.text = this.history[this.historyIndex];
            return this.text;
        }
        return this.text;
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

    // Create sidebar
    const sidebar = document.createElement("div");
    sidebar.className = "string-multiline-tag-editor-sidebar";
    sidebar.style.width = "220px";
    sidebar.style.minWidth = "220px";
    sidebar.style.maxWidth = "220px";
    sidebar.style.height = "100%";
    sidebar.style.background = "#222";
    sidebar.style.borderRight = "1px solid #444";
    sidebar.style.padding = "10px";
    sidebar.style.overflowY = "auto";
    sidebar.style.overflowX = "hidden";
    sidebar.style.fontSize = "11px";
    sidebar.style.flexShrink = "0";
    sidebar.style.display = "flex";
    sidebar.style.flexDirection = "column";

    // Create textarea
    const textarea = document.createElement("textarea");
    textarea.className = "comfy-multiline-input";
    textarea.value = state.text;
    textarea.placeholder = "Enter text with tags...\n\nExamples:\n[Alice] Hello!\n[Bob|seed:42] Hi!\ntext [char] more text";
    textarea.spellcheck = false;
    textarea.style.flex = "1 1 auto";
    textarea.style.fontFamily = "monospace";
    textarea.style.fontSize = "13px";
    textarea.style.padding = "10px";
    textarea.style.border = "none";
    textarea.style.background = "#1a1a1a";
    textarea.style.color = "#eee";
    textarea.style.outline = "none";
    textarea.style.margin = "0";
    textarea.style.resize = "none";
    textarea.style.minHeight = "0";
    textarea.style.width = "100%";

    editorContainer.appendChild(sidebar);
    editorContainer.appendChild(textarea);

    // Create the widget - this is the ONLY widget for this node
    const widget = node.addDOMWidget("text_output", "customtext", editorContainer, {
        getValue() {
            return textarea.value;
        },
        setValue(v) {
            textarea.value = v;
            state.text = v;
        }
    });

    widget.inputEl = textarea;
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

    // Populate characters from available voices
    const populateCharacters = () => {
        fetch("/api/tts_audio_suite/available_characters")
            .then(r => r.json())
            .then(data => {
                if (data.characters && Array.isArray(data.characters)) {
                    data.characters.forEach(char => {
                        const option = document.createElement("option");
                        option.value = char;
                        option.textContent = char;
                        charSelect.appendChild(option);
                    });
                    console.log(`✅ Loaded ${data.characters.length} available characters`);
                }
            })
            .catch(err => console.warn("Could not load characters:", err));
    };

    // Load characters after a short delay to ensure ComfyUI is ready
    setTimeout(populateCharacters, 500);

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

    // Parameter controls - placeholder section
    const paramSection = document.createElement("div");
    paramSection.style.marginBottom = "8px";
    paramSection.style.paddingBottom = "8px";
    paramSection.style.borderBottom = "1px solid #444";

    const paramLabel = document.createElement("div");
    paramLabel.textContent = "Parameters";
    paramLabel.style.fontWeight = "bold";
    paramLabel.style.marginBottom = "5px";
    paramLabel.style.fontSize = "11px";

    const paramInfo = document.createElement("div");
    paramInfo.style.fontSize = "10px";
    paramInfo.style.color = "#999";
    paramInfo.textContent = "Add parameters via 'Add Parameter' section";

    paramSection.appendChild(paramLabel);
    paramSection.appendChild(paramInfo);

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

    // Validation controls
    const validSection = document.createElement("div");

    const formatBtn = document.createElement("button");
    formatBtn.textContent = "Format";
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

    // Textarea input - add to history
    textarea.addEventListener("input", (e) => {
        state.addToHistory(e.target.value);
        state.saveToLocalStorage(storageKey);
        widget.callback?.(widget.value);
        historyStatus.textContent = state.getHistoryStatus();
    });

    // Undo/Redo buttons
    undoBtn.addEventListener("click", () => {
        textarea.value = state.undo();
        state.saveToLocalStorage(storageKey);
        widget.callback?.(widget.value);
        historyStatus.textContent = state.getHistoryStatus();
    });

    redoBtn.addEventListener("click", () => {
        textarea.value = state.redo();
        state.saveToLocalStorage(storageKey);
        widget.callback?.(widget.value);
        historyStatus.textContent = state.getHistoryStatus();
    });

    // Keyboard shortcuts for undo/redo
    textarea.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "z") {
            e.preventDefault();
            if (e.shiftKey) {
                textarea.value = state.redo();
            } else {
                textarea.value = state.undo();
            }
            state.saveToLocalStorage(storageKey);
            widget.callback?.(widget.value);
            historyStatus.textContent = state.getHistoryStatus();
        }
    });

    // Character select dropdown
    charSelect.addEventListener("change", () => {
        if (charSelect.value) {
            charInput.value = charSelect.value;
        }
    });

    // Add character button
    addCharBtn.addEventListener("click", () => {
        const char = charInput.value.trim() || charSelect.value;
        if (char) {
            state.lastCharacter = char;
            const selectionStart = textarea.selectionStart;
            const selectionEnd = textarea.selectionEnd;

            const tag = `[${char}]`;
            const newText = TagUtilities.insertTagAtPosition(
                textarea.value,
                tag,
                selectionStart,
                selectionStart !== selectionEnd,
                selectionStart,
                selectionEnd
            );

            textarea.value = newText;
            state.addToHistory(newText);
            state.saveToLocalStorage(storageKey);
            widget.callback?.(widget.value);
            historyStatus.textContent = state.getHistoryStatus();

            // Move cursor after tag
            setTimeout(() => {
                textarea.selectionStart = textarea.selectionEnd = selectionStart + tag.length + 1;
                textarea.focus();
            }, 0);
        }
    });

    // Format button - normalize spacing and structure
    formatBtn.addEventListener("click", () => {
        let text = textarea.value;
        // Normalize spacing around tags
        text = text.replace(/\s*\[\s*/g, "[").replace(/\s*\]/g, "]");
        // Ensure space after closing bracket
        text = text.replace(/\]([^\s])/g, "] $1");
        // Remove trailing spaces
        text = text.split("\n").map(line => line.trimEnd()).join("\n");

        textarea.value = text;
        state.addToHistory(text);
        state.saveToLocalStorage(storageKey);
        widget.callback?.(widget.value);
        historyStatus.textContent = state.getHistoryStatus();
        alert("✅ Text formatted!");
    });

    // Validation
    validateBtn.addEventListener("click", () => {
        const validation = TagUtilities.validateTagSyntax(textarea.value);
        if (validation.valid) {
            alert("✅ Tag syntax is valid!");
        } else {
            alert("❌ Syntax error: " + validation.error);
        }
    });

    // Preset button handlers
    Object.entries(presetButtons).forEach(([presetKey, buttons]) => {
        const presetNum = presetKey.split("_")[1];

        buttons.save.addEventListener("click", () => {
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
                alert(`✅ Preset ${presetNum} saved!\nCharacter: ${currentTag}\nLanguage: ${langSelect.value || 'none'}`);
            } else {
                alert("⚠️ Please select or enter a character first");
            }
        });

        buttons.load.addEventListener("click", () => {
            if (presetKey in state.presets) {
                const preset = state.presets[presetKey];
                charInput.value = preset.tag;
                state.lastCharacter = preset.tag;

                if (preset.parameters) {
                    if (preset.parameters.language) langSelect.value = preset.parameters.language;
                    if (preset.parameters.lastSeed) state.lastSeed = preset.parameters.lastSeed;
                    if (preset.parameters.lastTemperature) state.lastTemperature = preset.parameters.lastTemperature;
                }

                state.saveToLocalStorage(storageKey);
                alert(`✅ Preset ${presetNum} loaded!\nCharacter: ${preset.tag}`);
            } else {
                alert("⚠️ This preset is empty");
            }
        });

        buttons.del.addEventListener("click", () => {
            if (presetKey in state.presets) {
                delete state.presets[presetKey];
                state.saveToLocalStorage(storageKey);
                alert(`✅ Preset ${presetNum} deleted!`);
            } else {
                alert("⚠️ This preset is already empty");
            }
        });
    });

    // Middle mouse button panning support
    textarea.addEventListener("pointerdown", (e) => {
        if (e.button === 1) {
            app.canvas.processMouseDown(e);
        }
    });

    textarea.addEventListener("pointermove", (e) => {
        if ((e.buttons & 4) === 4) {
            app.canvas.processMouseMove(e);
        }
    });

    textarea.addEventListener("pointerup", (e) => {
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
