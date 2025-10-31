/**
 * 🏷️ Multiline TTS Tag Editor
 * Advanced multiline text editor with TTS tag support, undo/redo, and full persistence
 */

import { app } from "/scripts/app.js";
import { EditorState } from "./editor-state.js";
import { TagUtilities } from "./tag-utilities.js";
import { SyntaxHighlighter } from "./syntax-highlighter.js";
import { FontControls } from "./font-controls.js";
import { buildHistorySection, buildCharacterSection, buildLanguageSection, buildValidationSection } from "./widget-ui-builder.js";
import { buildParameterSection } from "./widget-parameter-section.js";
import { buildPresetSection } from "./widget-preset-system.js";
import { attachAllEventHandlers } from "./widget-event-handlers.js";


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
        // Update divider position to match new sidebar width
        if (resizeDivider) {
            resizeDivider.style.left = (newWidth - 3) + "px"; // 3px left + 3px right of border
        }
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
    textareaWrapper.style.flexDirection = "column";
    textareaWrapper.style.minHeight = "0";
    textareaWrapper.style.width = "100%";

    // Create contenteditable div - replaces both textarea and overlay
    const editor = document.createElement("div");
    editor.contentEditable = "true";
    editor.className = "comfy-multiline-input";
    editor.style.flex = "1 1 auto";
    editor.style.fontFamily = state.fontFamily;
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
        newSize = Math.max(2, Math.min(120, newSize)); // Clamp between 2px and 120px
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

    // Create font selector floating box (above editor)
    const fontBox = document.createElement("div");
    fontBox.style.background = "#2a2a2a";
    fontBox.style.border = "1px solid #444";
    fontBox.style.borderBottom = "1px solid #333";
    fontBox.style.padding = "8px 10px";
    fontBox.style.display = "flex";
    fontBox.style.gap = "12px";
    fontBox.style.alignItems = "center";
    fontBox.style.flexShrink = "0";

    // Font family dropdown
    const fontFamilyLabel = document.createElement("div");
    fontFamilyLabel.textContent = "Font:";
    fontFamilyLabel.style.fontWeight = "bold";
    fontFamilyLabel.style.fontSize = "10px";
    fontFamilyLabel.style.color = "#bbb";
    fontFamilyLabel.style.minWidth = "35px";

    const fontFamilySelect = document.createElement("select");
    fontFamilySelect.style.padding = "4px 6px";
    fontFamilySelect.style.fontSize = "10px";
    fontFamilySelect.style.background = "#1a1a1a";
    fontFamilySelect.style.color = "#eee";
    fontFamilySelect.style.border = "1px solid #555";
    fontFamilySelect.style.borderRadius = "2px";
    fontFamilySelect.style.cursor = "pointer";
    fontFamilySelect.style.flex = "1";

    // Add diverse fonts - web-safe alternatives (no external dependencies needed)
    // Includes programming-friendly monospace and general purpose fonts
    const fontFamilies = [
        // Monospace fonts (best for code/TTS)
        { label: "Monospace (System)", value: "monospace" },
        { label: "Courier New", value: "Courier New, monospace" },
        { label: "Courier", value: "Courier, monospace" },
        { label: "Lucida Console", value: "Lucida Console, monospace" },
        { label: "Lucida Typewriter", value: "Lucida Typewriter, monospace" },
        { label: "Liberation Mono", value: "Liberation Mono, monospace" },
        // Serif fonts
        { label: "Georgia", value: "Georgia, serif" },
        { label: "Times New Roman", value: "Times New Roman, serif" },
        { label: "Garamond", value: "Garamond, serif" },
        { label: "Palatino", value: "Palatino Linotype, serif" },
        // Sans-serif fonts
        { label: "Arial", value: "Arial, sans-serif" },
        { label: "Helvetica", value: "Helvetica, sans-serif" },
        { label: "Verdana", value: "Verdana, sans-serif" },
        { label: "Trebuchet MS", value: "Trebuchet MS, sans-serif" },
        { label: "Impact", value: "Impact, sans-serif" },
        // Decorative
        { label: "Comic Sans", value: "Comic Sans MS, cursive" }
    ];

    fontFamilies.forEach(font => {
        const option = document.createElement("option");
        option.value = font.value;
        option.textContent = font.label;
        fontFamilySelect.appendChild(option);
    });

    // Font size control
    const fontSizeLabel = document.createElement("div");
    fontSizeLabel.textContent = "Size:";
    fontSizeLabel.style.fontWeight = "bold";
    fontSizeLabel.style.fontSize = "10px";
    fontSizeLabel.style.color = "#bbb";
    fontSizeLabel.style.minWidth = "35px";

    const fontSizeInput = document.createElement("input");
    fontSizeInput.type = "number";
    fontSizeInput.min = "2";
    fontSizeInput.max = "120";
    fontSizeInput.value = state.fontSize;
    fontSizeInput.style.padding = "4px 6px";
    fontSizeInput.style.fontSize = "10px";
    fontSizeInput.style.background = "#1a1a1a";
    fontSizeInput.style.color = "#eee";
    fontSizeInput.style.border = "1px solid #555";
    fontSizeInput.style.borderRadius = "2px";
    fontSizeInput.style.width = "50px";

    const fontSizeDisplay = document.createElement("div");
    fontSizeDisplay.textContent = state.fontSize + "px";
    fontSizeDisplay.style.fontSize = "10px";
    fontSizeDisplay.style.color = "#999";
    fontSizeDisplay.style.minWidth = "30px";
    fontSizeDisplay.style.textAlign = "right";

    // Assemble font box
    fontBox.appendChild(fontFamilyLabel);
    fontBox.appendChild(fontFamilySelect);
    fontBox.appendChild(fontSizeLabel);
    fontBox.appendChild(fontSizeInput);
    fontBox.appendChild(fontSizeDisplay);

    // Set initial font family selection
    fontFamilySelect.value = state.fontFamily;

    // Add font box to textareaWrapper (above editor)
    textareaWrapper.appendChild(fontBox);
    textareaWrapper.appendChild(editor);

    // Create floating invisible divider on top of everything for resizing
    const resizeDivider = document.createElement("div");
    resizeDivider.style.position = "absolute";
    resizeDivider.style.top = "0";
    resizeDivider.style.width = "6px"; // Invisible grabable area (3px left, 3px right of border)
    resizeDivider.style.height = "100%";
    resizeDivider.style.cursor = "col-resize";
    resizeDivider.style.zIndex = "1000"; // On top of everything
    resizeDivider.style.userSelect = "none";
    resizeDivider.style.background = "transparent"; // Invisible
    editorContainer.appendChild(resizeDivider);

    // Update divider position when sidebar width changes (centered on border)
    const updateDividerPosition = () => {
        resizeDivider.style.left = (state.sidebarWidth - 3) + "px"; // 3px left + 3px right of border
    };
    updateDividerPosition();

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

    // Build sidebar sections using extracted modules
    const historyData = buildHistorySection(state, storageKey);
    const { historySection, undoBtn, redoBtn, historyStatus } = historyData;

    const charData = buildCharacterSection(state, storageKey);
    const { charSection, charSelect, charInput, addCharBtn } = charData;

    const langData = buildLanguageSection(state, storageKey);
    const { langSection, langSelect, addLangBtn } = langData;

    // Parameter controls - dynamic parameter selector
    // Build parameter section
    const paramData = buildParameterSection(state, storageKey, getPlainText, setEditorText, getCaretPos, setCaretPos, widget, historyStatus, editor);
    const { paramSection, paramTypeSelect, paramInputWrapper, addParamBtn, createParamInput, getCurrentParamInput, setCurrentParamInput } = paramData;
    let currentParamInput = paramData.getCurrentParamInput();

    // Preset controls
    // Build preset section
    const presetData = buildPresetSection(state, storageKey);
    const { presetSection, presetButtons, presetTitles, updatePresetGlows } = presetData;


    // Validation controls
    // Build validation section
    const validData = buildValidationSection();
    const { validSection, formatBtn, validateBtn } = validData;


    // Assemble sidebar
    sidebar.appendChild(historySection);
    sidebar.appendChild(charSection);
    sidebar.appendChild(langSection);
    sidebar.appendChild(paramSection);
    sidebar.appendChild(presetSection);
    sidebar.appendChild(validSection);

    // ==================== ATTACH EVENT HANDLERS ====================
    // Consolidates all addEventListener calls into a single module function
    attachAllEventHandlers(
        editor, state, widget, storageKey, getPlainText, setEditorText, getCaretPos, setCaretPos,
        undoBtn, redoBtn, historyStatus, charSelect, charInput, addCharBtn, langSelect, addLangBtn,
        paramTypeSelect, paramInputWrapper, addParamBtn, presetButtons, presetTitles, updatePresetGlows,
        formatBtn, validateBtn, fontFamilySelect, fontSizeInput, fontSizeDisplay, setFontSize,
        showNotification, resizeDivider, sidebar, setSidebarWidth, setUIScale
    );

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
