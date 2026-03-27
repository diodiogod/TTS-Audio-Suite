/**
 * 🏷️ Multiline TTS Tag Editor
 * Advanced multiline text editor with TTS tag support, undo/redo, and full persistence
 */

import { app } from "/scripts/app.js";
import { ChangeTracker } from "/scripts/changeTracker.js";
import { EditorState } from "./editor-state.js";
import { TagUtilities } from "./tag-utilities.js";
import { SyntaxHighlighter } from "./syntax-highlighter.js";
import { FontControls } from "./font-controls.js";
import { buildHistorySection, buildCharacterSection, buildLanguageSection, buildValidationSection } from "./widget-ui-builder.js";
import { buildParameterSection } from "./widget-parameter-section.js";
import { buildPresetSection } from "./widget-preset-system.js";
import { attachAllEventHandlers } from "./widget-event-handlers.js";
import { buildTabSystem } from "./widget-tabs.js";
import { buildInlineEditSection } from "./widget-inline-edit-section.js";


// Counter to ensure unique storage keys even when node.id is -1
let widgetCounter = 0;

const CHANGE_TRACKER_PATCH_FLAG = "__ttsTagEditorUndoRedoPatched";
const TAG_EDITOR_STYLESHEET_ID = "tts-tag-editor-styles";

const ensureTagEditorStylesheet = () => {
    if (document.getElementById(TAG_EDITOR_STYLESHEET_ID)) {
        return;
    }

    const link = document.createElement("link");
    link.id = TAG_EDITOR_STYLESHEET_ID;
    link.rel = "stylesheet";
    link.href = new URL("./string_multiline_tag_editor.css", import.meta.url).href;
    document.head.appendChild(link);
};

const isTagEditorFocused = () => {
    const activeElement = document.activeElement;
    return activeElement instanceof HTMLElement &&
        activeElement.classList.contains("comfy-multiline-input") &&
        !!activeElement.closest(".string-multiline-tag-editor-main");
};

const patchChangeTrackerUndoRedo = () => {
    const prototype = ChangeTracker?.prototype;
    if (!prototype || prototype[CHANGE_TRACKER_PATCH_FLAG] || typeof prototype.undoRedo !== "function") {
        return;
    }

    const originalUndoRedo = prototype.undoRedo;
    prototype.undoRedo = async function (event) {
        if (isTagEditorFocused()) {
            return false;
        }

        return await originalUndoRedo.call(this, event);
    };

    Object.defineProperty(prototype, CHANGE_TRACKER_PATCH_FLAG, {
        value: true,
        configurable: false,
        enumerable: false,
        writable: false
    });
};

patchChangeTrackerUndoRedo();
ensureTagEditorStylesheet();

// Create the widget
function addStringMultilineTagEditorWidget(node) {
    // Use widget counter as fallback when node.id is -1 (not yet assigned)
    const uniqueId = node.id !== -1 ? node.id : `widget_${widgetCounter++}`;
    const storageKey = `string_multiline_tag_editor_${uniqueId}`;

    // Don't load from localStorage yet - wait for workflow value first
    // We'll load localStorage in setValue if needed
    let state = new EditorState(); // Start with fresh state
    let isConfigured = false; // Flag to know if onConfigure already loaded the state
    let hasSetInitialValue = false; // Track if we've set the initial workflow value
    let onConfigureCallCount = 0; // Track how many times onConfigure has been called

    // Create a temporary state to check default text
    const defaultState = new EditorState();
    const defaultText = defaultState.text;

    // Create main editor container (this will be THE widget)
    const editorContainer = document.createElement("div");
    editorContainer.className = "string-multiline-tag-editor-main";
    editorContainer.style.display = "flex";
    editorContainer.style.gap = "0";
    editorContainer.style.width = "100%";
    editorContainer.style.height = "100%";
    editorContainer.style.overflow = "hidden";
    editorContainer.style.flexDirection = "column";
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

    const topBar = document.createElement("div");
    topBar.className = "string-multiline-tag-editor-topbar";

    const topBarNav = document.createElement("div");
    topBarNav.className = "string-multiline-tag-editor-nav";
    topBarNav.innerHTML = `
        <span class="is-active">Editor</span>
        <span>History</span>
        <span>Presets</span>
        <span>Library</span>
    `;

    topBar.appendChild(topBarNav);

    const shellBody = document.createElement("div");
    shellBody.className = "string-multiline-tag-editor-shell";

    // Create sidebar with resizable width and UI scaling
    const sidebar = document.createElement("div");
    sidebar.className = "string-multiline-tag-editor-sidebar";
    sidebar.style.width = state.sidebarWidth + "px";
    sidebar.style.minWidth = "150px";
    sidebar.style.maxWidth = "400px";
    sidebar.style.height = "100%";
    sidebar.style.background = "#222";
    sidebar.style.borderRight = "1px solid #444";
    sidebar.style.padding = "0";
    sidebar.style.overflow = "hidden";
    sidebar.style.fontSize = (11 * state.uiScale) + "px";
    sidebar.style.flexShrink = "0";
    sidebar.style.display = "flex";
    sidebar.style.flexDirection = "column";
    sidebar.style.position = "relative";

    const sidebarScrollContent = document.createElement("div");
    sidebarScrollContent.className = "string-multiline-tag-editor-sidebar-scroll-content";
    sidebarScrollContent.style.flex = "1 1 auto";
    sidebarScrollContent.style.minHeight = "0";
    sidebarScrollContent.style.padding = "10px";
    sidebarScrollContent.style.overflowY = "auto";
    sidebarScrollContent.style.overflowX = "hidden";
    sidebarScrollContent.style.display = "flex";
    sidebarScrollContent.style.flexDirection = "column";

    const sidebarScrollbar = document.createElement("div");
    sidebarScrollbar.className = "string-multiline-tag-editor-sidebar-scrollbar";

    const sidebarScrollbarThumb = document.createElement("div");
    sidebarScrollbarThumb.className = "string-multiline-tag-editor-sidebar-scrollbar-thumb";
    sidebarScrollbar.appendChild(sidebarScrollbarThumb);

    const updateSidebarScrollbar = () => {
        const visibleHeight = sidebarScrollContent.clientHeight;
        const scrollHeight = sidebarScrollContent.scrollHeight;
        const maxScrollTop = Math.max(0, scrollHeight - visibleHeight);
        const hasOverflow = maxScrollTop > 1;
        const topOffset = sidebarScrollContent.offsetTop + 6;
        const bottomOffset = Math.max(6, sidebar.clientHeight - (sidebarScrollContent.offsetTop + visibleHeight) + 6);

        sidebarScrollbar.style.top = `${topOffset}px`;
        sidebarScrollbar.style.bottom = `${bottomOffset}px`;

        sidebarScrollbar.style.opacity = hasOverflow ? "" : "0";
        sidebarScrollbar.style.pointerEvents = hasOverflow ? "auto" : "none";

        if (!hasOverflow) {
            sidebarScrollbarThumb.style.transform = "translateY(0)";
            sidebarScrollbarThumb.style.height = "0";
            return;
        }

        const trackHeight = visibleHeight - 8;
        const thumbHeight = Math.max(20, (visibleHeight / scrollHeight) * trackHeight);
        const availableTravel = Math.max(0, trackHeight - thumbHeight);
        const progress = maxScrollTop > 0 ? sidebarScrollContent.scrollTop / maxScrollTop : 0;
        const thumbOffset = progress * availableTravel;

        sidebarScrollbarThumb.style.height = `${thumbHeight}px`;
        sidebarScrollbarThumb.style.transform = `translateY(${thumbOffset}px)`;
    };

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
        updateSidebarScrollbar();
    };

    // Function to update UI scale
    const setUIScale = (factor) => {
        factor = Math.max(0.7, Math.min(1.5, factor)); // Clamp between 0.7 and 1.5
        state.uiScale = factor;
        sidebar.style.fontSize = (11 * factor) + "px";

        // Update all button and input sizes
        const buttons = sidebarScrollContent.querySelectorAll("button, input[type='text'], input[type='number'], select");
        buttons.forEach(btn => {
            const baseFontSize = 10;
            btn.style.fontSize = (baseFontSize * factor) + "px";
            btn.style.padding = (4 * factor) + "px " + (6 * factor) + "px";
        });

        state.saveToLocalStorage(storageKey);
        updateSidebarScrollbar();
    };

    // Create editor wrapper for contenteditable
    const textareaWrapper = document.createElement("div");
    textareaWrapper.className = "string-multiline-tag-editor-workspace";
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

    const editorStatusBar = document.createElement("div");
    editorStatusBar.className = "string-multiline-tag-editor-statusbar";

    const editorStatusChips = document.createElement("div");
    editorStatusChips.className = "string-multiline-tag-editor-chips";

    const charactersChip = document.createElement("span");
    charactersChip.className = "string-multiline-tag-editor-chip is-secondary";
    const inlineEditsChip = document.createElement("span");
    inlineEditsChip.className = "string-multiline-tag-editor-chip is-tertiary";
    editorStatusChips.appendChild(charactersChip);
    editorStatusChips.appendChild(inlineEditsChip);

    const editorStatusStats = document.createElement("div");
    editorStatusStats.className = "string-multiline-tag-editor-stats";

    editorStatusBar.appendChild(editorStatusChips);
    editorStatusBar.appendChild(editorStatusStats);

    const editorSurface = document.createElement("div");
    editorSurface.className = "string-multiline-tag-editor-surface";

    const lineGutter = document.createElement("div");
    lineGutter.className = "string-multiline-tag-editor-gutter";

    const lineGutterContent = document.createElement("div");
    lineGutterContent.className = "string-multiline-tag-editor-gutter-content";
    lineGutter.appendChild(lineGutterContent);

    const updateEditorMetrics = () => {
        const plainText = getPlainText();
        const characterTags = (plainText.match(/\[[^\]|]+(?:\|[^\]]+)?\]/g) || [])
            .map(tag => tag.slice(1, -1).split("|")[0])
            .filter(firstPart => firstPart && !firstPart.includes(":"));
        const uniqueCharacters = new Set(characterTags);
        const inlineEditCount = (plainText.match(/<[^<>\r\n]+>/g) || []).length;
        const wordCount = plainText.trim() ? plainText.trim().split(/\s+/).length : 0;
        const lineCount = plainText === "" ? 1 : plainText.split("\n").length;

        charactersChip.textContent = `Characters: ${uniqueCharacters.size}`;
        inlineEditsChip.textContent = `Inline Tags: ${inlineEditCount}`;
        const computedEditorStyle = window.getComputedStyle(editor);
        const lineHeight = parseFloat(computedEditorStyle.lineHeight) || (state.fontSize * 1.4);
        const visualRowCount = Math.max(lineCount, Math.ceil(editor.scrollHeight / lineHeight));
        editorStatusStats.textContent = `${lineCount} lines | ${wordCount} words | ${plainText.length} chars`;
        lineGutterContent.style.fontSize = `${state.fontSize}px`;
        lineGutterContent.style.lineHeight = `${lineHeight}px`;
        const gutterDigits = String(visualRowCount).length;
        lineGutter.style.flexBasis = `${Math.max(24, Math.ceil(gutterDigits * state.fontSize * 0.72) + 14)}px`;
        lineGutter.style.minWidth = lineGutter.style.flexBasis;

        const gutterFragment = document.createDocumentFragment();
        for (let row = 0; row < visualRowCount; row++) {
            const lineNumber = document.createElement("span");
            lineNumber.textContent = String(row + 1);
            gutterFragment.appendChild(lineNumber);
        }

        lineGutterContent.replaceChildren(gutterFragment);
    };

    // Function to update font size and persist it
    const setFontSize = (newSize) => {
        newSize = Math.max(2, Math.min(120, newSize)); // Clamp between 2px and 120px
        state.fontSize = newSize;
        editor.style.fontSize = newSize + "px";
        state.saveToLocalStorage(storageKey);
        updateEditorMetrics();
    };

    const setFontFamily = (newFamily) => {
        editor.style.fontFamily = newFamily;
        state.fontFamily = newFamily;
        state.saveToLocalStorage(storageKey);
        updateEditorMetrics();
    };

    // Initialize with text
    editor.textContent = state.text;

    const INTERNAL_MARKER_PATTERN = /(?:\x00)?(?:NUM_START|NUM_END|SRT_START|SRT_END|TAG_START|TAG_END|EDIT_START|EDIT_END|COMMA_START|COMMA_END|PERIOD_START|PERIOD_END|PUNCT_START|PUNCT_END|SPACE_START|SPACE_END)(?:\x00)?/g;

    const stripInternalMarkers = (text) => text.replace(INTERNAL_MARKER_PATTERN, "");

    const selectionIsInsideEditor = (selection) => {
        if (!selection || selection.rangeCount === 0) {
            return false;
        }

        const range = selection.getRangeAt(0);
        return editor.contains(range.commonAncestorContainer);
    };

    const getNodePlainText = (node) => {
        if (!node) {
            return "";
        }

        if (node.nodeType === Node.TEXT_NODE) {
            return node.textContent || "";
        }

        if (node.nodeName === "BR") {
            return "\n";
        }

        let text = "";
        node.childNodes.forEach(child => {
            text += getNodePlainText(child);
        });
        return text;
    };

    // Helper to get plain text (strip HTML for state management)
    const getPlainText = () => {
        return stripInternalMarkers(getNodePlainText(editor));
    };

    // Save caret position before update
    const getCaretPos = () => {
        const selection = window.getSelection();
        if (!selectionIsInsideEditor(selection)) {
            return state.lastCursorPosition || 0;
        }

        const range = selection.getRangeAt(0);
        const preRange = range.cloneRange();
        preRange.selectNodeContents(editor);
        preRange.setEnd(range.endContainer, range.endOffset);

        const fragment = preRange.cloneContents();
        const caretPos = stripInternalMarkers(getNodePlainText(fragment)).length;
        state.lastCursorPosition = caretPos;
        return caretPos;
    };

    // Restore caret position after update
    const setCaretPos = (pos) => {
        const selection = window.getSelection();
        const range = document.createRange();
        const plainTextLength = getPlainText().length;
        const targetPos = Math.max(0, Math.min(pos, plainTextLength));
        let charCount = 0;
        let nodeStack = [editor];
        let node;
        let foundStart = false;

        while (!foundStart && (node = nodeStack.pop())) {
            if (node.nodeType === Node.TEXT_NODE) {
                const nextCharCount = charCount + node.length;
                if (targetPos <= nextCharCount) {
                    range.setStart(node, targetPos - charCount);
                    foundStart = true;
                }
                charCount = nextCharCount;
            } else if (node.nodeName === "BR") {
                const nextCharCount = charCount + 1;
                if (targetPos <= nextCharCount) {
                    const parentNode = node.parentNode;
                    const nodeIndex = Array.prototype.indexOf.call(parentNode.childNodes, node);
                    range.setStart(parentNode, nodeIndex + 1);
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

        if (!foundStart) {
            range.selectNodeContents(editor);
            range.collapse(false);
            foundStart = true;
        }

        if (foundStart) {
            range.collapse(true);
            selection.removeAllRanges();
            selection.addRange(range);
            state.lastCursorPosition = targetPos;
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

        // Highlight character tags (square brackets) - bright cyan
        html = html.replace(
            /(\[[^\]]+\])/g,
            '\x00TAG_START\x00$1\x00TAG_END\x00'
        );

        // Highlight inline edit tags (angle brackets) - magenta
        html = html.replace(
            /(<[^<>\r\n]+>)/g,
            '\x00EDIT_START\x00$1\x00EDIT_END\x00'
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
            .replace(/\x00EDIT_START\x00(.*?)\x00EDIT_END\x00/g, '<span style="color: #ff66ff; font-weight: bold;">$1</span>')
            .replace(/\x00COMMA_START\x00(.*?)\x00COMMA_END\x00/g, '<span style="color: #66ff66; font-weight: bold;">$1</span>')
            .replace(/\x00PERIOD_START\x00(.*?)\x00PERIOD_END\x00/g, '<span style="color: #ffcc33; font-weight: bold;">$1</span>')
            .replace(/\x00PUNCT_START\x00(.*?)\x00PUNCT_END\x00/g, '<span style="color: #ff9999;">$1</span>')
            .replace(/\x00SPACE_START\x00(.*?)\x00SPACE_END\x00/g, '<span style="background: #2a2a2a; color: #eee;">$1</span>');

        // Update only if changed to avoid flicker
        if (editor.innerHTML !== html) {
            editor.innerHTML = html;
            setCaretPos(caretPos);
        }
        updateEditorMetrics();
    };

    // Update on input
    editor.addEventListener("input", () => {
        updateHighlights();
    });

    // Initial highlight
    updateHighlights();

    // Create font selector floating box (above editor)
    const fontBox = document.createElement("div");
    fontBox.className = "string-multiline-tag-editor-toolbar";
    fontBox.style.padding = "8px 10px";
    fontBox.style.display = "flex";
    fontBox.style.gap = "12px";
    fontBox.style.alignItems = "center";
    fontBox.style.flexShrink = "0";

    // Font family dropdown
    const fontFamilyLabel = document.createElement("div");
    fontFamilyLabel.textContent = "Font";
    fontFamilyLabel.className = "string-multiline-tag-editor-toolbar-label";
    fontFamilyLabel.style.minWidth = "35px";

    const fontFamilySelect = document.createElement("select");
    fontFamilySelect.style.padding = "4px 6px";
    fontFamilySelect.style.fontSize = "10px";
    fontFamilySelect.className = "string-multiline-tag-editor-toolbar-select";
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
    fontSizeLabel.textContent = "Size";
    fontSizeLabel.className = "string-multiline-tag-editor-toolbar-label";
    fontSizeLabel.style.minWidth = "35px";

    const fontSizeInput = document.createElement("input");
    fontSizeInput.type = "number";
    fontSizeInput.min = "2";
    fontSizeInput.max = "120";
    fontSizeInput.value = state.fontSize;
    fontSizeInput.style.padding = "4px 6px";
    fontSizeInput.style.fontSize = "10px";
    fontSizeInput.className = "string-multiline-tag-editor-toolbar-input";
    fontSizeInput.style.width = "50px";

    // Assemble font box
    fontBox.appendChild(fontFamilyLabel);
    fontBox.appendChild(fontFamilySelect);
    fontBox.appendChild(fontSizeLabel);
    fontBox.appendChild(fontSizeInput);

    // Set initial font family selection
    fontFamilySelect.value = state.fontFamily;

    const topActions = document.createElement("div");
    topActions.className = "string-multiline-tag-editor-top-actions";

    // Create floating invisible divider on top of everything for resizing
    const resizeDivider = document.createElement("div");
    resizeDivider.style.position = "absolute";
    resizeDivider.style.top = "56px";
    resizeDivider.style.width = "6px"; // Invisible grabable area (3px left, 3px right of border)
    resizeDivider.style.bottom = "0";
    resizeDivider.style.height = "auto";
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

    editorSurface.appendChild(editor);
    editorSurface.prepend(lineGutter);
    textareaWrapper.appendChild(editorStatusBar);
    textareaWrapper.appendChild(editorSurface);
    shellBody.appendChild(sidebar);
    shellBody.appendChild(textareaWrapper);
    editorContainer.appendChild(topBar);
    editorContainer.appendChild(shellBody);

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
            // Skip loading if we've already processed the workflow value via onConfigure
            if (isConfigured && hasSetInitialValue) {
                return;
            }

            // Load localStorage for this specific node only when setValue is called
            // This ensures we don't mix up data between different node instances
            if (state.text === defaultText && !hasSetInitialValue) {
                // State hasn't been customized yet - try loading from localStorage
                const savedState = EditorState.loadFromLocalStorage(storageKey);
                if (savedState.text && savedState.text !== defaultText) {
                    Object.assign(state, savedState);
                }
            }

            // Priority order:
            // 1. If localStorage has custom data (not default) → use it (persistent edits)
            // 2. If workflow has custom data (not default) → use it (shared workflow)
            // 3. Otherwise use workflow value or default
            if (state.text && state.text !== defaultText && !hasSetInitialValue) {
                // localStorage has custom data - keep it (same session, persistent edits)
                setEditorText(state.text);
                hasSetInitialValue = true;
            } else if (v && v !== defaultText && !hasSetInitialValue) {
                // Workflow has custom data - use it (first load of shared workflow)
                setEditorText(v);
                state.text = v;
                hasSetInitialValue = true;
            } else if (!hasSetInitialValue) {
                // Both are default or empty - use workflow value or default
                const textToUse = v || defaultText;
                setEditorText(textToUse);
                state.text = textToUse;
                hasSetInitialValue = true;
            }
        }
    });

    widget.inputEl = editor;
    widget.options.minNodeSize = [900, 600];
    widget.options.maxWidth = 1400;

    // Ensure widget value is properly loaded from workflow
    // ComfyUI loads widget values, but we need to handle them properly
    const originalSetValue = widget.setValue.bind(widget);
    widget.setValue = function(v) {
        originalSetValue(v);
        if (v !== undefined) {
            widget.value = v;
        }
    };

    // Hook into node's onConfigure to load workflow values
    // This is called after node creation with the workflow data
    const originalOnConfigure = node.onConfigure?.bind(node) || (() => {});
    node.onConfigure = function(info) {
        onConfigureCallCount++;
        const result = originalOnConfigure(info);

        // Load workflow value when onConfigure is called (user opened a file)
        if (info && Array.isArray(info.widgets_values) && info.widgets_values[0]) {
            const workflowValue = info.widgets_values[0];

            // Try to load full state from localStorage first (includes history)
            const savedState = EditorState.loadFromLocalStorage(storageKey);
            if (savedState && savedState.text && savedState.history && savedState.history.length > 0) {
                // We have localStorage with history - this means we had local edits
                Object.assign(state, savedState);

                // Use onConfigureCallCount to distinguish reload from new workflow
                // First call = initial load, preserve localStorage (it's either reload or user edits)
                // Subsequent calls = file opened or workflow reloaded, reset if text changed
                if (onConfigureCallCount === 1) {
                    // First time seeing this node - keep the loaded history
                    // Just make sure text matches workflow so they're in sync
                    state.text = workflowValue;
                } else {
                    // Subsequent calls - check if workflow changed
                    if (state.lastWorkflowValue && workflowValue !== state.lastWorkflowValue) {
                        // Workflow value changed - user opened a different file
                        state.text = workflowValue;
                        state.history = [{text: workflowValue, caretPos: 0}];
                        state.historyIndex = 0;
                    } else {
                        // Workflow unchanged or first time tracking it - preserve history (page reload)
                        state.text = workflowValue;
                    }
                }

                state.lastWorkflowValue = workflowValue;
                setEditorText(state.text);
            } else {
                // No localStorage or no history - just use workflow value
                setEditorText(workflowValue);
                state.text = workflowValue;
                state.history = [{text: workflowValue, caretPos: 0}];
                state.historyIndex = 0;
                state.lastWorkflowValue = workflowValue;
            }

            widget.value = workflowValue;
            historyStatus.textContent = state.getHistoryStatus();
            isConfigured = true; // Mark that we've configured from workflow
            hasSetInitialValue = true; // Mark that we've set the initial value from workflow
        }

        return result;
    };

    // Set initial node size on creation
    setTimeout(() => {
        node.setSize([900, 600]);
    }, 0);

    // ==================== SIDEBAR CONTROLS ====================

    // Build sidebar sections using extracted modules
    const historyData = buildHistorySection(state, storageKey);
    const { historySection, undoBtn, redoBtn, historyStatus } = historyData;
    historySection.classList.add("string-multiline-tag-editor-history");

    const charData = buildCharacterSection(state, storageKey);
    const { charSection, charSelect, charInput, addCharBtn } = charData;
    charSection.classList.add("string-multiline-tag-editor-panel-section");

    const langData = buildLanguageSection(state, storageKey);
    const { langSection, langSelect, addLangBtn } = langData;
    langSection.classList.add("string-multiline-tag-editor-panel-section");

    // Parameter controls - dynamic parameter selector
    // Build parameter section
    const paramData = buildParameterSection(state, storageKey, getPlainText, setEditorText, getCaretPos, setCaretPos, widget, historyStatus, editor);
    const { paramSection, paramTypeSelect, paramInputWrapper, addParamBtn, createParamInput, getCurrentParamInput, setCurrentParamInput } = paramData;
    let currentParamInput = paramData.getCurrentParamInput();
    paramSection.classList.add("string-multiline-tag-editor-panel-section");

    // Preset controls
    // Build preset section
    const presetData = buildPresetSection(state, storageKey);
    const { presetSection, presetButtons, presetTitles, updatePresetGlows } = presetData;
    presetSection.classList.add("string-multiline-tag-editor-panel-section");

    // Validation controls
    // Build validation section
    const validData = buildValidationSection();
    const { validSection, formatBtn, validateBtn } = validData;
    validSection.classList.add("string-multiline-tag-editor-validation");
    formatBtn.classList.add("string-multiline-tag-editor-footer-btn");
    validateBtn.classList.add("string-multiline-tag-editor-footer-btn", "is-primary");

    // Build inline edit section
    const inlineEditData = buildInlineEditSection(state, storageKey);
    const {
        inlineEditSection,
        paraSelect, paraIterSlider, addParaBtn,
        emotionSelect, emotionIterSlider, addEmotionBtn,
        styleSelect, styleIterSlider, addStyleBtn,
        speedSelect, speedIterSlider, addSpeedBtn,
        restorePassSlider, restoreRefInput, addRestoreBtn
    } = inlineEditData;
    inlineEditSection.classList.add("string-multiline-tag-editor-inline-section");

    // Build tab system
    const tabData = buildTabSystem(state, storageKey);
    const { tabContainer, charParamContent, inlineEditContent, switchTab } = tabData;
    tabContainer.classList.add("string-multiline-tag-editor-tab-system");
    charParamContent.classList.add("string-multiline-tag-editor-tab-content");
    inlineEditContent.classList.add("string-multiline-tag-editor-tab-content");

    // Assemble Character/Parameters tab
    charParamContent.appendChild(charSection);
    charParamContent.appendChild(langSection);
    charParamContent.appendChild(paramSection);
    charParamContent.appendChild(presetSection);

    // Assemble Inline Edit tab
    inlineEditContent.appendChild(inlineEditSection);

    // Assemble header and shell
    topActions.appendChild(formatBtn);
    topActions.appendChild(validateBtn);
    topBar.appendChild(topActions);
    topBar.appendChild(fontBox);
    topBar.appendChild(historySection);

    const sidebarHeader = document.createElement("div");
    sidebarHeader.className = "string-multiline-tag-editor-sidebar-header";
    const tabHeaderStrip = tabContainer.firstElementChild;
    if (tabHeaderStrip) {
        sidebarHeader.appendChild(tabHeaderStrip);
    }

    // Assemble sidebar
    sidebar.appendChild(sidebarHeader);
    sidebarScrollContent.appendChild(charParamContent);
    sidebarScrollContent.appendChild(inlineEditContent);
    sidebar.appendChild(sidebarScrollContent);
    sidebar.appendChild(sidebarScrollbar);

    // ==================== ATTACH EVENT HANDLERS ====================
    // Consolidates all addEventListener calls into a single module function
    attachAllEventHandlers(
        editor, state, widget, storageKey, getPlainText, setEditorText, getCaretPos, setCaretPos,
        undoBtn, redoBtn, historyStatus, charSelect, charInput, addCharBtn, langSelect, addLangBtn,
        paramTypeSelect, paramInputWrapper, addParamBtn, presetButtons, presetTitles, updatePresetGlows,
        formatBtn, validateBtn, fontFamilySelect, fontSizeInput, null, setFontSize, setFontFamily,
        showNotification, resizeDivider, sidebar, setSidebarWidth, setUIScale,
        // Inline edit controls
        paraSelect, paraIterSlider, addParaBtn,
        emotionSelect, emotionIterSlider, addEmotionBtn,
        styleSelect, styleIterSlider, addStyleBtn,
        speedSelect, speedIterSlider, addSpeedBtn,
        restorePassSlider, restoreRefInput, addRestoreBtn
    );

    // Store state when node is removed
    widget.onRemove = () => {
        state.saveToLocalStorage(storageKey);
    };

    // Initialize history display
    historyStatus.textContent = state.getHistoryStatus();

    editor.addEventListener("scroll", () => {
        lineGutterContent.style.transform = `translateY(${-editor.scrollTop}px)`;
    });
    sidebarScrollContent.addEventListener("scroll", updateSidebarScrollbar);

    if (typeof ResizeObserver !== "undefined") {
        const resizeObserver = new ResizeObserver(() => {
            updateEditorMetrics();
            updateSidebarScrollbar();
        });
        resizeObserver.observe(editorSurface);
        resizeObserver.observe(editor);
        resizeObserver.observe(sidebar);
        resizeObserver.observe(sidebarScrollContent);
    }

    requestAnimationFrame(updateSidebarScrollbar);

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

                // Remove the default widget from the widgets array so it doesn't render
                this.widgets.shift();

                // Create our custom widget (becomes the FIRST widget now)
                addStringMultilineTagEditorWidget(this);
            };
        }
    }
});
