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
import { SRTTimingDragController, buildSRTTimingMarkup } from "./string_multiline_tag_editor_timing_drag.js";
import { SRTCueEditController, buildSRTCueNumberMarkup } from "./string_multiline_tag_editor_srt_cue_ops.js";


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

    const updateRangeFill = (input) => {
        if (!(input instanceof HTMLInputElement) || input.type !== "range") {
            return;
        }

        const min = Number(input.min || 0);
        const max = Number(input.max || 100);
        const value = Number(input.value || min);
        const ratio = max > min ? ((value - min) / (max - min)) * 100 : 0;
        input.style.background = `linear-gradient(90deg, rgba(0, 225, 174, 0.88) 0%, rgba(0, 225, 174, 0.88) ${ratio}%, rgba(53, 53, 52, 0.9) ${ratio}%, rgba(53, 53, 52, 0.9) 100%)`;
    };

    const topBar = document.createElement("div");
    topBar.className = "string-multiline-tag-editor-topbar";

    const topBarNav = document.createElement("div");
    topBarNav.className = "string-multiline-tag-editor-nav";

    const topNavItems = new Map();
    [
        ["editor", "Editor"],
        ["history", "History"],
        ["presets", "Presets"],
        ["library", "Library"]
    ].forEach(([viewKey, label]) => {
        const item = document.createElement("span");
        item.textContent = label;
        item.dataset.view = viewKey;
        if (viewKey === "editor") {
            item.classList.add("is-active");
        }
        topBarNav.appendChild(item);
        topNavItems.set(viewKey, item);
    });

    topBar.appendChild(topBarNav);

    const contentStage = document.createElement("div");
    contentStage.className = "string-multiline-tag-editor-stage";

    const shellBody = document.createElement("div");
    shellBody.className = "string-multiline-tag-editor-shell";

    const auxiliaryView = document.createElement("div");
    auxiliaryView.className = "string-multiline-tag-editor-aux-view";

    const auxiliaryHeader = document.createElement("div");
    auxiliaryHeader.className = "string-multiline-tag-editor-aux-header";

    const auxiliaryTitle = document.createElement("h3");
    auxiliaryTitle.className = "string-multiline-tag-editor-aux-title";

    const auxiliaryDescription = document.createElement("p");
    auxiliaryDescription.className = "string-multiline-tag-editor-aux-description";

    auxiliaryHeader.appendChild(auxiliaryTitle);
    auxiliaryHeader.appendChild(auxiliaryDescription);

    const auxiliaryContent = document.createElement("div");
    auxiliaryContent.className = "string-multiline-tag-editor-aux-content";

    auxiliaryView.appendChild(auxiliaryHeader);
    auxiliaryView.appendChild(auxiliaryContent);

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

    const editorScrollbar = document.createElement("div");
    editorScrollbar.className = "string-multiline-tag-editor-editor-scrollbar";

    const editorScrollbarThumb = document.createElement("div");
    editorScrollbarThumb.className = "string-multiline-tag-editor-editor-scrollbar-thumb";
    editorScrollbar.appendChild(editorScrollbarThumb);
    let editorScrollbarDragState = null;
    let renderedLogicalLineHtmlParts = [""];
    const EDITOR_LOGICAL_LINE_CLASS = "string-multiline-tag-editor-editor-line";

    const isEditorLogicalLineElement = (node) => (
        node instanceof HTMLElement &&
        node.classList.contains(EDITOR_LOGICAL_LINE_CLASS)
    );

    const getRenderedLogicalLineElements = () => (
        Array.from(editor.children).filter((child) => isEditorLogicalLineElement(child))
    );

    const getRenderedLogicalLineHeights = (lineCount, lineHeight) => {
        const renderedLineElements = getRenderedLogicalLineElements();
        if (!renderedLineElements.length) {
            return Array.from({ length: lineCount }, () => lineHeight);
        }

        return Array.from({ length: lineCount }, (_, row) => (
            Math.max(lineHeight, renderedLineElements[row]?.getBoundingClientRect().height || lineHeight)
        ));
    };

    const updateEditorScrollbar = () => {
        const visibleHeight = editor.clientHeight;
        const scrollHeight = editor.scrollHeight;
        const maxScrollTop = Math.max(0, scrollHeight - visibleHeight);
        const hasOverflow = maxScrollTop > 1;

        editorScrollbar.style.opacity = hasOverflow ? "" : "0";
        editorScrollbar.style.pointerEvents = hasOverflow ? "auto" : "none";

        if (!hasOverflow) {
            editorScrollbarThumb.style.transform = "translateY(0)";
            editorScrollbarThumb.style.height = "0";
            return;
        }

        const trackHeight = Math.max(0, editorScrollbar.clientHeight);
        const thumbHeight = Math.max(20, (visibleHeight / scrollHeight) * trackHeight);
        const availableTravel = Math.max(0, trackHeight - thumbHeight);
        const progress = maxScrollTop > 0 ? editor.scrollTop / maxScrollTop : 0;
        const thumbOffset = progress * availableTravel;

        editorScrollbarThumb.style.height = `${thumbHeight}px`;
        editorScrollbarThumb.style.transform = `translateY(${thumbOffset}px)`;
    };

    const stopEditorScrollbarDrag = () => {
        if (!editorScrollbarDragState) {
            return;
        }

        window.removeEventListener("pointermove", handleEditorScrollbarPointerMove, true);
        window.removeEventListener("pointerup", stopEditorScrollbarDrag, true);
        window.removeEventListener("pointercancel", stopEditorScrollbarDrag, true);
        editorSurface.classList.remove("is-dragging-scrollbar");
        editorScrollbarDragState = null;
    };

    const handleEditorScrollbarPointerMove = (event) => {
        if (!editorScrollbarDragState) {
            return;
        }

        event.preventDefault();
        event.stopPropagation();

        const deltaY = event.clientY - editorScrollbarDragState.startY;
        const scrollDelta = editorScrollbarDragState.availableTravel > 0
            ? (deltaY / editorScrollbarDragState.availableTravel) * editorScrollbarDragState.maxScrollTop
            : 0;
        editor.scrollTop = Math.max(0, Math.min(
            editorScrollbarDragState.startScrollTop + scrollDelta,
            editorScrollbarDragState.maxScrollTop
        ));
        updateEditorScrollbar();
    };

    editorScrollbarThumb.addEventListener("pointerdown", (event) => {
        const visibleHeight = editor.clientHeight;
        const scrollHeight = editor.scrollHeight;
        const maxScrollTop = Math.max(0, scrollHeight - visibleHeight);
        if (event.button !== 0 || maxScrollTop <= 0) {
            return;
        }

        const trackHeight = Math.max(0, editorScrollbar.clientHeight);
        const thumbHeight = editorScrollbarThumb.getBoundingClientRect().height;
        const availableTravel = Math.max(0, trackHeight - thumbHeight);

        event.preventDefault();
        event.stopPropagation();

        editorScrollbarDragState = {
            startY: event.clientY,
            startScrollTop: editor.scrollTop,
            maxScrollTop,
            availableTravel
        };

        editorSurface.classList.add("is-dragging-scrollbar");
        window.addEventListener("pointermove", handleEditorScrollbarPointerMove, true);
        window.addEventListener("pointerup", stopEditorScrollbarDrag, true);
        window.addEventListener("pointercancel", stopEditorScrollbarDrag, true);
    });

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
        const paddingTop = parseFloat(computedEditorStyle.paddingTop || "0");
        const paddingBottom = parseFloat(computedEditorStyle.paddingBottom || "0");
        editorStatusStats.textContent = `${lineCount} lines | ${wordCount} words | ${plainText.length} chars`;
        lineGutterContent.style.fontSize = `${state.fontSize}px`;
        lineGutterContent.style.fontFamily = computedEditorStyle.fontFamily;
        lineGutterContent.style.lineHeight = `${lineHeight}px`;
        lineGutterContent.style.paddingTop = `${paddingTop}px`;
        lineGutterContent.style.paddingBottom = `${paddingBottom}px`;
        const gutterDigits = String(lineCount).length;
        lineGutter.style.flexBasis = `${Math.max(24, Math.ceil(gutterDigits * state.fontSize * 0.72) + 14)}px`;
        lineGutter.style.minWidth = lineGutter.style.flexBasis;

        const logicalLineHeights = getRenderedLogicalLineHeights(lineCount, lineHeight);
        const contentHeight = logicalLineHeights.reduce((sum, height) => sum + height, 0);
        lineGutterContent.style.height = `${Math.max(editor.scrollHeight, paddingTop + contentHeight + paddingBottom)}px`;
        const gutterFragment = document.createDocumentFragment();
        for (let row = 0; row < lineCount; row++) {
            const lineNumber = document.createElement("span");
            lineNumber.textContent = String(row + 1);
            lineNumber.style.height = `${logicalLineHeights[row] ?? lineHeight}px`;
            lineNumber.style.lineHeight = `${lineHeight}px`;
            gutterFragment.appendChild(lineNumber);
        }

        lineGutterContent.replaceChildren(gutterFragment);
        updateEditorScrollbar();
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

    const INTERNAL_MARKER_PATTERN = /(?:\x00)?(?:NUM_START(?:_\d+)?|NUM_END|SRT_START|SRT_END|TAG_START|TAG_END|EDIT_START|EDIT_END|COMMA_START|COMMA_END|PERIOD_START|PERIOD_END|PUNCT_START|PUNCT_END|SPACE_START|SPACE_END)(?:\x00)?/g;

    const stripInternalMarkers = (text) => text.replace(INTERNAL_MARKER_PATTERN, "");
    let timingDragController = null;
    let cueEditController = null;

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
            const parentNode = node.parentNode;
            if (isEditorLogicalLineElement(parentNode) && parentNode.childNodes.length === 1) {
                return "";
            }
            return "\n";
        }

        let text = "";
        const childNodes = Array.from(node.childNodes);
        childNodes.forEach((child, index) => {
            text += getNodePlainText(child);
            if (isEditorLogicalLineElement(child) && index < childNodes.length - 1) {
                text += "\n";
            }
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
        let foundStart = false;

        const setCaretPosWithinNode = (rootNode, localTargetPos) => {
            if (localTargetPos <= 0) {
                range.setStart(rootNode, 0);
                return true;
            }

            let charCount = 0;
            let nodeStack = [rootNode];
            let node;

            while ((node = nodeStack.pop())) {
                if (node.nodeType === Node.TEXT_NODE) {
                    const nextCharCount = charCount + node.length;
                    if (localTargetPos <= nextCharCount) {
                        range.setStart(node, localTargetPos - charCount);
                        return true;
                    }
                    charCount = nextCharCount;
                } else if (node.nodeName === "BR") {
                    const parentNode = node.parentNode;
                    const isPlaceholderBreak = isEditorLogicalLineElement(parentNode) && parentNode.childNodes.length === 1;
                    if (isPlaceholderBreak) {
                        continue;
                    }

                    const nextCharCount = charCount + 1;
                    if (localTargetPos <= nextCharCount) {
                        range.setStartAfter(node);
                        return true;
                    }
                    charCount = nextCharCount;
                } else {
                    let i = node.childNodes.length;
                    while (i--) {
                        nodeStack.push(node.childNodes[i]);
                    }
                }
            }

            range.selectNodeContents(rootNode);
            range.collapse(false);
            return true;
        };

        const renderedLogicalLineElements = getRenderedLogicalLineElements();
        if (renderedLogicalLineElements.length) {
            let remainingPos = targetPos;
            for (let lineIndex = 0; lineIndex < renderedLogicalLineElements.length; lineIndex++) {
                const lineElement = renderedLogicalLineElements[lineIndex];
                const lineTextLength = stripInternalMarkers(getNodePlainText(lineElement)).length;

                if (remainingPos <= lineTextLength) {
                    foundStart = setCaretPosWithinNode(lineElement, remainingPos);
                    break;
                }

                remainingPos -= lineTextLength;

                if (lineIndex < renderedLogicalLineElements.length - 1) {
                    if (remainingPos === 1) {
                        foundStart = setCaretPosWithinNode(renderedLogicalLineElements[lineIndex + 1], 0);
                        break;
                    }
                    remainingPos -= 1;
                }
            }
        }

        let charCount = 0;
        let nodeStack = [editor];
        let node;
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
                    range.setStartAfter(node);
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
        let cueNumberIndex = 0;
        let timingHandleIndex = 0;

        // Highlight SRT sequence numbers - bright red
        html = html.replace(
            /^(\d+)\s*\n(\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3})/gm,
            (_, cueNumber, timingLine) => `\x00NUM_START_${cueNumberIndex++}\x00${cueNumber}\x00NUM_END\x00\n${timingLine}`
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
            .replace(/\x00NUM_START_(\d+)\x00(.*?)\x00NUM_END\x00/g, (_, cueIndex, cueNumber) => buildSRTCueNumberMarkup(cueNumber, Number(cueIndex)))
            .replace(/\x00SRT_START\x00(.*?)\x00SRT_END\x00/g, (_, timingText) => buildSRTTimingMarkup(stripInternalMarkers(timingText).replace(/&gt;/g, ">"), timingHandleIndex++))
            .replace(/\x00TAG_START\x00(.*?)\x00TAG_END\x00/g, '<span style="color: #38d7ae; font-weight: 700;">$1</span>')
            .replace(/\x00EDIT_START\x00(.*?)\x00EDIT_END\x00/g, '<span style="color: #a6d700; font-weight: 700;">$1</span>')
            .replace(/\x00COMMA_START\x00(.*?)\x00COMMA_END\x00/g, '<span style="color: #7bd6a7; font-weight: bold;">$1</span>')
            .replace(/\x00PERIOD_START\x00(.*?)\x00PERIOD_END\x00/g, '<span style="color: #e3be69; font-weight: bold;">$1</span>')
            .replace(/\x00PUNCT_START\x00(.*?)\x00PUNCT_END\x00/g, '<span style="color: #f0a1a1;">$1</span>')
            .replace(/\x00SPACE_START\x00(.*?)\x00SPACE_END\x00/g, '<span style="background: #2a2a2a; color: #eee;">$1</span>');

        renderedLogicalLineHtmlParts = html.split("\n");
        html = renderedLogicalLineHtmlParts.map((lineHtml, index) => (
            `<div class="${EDITOR_LOGICAL_LINE_CLASS}" data-line-index="${index}">${lineHtml && lineHtml.length ? lineHtml : "<br>"}</div>`
        )).join("");

        // Update only if changed to avoid flicker
        if (editor.innerHTML !== html) {
            editor.innerHTML = html;
            setCaretPos(caretPos);
        }
        timingDragController?.syncActiveHandle();
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
    editorSurface.appendChild(editorScrollbar);
    textareaWrapper.appendChild(editorStatusBar);
    textareaWrapper.appendChild(editorSurface);
    shellBody.appendChild(sidebar);
    shellBody.appendChild(textareaWrapper);
    contentStage.appendChild(shellBody);
    contentStage.appendChild(auxiliaryView);
    editorContainer.appendChild(topBar);
    editorContainer.appendChild(contentStage);

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
    timingDragController = new SRTTimingDragController({
        rootElement: editorContainer,
        editor,
        getPlainText,
        setEditorText,
        getCaretPos,
        setCaretPos,
        state,
        storageKey,
        widget,
        historyStatus,
        showNotification
    });

    cueEditController = new SRTCueEditController({
        editor,
        getPlainText,
        setEditorText,
        getCaretPos,
        setCaretPos,
        state,
        storageKey,
        widget,
        historyStatus,
        showNotification
    });

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

    const summarizeText = (text, maxLength = 180) => {
        const compact = (text || "").replace(/\s+/g, " ").trim();
        if (!compact) {
            return "Empty";
        }
        return compact.length > maxLength ? `${compact.slice(0, maxLength)}...` : compact;
    };

    const renderAuxiliaryEmptyState = (title, description) => {
        const emptyState = document.createElement("div");
        emptyState.className = "string-multiline-tag-editor-empty-state";

        const emptyTitle = document.createElement("strong");
        emptyTitle.textContent = title;

        const emptyDescription = document.createElement("span");
        emptyDescription.textContent = description;

        emptyState.appendChild(emptyTitle);
        emptyState.appendChild(emptyDescription);
        auxiliaryContent.replaceChildren(emptyState);
    };

    const restoreHistoryEntryAt = (entryIndex) => {
        const entry = state.history[entryIndex];
        if (!entry) {
            return;
        }

        state.historyIndex = entryIndex;
        setEditorText(entry.text);
        state.saveToLocalStorage(storageKey);
        widget.callback?.(widget.value);
        historyStatus.textContent = state.getHistoryStatus();

        setTimeout(() => {
            setCaretPos(entry.caretPos || 0);
            editor.focus();
        }, 0);
    };

    const insertSnippetAtCursor = (snippet, label = "Snippet") => {
        const text = getPlainText();
        const currentCaretPos = getCaretPos();
        const before = text.substring(0, currentCaretPos);
        const after = text.substring(currentCaretPos);
        const prefix = before && !/[\s\n]$/.test(before) ? "\n" : "";
        const suffix = after && !/^[\s\n]/.test(after) ? "\n" : "";
        const newText = `${before}${prefix}${snippet}${suffix}${after}`;
        const newCaretPos = before.length + prefix.length + snippet.length;

        setEditorText(newText);
        setTimeout(() => {
            setCaretPos(newCaretPos);
            state.addToHistory(newText, newCaretPos);
            state.saveToLocalStorage(storageKey);
            editor.focus();
        }, 0);
        widget.callback?.(widget.value);
        historyStatus.textContent = state.getHistoryStatus();
        showNotification(`✅ ${label} inserted`);
    };

    const renderHistoryView = () => {
        auxiliaryTitle.textContent = "History";
        auxiliaryDescription.textContent = "Restore previous snapshots from the editor timeline.";

        const historyLayout = document.createElement("div");
        historyLayout.className = "string-multiline-tag-editor-aux-list";

        const historyToolbar = document.createElement("div");
        historyToolbar.className = "string-multiline-tag-editor-aux-toolbar";

        const clearHistoryBtn = document.createElement("button");
        clearHistoryBtn.className = "string-multiline-tag-editor-aux-btn";
        clearHistoryBtn.textContent = "Clear History";
        clearHistoryBtn.addEventListener("click", () => {
            const currentText = getPlainText();
            const currentCaretPos = getCaretPos();
            state.history = [{ text: currentText, caretPos: currentCaretPos }];
            state.historyIndex = 0;
            state.saveToLocalStorage(storageKey);
            historyStatus.textContent = state.getHistoryStatus();
            showNotification("✅ History cleared");
            renderHistoryView();
        });

        historyToolbar.appendChild(clearHistoryBtn);
        historyLayout.appendChild(historyToolbar);

        if (!state.history.length) {
            const emptyState = document.createElement("div");
            emptyState.className = "string-multiline-tag-editor-empty-state";
            const emptyTitle = document.createElement("strong");
            emptyTitle.textContent = "No history yet";
            const emptyDescription = document.createElement("span");
            emptyDescription.textContent = "Start editing text and snapshots will show up here.";
            emptyState.appendChild(emptyTitle);
            emptyState.appendChild(emptyDescription);
            historyLayout.appendChild(emptyState);
            auxiliaryContent.replaceChildren(historyLayout);
            return;
        }

        const historyList = document.createElement("div");
        historyList.className = "string-multiline-tag-editor-aux-list";

        for (let index = state.history.length - 1; index >= 0; index--) {
            const entry = state.history[index];
            const card = document.createElement("div");
            card.className = "string-multiline-tag-editor-aux-card";
            if (index === state.historyIndex) {
                card.classList.add("is-active");
            }

            const cardTop = document.createElement("div");
            cardTop.className = "string-multiline-tag-editor-aux-card-top";

            const title = document.createElement("div");
            title.className = "string-multiline-tag-editor-aux-card-title";
            title.textContent = `Step ${index + 1}`;

            const meta = document.createElement("div");
            meta.className = "string-multiline-tag-editor-aux-card-meta";
            meta.textContent = index === state.historyIndex ? "Current" : `${entry.text.length} chars`;

            cardTop.appendChild(title);
            cardTop.appendChild(meta);

            const preview = document.createElement("pre");
            preview.className = "string-multiline-tag-editor-aux-preview";
            preview.textContent = summarizeText(entry.text, 240);

            const actions = document.createElement("div");
            actions.className = "string-multiline-tag-editor-aux-actions";

            const restoreBtn = document.createElement("button");
            restoreBtn.className = "string-multiline-tag-editor-aux-btn is-primary";
            restoreBtn.textContent = "Restore";
            restoreBtn.addEventListener("click", () => {
                restoreHistoryEntryAt(index);
                activateTopView("editor");
            });

            actions.appendChild(restoreBtn);
            card.appendChild(cardTop);
            card.appendChild(preview);
            card.appendChild(actions);
            historyList.appendChild(card);
        }

        historyLayout.appendChild(historyList);
        auxiliaryContent.replaceChildren(historyLayout);
    };

    const renderPresetsView = () => {
        auxiliaryTitle.textContent = "Presets";
        auxiliaryDescription.textContent = "Manage reusable quick slots for tags, characters, and selected snippets.";

        const presetGrid = document.createElement("div");
        presetGrid.className = "string-multiline-tag-editor-aux-grid";

        for (let i = 1; i <= 3; i++) {
            const presetKey = `preset_${i}`;
            const preset = state.presets[presetKey];
            const card = document.createElement("div");
            card.className = "string-multiline-tag-editor-aux-card";

            const cardTop = document.createElement("div");
            cardTop.className = "string-multiline-tag-editor-aux-card-top";

            const title = document.createElement("div");
            title.className = "string-multiline-tag-editor-aux-card-title";
            title.textContent = `P${i}`;

            const meta = document.createElement("div");
            meta.className = "string-multiline-tag-editor-aux-card-meta";
            meta.textContent = preset ? (preset.isComplexTag ? "Saved snippet" : "Character preset") : "Empty slot";

            cardTop.appendChild(title);
            cardTop.appendChild(meta);

            const preview = document.createElement("pre");
            preview.className = "string-multiline-tag-editor-aux-preview";
            preview.textContent = preset ? preset.tag : "No preset saved in this slot yet.";

            const note = document.createElement("div");
            note.className = "string-multiline-tag-editor-aux-note";
            if (preset?.parameters?.language) {
                note.textContent = `Language: ${preset.parameters.language.toUpperCase()}`;
            } else if (preset) {
                note.textContent = "Uses the saved tag/snippet exactly as stored.";
            } else {
                note.textContent = "Save current character settings or selected editor text.";
            }

            const actions = document.createElement("div");
            actions.className = "string-multiline-tag-editor-aux-actions";

            const saveBtn = document.createElement("button");
            saveBtn.className = "string-multiline-tag-editor-aux-btn";
            saveBtn.textContent = "Save";
            saveBtn.addEventListener("click", () => {
                presetButtons[presetKey]?.save?.click?.();
                setTimeout(() => renderPresetsView(), 0);
            });

            const loadBtn = document.createElement("button");
            loadBtn.className = "string-multiline-tag-editor-aux-btn is-primary";
            loadBtn.textContent = "Insert";
            loadBtn.addEventListener("click", () => {
                presetButtons[presetKey]?.load?.click?.();
                setTimeout(() => activateTopView("editor"), 0);
            });

            const deleteBtn = document.createElement("button");
            deleteBtn.className = "string-multiline-tag-editor-aux-btn";
            deleteBtn.textContent = "Delete";
            deleteBtn.addEventListener("click", () => {
                presetButtons[presetKey]?.del?.click?.();
                setTimeout(() => renderPresetsView(), 0);
            });

            actions.appendChild(saveBtn);
            actions.appendChild(loadBtn);
            actions.appendChild(deleteBtn);

            card.appendChild(cardTop);
            card.appendChild(preview);
            card.appendChild(note);
            card.appendChild(actions);
            presetGrid.appendChild(card);
        }

        auxiliaryContent.replaceChildren(presetGrid);
    };

    const renderLibraryView = () => {
        auxiliaryTitle.textContent = "Library";
        auxiliaryDescription.textContent = "Consult the tag guides directly in the editor: character switching, per-segment parameters, inline edit workflow notes, and the SRT editing cheat sheet.";

        const libraryGroups = [
            {
                title: "Character Switching Guide",
                intro: "Use square-bracket tags to swap speakers and optionally language without leaving the same text field.",
                rows: [
                    { syntax: "[Alice]", purpose: "Switch active speaker", notes: "Uses the named character for following text until another speaker tag appears." },
                    { syntax: "[en:Alice]", purpose: "Set language and speaker together", notes: "Useful when one segment needs a different language voice or pronunciation context." },
                    { syntax: "[pause:1s]", purpose: "Insert silence", notes: "Duration can use values like 500ms, 1s, 2.5s." }
                ],
                bullets: [
                    "Character names are case-insensitive and unknown characters fall back safely.",
                    "Language-aware switching uses `[lang:character]` and works alongside narrator text.",
                    "Voice files are matched from filenames, not folder names."
                ]
            },
            {
                title: "Parameter Switching Guide",
                intro: "Override generation behavior per segment without changing node-level defaults.",
                rows: [
                    { syntax: "[Alice|seed:42]", purpose: "Fix or vary randomness per segment", notes: "Keeps a stable seed for that section only." },
                    { syntax: "[Alice|temperature:0.7]", purpose: "Adjust expressive variation", notes: "Higher values are looser; lower values are more controlled." },
                    { syntax: "[en:Alice|seed:42|temperature:0.7]", purpose: "Combine language plus multiple overrides", notes: "Stack parameters with `|` after the speaker or language+speaker prefix." }
                ],
                bullets: [
                    "Order is flexible: `[seed:42|Alice]` and `[Alice|seed:42]` are both valid.",
                    "Supported parameters vary by engine, unsupported ones are ignored with warnings.",
                    "Useful aliases include `temp`, `cfg_weight`, `exag`, `topk`, and `topp`."
                ]
            },
            {
                title: "Inline Edit Tags Guide",
                intro: "These tags are for convenience when you want segment-level Step Audio EditX processing without building separate TTS -> Edit chains.",
                rows: [
                    { syntax: "<Laughter> / <Laughter:2>", purpose: "Insert laughter", notes: "Paralinguistic insertion. Position matters because the sound is inserted where the tag appears." },
                    { syntax: "<Breathing>", purpose: "Insert breathing", notes: "Useful for pauses, fatigue, or realism between spoken phrases." },
                    { syntax: "<Sigh>", purpose: "Insert sigh", notes: "Good for resignation, frustration, or relief beats." },
                    { syntax: "<Uhm>", purpose: "Insert hesitation", notes: "Adds an 'uhm' hesitation sound at the tag position." },
                    { syntax: "<Surprise-oh> / <Surprise-ah> / <Surprise-wa>", purpose: "Insert surprise reactions", notes: "Three surprise variants for different expressive tones." },
                    { syntax: "<Confirmation-en>", purpose: "Insert confirmation sound", notes: "Short confirming reaction inserted inline." },
                    { syntax: "<Question-ei>", purpose: "Insert questioning sound", notes: "Useful before or around uncertain dialogue." },
                    { syntax: "<Dissatisfaction-hnn>", purpose: "Insert dissatisfied reaction", notes: "Adds a disapproving or displeased 'hnn' sound." },
                    { syntax: "<emotion:VALUE> / <emotion:VALUE:ITERATIONS>", purpose: "Apply whole-segment emotion", notes: "Available values: happy, sad, angry, excited, calm, fearful, surprised, disgusted, confusion, empathy, embarrass, depressed, coldness, admiration." },
                    { syntax: "<style:VALUE> / <style:VALUE:ITERATIONS>", purpose: "Apply whole-segment style", notes: "Available values include whisper, serious, child, older, pure, sister, sweet, exaggerated, ethereal, warm, comfort, authority, chat, radio, soulful, gentle, story, vivid, program, news, advertising, roar, murmur, shout, deeply, loudly, arrogant, friendly." },
                    { syntax: "<speed:faster> / <speed:slower> / <speed:more_faster> / <speed:more_slower>", purpose: "Adjust whole-segment speed", notes: "Speed tags affect the full segment, not a point insertion." },
                    { syntax: "<restore>", purpose: "Basic voice restoration", notes: "Runs 1 voice-conversion restore pass using the original pre-edit audio as the reference." },
                    { syntax: "<restore:2>", purpose: "Stronger restoration", notes: "Runs 2 restore passes using the original clean pre-edit audio as the reference." },
                    { syntax: "<restore:1@2>", purpose: "Restore from an intermediate edit-step reference", notes: "`N@M` means: run N restore passes using edit-step M as the reference audio, not restore pass M. Example timeline: `<style:whisper:2> <Laughter:3> <restore:1@2>` means whisper creates edit steps 1-2, laughter creates edit steps 3-5, then restore runs last using edit step 2 as reference so it keeps the whisper character but removes later degradation." },
                    { syntax: "<A|B|C> or <A><B><C>", purpose: "Combine multiple inline tags", notes: "Both pipe-separated and separate-tag forms work. Processing order is emotion/style/speed first, paralinguistics second, restore last." }
                ],
                bullets: [
                    "Use inline tags for convenience and selective segment editing. Use the separate Audio Editor node for maximum manual control.",
                    "Processing order is emotion/style/speed first, then paralinguistic insertion, then restore last.",
                    "Position matters for paralinguistic tags like `<Laughter>` and `<Breathing>`, but not for whole-segment tags like emotion, style, speed, and restore.",
                    "`<restore>` and `<restore:N>` use the original clean pre-edit audio as reference. `<restore:N@M>` switches the reference to edit step M, not restore pass M.",
                    "Example: `<style:whisper:2> <Laughter:3> <restore:1@2>` means restore runs after everything else, but it aims back at the audio from whisper step 2 so you keep the whisper feel and drop the later laughter damage.",
                    "If you want stronger laughter or reaction effects, include supporting spoken text too, not just the tag."
                ]
            },
            {
                title: "SRT Editing Guide",
                intro: "Use these subtitle-specific actions when the editor is showing SRT content and you need to retime, merge, or split cues directly inside the node.",
                rows: [
                    { syntax: "Drag start time", purpose: "Retiming cue start", notes: "Drag the left timestamp horizontally to move only the cue start." },
                    { syntax: "Drag end time", purpose: "Retiming cue end", notes: "Drag the right timestamp horizontally to move only the cue end." },
                    { syntax: "Drag -->", purpose: "Move whole cue", notes: "Drag the arrow segment to move the full subtitle without changing its duration." },
                    { syntax: "Shift + drag start/end", purpose: "Keep adjacent gap stable", notes: "Moves the neighboring cue boundary by the same delta so the existing gap remains intact." },
                    { syntax: "Alt + click cue number", purpose: "Merge with next cue", notes: "Keeps the current cue start, uses the next cue end, joins the text, and renumbers later cues." },
                    { syntax: "Alt + Shift + click cue number", purpose: "Merge with previous cue", notes: "Merges the selected cue backward into the previous one and keeps timing across the full combined span." },
                    { syntax: "Ctrl + Shift + Enter", purpose: "Split cue at caret", notes: "Splits subtitle text at the caret and estimates the new boundary time from the text proportion on both sides." }
                ],
                bullets: [
                    "Merge is useful when short subtitle chunks sound too abrupt and you want a larger TTS segment.",
                    "Split uses text proportion plus punctuation bias as a first-pass timing estimate, so review the result if the pacing is critical.",
                    "Cue numbers are now action targets for merge, while timing lines remain action targets for retiming.",
                    "For the full guide, see `docs/MULTILINE_TTS_TAG_EDITOR_GUIDE.md`."
                ]
            }
        ];

        const libraryLayout = document.createElement("div");
        libraryLayout.className = "string-multiline-tag-editor-library";

        libraryGroups.forEach(group => {
            const section = document.createElement("section");
            section.className = "string-multiline-tag-editor-library-section";

            const title = document.createElement("h4");
            title.className = "string-multiline-tag-editor-library-title";
            title.textContent = group.title;

            const intro = document.createElement("p");
            intro.className = "string-multiline-tag-editor-library-intro";
            intro.textContent = group.intro;

            const bulletList = document.createElement("div");
            bulletList.className = "string-multiline-tag-editor-guide-list";
            group.bullets.forEach(text => {
                const bullet = document.createElement("div");
                bullet.className = "string-multiline-tag-editor-guide-item";
                bullet.textContent = text;
                bulletList.appendChild(bullet);
            });

            const table = document.createElement("div");
            table.className = "string-multiline-tag-editor-reference-table";

            const header = document.createElement("div");
            header.className = "string-multiline-tag-editor-reference-row is-header";
            ["Syntax", "Purpose", "Notes"].forEach(text => {
                const cell = document.createElement("div");
                cell.textContent = text;
                header.appendChild(cell);
            });
            table.appendChild(header);

            group.rows.forEach(row => {
                const rowEl = document.createElement("div");
                rowEl.className = "string-multiline-tag-editor-reference-row";

                const syntax = document.createElement("code");
                syntax.className = "string-multiline-tag-editor-reference-syntax";
                syntax.textContent = row.syntax;

                const purpose = document.createElement("div");
                purpose.textContent = row.purpose;

                const notes = document.createElement("div");
                notes.textContent = row.notes;

                rowEl.appendChild(syntax);
                rowEl.appendChild(purpose);
                rowEl.appendChild(notes);
                table.appendChild(rowEl);
            });

            section.appendChild(title);
            section.appendChild(intro);
            section.appendChild(bulletList);
            section.appendChild(table);
            libraryLayout.appendChild(section);
        });

        auxiliaryContent.replaceChildren(libraryLayout);
    };

    const activateTopView = (viewKey) => {
        topNavItems.forEach((item, key) => {
            item.classList.toggle("is-active", key === viewKey);
        });

        state.activeTopView = viewKey;
        state.saveToLocalStorage(storageKey);

        const isEditorView = viewKey === "editor";
        shellBody.style.display = isEditorView ? "flex" : "none";
        auxiliaryView.style.display = isEditorView ? "none" : "flex";
        resizeDivider.style.display = isEditorView ? "block" : "none";

        if (isEditorView) {
            requestAnimationFrame(updateSidebarScrollbar);
            return;
        }

        if (viewKey === "history") {
            renderHistoryView();
        } else if (viewKey === "presets") {
            renderPresetsView();
        } else if (viewKey === "library") {
            renderLibraryView();
        }
    };

    topNavItems.forEach((item, key) => {
        item.addEventListener("click", () => activateTopView(key));
    });

    sidebar.querySelectorAll('input[type="range"]').forEach(input => {
        updateRangeFill(input);
        input.addEventListener("input", () => updateRangeFill(input));
    });

    // Store state when node is removed
    widget.onRemove = () => {
        stopEditorScrollbarDrag();
        timingDragController?.dispose();
        cueEditController?.dispose();
        state.saveToLocalStorage(storageKey);
    };

    // Initialize history display
    historyStatus.textContent = state.getHistoryStatus();
    activateTopView(state.activeTopView || "editor");

    editor.addEventListener("scroll", () => {
        lineGutterContent.style.transform = `translateY(${-editor.scrollTop}px)`;
        updateEditorScrollbar();
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
    requestAnimationFrame(updateEditorScrollbar);

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
