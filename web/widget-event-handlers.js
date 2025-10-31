/**
 * 🏷️ Widget Event Handlers
 * All event handler attachments for the editor
 * Extracted for modularity - preserves 100% of original logic
 */

import { TagUtilities } from "./tag-utilities.js";

export function attachAllEventHandlers(
    editor, state, widget, storageKey, getPlainText, setEditorText, getCaretPos, setCaretPos,
    undoBtn, redoBtn, historyStatus, charSelect, charInput, addCharBtn, langSelect,
    paramTypeSelect, paramInputWrapper, addParamBtn, presetButtons, presetTitles, updatePresetGlows,
    formatBtn, validateBtn, fontFamilySelect, fontSizeInput, fontSizeDisplay, setFontSize,
    showNotification, resizeDivider, sidebar, setSidebarWidth, setUIScale
) {
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
        state.saveToLocalStorage(storageKey);
        widget.callback?.(widget.value);

        if (historyDebounceTimer !== null) {
            clearTimeout(historyDebounceTimer);
        }

        historyDebounceTimer = setTimeout(() => {
            flushHistory();
            historyStatus.textContent = state.getHistoryStatus();
            historyDebounceTimer = null;
        }, 500);
    });

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

    // Undo/Redo buttons
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
        if (document.activeElement === editor && e.altKey && e.key === "z") {
            e.preventDefault();
            let entry = e.shiftKey ? state.redo() : state.undo();
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
            const delta = e.deltaY > 0 ? -1 : 1;
            let newSize = state.fontSize + delta;
            newSize = Math.max(2, Math.min(120, newSize));
            setFontSize(newSize);
            fontSizeInput.value = newSize;
            fontSizeDisplay.textContent = newSize + "px";
        }
    });

    // Font family selector change
    fontFamilySelect.addEventListener("change", () => {
        editor.style.fontFamily = fontFamilySelect.value;
        state.fontFamily = fontFamilySelect.value;
        state.saveToLocalStorage(storageKey);
    });

    // Font size input change
    fontSizeInput.addEventListener("change", () => {
        let newSize = parseInt(fontSizeInput.value) || 14;
        newSize = Math.max(2, Math.min(120, newSize));
        setFontSize(newSize);
        fontSizeInput.value = newSize;
    });

    // Font size input live preview
    fontSizeInput.addEventListener("input", () => {
        let newSize = parseInt(fontSizeInput.value) || state.fontSize;
        newSize = Math.max(2, Math.min(120, newSize));
        editor.style.fontSize = newSize + "px";
        fontSizeDisplay.textContent = newSize + "px";
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
        if (!char) return;

        state.lastCharacter = char;
        const text = getPlainText();
        const caretPos = getCaretPos();

        const result = TagUtilities.modifyTagContent(text, caretPos, (tagContent) => {
            const parts = tagContent.split("|");
            const firstPartIsParam = parts[0].includes(":");
            if (firstPartIsParam) {
                parts.unshift(char);
            } else {
                parts[0] = char;
            }
            return parts.join("|");
        });

        if (result) {
            setEditorText(result.newText);
            setTimeout(() => {
                setCaretPos(result.newCaretPos);
                state.addToHistory(result.newText, result.newCaretPos);
                state.saveToLocalStorage(storageKey);
            }, 0);
        } else {
            const charTag = `[${char}]`;
            const newText = text.substring(0, caretPos) + charTag + " " + text.substring(caretPos);
            const newCaretPos = caretPos + charTag.length + 1;

            setEditorText(newText);
            setTimeout(() => {
                setCaretPos(newCaretPos);
                state.addToHistory(newText, newCaretPos);
                state.saveToLocalStorage(storageKey);
                editor.focus();
            }, 0);
        }
        widget.callback?.(widget.value);
        historyStatus.textContent = state.getHistoryStatus();
    });

    // Format button
    formatBtn.addEventListener("click", () => {
        let text = getPlainText();
        text = text.replace(/\s*\[\s*/g, "[").replace(/\s*\]/g, "]");
        text = text.replace(/([^\n\[])\[/g, "$1 [");
        text = text.replace(/\]([^\s\n])/g, "] $1");
        text = text.split("\n").map(line => line.trimEnd()).join("\n");

        setEditorText(text);
        setTimeout(() => {
            state.addToHistory(text, 0);
            state.saveToLocalStorage(storageKey);
            setCaretPos(0);
        }, 0);
        widget.callback?.(widget.value);
        historyStatus.textContent = state.getHistoryStatus();
    });

    // Validate button
    validateBtn.addEventListener("click", () => {
        const validation = TagUtilities.validateTagSyntax(getPlainText());
        if (validation.valid) {
            showNotification("✅ Tag syntax is valid!");
        } else {
            showNotification("❌ " + validation.error);
        }
    });

    // Preset buttons
    Object.entries(presetButtons).forEach(([presetKey, buttons]) => {
        const presetNum = presetKey.split("_")[1];

        buttons.save.addEventListener("click", () => {
            // First check if user selected text in editor (like [de:Alice|seed:42|temp:0.8])
            const selection = window.getSelection();
            let selectedText = "";

            if (selection.toString().length > 0) {
                selectedText = selection.toString();
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
                const currentCaretPos = getCaretPos(); // Get current caret position when button is clicked
                const text = getPlainText();

                // Always insert the preset tag at the current caret position
                const newText = text.substring(0, currentCaretPos) + preset.tag + " " + text.substring(currentCaretPos);
                setEditorText(newText);

                // Move caret to after the inserted text
                const newCaretPos = currentCaretPos + preset.tag.length + 1; // +1 for the space after tag
                setTimeout(() => {
                    setCaretPos(newCaretPos);
                    state.addToHistory(newText, newCaretPos);
                    state.saveToLocalStorage(storageKey);
                    editor.focus();
                }, 0);
                widget.callback?.(widget.value);
                historyStatus.textContent = state.getHistoryStatus();

                // If it's a simple character preset, also update the sidebar for convenience
                if (!preset.isComplexTag && preset.tag) {
                    charInput.value = preset.tag;
                    state.lastCharacter = preset.tag;

                    if (preset.parameters) {
                        if (preset.parameters.language) langSelect.value = preset.parameters.language;
                        if (preset.parameters.lastSeed) state.lastSeed = preset.parameters.lastSeed;
                        if (preset.parameters.lastTemperature) state.lastTemperature = preset.parameters.lastTemperature;
                    }
                }

                showNotification(`✅ Preset ${presetNum} inserted at cursor`);
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

    // Divider resize
    let isResizing = false;
    let lastClickTime = 0;
    let lastClickX = 0;
    let initialMouseX = 0;
    let initialSidebarWidth = 0;

    resizeDivider.addEventListener("mousedown", (e) => {
        const currentTime = Date.now();
        if (currentTime - lastClickTime < 300 && Math.abs(e.clientX - lastClickX) < 5) {
            setSidebarWidth(220);
            setUIScale(1.0);
            setFontSize(14);
            showNotification("🔄 Reset: Sidebar width, UI scale, and font size to defaults");
            lastClickTime = 0;
            return;
        }
        lastClickTime = currentTime;
        lastClickX = e.clientX;

        initialMouseX = e.clientX;
        initialSidebarWidth = state.sidebarWidth;
        isResizing = true;
        e.preventDefault();
    });

    document.addEventListener("mousemove", (e) => {
        if (!isResizing) return;
        const delta = e.clientX - initialMouseX;
        const newWidth = initialSidebarWidth + delta;
        setSidebarWidth(newWidth);
    });

    document.addEventListener("mouseup", () => {
        isResizing = false;
    });

    // Ctrl+wheel on sidebar to change UI scale
    sidebar.addEventListener("wheel", (e) => {
        if ((e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            const delta = e.deltaY > 0 ? -0.1 : 0.1;
            const newScale = state.uiScale + delta;
            setUIScale(newScale);
        }
    });

    // Pointer events for canvas interaction
    editor.addEventListener("pointermove", (e) => {
        if ((e.buttons & 4) === 4) {
            window.app.canvas.processMouseMove(e);
        }
    });

    editor.addEventListener("pointerup", (e) => {
        if (e.button === 1) {
            window.app.canvas.processMouseUp(e);
        }
    });
}
