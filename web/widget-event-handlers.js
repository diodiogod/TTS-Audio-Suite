/**
 * 🏷️ Widget Event Handlers
 * All event handler attachments for the editor
 * Extracted for modularity - preserves 100% of original logic
 */

import { TagUtilities } from "./tag-utilities.js";
import { isLanguageCode } from "./language-constants.js";

export function attachAllEventHandlers(
    editor, state, widget, storageKey, getPlainText, setEditorText, getCaretPos, setCaretPos,
    undoBtn, redoBtn, historyStatus, charSelect, charInput, addCharBtn, langSelect, addLangBtn,
    paramTypeSelect, paramInputWrapper, addParamBtn, presetButtons, presetTitles, updatePresetGlows,
    formatBtn, validateBtn, fontFamilySelect, fontSizeInput, fontSizeDisplay, setFontSize,
    showNotification, resizeDivider, sidebar, setSidebarWidth, setUIScale
) {
    // Block ComfyUI shortcuts when editor is focused, but allow Enter and Alt combinations
    editor.addEventListener("keydown", (e) => {
        // Don't block Enter or Alt key combinations - we have handlers for those
        if (e.key !== "Enter" && !e.altKey) {
            e.stopPropagation();
            e.stopImmediatePropagation();
        }
    }, true); // Use capture phase to intercept before other handlers

    // Manually handle Enter key to insert newline
    editor.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && document.activeElement === editor) {
            e.preventDefault();
            e.stopPropagation();
            document.execCommand("insertLineBreak");
            // Trigger input event to update history
            setTimeout(() => {
                editor.dispatchEvent(new Event("input", { bubbles: true }));
            }, 0);
        }
    });

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

    // Keyboard shortcuts for undo/redo and tag/preset insertion
    editor.addEventListener("keydown", (e) => {
        if (document.activeElement !== editor) return;

        if (e.altKey && !e.ctrlKey && !e.metaKey && !e.shiftKey) {
            // Alt+L: Add Language Tag
            if (e.key === "l" || e.key === "L") {
                e.preventDefault();
                addLangBtn.click();
            }
            // Alt+C: Add Character
            else if (e.key === "c" || e.key === "C") {
                e.preventDefault();
                addCharBtn.click();
            }
            // Alt+P: Add Parameter
            else if (e.key === "p" || e.key === "P") {
                e.preventDefault();
                addParamBtn.click();
            }
            // Alt+1/2/3: Load Preset
            else if (e.key === "1") {
                e.preventDefault();
                presetButtons.preset_1?.load?.click?.();
            }
            else if (e.key === "2") {
                e.preventDefault();
                presetButtons.preset_2?.load?.click?.();
            }
            else if (e.key === "3") {
                e.preventDefault();
                presetButtons.preset_3?.load?.click?.();
            }
        }
        // Alt+Z: Undo, Alt+Shift+Z: Redo (also allow with shift)
        if (e.altKey && !e.ctrlKey && !e.metaKey && (e.key === "z" || e.key === "Z")) {
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

    // Helper to get selected text and its position
    const getSelection = () => {
        const sel = window.getSelection();
        if (sel.toString().length === 0) return null;

        const range = sel.getRangeAt(0);
        const preRange = range.cloneRange();
        preRange.selectNodeContents(editor);
        preRange.setEnd(range.startContainer, range.startOffset);
        const start = preRange.toString().length;
        const end = start + range.toString().length;

        return { start, end, text: range.toString() };
    };

    // Add character button
    addCharBtn.addEventListener("click", () => {
        const char = charInput.value.trim() || charSelect.value;
        if (!char) return;

        state.lastCharacter = char;
        const text = getPlainText();

        // Check if text is selected
        const selection = getSelection();
        let caretPos;
        if (selection && selection.text.match(/^\s*\[/)) {
            // Selected text starts with a tag - find position right after the opening bracket
            const leadingWhitespace = selection.text.match(/^\s*/)[0].length;
            caretPos = selection.start + leadingWhitespace + 1; // position after [
        } else {
            caretPos = selection ? selection.start : getCaretPos();
        }

        const result = TagUtilities.modifyTagContent(text, caretPos, (tagContent) => {
            const parts = tagContent.split("|");
            const firstPart = parts[0];

            // Check if first part has a colon
            if (firstPart.includes(":")) {
                const colonIndex = firstPart.indexOf(":");
                const beforeColon = firstPart.substring(0, colonIndex);

                // Check if it's a language code (not a parameter)
                if (isLanguageCode(beforeColon)) {
                    // It's a language:character format, replace the character part
                    const langCode = beforeColon;
                    parts[0] = `${langCode}:${char}`;
                } else {
                    // It's a parameter (like seed:5), prepend character
                    parts.unshift(char);
                }
            } else {
                // No colon, just a character name - replace it
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
            // No existing tag found - create new one
            const charTag = `[${char}]`;
            let newText, newCaretPos;

            if (selection) {
                // Selected text: insert tag at beginning of selection
                newText = text.substring(0, selection.start) + charTag + " " + text.substring(selection.start);
                newCaretPos = selection.start + charTag.length + 1;
            } else {
                // No selection: insert at caret position
                newText = text.substring(0, caretPos) + charTag + " " + text.substring(caretPos);
                newCaretPos = caretPos + charTag.length + 1;
            }

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

    // Add language tag button
    addLangBtn.addEventListener("click", () => {
        const lang = langSelect.value;
        if (!lang) return;

        state.lastLanguage = lang;
        const text = getPlainText();

        // Check if text is selected
        const selection = getSelection();
        let caretPos;
        if (selection && selection.text.match(/^\s*\[/)) {
            // Selected text starts with a tag - find position right after the opening bracket
            const leadingWhitespace = selection.text.match(/^\s*/)[0].length;
            caretPos = selection.start + leadingWhitespace + 1; // position after [
        } else {
            caretPos = selection ? selection.start : getCaretPos();
        }

        // Try to modify existing tag
        const result = TagUtilities.modifyTagContent(text, caretPos, (tagContent) => {
            // Check if already has language code (contains colon in first part before pipe)
            const pipeIndex = tagContent.indexOf("|");
            const firstPart = pipeIndex === -1 ? tagContent : tagContent.substring(0, pipeIndex);

            if (firstPart.includes(":")) {
                const colonIndex = firstPart.indexOf(":");
                const beforeColon = firstPart.substring(0, colonIndex);

                // Check if it's a language code using the supported languages list
                if (isLanguageCode(beforeColon)) {
                    // It's already a language tag - only update if language is different
                    if (beforeColon === lang) {
                        // Same language already exists, return unchanged
                        return tagContent;
                    } else {
                        // Replace the language part with new language
                        const charPart = firstPart.substring(colonIndex + 1);
                        const rest = pipeIndex === -1 ? "" : tagContent.substring(pipeIndex);
                        return `${lang}:${charPart}${rest}`;
                    }
                } else {
                    // It's a parameter tag (like seed:5), insert language with pipe separator: [de:|seed:5|steps:3]
                    return `${lang}:|${tagContent}`;
                }
            } else {
                // No colon at all, just a character name - prepend language
                return `${lang}:${tagContent}`;
            }
        });

        if (result) {
            // Modified existing tag
            setEditorText(result.newText);
            setTimeout(() => {
                setCaretPos(result.newCaretPos);
                state.addToHistory(result.newText, result.newCaretPos);
                state.saveToLocalStorage(storageKey);
            }, 0);
        } else {
            // Create new language tag
            const langTag = `[${lang}:]`;
            let newText, newCaretPos;

            if (selection) {
                // Selected text: insert tag at beginning of selection
                newText = text.substring(0, selection.start) + langTag + " " + text.substring(selection.start);
                newCaretPos = selection.start + langTag.length + 1;
            } else {
                // No selection: insert at caret position
                newText = text.substring(0, caretPos) + langTag + " " + text.substring(caretPos);
                newCaretPos = caretPos + langTag.length + 1;
            }

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
        // Normalize spacing around brackets: multiple spaces/tabs become single space
        text = text.replace(/[ \t]+\[/g, " [").replace(/\[[ \t]+/g, "[");
        // Remove spaces/tabs before ], but keep newlines
        text = text.replace(/[ \t]+\]/g, "]").replace(/\][ \t]+/g, "]");
        // Add space before tags only if not after newline or space
        text = text.replace(/([^\n\s\[])\[/g, "$1 [");
        // Add space after closing bracket if followed by non-space/non-newline
        text = text.replace(/\]([^\s\n])/g, "] $1");
        // Trim end of each line but preserve newlines
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
