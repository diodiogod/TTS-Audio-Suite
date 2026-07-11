/**
 * 🏷️ Widget Event Handlers
 * All event handler attachments for the editor
 * Extracted for modularity - preserves 100% of original logic
 */

import { TagUtilities } from "./tag-utilities.js";
import { isLanguageCode } from "./language-constants.js";

export function attachAllEventHandlers(
    editor, state, widget, storageKey, getPlainText, setEditorText, getCaretPos, setCaretPos, getSelectionRange,
    undoBtn, redoBtn, historyStatus, charSelect, charInput, addCharBtn, langSelect, addLangBtn,
    paramTypeSelect, paramInputWrapper, addParamBtn, presetButtons, presetTitles, updatePresetGlows,
    formatBtn, validateBtn, fontFamilySelect, fontSizeInput, fontSizeDisplay, setFontSize, setFontFamily,
    showNotification, resizeDivider, sidebar, setSidebarWidth, setUIScale, setSidebarResizeActive,
    inlineTagControls,
    openFindReplace, focusNextFindMatch, focusPreviousFindMatch
) {
    const {
        inlineEngineSelect,
        step: {
            paraSelect, paraIterSlider, addParaBtn,
            emotionSelect, emotionIterSlider, addEmotionBtn,
            styleSelect, styleIterSlider, addStyleBtn,
            speedSelect, speedIterSlider, addSpeedBtn,
            restorePassSlider, restoreRefInput, addRestoreBtn,
        },
        higgs: {
            emotionSelect: higgsEmotionSelect,
            addEmotionBtn: addHiggsEmotionBtn,
            styleSelect: higgsStyleSelect,
            addStyleBtn: addHiggsStyleBtn,
            prosodySelect: higgsProsodySelect,
            addProsodyBtn: addHiggsProsodyBtn,
            sfxSelect: higgsSfxSelect,
            addSfxBtn: addHiggsSfxBtn,
        },
        cosy: {
            singleTagSelect: cosySingleTagSelect,
            addSingleTagBtn,
            wrapperTagSelect: cosyWrapperTagSelect,
            addWrapperTagBtn,
        },
        omnivoice: {
            tagSelect: omnivoiceTagSelect,
            addTagBtn: addOmniVoiceTagBtn,
        },
        indexTTS: {
            vectorModeSelect: indexTTSVectorModeSelect,
            addVectorBtn: addIndexTTSVectorBtn,
            namedEmotionSelect: indexTTSNamedEmotionSelect,
            namedOperationSelect: indexTTSNamedOperationSelect,
            namedValueInput: indexTTSNamedValueInput,
            addNamedEmotionBtn: addIndexTTSNamedEmotionBtn,
            presetSelect: indexTTSPresetSelect,
            addTextPresetBtn: addIndexTTSTextPresetBtn,
            emotionTextInput: indexTTSEmotionTextInput,
            addEmotionTextBtn: addIndexTTSEmotionTextBtn,
        },
    } = inlineTagControls;

    // Block ComfyUI shortcuts when editor is focused, but allow Enter, Alt, and Ctrl combinations
    editor.addEventListener("keydown", (e) => {
        // Don't block Enter, Alt, or Ctrl key combinations (allow copy/paste/cut)
        if (e.key !== "Enter" && !e.altKey && !e.ctrlKey && !e.metaKey) {
            e.stopPropagation();
            e.stopImmediatePropagation();
        }
    }, true); // Use capture phase to intercept before other handlers

    inlineEngineSelect?.addEventListener("change", () => {
        const plainText = getPlainText();
        setEditorText(plainText);
    });

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
    let lastHistoryText = state.history[state.historyIndex]?.text ?? getPlainText();

    const flushHistory = () => {
        const plainText = getPlainText();
        const currentHistoryText = state.history[state.historyIndex]?.text;

        if (plainText !== lastHistoryText && plainText !== currentHistoryText) {
            const caretPos = getCaretPos();
            state.addToHistory(plainText, caretPos);
        }

        lastHistoryText = plainText;
    };

    const flushPendingHistory = () => {
        if (historyDebounceTimer !== null) {
            clearTimeout(historyDebounceTimer);
            historyDebounceTimer = null;
        }

        flushHistory();
        historyStatus.textContent = state.getHistoryStatus();
    };

    const getClampedCaretPos = (targetText, preferredPos, fallbackPos = 0) => {
        const basePos = preferredPos ?? fallbackPos ?? 0;
        return Math.max(0, Math.min(basePos, targetText.length));
    };

    const mapCaretPosBetweenTexts = (fromText, toText, caretPos, fallbackPos = 0) => {
        const safeCaretPos = getClampedCaretPos(fromText, caretPos, fallbackPos);

        if (fromText === toText) {
            return getClampedCaretPos(toText, safeCaretPos, fallbackPos);
        }

        const maxPrefix = Math.min(fromText.length, toText.length);
        let prefixLength = 0;
        while (prefixLength < maxPrefix && fromText[prefixLength] === toText[prefixLength]) {
            prefixLength++;
        }

        let fromSuffixIndex = fromText.length;
        let toSuffixIndex = toText.length;
        while (
            fromSuffixIndex > prefixLength &&
            toSuffixIndex > prefixLength &&
            fromText[fromSuffixIndex - 1] === toText[toSuffixIndex - 1]
        ) {
            fromSuffixIndex--;
            toSuffixIndex--;
        }

        const suffixLength = fromText.length - fromSuffixIndex;
        const fromChangeEnd = fromText.length - suffixLength;
        const toChangeEnd = toText.length - suffixLength;

        if (safeCaretPos <= prefixLength) {
            return safeCaretPos;
        }

        if (safeCaretPos >= fromChangeEnd) {
            const suffixOffset = safeCaretPos - fromChangeEnd;
            return getClampedCaretPos(toText, toChangeEnd + suffixOffset, fallbackPos);
        }

        const changedRegionOffset = safeCaretPos - prefixLength;
        return getClampedCaretPos(toText, prefixLength + changedRegionOffset, fallbackPos);
    };

    const restoreEditorHistoryEntry = (entry, preferredCaretPos = null) => {
        const restoredCaretPos = getClampedCaretPos(entry.text, preferredCaretPos, entry.caretPos);
        setEditorText(entry.text);
        lastHistoryText = entry.text;
        setTimeout(() => {
            editor.focus();
            setCaretPos(restoredCaretPos);
        }, 0);
        state.saveToLocalStorage(storageKey);
        widget.callback?.(widget.value);
        historyStatus.textContent = state.getHistoryStatus();
    };

    const commitEditorTextChange = (newText, newCaretPos, { focusEditor = true } = {}) => {
        setEditorText(newText);
        state.text = newText;
        state.addToHistory(newText, newCaretPos);
        state.saveToLocalStorage(storageKey);
        lastHistoryText = newText;
        widget.value = newText;
        widget.callback?.(newText);
        historyStatus.textContent = state.getHistoryStatus();

        setTimeout(() => {
            if (focusEditor) {
                editor.focus();
            }
            setCaretPos(newCaretPos);
        }, 0);
    };

    editor.addEventListener("input", (e) => {
        const plainText = getPlainText();
        // Update state.text immediately so it's current when saved to localStorage
        state.text = plainText;
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

    editor.addEventListener("copy", (e) => {
        // Allow copy to work, but prevent ComfyUI from receiving the event
        // Critical for ComfyUI v0.3.75+ which intercepts clipboard events
        e.stopPropagation();

        const selection = getSelectionRange();
        if (!selection?.text || !e.clipboardData) {
            return;
        }

        e.preventDefault();
        e.clipboardData.setData("text/plain", selection.text);
    });

    editor.addEventListener("cut", (e) => {
        // Allow cut to work, but prevent ComfyUI from receiving the event
        e.stopPropagation();

        const selection = getSelectionRange();
        if (!selection?.text || !e.clipboardData) {
            setTimeout(() => {
                flushHistory();
                historyStatus.textContent = state.getHistoryStatus();
            }, 0);
            return;
        }

        e.preventDefault();
        e.clipboardData.setData("text/plain", selection.text);

        const plainText = getPlainText();
        const newText = plainText.substring(0, selection.start) + plainText.substring(selection.end);
        commitEditorTextChange(newText, selection.start);
    });

    editor.addEventListener("paste", (e) => {
        // Stop propagation AFTER paste completes to prevent ComfyUI from pasting nodes
        e.stopPropagation();

        if (!e.clipboardData) {
            return;
        }

        const pastedText = e.clipboardData.getData("text/plain");
        if (typeof pastedText !== "string") {
            return;
        }

        e.preventDefault();

        const selection = getSelectionRange();
        const plainText = getPlainText();
        const insertStart = selection ? selection.start : getCaretPos();
        const insertEnd = selection ? selection.end : insertStart;
        const normalizedText = pastedText.replace(/\r\n?/g, "\n");
        const newText = plainText.substring(0, insertStart) + normalizedText + plainText.substring(insertEnd);
        const newCaretPos = insertStart + normalizedText.length;

        commitEditorTextChange(newText, newCaretPos);
    });

    const rememberCaretPosition = () => {
        const caretPos = getCaretPos();
        state.lastCursorPosition = caretPos;

        if (state.historyIndex >= 0 && state.history[state.historyIndex]) {
            state.history[state.historyIndex].caretPos = caretPos;
        }
    };

    editor.addEventListener("mouseup", rememberCaretPosition);
    editor.addEventListener("keyup", rememberCaretPosition);
    editor.addEventListener("focus", rememberCaretPosition);

    const applyHistoryStep = (direction) => {
        flushPendingHistory();
        const currentText = getPlainText();
        const currentCaretPos = getCaretPos();
        const targetEntry = direction === "redo" ? state.redo() : state.undo();
        const mappedCaretPos = mapCaretPosBetweenTexts(currentText, targetEntry.text, currentCaretPos, targetEntry.caretPos);
        restoreEditorHistoryEntry(targetEntry, mappedCaretPos);
    };

    // Undo/Redo buttons
    undoBtn.addEventListener("click", () => {
        applyHistoryStep("undo");
    });

    redoBtn.addEventListener("click", () => {
        applyHistoryStep("redo");
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
        const isPrimaryModifier = (e.ctrlKey || e.metaKey) && !e.altKey;
        if (isPrimaryModifier) {
            if (e.key === "f" || e.key === "F") {
                e.preventDefault();
                e.stopPropagation();
                e.stopImmediatePropagation();
                openFindReplace("find");
                return;
            }

            if (e.key === "h" || e.key === "H") {
                e.preventDefault();
                e.stopPropagation();
                e.stopImmediatePropagation();
                openFindReplace("replace");
                return;
            }

            const isUndo = (e.key === "z" || e.key === "Z") && !e.shiftKey;
            const isRedo = (e.key === "y" || e.key === "Y") || ((e.key === "z" || e.key === "Z") && e.shiftKey);

            if (isUndo || isRedo) {
                e.preventDefault();
                e.stopPropagation();
                e.stopImmediatePropagation();
                applyHistoryStep(isRedo ? "redo" : "undo");
                return;
            }
        }

        // Alt+Z: Undo, Alt+Shift+Z: Redo (also allow with shift)
        if (e.altKey && !e.ctrlKey && !e.metaKey && (e.key === "z" || e.key === "Z")) {
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();
            applyHistoryStep(e.shiftKey ? "redo" : "undo");
            return;
        }

        if (e.key === "F3") {
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();
            if (e.shiftKey) {
                focusPreviousFindMatch();
            } else {
                focusNextFindMatch();
            }
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
            if (fontSizeDisplay) {
                fontSizeDisplay.textContent = newSize + "px";
            }
        }
    });

    // Font family selector change
    fontFamilySelect.addEventListener("change", () => {
        setFontFamily(fontFamilySelect.value);
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
        if (fontSizeDisplay) {
            fontSizeDisplay.textContent = newSize + "px";
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

    // Helper to get selected text and its position
    const getSelection = () => {
        const selection = getSelectionRange();
        if (!selection?.text) {
            return null;
        }

        return selection;
    };

    const replaceCharacterComponentInColonTag = (text, caretPos, character) => {
        const tagStart = text.lastIndexOf("[", caretPos);
        const tagEnd = text.indexOf("]", caretPos);
        if (tagStart < 0 || tagEnd < caretPos) return null;

        const content = text.slice(tagStart + 1, tagEnd);
        const colonIndex = content.indexOf(":");
        if (colonIndex <= 0) return null;

        const first = content.slice(0, colonIndex).trim();
        const secondEndRelative = content.indexOf("|", colonIndex);
        const secondEnd = secondEndRelative >= 0 ? secondEndRelative : content.length;
        const second = content.slice(colonIndex + 1, secondEnd).trim();
        const parameterKey = /^(seed|temp|temperature|cfg|cfg_weight|emotion_alpha|vector|emotion|speed|top_p|top_k|steps|instruction|quality|sound_event|ambient_sound)$/i;
        if (parameterKey.test(first) || /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$/.test(second)) return null;

        const caretRelative = caretPos - (tagStart + 1);
        const replacingFirst = caretRelative <= colonIndex;
        const componentStart = replacingFirst ? 0 : colonIndex + 1;
        const componentEnd = replacingFirst ? colonIndex : secondEnd;
        const newContent = content.slice(0, componentStart) + character + content.slice(componentEnd);
        return {
            newText: text.slice(0, tagStart + 1) + newContent + text.slice(tagEnd),
            newCaretPos: tagStart + 1 + componentStart + character.length,
        };
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

        const colonComponentResult = replaceCharacterComponentInColonTag(text, caretPos, char);
        if (colonComponentResult) {
            setEditorText(colonComponentResult.newText);
            setTimeout(() => {
                setCaretPos(colonComponentResult.newCaretPos);
                state.addToHistory(colonComponentResult.newText, colonComponentResult.newCaretPos);
                state.saveToLocalStorage(storageKey);
                editor.focus();
            }, 0);
            widget.callback?.(widget.value);
            historyStatus.textContent = state.getHistoryStatus();
            return;
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
        const text = getPlainText();
        const bracketValidation = TagUtilities.validateTagSyntax(text);
        if (!bracketValidation.valid) {
            showNotification("❌ " + bracketValidation.error);
            return;
        }

        const inlineSyntaxValidation = TagUtilities.validateInlineSyntax(text);
        if (!inlineSyntaxValidation.valid) {
            showNotification("❌ " + inlineSyntaxValidation.error, 3000);
            return;
        }

        const targetEngine = inlineEngineSelect?.value || state.activeInlineTagEngine || "step_audio_editx";
        const inlineValidation = TagUtilities.validateInlineTags(text, targetEngine);

        if (inlineValidation.unknownTags.length > 0) {
            showNotification(`❌ Found ${inlineValidation.unknownTags.length} unrecognized inline tag(s)`, 3000);
            return;
        }

        if (inlineValidation.foreignTags.length > 0) {
            const engineLabel = inlineEngineSelect?.selectedOptions?.[0]?.textContent || targetEngine;
            const convertibleCount = inlineValidation.convertibleTags.length;
            const skippedCount = inlineValidation.foreignTags.length - convertibleCount;

            const baseMessage = `Found ${inlineValidation.foreignTags.length} inline tag(s) that do not belong to ${engineLabel}.`;

            if (convertibleCount > 0) {
                let message = baseMessage;
                message += `\n\nConvert ${convertibleCount} safe tag(s) now?`;
                if (skippedCount > 0) {
                    message += `\n${skippedCount} tag(s) have no safe equivalent and will stay unchanged.`;
                }
                if (window.confirm(message)) {
                    const conversion = TagUtilities.convertInlineTagsForEngine(text, targetEngine);
                    if (conversion.converted > 0 && conversion.text !== text) {
                        commitEditorTextChange(conversion.text, getCaretPos());
                        showNotification(`✓ Converted ${conversion.converted} inline tag(s)${conversion.skipped ? `, skipped ${conversion.skipped}` : ""}`, 3000);
                        return;
                    }
                }

                showNotification(`⚠️ Found ${inlineValidation.foreignTags.length} foreign inline tag(s) for ${engineLabel}. ${convertibleCount} can be auto-converted, ${skippedCount} cannot.`, 4000);
                return;
            }

            showNotification(`⚠️ Found ${inlineValidation.foreignTags.length} foreign inline tag(s) for ${engineLabel}; no safe auto-conversion is defined yet`, 4000);
            return;
        }

        showNotification("✅ Tags match the selected engine!", 2000);
    });

    // Preset buttons
    Object.entries(presetButtons).forEach(([presetKey, buttons]) => {
        const presetNum = presetKey.split("_")[1];

        buttons.save.addEventListener("click", () => {
            // First check if user selected text in editor (like [de:Alice|seed:42|temp:0.8])
            const selection = getSelectionRange();
            let selectedText = "";

            if (selection?.text?.length > 0) {
                selectedText = selection.text;
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
            setSidebarResizeActive(false);
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
        setSidebarResizeActive(true);
        e.preventDefault();
    });

    document.addEventListener("mousemove", (e) => {
        if (!isResizing) return;
        const delta = e.clientX - initialMouseX;
        const newWidth = initialSidebarWidth + delta;
        setSidebarWidth(newWidth);
    });

    document.addEventListener("mouseup", () => {
        if (isResizing) {
            setSidebarResizeActive(false);
        }
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

    // ==================== INLINE EDIT TAG INSERTION ====================

    // Helper to modify inline edit tag content (similar to TagUtilities.modifyTagContent but for <...>)
    const modifyInlineEditTag = (text, caretPos, newTagPart) => {
        // Check if caret is right after an inline edit tag >
        let isRightAfterTag = caretPos > 0 && text[caretPos - 1] === ">";
        let spaceAfterTag = false;
        if (!isRightAfterTag && caretPos > 1 && text[caretPos - 1] === " " && text[caretPos - 2] === ">") {
            isRightAfterTag = true;
            spaceAfterTag = true;
        }

        // Check if caret is INSIDE an inline edit tag (between < and >)
        let isInsideTag = false;
        let tagStart = -1;
        let tagEnd = -1;

        if (!isRightAfterTag) {
            // Look backwards for < without hitting >
            let bracketDepth = 0;
            for (let i = caretPos - 1; i >= 0; i--) {
                if (text[i] === ">") {
                    bracketDepth++;
                } else if (text[i] === "<") {
                    if (bracketDepth === 0) {
                        tagStart = i;
                        // Find matching >
                        for (let j = i + 1; j < text.length; j++) {
                            if (text[j] === ">") {
                                tagEnd = j;
                                if (tagEnd >= caretPos) {
                                    isInsideTag = true;
                                }
                                break;
                            }
                        }
                        break;
                    } else {
                        bracketDepth--;
                    }
                }
            }
        } else {
            // Right after tag - find the tag we're right after
            let searchPos = spaceAfterTag ? caretPos - 2 : caretPos - 1;
            let bracketDepth = 0;
            for (let i = searchPos - 1; i >= 0; i--) {
                if (text[i] === ">") {
                    bracketDepth++;
                } else if (text[i] === "<") {
                    if (bracketDepth === 0) {
                        tagStart = i;
                        tagEnd = searchPos;
                        isInsideTag = false; // We're after it, not inside
                        break;
                    } else {
                        bracketDepth--;
                    }
                }
            }
        }

        if (tagStart !== -1 && tagEnd !== -1) {
            // Found a tag - modify it
            const tagContent = text.substring(tagStart + 1, tagEnd);
            const parts = tagContent.split("|");

            // Check if this tag type already exists
            const tagType = newTagPart.split(":")[0];
            const existingIndex = parts.findIndex(part => part.startsWith(tagType + ":") || part === tagType);

            let newContent;
            if (existingIndex !== -1) {
                // Replace existing tag of same type
                parts[existingIndex] = newTagPart;
                newContent = parts.join("|");
            } else {
                // Add new tag with pipe separator
                newContent = tagContent + "|" + newTagPart;
            }

            const newTag = `<${newContent}>`;
            const newText = text.substring(0, tagStart) + newTag + text.substring(tagEnd + 1);
            const newCaretPos = tagStart + newTag.length;

            return { newText, newCaretPos, modified: true };
        }

        // No tag found - insert new tag
        return { modified: false };
    };

    const insertTextSnippet = (snippet, {
        caretOffset = snippet.length,
        notification = `✓ Inserted: ${snippet}`,
    } = {}) => {
        const selection = getSelection();
        const plainText = getPlainText();
        const insertStart = selection ? selection.start : getCaretPos();
        const insertEnd = selection ? selection.end : insertStart;
        const newText = plainText.substring(0, insertStart) + snippet + plainText.substring(insertEnd);
        commitEditorTextChange(newText, insertStart + caretOffset);
        showNotification(notification, 1500);
    };

    const wrapSelectionWithTag = (tagName) => {
        const selection = getSelection();
        if (selection) {
            const wrappedText = `<${tagName}>${selection.text}</${tagName}>`;
            const plainText = getPlainText();
            const newText = plainText.substring(0, selection.start) + wrappedText + plainText.substring(selection.end);
            commitEditorTextChange(newText, selection.start + wrappedText.length);
            showNotification(`✓ Wrapped with <${tagName}>`, 1500);
            return;
        }

        const placeholder = "text";
        const snippet = `<${tagName}>${placeholder}</${tagName}>`;
        const caretOffset = tagName.length + 2;
        insertTextSnippet(snippet, {
            caretOffset,
            notification: `✓ Inserted: ${snippet}`,
        });
    };

    const replaceTagAroundCaret = (text, caretPos, matcher, replacement) => {
        matcher.lastIndex = 0;
        let match;
        while ((match = matcher.exec(text)) !== null) {
            const start = match.index;
            const end = start + match[0].length;
            if (caretPos < start || caretPos > end) {
                continue;
            }

            return {
                modified: true,
                newText: text.substring(0, start) + replacement + text.substring(end),
                newCaretPos: start + replacement.length,
            };
        }
        return { modified: false };
    };

    const insertOrReplaceHiggsTag = (category, value) => {
        const caretPos = getCaretPos();
        const plainText = getPlainText();
        const replacement = `<|${category}:${value}|>`;

        const result = replaceTagAroundCaret(
            plainText,
            caretPos,
            new RegExp(
                `<\\|(?:emotion|style|prosody|sfx):[^|>]+\\|>|<(?:emotion|style|prosody|sfx):[^>]+>|<(?:Laughter|Breathing|Sigh|Uhm|Surprise-oh|Surprise-ah|Surprise-wa|Confirmation-en|Question-ei|Dissatisfaction-hnn)(?::\\d+)?>`,
                "g"
            ),
            replacement
        );

        if (result.modified) {
            commitEditorTextChange(result.newText, result.newCaretPos);
            showNotification("✓ Updated inline tag", 1500);
            return;
        }

        insertTextSnippet(replacement, {
            notification: `✓ Inserted: ${replacement}`,
        });
    };

    // Helper function to insert Step Audio EditX inline tag (with pipe-separator support)
    const insertStepInlineTag = (tagPart) => {
        const caretPos = getCaretPos();
        const plainText = getPlainText();

        // Try to modify existing inline edit tag
        const result = modifyInlineEditTag(plainText, caretPos, tagPart);

        if (result.modified) {
            // Modified existing tag
            setEditorText(result.newText);
            state.text = result.newText;
            state.addToHistory(result.newText, result.newCaretPos);
            state.saveToLocalStorage(storageKey);
            widget.callback?.(widget.value);
            historyStatus.textContent = state.getHistoryStatus();

            setTimeout(() => {
                editor.focus();
                setCaretPos(result.newCaretPos);
            }, 0);
            showNotification(`✓ Updated inline tag`, 1500);
        } else {
            // Create new tag
            const newTag = `<${tagPart}>`;
            const before = plainText.substring(0, caretPos);
            const after = plainText.substring(caretPos);
            const newText = before + newTag + after;

            setEditorText(newText);
            state.text = newText;
            state.addToHistory(newText, caretPos + newTag.length);
            state.saveToLocalStorage(storageKey);
            widget.callback?.(widget.value);
            historyStatus.textContent = state.getHistoryStatus();

            setTimeout(() => {
                editor.focus();
                setCaretPos(caretPos + newTag.length);
            }, 0);
            showNotification(`✓ Inserted: ${newTag}`, 1500);
        }
    };

    // Paralinguistic tag insertion
    addParaBtn.addEventListener("click", () => {
        const type = paraSelect.value;
        if (!type) {
            showNotification("⚠️ Select a paralinguistic type first", 2000);
            return;
        }

        const iterations = paraIterSlider.value;
        const tagPart = iterations === "1" ? type : `${type}:${iterations}`;
        insertStepInlineTag(tagPart);
    });

    // Emotion tag insertion
    addEmotionBtn.addEventListener("click", () => {
        const emotion = emotionSelect.value;
        if (!emotion) {
            showNotification("⚠️ Select an emotion first", 2000);
            return;
        }

        const iterations = emotionIterSlider.value;
        const tagPart = iterations === "1" ? `emotion:${emotion}` : `emotion:${emotion}:${iterations}`;
        insertStepInlineTag(tagPart);
    });

    // Style tag insertion
    addStyleBtn.addEventListener("click", () => {
        const style = styleSelect.value;
        if (!style) {
            showNotification("⚠️ Select a style first", 2000);
            return;
        }

        const iterations = styleIterSlider.value;
        const tagPart = iterations === "1" ? `style:${style}` : `style:${style}:${iterations}`;
        insertStepInlineTag(tagPart);
    });

    // Speed tag insertion
    addSpeedBtn.addEventListener("click", () => {
        const speed = speedSelect.value;
        if (!speed) {
            showNotification("⚠️ Select a speed first", 2000);
            return;
        }

        const iterations = speedIterSlider.value;
        const tagPart = iterations === "1" ? `speed:${speed}` : `speed:${speed}:${iterations}`;
        insertStepInlineTag(tagPart);
    });

    // Restore tag insertion
    addRestoreBtn.addEventListener("click", () => {
        const passes = restorePassSlider.value;
        const refIter = restoreRefInput.value.trim();

        let tagPart;
        if (refIter) {
            // Format: restore:N@M where N=passes, M=reference iteration
            tagPart = `restore:${passes}@${refIter}`;
        } else if (passes === "1") {
            // Simple restore
            tagPart = "restore";
        } else {
            // Multiple passes
            tagPart = `restore:${passes}`;
        }

        insertStepInlineTag(tagPart);
    });

    const bindHiggsTagInsert = (category, select, button, missingMessage) => {
        button.addEventListener("click", () => {
            const value = select.value;
            if (!value) {
                showNotification(missingMessage, 2000);
                return;
            }

            insertOrReplaceHiggsTag(category, value);
        });
    };

    bindHiggsTagInsert("emotion", higgsEmotionSelect, addHiggsEmotionBtn, "⚠️ Select a Higgs emotion first");
    bindHiggsTagInsert("style", higgsStyleSelect, addHiggsStyleBtn, "⚠️ Select a Higgs style first");
    bindHiggsTagInsert("prosody", higgsProsodySelect, addHiggsProsodyBtn, "⚠️ Select a Higgs prosody tag first");
    bindHiggsTagInsert("sfx", higgsSfxSelect, addHiggsSfxBtn, "⚠️ Select a Higgs SFX tag first");

    addSingleTagBtn.addEventListener("click", () => {
        const tagName = cosySingleTagSelect.value;
        if (!tagName) {
            showNotification("⚠️ Select a CosyVoice3 tag first", 2000);
            return;
        }

        const tag = `<${tagName}>`;
        insertTextSnippet(tag);
    });

    addWrapperTagBtn.addEventListener("click", () => {
        const tagName = cosyWrapperTagSelect.value;
        if (!tagName) {
            showNotification("⚠️ Select a CosyVoice3 wrapper tag first", 2000);
            return;
        }

        wrapSelectionWithTag(tagName);
    });

    addOmniVoiceTagBtn.addEventListener("click", () => {
        const tagName = omnivoiceTagSelect.value;
        if (!tagName) {
            showNotification("⚠️ Select an OmniVoice tag first", 2000);
            return;
        }

        const tag = `<${tagName}>`;
        insertTextSnippet(tag);
    });

    const isIndexTTSEmotionTag = (tag) => {
        if (/^\[(?:vector|emotion):/i.test(tag)) return true;
        const content = tag.slice(1, -1);
        const emotionNames = "happy|angry|sad|afraid|disgusted|melancholic|surprised|calm";
        return content.split("|").every(part => new RegExp(`^(?:${emotionNames}):[+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)$`, "i").test(part.trim()));
    };

    const insertOrReplaceIndexTTSTag = (replacement) => {
        const text = getPlainText();
        const caretPos = getCaretPos();
        const start = text.lastIndexOf("[", caretPos);
        const closingIndex = text.indexOf("]", caretPos);
        if (start >= 0 && closingIndex >= caretPos) {
            const existing = text.slice(start, closingIndex + 1);
            if (isIndexTTSEmotionTag(existing)) {
                const newText = text.slice(0, start) + replacement + text.slice(closingIndex + 1);
                commitEditorTextChange(newText, start + replacement.length);
                showNotification(`✓ Updated: ${replacement}`, 1500);
                return;
            }
        }
        insertTextSnippet(replacement);
    };

    addIndexTTSVectorBtn.addEventListener("click", () => {
        const relative = indexTTSVectorModeSelect.value === "delta";
        const values = Array(8).fill(relative ? "+0" : "0");
        insertOrReplaceIndexTTSTag(`[vector:${values.join(",")}]`);
    });

    addIndexTTSNamedEmotionBtn.addEventListener("click", () => {
        const emotion = indexTTSNamedEmotionSelect.value;
        const magnitude = Math.max(0, Math.min(1.2, Number(indexTTSNamedValueInput.value) || 0));
        const operation = indexTTSNamedOperationSelect.value;
        const prefix = operation === "positive" ? "+" : operation === "negative" ? "-" : "";
        insertOrReplaceIndexTTSTag(`[${emotion}:${prefix}${magnitude}]`);
    });

    addIndexTTSTextPresetBtn.addEventListener("click", () => {
        const preset = indexTTSPresetSelect.value;
        if (!preset) {
            showNotification("⚠️ Select an IndexTTS text preset first", 2000);
            return;
        }
        const selectedOption = indexTTSPresetSelect.selectedOptions[0];
        if (selectedOption?.dataset.presetType === "vector") {
            const values = JSON.parse(selectedOption.dataset.vectorValues || "[]");
            if (values.length !== 8) {
                showNotification("⚠️ This vector preset is invalid", 2000);
                return;
            }
            insertOrReplaceIndexTTSTag(`[vector:${values.join(",")}]`);
            return;
        }
        insertOrReplaceIndexTTSTag(`[emotion:${preset}]`);
    });

    addIndexTTSEmotionTextBtn.addEventListener("click", () => {
        const description = indexTTSEmotionTextInput.value.trim();
        if (!description) {
            showNotification("⚠️ Enter an emotion description first", 2000);
            return;
        }
        const quoted = !description.includes('"')
            ? `"${description}"`
            : !description.includes("'")
                ? `'${description}'`
                : `"${description.replaceAll('"', '”')}"`;
        insertOrReplaceIndexTTSTag(`[emotion:${quoted}]`);
    });

    return {
        beginExternalTransaction() {
            flushPendingHistory();
            return {
                originalText: getPlainText(),
                originalCaretPos: getCaretPos(),
            };
        },
        previewExternalTransaction(transaction, newText, caretPos) {
            setEditorText(newText);
            state.text = newText;
            widget.value = newText;
            widget.callback?.(newText);
            if (Number.isFinite(caretPos)) state.lastCursorPosition = caretPos;
        },
        cancelExternalTransaction(transaction) {
            const originalText = transaction.originalText;
            const caretPos = getClampedCaretPos(originalText, transaction.originalCaretPos, 0);
            setEditorText(originalText);
            state.text = originalText;
            widget.value = originalText;
            widget.callback?.(originalText);
            lastHistoryText = originalText;
            state.saveToLocalStorage(storageKey);
            historyStatus.textContent = state.getHistoryStatus();
            setTimeout(() => setCaretPos(caretPos), 0);
        },
        commitExternalTransaction(transaction, finalText, caretPos) {
            const finalCaretPos = getClampedCaretPos(finalText, caretPos, transaction.originalCaretPos);
            setEditorText(finalText);
            state.text = finalText;
            if (finalText !== transaction.originalText) {
                state.addToHistory(finalText, finalCaretPos);
            }
            widget.value = finalText;
            widget.callback?.(finalText);
            lastHistoryText = finalText;
            state.saveToLocalStorage(storageKey);
            historyStatus.textContent = state.getHistoryStatus();
            setTimeout(() => setCaretPos(finalCaretPos), 0);
        },
    };
}
