/**
 * 🏷️ Widget Parameter Section
 * Builds the parameter control section with dynamic input types
 * Extracted for modularity - preserves 100% of original logic
 */

import { TagUtilities } from "./tag-utilities.js";

export function buildParameterSection(state, storageKey, getPlainText, setEditorText, getCaretPos, setCaretPos, widget, historyStatus, editor) {
    const paramSection = document.createElement("div");
    paramSection.style.marginBottom = "8px";
    paramSection.style.paddingBottom = "8px";
    paramSection.style.borderBottom = "1px solid #444";

    const paramLabel = document.createElement("div");
    paramLabel.textContent = "Add Parameter";
    paramLabel.style.fontWeight = "bold";
    paramLabel.style.marginBottom = "5px";
    paramLabel.style.fontSize = "11px";

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

    const paramInputWrapper = document.createElement("div");
    paramInputWrapper.style.marginBottom = "5px";

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

    paramTypeSelect.addEventListener("change", () => {
        paramInputWrapper.innerHTML = "";
        if (paramTypeSelect.value) {
            currentParamInput = createParamInput(paramTypeSelect.value);
            paramInputWrapper.appendChild(currentParamInput);
            state.lastParameterType = paramTypeSelect.value;
            state.saveToLocalStorage(storageKey);
        } else {
            currentParamInput = null;
        }
    });

    if (state.lastParameterType) {
        paramTypeSelect.value = state.lastParameterType;
        const changeEvent = new Event("change");
        paramTypeSelect.dispatchEvent(changeEvent);
    }

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

    // Add parameter button click handler
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
            const paramType = paramTypeSelect.value;
            const paramRegex = new RegExp(`${paramType}:[^|\\]]+`);

            if (paramRegex.test(tagContent)) {
                // Replace existing parameter
                return tagContent.replace(paramRegex, paramStr);
            } else {
                // Add new parameter
                return `${tagContent}|${paramStr}`;
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
            // Create new parameter tag
            const paramTag = `[${paramStr}]`;
            let newText, newCaretPos;

            if (selection) {
                // Selected text: insert tag at beginning of selection
                newText = text.substring(0, selection.start) + paramTag + " " + text.substring(selection.start);
                newCaretPos = selection.start + paramTag.length;
            } else {
                // No selection: insert at caret position
                newText = text.substring(0, caretPos) + paramTag + " " + text.substring(caretPos);
                newCaretPos = caretPos + paramTag.length;
            }

            setEditorText(newText);
            setTimeout(() => {
                setCaretPos(newCaretPos);
                state.addToHistory(newText, newCaretPos);
                state.saveToLocalStorage(storageKey);
            }, 0);
        }
        widget.callback?.(widget.value);
        if (historyStatus) {
            historyStatus.textContent = state.getHistoryStatus();
        }
    });

    paramSection.appendChild(paramLabel);
    paramSection.appendChild(paramTypeSelect);
    paramSection.appendChild(paramInputWrapper);
    paramSection.appendChild(addParamBtn);

    return {
        paramSection,
        paramTypeSelect,
        paramInputWrapper,
        addParamBtn,
        createParamInput,
        getCurrentParamInput: () => currentParamInput,
        setCurrentParamInput: (input) => { currentParamInput = input; }
    };
}
