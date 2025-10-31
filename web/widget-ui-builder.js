/**
 * 🏷️ Widget UI Builder
 * Builds all UI sections for the string multiline tag editor
 * Extracted for modularity - preserves 100% of original logic
 */

import { TagUtilities } from "./tag-utilities.js";
import { loadSupportedLanguages } from "./language-constants.js";

// Load supported languages into memory on module load
loadSupportedLanguages();

export function buildHistorySection(state, storageKey) {
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
    undoBtn.title = "Undo (Alt+Z)";
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
    redoBtn.title = "Redo (Alt+Shift+Z)";
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

    return { historySection, undoBtn, redoBtn, historyStatus };
}

export function buildCharacterSection(state, storageKey) {
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

    const populateCharacters = async () => {
        try {
            let characters = [];
            if (state.discoveredCharacters && typeof state.discoveredCharacters === 'object') {
                if (!Array.isArray(state.discoveredCharacters)) {
                    characters = Object.keys(state.discoveredCharacters);
                } else {
                    characters = state.discoveredCharacters;
                }
            }

            if (characters.length === 0) {
                try {
                    const response = await fetch("/api/tts-audio-suite/available-characters");
                    if (response.ok) {
                        const data = await response.json();
                        if (data.characters && Array.isArray(data.characters)) {
                            characters = data.characters;
                        }
                    }
                } catch (err) {}
            }

            if (characters.length === 0) {
                characters = ["Alice", "Bob", "Charlie", "Diana", "Emma", "Frank", "Grace", "Henry"];
            }

            characters.forEach(char => {
                const option = document.createElement("option");
                option.value = char;
                option.textContent = char;
                charSelect.appendChild(option);
            });
            console.log(`✅ Loaded ${characters.length} character voices`);
        } catch (err) {
            console.warn("Could not populate characters:", err);
        }
    };

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

    return { charSection, charSelect, charInput, addCharBtn };
}

export function buildLanguageSection(state, storageKey) {
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
    langSelect.style.marginBottom = "4px";
    langSelect.style.padding = "3px";
    langSelect.style.fontSize = "10px";
    langSelect.style.background = "#2a2a2a";
    langSelect.style.color = "#eee";
    langSelect.style.border = "1px solid #444";
    langSelect.innerHTML = "<option value=''>Select...</option>";

    const populateLanguages = async () => {
        try {
            const response = await fetch("/api/tts-audio-suite/available-languages");
            if (response.ok) {
                const data = await response.json();
                if (data.languages && Array.isArray(data.languages)) {
                    langSelect.innerHTML = "<option value=''>Select...</option>";
                    data.languages.forEach(lang => {
                        const option = document.createElement("option");
                        option.value = lang;
                        option.textContent = lang.toUpperCase();
                        langSelect.appendChild(option);
                    });
                    // Restore saved language AFTER options are populated
                    langSelect.value = state.lastLanguage;
                    console.log(`✅ Loaded ${data.languages.length} language codes`);
                }
            }
        } catch (err) {
            console.warn("Could not load languages from API, using fallback:", err);
            // Fallback to hardcoded list
            const fallbackLanguages = ["en", "de", "fr", "ja", "es", "it", "pt", "th", "no"];
            langSelect.innerHTML = "<option value=''>Select...</option>";
            fallbackLanguages.forEach(lang => {
                const option = document.createElement("option");
                option.value = lang;
                option.textContent = lang.toUpperCase();
                langSelect.appendChild(option);
            });
            // Restore saved language AFTER options are populated
            langSelect.value = state.lastLanguage;
        }
    };

    populateLanguages();

    langSelect.addEventListener("change", () => {
        state.lastLanguage = langSelect.value;
        state.saveToLocalStorage(storageKey);
    });

    const addLangBtn = document.createElement("button");
    addLangBtn.textContent = "Add Language Tag";
    addLangBtn.title = "Insert language tag [lang:] or [lang:Character] at cursor";
    addLangBtn.style.width = "100%";
    addLangBtn.style.padding = "4px";
    addLangBtn.style.cursor = "pointer";
    addLangBtn.style.fontSize = "10px";
    addLangBtn.style.background = "#3a3a3a";
    addLangBtn.style.color = "#eee";
    addLangBtn.style.border = "1px solid #555";
    addLangBtn.style.borderRadius = "2px";

    langSection.appendChild(langLabel);
    langSection.appendChild(langSelect);
    langSection.appendChild(addLangBtn);

    return { langSection, langSelect, addLangBtn };
}

export function buildValidationSection() {
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

    return { validSection, formatBtn, validateBtn };
}
