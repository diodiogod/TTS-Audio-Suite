/**
 * 🏷️ Widget Preset System
 * Manages preset UI and logic for saving/loading text
 * Extracted for modularity - preserves 100% of original logic
 */

export function buildPresetSection(state, storageKey) {
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
    const presetTitles = {};

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
        presetTitles[presetKey] = presetTitle;

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

    const updatePresetGlows = () => {
        Object.entries(presetButtons).forEach(([presetKey, buttons]) => {
            const presetNum = presetKey.split("_")[1];
            if (presetKey in state.presets && state.presets[presetKey]) {
                const preset = state.presets[presetKey];
                let displayName = preset.tag;
                const availableWidth = state.sidebarWidth - 30;
                const maxChars = Math.max(8, Math.floor(availableWidth / 6));

                if (displayName.length > maxChars) {
                    displayName = displayName.substring(0, maxChars) + "...";
                }

                presetTitles[presetKey].textContent = displayName;
                buttons.load.style.background = "#00cc00";
                buttons.load.style.boxShadow = "0 0 8px #00cc00";
            } else {
                presetTitles[presetKey].textContent = `P${presetNum}`;
                buttons.load.style.background = "#3a3a3a";
                buttons.load.style.boxShadow = "none";
            }
        });
    };

    updatePresetGlows();

    return {
        presetSection,
        presetButtons,
        presetTitles,
        updatePresetGlows
    };
}
