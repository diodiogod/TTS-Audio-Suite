const TABS = [
    ["char-param", "Character", "Choose a character or language for one segment. Save and load tag presets here."],
    ["parameters", "Parameters", "Set generation or steering parameters for one segment. Example [Alice|seed:42]."],
    ["inline-edit", "Inline Tags", "Add cues the selected engine handles in the text. Example <breath>."],
];

export function buildTabSystem(state, storageKey, onTabChange) {
    const tabHeaders = document.createElement("div");
    tabHeaders.className = "string-multiline-tag-editor-tabs";
    tabHeaders.setAttribute("role", "group");
    tabHeaders.setAttribute("aria-label", "Tag editor tabs");
    const tabs = new Map();

    for (const [key, label, tooltip] of TABS) {
        const button = document.createElement("button");
        button.type = "button";
        const text = document.createElement("span");
        text.textContent = label;
        button.appendChild(text);
        button.title = tooltip;
        button.dataset.tab = key;
        button.addEventListener("click", () => {
            switchTab(key);
            state.saveToLocalStorage(storageKey);
            onTabChange();
        });

        const panel = document.createElement("div");
        panel.className = "string-multiline-tag-editor-tab-content";
        panel.dataset.content = key;
        Object.assign(panel.style, { flexDirection: "column", overflowX: "hidden", flex: "0 0 auto" });
        tabs.set(key, { button, panel });
        tabHeaders.appendChild(button);
    }

    function switchTab(key) {
        state.activeTab = tabs.has(key) ? key : TABS[0][0];
        for (const [tabKey, { button, panel }] of tabs) {
            const active = tabKey === state.activeTab;
            button.setAttribute("aria-pressed", String(active));
            panel.style.display = active ? "flex" : "none";
        }
    }
    switchTab(state.activeTab);

    return {
        tabHeaders,
        characterContent: tabs.get("char-param").panel,
        parameterContent: tabs.get("parameters").panel,
        inlineEditContent: tabs.get("inline-edit").panel,
        switchTab,
    };
}
