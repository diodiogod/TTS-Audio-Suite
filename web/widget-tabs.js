/**
 * 🏷️ Widget Tabs System
 * Creates tabbed interface for Character/Parameters and Inline Edit sections
 */

export function buildTabSystem(state, storageKey) {
    // Tab container
    const tabContainer = document.createElement("div");
    tabContainer.style.display = "flex";
    tabContainer.style.flexDirection = "column";
    tabContainer.style.height = "100%";
    tabContainer.style.overflow = "hidden";

    // Tab headers
    const tabHeaders = document.createElement("div");
    tabHeaders.style.display = "flex";
    tabHeaders.style.borderBottom = "1px solid #444";
    tabHeaders.style.marginBottom = "10px";
    tabHeaders.style.flexShrink = "0";

    const charParamTab = document.createElement("div");
    charParamTab.textContent = "Character/Parameters";
    charParamTab.style.flex = "1";
    charParamTab.style.padding = "8px 10px 6px 10px";
    charParamTab.style.textAlign = "center";
    charParamTab.style.cursor = "pointer";
    charParamTab.style.fontSize = "11px";
    charParamTab.style.fontWeight = "bold";
    charParamTab.style.background = "linear-gradient(to bottom, #3a3a3a 0%, #2e2e2e 100%)";
    charParamTab.style.borderRight = "1px solid #1a1a1a";
    charParamTab.style.borderTop = "2px solid #555";
    charParamTab.style.borderLeft = "1px solid #555";
    charParamTab.style.borderBottom = "none";
    charParamTab.style.clipPath = "polygon(8% 0%, 92% 0%, 100% 100%, 0% 100%)";
    charParamTab.style.transition = "all 0.2s ease";
    charParamTab.style.position = "relative";
    charParamTab.style.marginRight = "2px";
    charParamTab.style.whiteSpace = "nowrap";
    charParamTab.style.overflow = "hidden";
    charParamTab.style.textOverflow = "ellipsis";
    charParamTab.dataset.tab = "char-param";

    const inlineEditTab = document.createElement("div");
    inlineEditTab.textContent = "Inline Edit";
    inlineEditTab.style.flex = "1";
    inlineEditTab.style.padding = "8px 10px 6px 10px";
    inlineEditTab.style.textAlign = "center";
    inlineEditTab.style.cursor = "pointer";
    inlineEditTab.style.fontSize = "11px";
    inlineEditTab.style.fontWeight = "bold";
    inlineEditTab.style.background = "linear-gradient(to bottom, #2a2a2a 0%, #222 100%)";
    inlineEditTab.style.borderTop = "2px solid #444";
    inlineEditTab.style.borderRight = "1px solid #555";
    inlineEditTab.style.borderBottom = "none";
    inlineEditTab.style.clipPath = "polygon(8% 0%, 92% 0%, 100% 100%, 0% 100%)";
    inlineEditTab.style.transition = "all 0.2s ease";
    inlineEditTab.style.position = "relative";
    inlineEditTab.style.whiteSpace = "nowrap";
    inlineEditTab.style.overflow = "hidden";
    inlineEditTab.style.textOverflow = "ellipsis";
    inlineEditTab.dataset.tab = "inline-edit";

    tabHeaders.appendChild(charParamTab);
    tabHeaders.appendChild(inlineEditTab);

    // Tab content containers
    const charParamContent = document.createElement("div");
    charParamContent.style.display = "flex";
    charParamContent.style.flexDirection = "column";
    charParamContent.style.overflowY = "auto";
    charParamContent.style.overflowX = "hidden";
    charParamContent.style.flex = "1";
    charParamContent.dataset.content = "char-param";

    const inlineEditContent = document.createElement("div");
    inlineEditContent.style.display = "none";
    inlineEditContent.style.flexDirection = "column";
    inlineEditContent.style.overflowY = "auto";
    inlineEditContent.style.overflowX = "hidden";
    inlineEditContent.style.flex = "1";
    inlineEditContent.dataset.content = "inline-edit";

    // Tab switching logic
    const switchTab = (tabName) => {
        // Update tab headers
        if (tabName === "char-param") {
            // Active tab
            charParamTab.style.background = "linear-gradient(to bottom, #4a4a4a 0%, #3a3a3a 100%)";
            charParamTab.style.borderTop = "2px solid #0af";
            charParamTab.style.color = "#fff";
            charParamTab.style.zIndex = "10";
            // Inactive tab
            inlineEditTab.style.background = "linear-gradient(to bottom, #2a2a2a 0%, #222 100%)";
            inlineEditTab.style.borderTop = "2px solid #444";
            inlineEditTab.style.color = "#999";
            inlineEditTab.style.zIndex = "5";
            // Content
            charParamContent.style.display = "flex";
            inlineEditContent.style.display = "none";
        } else if (tabName === "inline-edit") {
            // Inactive tab
            charParamTab.style.background = "linear-gradient(to bottom, #2a2a2a 0%, #222 100%)";
            charParamTab.style.borderTop = "2px solid #444";
            charParamTab.style.color = "#999";
            charParamTab.style.zIndex = "5";
            // Active tab
            inlineEditTab.style.background = "linear-gradient(to bottom, #4a4a4a 0%, #3a3a3a 100%)";
            inlineEditTab.style.borderTop = "2px solid #0af";
            inlineEditTab.style.color = "#fff";
            inlineEditTab.style.zIndex = "10";
            // Content
            charParamContent.style.display = "none";
            inlineEditContent.style.display = "flex";
        }

        // Save active tab to state
        state.activeTab = tabName;
        state.saveToLocalStorage(storageKey);
    };

    charParamTab.addEventListener("click", () => switchTab("char-param"));
    inlineEditTab.addEventListener("click", () => switchTab("inline-edit"));

    // Assemble tab system
    tabContainer.appendChild(tabHeaders);
    tabContainer.appendChild(charParamContent);
    tabContainer.appendChild(inlineEditContent);

    // Load saved tab state (default to char-param)
    const activeTab = state.activeTab || "char-param";
    switchTab(activeTab);

    return {
        tabContainer,
        charParamContent,
        inlineEditContent,
        switchTab
    };
}
