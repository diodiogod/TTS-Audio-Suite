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
    charParamTab.textContent = "Character";
    charParamTab.style.flex = "1";
    charParamTab.style.padding = "8px 10px 6px 10px";
    charParamTab.style.textAlign = "center";
    charParamTab.style.cursor = "pointer";
    charParamTab.style.fontSize = "11px";
    charParamTab.style.fontWeight = "bold";
    charParamTab.style.background = "rgba(53, 53, 52, 0.72)";
    charParamTab.style.border = "1px solid transparent";
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
    inlineEditTab.style.background = "rgba(53, 53, 52, 0.52)";
    inlineEditTab.style.border = "1px solid transparent";
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
    charParamContent.style.overflowY = "visible";
    charParamContent.style.overflowX = "hidden";
    charParamContent.style.flex = "0 0 auto";
    charParamContent.dataset.content = "char-param";

    const inlineEditContent = document.createElement("div");
    inlineEditContent.style.display = "none";
    inlineEditContent.style.flexDirection = "column";
    inlineEditContent.style.overflowY = "visible";
    inlineEditContent.style.overflowX = "hidden";
    inlineEditContent.style.flex = "0 0 auto";
    inlineEditContent.dataset.content = "inline-edit";

    // Tab switching logic
    const switchTab = (tabName) => {
        // Update tab headers
        if (tabName === "char-param") {
            // Active tab
            charParamTab.style.background = "linear-gradient(135deg, rgba(0, 225, 174, 0.24) 0%, rgba(0, 56, 41, 0.82) 100%)";
            charParamTab.style.border = "1px solid rgba(0, 225, 174, 0.24)";
            charParamTab.style.color = "#00e1ae";
            charParamTab.style.zIndex = "10";
            // Inactive tab
            inlineEditTab.style.background = "rgba(53, 53, 52, 0.52)";
            inlineEditTab.style.border = "1px solid transparent";
            inlineEditTab.style.color = "#8e8e8e";
            inlineEditTab.style.zIndex = "5";
            // Content
            charParamContent.style.display = "flex";
            inlineEditContent.style.display = "none";
        } else if (tabName === "inline-edit") {
            // Inactive tab
            charParamTab.style.background = "rgba(53, 53, 52, 0.52)";
            charParamTab.style.border = "1px solid transparent";
            charParamTab.style.color = "#8e8e8e";
            charParamTab.style.zIndex = "5";
            // Active tab
            inlineEditTab.style.background = "linear-gradient(135deg, rgba(0, 225, 174, 0.24) 0%, rgba(0, 56, 41, 0.82) 100%)";
            inlineEditTab.style.border = "1px solid rgba(0, 225, 174, 0.24)";
            inlineEditTab.style.color = "#00e1ae";
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
