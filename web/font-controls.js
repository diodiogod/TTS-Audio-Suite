/**
 * 🏷️ Font Controls Component
 * Font family and size selector UI component
 */

export class FontControls {
    static createFontBox(state) {
        const fontBox = document.createElement("div");
        fontBox.style.background = "#2a2a2a";
        fontBox.style.border = "1px solid #444";
        fontBox.style.borderBottom = "1px solid #333";
        fontBox.style.padding = "8px 10px";
        fontBox.style.display = "flex";
        fontBox.style.gap = "12px";
        fontBox.style.alignItems = "center";
        fontBox.style.flexShrink = "0";

        // Font family dropdown
        const fontFamilyLabel = document.createElement("div");
        fontFamilyLabel.textContent = "Font:";
        fontFamilyLabel.style.fontWeight = "bold";
        fontFamilyLabel.style.fontSize = "10px";
        fontFamilyLabel.style.color = "#bbb";
        fontFamilyLabel.style.minWidth = "35px";

        const fontFamilySelect = document.createElement("select");
        fontFamilySelect.style.padding = "4px 6px";
        fontFamilySelect.style.fontSize = "10px";
        fontFamilySelect.style.background = "#1a1a1a";
        fontFamilySelect.style.color = "#eee";
        fontFamilySelect.style.border = "1px solid #555";
        fontFamilySelect.style.borderRadius = "2px";
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
        fontSizeLabel.textContent = "Size:";
        fontSizeLabel.style.fontWeight = "bold";
        fontSizeLabel.style.fontSize = "10px";
        fontSizeLabel.style.color = "#bbb";
        fontSizeLabel.style.minWidth = "35px";

        const fontSizeInput = document.createElement("input");
        fontSizeInput.type = "number";
        fontSizeInput.min = "2";
        fontSizeInput.max = "120";
        fontSizeInput.value = state.fontSize;
        fontSizeInput.style.padding = "4px 6px";
        fontSizeInput.style.fontSize = "10px";
        fontSizeInput.style.background = "#1a1a1a";
        fontSizeInput.style.color = "#eee";
        fontSizeInput.style.border = "1px solid #555";
        fontSizeInput.style.borderRadius = "2px";
        fontSizeInput.style.width = "50px";

        const fontSizeDisplay = document.createElement("div");
        fontSizeDisplay.textContent = state.fontSize + "px";
        fontSizeDisplay.style.fontSize = "10px";
        fontSizeDisplay.style.color = "#999";
        fontSizeDisplay.style.minWidth = "30px";
        fontSizeDisplay.style.textAlign = "right";

        // Assemble font box
        fontBox.appendChild(fontFamilyLabel);
        fontBox.appendChild(fontFamilySelect);
        fontBox.appendChild(fontSizeLabel);
        fontBox.appendChild(fontSizeInput);
        fontBox.appendChild(fontSizeDisplay);

        // Set initial font family selection
        fontFamilySelect.value = state.fontFamily;

        return {
            fontBox,
            fontFamilySelect,
            fontSizeInput,
            fontSizeDisplay
        };
    }
}
