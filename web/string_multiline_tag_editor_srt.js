/**
 * SRT-Specific Features for Multiline TTS Tag Editor
 * Handles SRT format detection, validation, highlighting, and per-entry operations
 */

class SRTManager {
    /**
     * Parse SRT format text into structured entries
     */
    static parseSRT(text) {
        const entries = [];
        const entryBlocks = text.trim().split(/\n\s*\n/);

        for (const block of entryBlocks) {
            if (!block.trim()) continue;

            const lines = block.trim().split("\n");
            if (lines.length < 3) continue;

            const indexMatch = lines[0].match(/^\d+$/);
            if (!indexMatch) continue;

            const timingMatch = lines[1].match(/(\d{2}):(\d{2}):(\d{2}),(\d{3})\s+-->\s+(\d{2}):(\d{2}):(\d{2}),(\d{3})/);
            if (!timingMatch) continue;

            const entry = {
                index: parseInt(lines[0]),
                startTime: lines[1].split(" --> ")[0],
                endTime: lines[1].split(" --> ")[1],
                text: lines.slice(2).join("\n"),
                originalBlock: block
            };

            entries.push(entry);
        }

        return entries;
    }

    /**
     * Detect if text is in SRT format
     */
    static isSRT(text) {
        const srtPattern = /^\d+\s*\n\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}\s*\n/m;
        return srtPattern.test(text);
    }

    /**
     * Validate SRT structure
     */
    static validateSRT(text) {
        const issues = [];

        if (!this.isSRT(text)) {
            return { valid: false, issues: ["Not valid SRT format"] };
        }

        const entries = this.parseSRT(text);
        if (entries.length === 0) {
            return { valid: false, issues: ["No valid SRT entries found"] };
        }

        // Check for timing overlaps
        for (let i = 0; i < entries.length - 1; i++) {
            const current = entries[i];
            const next = entries[i + 1];

            const currentEnd = this.timeToMs(current.endTime);
            const nextStart = this.timeToMs(next.startTime);

            if (currentEnd > nextStart) {
                issues.push(`Entry ${current.index} overlaps with entry ${next.index}`);
            }
        }

        // Check for gaps
        for (let i = 0; i < entries.length - 1; i++) {
            const current = entries[i];
            const next = entries[i + 1];

            const currentEnd = this.timeToMs(current.endTime);
            const nextStart = this.timeToMs(next.startTime);

            if (nextStart - currentEnd > 5000) {
                // More than 5 seconds gap
                issues.push(`Large gap between entry ${current.index} and ${next.index} (${((nextStart - currentEnd) / 1000).toFixed(1)}s)`);
            }
        }

        return { valid: issues.length === 0, issues };
    }

    /**
     * Convert timestamp to milliseconds
     */
    static timeToMs(timeStr) {
        const match = timeStr.match(/(\d{2}):(\d{2}):(\d{2}),(\d{3})/);
        if (!match) return 0;

        const [, hours, minutes, seconds, ms] = match;
        return parseInt(hours) * 3600000 + parseInt(minutes) * 60000 + parseInt(seconds) * 1000 + parseInt(ms);
    }

    /**
     * Convert milliseconds to SRT timestamp format
     */
    static msToTime(ms) {
        const hours = Math.floor(ms / 3600000);
        const minutes = Math.floor((ms % 3600000) / 60000);
        const seconds = Math.floor((ms % 60000) / 1000);
        const milliseconds = ms % 1000;

        return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")},${String(milliseconds).padStart(3, "0")}`;
    }

    /**
     * Get current entry index based on cursor position
     */
    static getCurrentEntryIndex(text, cursorPos) {
        const entries = this.parseSRT(text);
        let currentPos = 0;

        for (let i = 0; i < entries.length; i++) {
            const entryStart = text.indexOf(entries[i].originalBlock, currentPos);
            const entryEnd = entryStart + entries[i].originalBlock.length;

            if (cursorPos >= entryStart && cursorPos <= entryEnd) {
                return i;
            }

            currentPos = entryEnd;
        }

        return -1;
    }

    /**
     * Update a specific SRT entry
     */
    static updateEntry(text, entryIndex, newContent) {
        const entries = this.parseSRT(text);

        if (entryIndex < 0 || entryIndex >= entries.length) {
            return text;
        }

        const entry = entries[entryIndex];
        const oldBlock = entry.originalBlock;
        const newBlock = `${entry.index}\n${entry.startTime} --> ${entry.endTime}\n${newContent}`;

        return text.replace(oldBlock, newBlock);
    }

    /**
     * Apply tag to specific entry or range of entries
     */
    static applyTagToEntries(text, tag, startIndex, endIndex = -1) {
        const entries = this.parseSRT(text);
        endIndex = endIndex === -1 ? startIndex : endIndex;

        let result = text;

        for (let i = Math.max(0, startIndex); i <= Math.min(entries.length - 1, endIndex); i++) {
            const entry = entries[i];
            const newText = `${tag} ${entry.text}`;
            result = this.updateEntry(result, i, newText);
        }

        return result;
    }
}

/**
 * SRT UI Manager - handles SRT-specific sidebar controls
 */
class SRTUIManager {
    constructor(container, textarea, state) {
        this.container = container;
        this.textarea = textarea;
        this.state = state;
        this.isSRTMode = false;
        this.currentEntryIndex = -1;
        this.srtEntries = [];
    }

    /**
     * Create SRT-specific controls
     */
    createSRTControls() {
        const srtSection = document.createElement("div");
        srtSection.className = "srt-controls-section";
        srtSection.style.marginBottom = "15px";
        srtSection.style.paddingBottom = "15px";
        srtSection.style.borderBottom = "1px solid #444";

        const srtLabel = document.createElement("div");
        srtLabel.textContent = "🎬 SRT Mode";
        srtLabel.style.fontWeight = "bold";
        srtLabel.style.marginBottom = "8px";

        // Entry navigation
        const navContainer = document.createElement("div");
        navContainer.style.marginBottom = "8px";

        const prevBtn = document.createElement("button");
        prevBtn.textContent = "← Prev";
        prevBtn.style.flex = "1";
        prevBtn.style.padding = "4px";
        prevBtn.style.cursor = "pointer";
        prevBtn.style.marginRight = "3px";

        const entryIndicator = document.createElement("div");
        entryIndicator.style.fontSize = "11px";
        entryIndicator.style.textAlign = "center";
        entryIndicator.style.padding = "4px";
        entryIndicator.textContent = "Entry -/-";

        const nextBtn = document.createElement("button");
        nextBtn.textContent = "Next →";
        nextBtn.style.flex = "1";
        nextBtn.style.padding = "4px";
        nextBtn.style.cursor = "pointer";
        nextBtn.style.marginLeft = "3px";

        navContainer.appendChild(prevBtn);
        navContainer.appendChild(entryIndicator);
        navContainer.appendChild(nextBtn);

        // Validation button
        const validateBtn = document.createElement("button");
        validateBtn.textContent = "Validate SRT";
        validateBtn.style.width = "100%";
        validateBtn.style.padding = "5px";
        validateBtn.style.cursor = "pointer";
        validateBtn.style.marginBottom = "8px";

        // Per-entry tag application
        const applyTagLabel = document.createElement("div");
        applyTagLabel.textContent = "Apply to:";
        applyTagLabel.style.fontSize = "11px";
        applyTagLabel.style.marginBottom = "5px";

        const applyOptions = document.createElement("div");
        applyOptions.style.display = "flex";
        applyOptions.style.gap = "3px";
        applyOptions.style.marginBottom = "8px";

        const applyCurrentBtn = document.createElement("button");
        applyCurrentBtn.textContent = "Current";
        applyCurrentBtn.style.flex = "1";
        applyCurrentBtn.style.padding = "3px";
        applyCurrentBtn.style.fontSize = "11px";
        applyCurrentBtn.style.cursor = "pointer";

        const applyAllBtn = document.createElement("button");
        applyAllBtn.textContent = "All";
        applyAllBtn.style.flex = "1";
        applyAllBtn.style.padding = "3px";
        applyAllBtn.style.fontSize = "11px";
        applyAllBtn.style.cursor = "pointer";

        applyOptions.appendChild(applyCurrentBtn);
        applyOptions.appendChild(applyAllBtn);

        srtSection.appendChild(srtLabel);
        srtSection.appendChild(navContainer);
        srtSection.appendChild(validateBtn);
        srtSection.appendChild(applyTagLabel);
        srtSection.appendChild(applyOptions);

        // Event handlers
        prevBtn.addEventListener("click", () => {
            if (this.currentEntryIndex > 0) {
                this.jumpToEntry(this.currentEntryIndex - 1);
            }
        });

        nextBtn.addEventListener("click", () => {
            if (this.currentEntryIndex < this.srtEntries.length - 1) {
                this.jumpToEntry(this.currentEntryIndex + 1);
            }
        });

        validateBtn.addEventListener("click", () => {
            const validation = SRTManager.validateSRT(this.textarea.value);
            if (validation.valid) {
                alert("✅ SRT format is valid!");
            } else {
                alert("⚠️ SRT Issues:\n" + validation.issues.join("\n"));
            }
        });

        this.updateEntryIndicator = () => {
            if (this.currentEntryIndex >= 0 && this.currentEntryIndex < this.srtEntries.length) {
                entryIndicator.textContent = `Entry ${this.currentEntryIndex + 1}/${this.srtEntries.length}`;
            }
        };

        return srtSection;
    }

    /**
     * Jump to a specific SRT entry
     */
    jumpToEntry(index) {
        if (index < 0 || index >= this.srtEntries.length) return;

        this.currentEntryIndex = index;
        const entry = this.srtEntries[index];

        // Find and scroll to entry in textarea
        const blockStart = this.textarea.value.indexOf(entry.originalBlock);
        if (blockStart !== -1) {
            this.textarea.selectionStart = blockStart;
            this.textarea.selectionEnd = blockStart + entry.originalBlock.length;
            this.textarea.focus();

            // Scroll into view
            this.textarea.scrollTop = (blockStart / this.textarea.value.length) * this.textarea.scrollHeight;
        }

        this.updateEntryIndicator?.();
    }

    /**
     * Update SRT mode state
     */
    updateSRTMode(text) {
        const wasSRTMode = this.isSRTMode;
        this.isSRTMode = SRTManager.isSRT(text);

        if (this.isSRTMode && !wasSRTMode) {
            console.log("🎬 SRT mode activated");
        } else if (!this.isSRTMode && wasSRTMode) {
            console.log("📝 Switched to plain text mode");
        }

        if (this.isSRTMode) {
            this.srtEntries = SRTManager.parseSRT(text);
            this.currentEntryIndex = SRTManager.getCurrentEntryIndex(text, this.textarea.selectionStart);
            this.updateEntryIndicator?.();
        }
    }
}

// Export for use in main editor
export { SRTManager, SRTUIManager };
