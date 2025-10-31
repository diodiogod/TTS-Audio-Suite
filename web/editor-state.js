/**
 * 🏷️ Editor State Management
 * Handles persistent state for the multiline TTS tag editor
 */

export class EditorState {
    constructor() {
        this.text = "";
        this.history = [];
        this.historyIndex = -1;
        this.presets = {};
        this.lastCharacter = "";
        this.lastLanguage = "";
        this.lastSeed = 0;
        this.lastTemperature = 0.7;
        this.lastPauseDuration = "1s";
        this.lastParameterType = "";
        this.sidebarExpanded = true;
        this.lastCursorPosition = 0;
        this.discoveredCharacters = {};
        this.fontSize = 14;
        this.fontFamily = "monospace";
        this.sidebarWidth = 220;
        this.uiScale = 1.0;
    }

    addToHistory(text, caretPos = 0) {
        if (this.historyIndex < this.history.length - 1) {
            this.history = this.history.slice(0, this.historyIndex + 1);
        }

        this.history.push({ text, caretPos });
        this.historyIndex = this.history.length - 1;

        if (this.history.length > 100) {
            this.history = this.history.slice(-100);
            this.historyIndex = this.history.length - 1;
        }

        this.text = text;
    }

    undo() {
        if (this.historyIndex > 0) {
            this.historyIndex--;
            const entry = this.history[this.historyIndex];
            this.text = entry.text;
            return entry;
        }
        const current = this.history[this.historyIndex] || { text: this.text, caretPos: 0 };
        return current;
    }

    redo() {
        if (this.historyIndex < this.history.length - 1) {
            this.historyIndex++;
            const entry = this.history[this.historyIndex];
            this.text = entry.text;
            return entry;
        }
        const current = this.history[this.historyIndex] || { text: this.text, caretPos: 0 };
        return current;
    }

    getHistoryStatus() {
        return `${this.historyIndex + 1}/${this.history.length}`;
    }

    serialize() {
        return JSON.stringify(this);
    }

    static deserialize(json) {
        try {
            const data = JSON.parse(json);
            const state = new EditorState();
            Object.assign(state, data);
            return state;
        } catch (e) {
            console.warn("Failed to deserialize state:", e);
            return new EditorState();
        }
    }

    saveToLocalStorage(key) {
        localStorage.setItem(key, this.serialize());
    }

    static loadFromLocalStorage(key) {
        const json = localStorage.getItem(key);
        if (json) {
            return EditorState.deserialize(json);
        }
        return new EditorState();
    }
}
