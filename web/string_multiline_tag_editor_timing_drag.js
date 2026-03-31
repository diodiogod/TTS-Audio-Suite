const TIMING_LINE_REGEX = /(\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2},\d{3})/g;
const MIN_CUE_DURATION_MS = 50;
const MAX_SRT_TIME_MS = 359999999;
const BASE_MS_PER_PIXEL = 40;
const BASE_SNAP_MS = 50;
const FINE_MS_PER_PIXEL = 10;
const FINE_SNAP_MS = 10;

const clamp = (value, min, max) => {
    const safeMax = Math.max(min, max);
    return Math.min(Math.max(value, min), safeMax);
};

const formatDeltaSeconds = (deltaMs) => {
    const sign = deltaMs >= 0 ? "+" : "-";
    const seconds = (Math.abs(deltaMs) / 1000).toFixed(2).replace(/\.?0+$/, "");
    return `${sign}${seconds}s`;
};

export const timeToMs = (timeStr) => {
    const match = timeStr.match(/^(\d{2}):(\d{2}):(\d{2}),(\d{3})$/);
    if (!match) {
        return 0;
    }

    const [, hours, minutes, seconds, milliseconds] = match;
    return (
        Number(hours) * 3600000 +
        Number(minutes) * 60000 +
        Number(seconds) * 1000 +
        Number(milliseconds)
    );
};

export const msToTime = (value) => {
    const ms = clamp(Math.round(value), 0, MAX_SRT_TIME_MS);
    const hours = Math.floor(ms / 3600000);
    const minutes = Math.floor((ms % 3600000) / 60000);
    const seconds = Math.floor((ms % 60000) / 1000);
    const milliseconds = ms % 1000;

    return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")},${String(milliseconds).padStart(3, "0")}`;
};

export const collectSRTTimingRanges = (text) => {
    const ranges = [];
    TIMING_LINE_REGEX.lastIndex = 0;

    let match;
    let entryIndex = 0;

    while ((match = TIMING_LINE_REGEX.exec(text)) !== null) {
        const [fullMatch, startTime, endTime] = match;
        ranges.push({
            entryIndex,
            fullMatch,
            startOffset: match.index,
            endOffset: match.index + fullMatch.length,
            startText: startTime,
            endText: endTime,
            startMs: timeToMs(startTime),
            endMs: timeToMs(endTime)
        });
        entryIndex += 1;
    }

    return ranges;
};

export const buildSRTTimingMarkup = (timingText, entryIndex) => {
    const match = timingText.match(/^(\d{2}:\d{2}:\d{2},\d{3})(\s+-->\s+)(\d{2}:\d{2}:\d{2},\d{3})$/);
    if (!match) {
        return `<span class="string-multiline-tag-editor-srt-timing">${timingText}</span>`;
    }

    const [, startTime, arrow, endTime] = match;
    const cueNumber = entryIndex + 1;

    return [
        `<span class="string-multiline-tag-editor-srt-timing" data-srt-entry-index="${entryIndex}" title="Cue ${cueNumber}. Drag start or end to retime. Drag the arrow to move the whole cue.">`,
        `<span class="string-multiline-tag-editor-srt-timing-part is-start" data-srt-entry-index="${entryIndex}" data-srt-drag-part="start" title="Drag to change cue ${cueNumber} start. Hold Shift to move the previous cue end by the same amount and keep the gap.">${startTime}</span>`,
        `<span class="string-multiline-tag-editor-srt-timing-part is-range" data-srt-entry-index="${entryIndex}" data-srt-drag-part="range" title="Drag to move cue ${cueNumber} without changing its duration.">${arrow}</span>`,
        `<span class="string-multiline-tag-editor-srt-timing-part is-end" data-srt-entry-index="${entryIndex}" data-srt-drag-part="end" title="Drag to change cue ${cueNumber} end. Hold Shift to move the next cue start by the same amount and keep the gap.">${endTime}</span>`,
        `</span>`
    ].join("");
};

export const applySRTTimingDrag = (text, entryIndex, part, deltaMs, { rippleNeighbor = false } = {}) => {
    const ranges = collectSRTTimingRanges(text);
    const target = ranges[entryIndex];
    if (!target) {
        return {
            text,
            changed: false,
            appliedDeltaMs: 0
        };
    }

    let newStartMs = target.startMs;
    let newEndMs = target.endMs;
    let linkedPreviousEndMs = null;
    let linkedNextStartMs = null;

    if (part === "start") {
        let minDeltaMs = -target.startMs;
        let maxDeltaMs = (target.endMs - MIN_CUE_DURATION_MS) - target.startMs;

        if (rippleNeighbor && entryIndex > 0) {
            const previous = ranges[entryIndex - 1];
            minDeltaMs = Math.max(minDeltaMs, (previous.startMs + MIN_CUE_DURATION_MS) - previous.endMs);
        }

        const appliedDeltaMs = clamp(deltaMs, minDeltaMs, maxDeltaMs);
        newStartMs = target.startMs + appliedDeltaMs;

        if (rippleNeighbor && entryIndex > 0) {
            const previous = ranges[entryIndex - 1];
            linkedPreviousEndMs = previous.endMs + appliedDeltaMs;
        }
    } else if (part === "end") {
        let minDeltaMs = (target.startMs + MIN_CUE_DURATION_MS) - target.endMs;
        let maxDeltaMs = MAX_SRT_TIME_MS - target.endMs;

        if (rippleNeighbor && entryIndex < ranges.length - 1) {
            const next = ranges[entryIndex + 1];
            maxDeltaMs = Math.min(maxDeltaMs, (next.endMs - MIN_CUE_DURATION_MS) - next.startMs);
        }

        const appliedDeltaMs = clamp(deltaMs, minDeltaMs, maxDeltaMs);
        newEndMs = target.endMs + appliedDeltaMs;

        if (rippleNeighbor && entryIndex < ranges.length - 1) {
            const next = ranges[entryIndex + 1];
            linkedNextStartMs = next.startMs + appliedDeltaMs;
        }
    } else if (part === "range") {
        const durationMs = Math.max(MIN_CUE_DURATION_MS, target.endMs - target.startMs);
        const maxStartMs = Math.max(0, MAX_SRT_TIME_MS - durationMs);
        newStartMs = clamp(target.startMs + deltaMs, 0, maxStartMs);
        newEndMs = newStartMs + durationMs;
    } else {
        return {
            text,
            changed: false,
            appliedDeltaMs: 0
        };
    }

    const replacements = new Map();
    replacements.set(entryIndex, `${msToTime(newStartMs)} --> ${msToTime(newEndMs)}`);

    if (linkedPreviousEndMs !== null && entryIndex > 0) {
        const previous = ranges[entryIndex - 1];
        replacements.set(entryIndex - 1, `${previous.startText} --> ${msToTime(linkedPreviousEndMs)}`);
    }

    if (linkedNextStartMs !== null && entryIndex < ranges.length - 1) {
        const next = ranges[entryIndex + 1];
        replacements.set(entryIndex + 1, `${msToTime(linkedNextStartMs)} --> ${next.endText}`);
    }

    let nextText = text;
    Array.from(replacements.keys())
        .sort((left, right) => right - left)
        .forEach((replacementIndex) => {
            const range = ranges[replacementIndex];
            nextText = `${nextText.slice(0, range.startOffset)}${replacements.get(replacementIndex)}${nextText.slice(range.endOffset)}`;
        });

    const appliedDeltaMs =
        part === "start" ? newStartMs - target.startMs :
        part === "end" ? newEndMs - target.endMs :
        newStartMs - target.startMs;

    return {
        text: nextText,
        changed: nextText !== text,
        appliedDeltaMs,
        linkedPrevious: linkedPreviousEndMs !== null,
        linkedNext: linkedNextStartMs !== null
    };
};

export class SRTTimingDragController {
    constructor({
        rootElement,
        editor,
        getPlainText,
        setEditorText,
        getCaretPos,
        setCaretPos,
        state,
        storageKey,
        widget,
        historyStatus,
        showNotification
    }) {
        this.rootElement = rootElement;
        this.editor = editor;
        this.getPlainText = getPlainText;
        this.setEditorText = setEditorText;
        this.getCaretPos = getCaretPos;
        this.setCaretPos = setCaretPos;
        this.state = state;
        this.storageKey = storageKey;
        this.widget = widget;
        this.historyStatus = historyStatus;
        this.showNotification = showNotification;
        this.dragState = null;

        this.handlePointerDown = this.handlePointerDown.bind(this);
        this.handlePointerMove = this.handlePointerMove.bind(this);
        this.handlePointerUp = this.handlePointerUp.bind(this);
        this.cancelDrag = this.cancelDrag.bind(this);

        this.editor.addEventListener("pointerdown", this.handlePointerDown, true);
    }

    dispose() {
        this.editor.removeEventListener("pointerdown", this.handlePointerDown, true);
        this.removeGlobalDragListeners();
        this.rootElement.classList.remove("is-dragging-srt");
        this.dragState = null;
        this.syncActiveHandle();
    }

    syncActiveHandle() {
        this.editor.querySelectorAll(".string-multiline-tag-editor-srt-timing.is-active, .string-multiline-tag-editor-srt-timing-part.is-active").forEach((element) => {
            element.classList.remove("is-active");
        });

        if (!this.dragState) {
            return;
        }

        const timingSelector = `.string-multiline-tag-editor-srt-timing[data-srt-entry-index="${this.dragState.entryIndex}"]`;
        const timingElement = this.editor.querySelector(timingSelector);
        timingElement?.classList.add("is-active");

        const handleSelector = `${timingSelector} [data-srt-drag-part="${this.dragState.part}"]`;
        this.editor.querySelector(handleSelector)?.classList.add("is-active");
    }

    handlePointerDown(event) {
        if (event.button !== 0) {
            return;
        }

        const handle = event.target.closest("[data-srt-drag-part]");
        if (!handle || !this.editor.contains(handle)) {
            return;
        }

        const entryIndex = Number(handle.dataset.srtEntryIndex);
        const part = handle.dataset.srtDragPart;
        if (!Number.isInteger(entryIndex) || !part) {
            return;
        }

        const baseText = this.getPlainText();
        const timingRanges = collectSRTTimingRanges(baseText);
        if (!timingRanges[entryIndex]) {
            return;
        }

        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation?.();

        this.dragState = {
            entryIndex,
            part,
            startX: event.clientX,
            baseText,
            lastText: baseText,
            initialCaretPos: this.getCaretPos(),
            appliedDeltaMs: 0,
            rippleNeighbor: false
        };

        this.rootElement.classList.add("is-dragging-srt");
        this.syncActiveHandle();
        this.addGlobalDragListeners();
    }

    handlePointerMove(event) {
        if (!this.dragState) {
            return;
        }

        event.preventDefault();
        event.stopPropagation();

        const msPerPixel = event.altKey ? FINE_MS_PER_PIXEL : BASE_MS_PER_PIXEL;
        const snapMs = event.altKey ? FINE_SNAP_MS : BASE_SNAP_MS;
        const rawDeltaMs = (event.clientX - this.dragState.startX) * msPerPixel;
        const snappedDeltaMs = Math.round(rawDeltaMs / snapMs) * snapMs;
        const rippleNeighbor = event.shiftKey && this.dragState.part !== "range";

        const result = applySRTTimingDrag(
            this.dragState.baseText,
            this.dragState.entryIndex,
            this.dragState.part,
            snappedDeltaMs,
            { rippleNeighbor }
        );

        this.dragState.appliedDeltaMs = result.appliedDeltaMs;
        this.dragState.rippleNeighbor = rippleNeighbor;

        if (result.text === this.dragState.lastText) {
            return;
        }

        this.setEditorText(result.text);
        this.dragState.lastText = result.text;
        this.syncActiveHandle();
    }

    handlePointerUp(event) {
        if (!this.dragState) {
            return;
        }

        event.preventDefault();
        event.stopPropagation();

        const finishedDrag = this.dragState;
        this.removeGlobalDragListeners();
        this.rootElement.classList.remove("is-dragging-srt");

        const finalText = this.getPlainText();
        const changed = finalText !== finishedDrag.baseText;

        this.dragState = null;
        this.syncActiveHandle();

        if (!changed) {
            return;
        }

        this.state.addToHistory(finalText, finishedDrag.initialCaretPos);
        this.state.saveToLocalStorage(this.storageKey);
        this.widget.value = finalText;
        this.widget.callback?.(finalText);
        this.historyStatus.textContent = this.state.getHistoryStatus();

        const actionLabel =
            finishedDrag.part === "range" ? "moved" :
            finishedDrag.part === "start" ? "start" :
            "end";
        const linkLabel =
            finishedDrag.rippleNeighbor && finishedDrag.part === "start" ? " | kept prev gap" :
            finishedDrag.rippleNeighbor && finishedDrag.part === "end" ? " | kept next gap" :
            "";

        this.showNotification(
            `Cue ${finishedDrag.entryIndex + 1} ${actionLabel} ${formatDeltaSeconds(finishedDrag.appliedDeltaMs)}${linkLabel}`,
            1800
        );

        setTimeout(() => {
            this.editor.focus();
            this.setCaretPos(finishedDrag.initialCaretPos);
        }, 0);
    }

    cancelDrag() {
        if (!this.dragState) {
            return;
        }

        this.setEditorText(this.dragState.baseText);
        this.removeGlobalDragListeners();
        this.rootElement.classList.remove("is-dragging-srt");
        this.dragState = null;
        this.syncActiveHandle();
    }

    addGlobalDragListeners() {
        window.addEventListener("pointermove", this.handlePointerMove, true);
        window.addEventListener("pointerup", this.handlePointerUp, true);
        window.addEventListener("pointercancel", this.cancelDrag, true);
        window.addEventListener("blur", this.cancelDrag, true);
    }

    removeGlobalDragListeners() {
        window.removeEventListener("pointermove", this.handlePointerMove, true);
        window.removeEventListener("pointerup", this.handlePointerUp, true);
        window.removeEventListener("pointercancel", this.cancelDrag, true);
        window.removeEventListener("blur", this.cancelDrag, true);
    }
}
