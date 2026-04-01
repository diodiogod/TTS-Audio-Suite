import { msToTime, timeToMs } from "./string_multiline_tag_editor_timing_drag.js";

const TIMING_LINE_PATTERN = /^(\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2},\d{3})$/;
const MIN_SPLIT_CUE_DURATION_MS = 400;
const SPLIT_SNAP_MS = 10;

const clamp = (value, min, max) => {
    const safeMax = Math.max(min, max);
    return Math.min(Math.max(value, min), safeMax);
};

const normalizeSRTText = (text) => (text || "").replace(/\r\n?/g, "\n");

const buildLineStarts = (text) => {
    const lines = text.split("\n");
    const lineStarts = [];
    let offset = 0;

    lines.forEach((line) => {
        lineStarts.push(offset);
        offset += line.length + 1;
    });

    return { lines, lineStarts };
};

const isCueStart = (lines, lineIndex) => {
    if (lineIndex < 0 || lineIndex >= lines.length - 1) {
        return false;
    }

    return /^\d+$/.test(lines[lineIndex].trim()) && TIMING_LINE_PATTERN.test(lines[lineIndex + 1].trim());
};

export const parseSRTCues = (text) => {
    const normalizedText = normalizeSRTText(text);
    const { lines, lineStarts } = buildLineStarts(normalizedText);
    const cues = [];

    let lineIndex = 0;
    while (lineIndex < lines.length) {
        while (lineIndex < lines.length && lines[lineIndex].trim() === "") {
            lineIndex += 1;
        }

        if (!isCueStart(lines, lineIndex)) {
            lineIndex += 1;
            continue;
        }

        const number = Number(lines[lineIndex].trim());
        const timingMatch = lines[lineIndex + 1].trim().match(TIMING_LINE_PATTERN);
        const textStartLineIndex = lineIndex + 2;
        let scanIndex = textStartLineIndex;

        while (scanIndex < lines.length) {
            if (lines[scanIndex].trim() === "") {
                let nextNonBlankIndex = scanIndex;
                while (nextNonBlankIndex < lines.length && lines[nextNonBlankIndex].trim() === "") {
                    nextNonBlankIndex += 1;
                }

                if (isCueStart(lines, nextNonBlankIndex)) {
                    break;
                }
            }

            scanIndex += 1;
        }

        let textEndLineExclusive = scanIndex;
        while (textEndLineExclusive > textStartLineIndex && lines[textEndLineExclusive - 1].trim() === "") {
            textEndLineExclusive -= 1;
        }

        const cueText = lines.slice(textStartLineIndex, textEndLineExclusive).join("\n");
        const textStartOffset = textStartLineIndex < lineStarts.length ? lineStarts[textStartLineIndex] : normalizedText.length;
        const textEndOffset = textStartOffset + cueText.length;
        const [, startText, endText] = timingMatch;

        cues.push({
            entryIndex: cues.length,
            number,
            startText,
            endText,
            startMs: timeToMs(startText),
            endMs: timeToMs(endText),
            text: cueText,
            textStartOffset,
            textEndOffset
        });

        lineIndex = scanIndex;
    }

    return {
        text: normalizedText,
        cues
    };
};

const serializeSRTCues = (cues) => cues.map((cue, cueIndex) => (
    `${cueIndex + 1}\n${msToTime(cue.startMs)} --> ${msToTime(cue.endMs)}\n${cue.text || ""}`
)).join("\n\n");

const trimCueBoundaryStart = (text) => text.replace(/^\s+/, "");
const trimCueBoundaryEnd = (text) => text.replace(/\s+$/, "");

const joinCueTexts = (firstText, secondText) => {
    const leftText = trimCueBoundaryEnd(firstText || "");
    const rightText = trimCueBoundaryStart(secondText || "");

    if (!leftText) {
        return rightText;
    }

    if (!rightText) {
        return leftText;
    }

    return `${leftText} ${rightText}`;
};

const getSplitWeight = (text, { biasTerminalPunctuation = false } = {}) => {
    const compactLength = (text || "").replace(/\s+/g, "").length;
    let weight = Math.max(1, compactLength);

    if (biasTerminalPunctuation) {
        const trimmed = text.trimEnd();
        if (/[.!?…]$/.test(trimmed)) {
            weight += 6;
        } else if (/[,;:]$/.test(trimmed)) {
            weight += 3;
        }
    }

    return weight;
};

export const buildSRTCueNumberMarkup = (numberText, entryIndex) => (
    `<span class="string-multiline-tag-editor-srt-number" data-srt-entry-index="${entryIndex}" title="Cue ${entryIndex + 1}. Alt+click to merge with next. Alt+Shift+click to merge with previous.">${numberText}</span>`
);

export const mergeSRTCues = (text, entryIndex, direction = "next") => {
    const parsed = parseSRTCues(text);
    const { cues } = parsed;

    if (cues.length < 2) {
        return {
            text: parsed.text,
            changed: false,
            reason: "not_enough_cues"
        };
    }

    const baseIndex = direction === "previous" ? entryIndex - 1 : entryIndex;
    const mergeIndex = baseIndex + 1;

    if (baseIndex < 0 || mergeIndex >= cues.length) {
        return {
            text: parsed.text,
            changed: false,
            reason: direction === "previous" ? "no_previous_cue" : "no_next_cue"
        };
    }

    const firstCue = cues[baseIndex];
    const secondCue = cues[mergeIndex];
    const leftText = trimCueBoundaryEnd(firstCue.text || "");
    const rightText = trimCueBoundaryStart(secondCue.text || "");
    const separator = leftText && rightText ? " " : "";
    const mergedText = `${leftText}${separator}${rightText}`;

    const mergedCue = {
        ...firstCue,
        endMs: secondCue.endMs,
        text: mergedText
    };

    const nextCues = [
        ...cues.slice(0, baseIndex),
        mergedCue,
        ...cues.slice(mergeIndex + 1)
    ];
    const nextText = serializeSRTCues(nextCues);
    const nextParsed = parseSRTCues(nextText);
    const mergedParsedCue = nextParsed.cues[baseIndex];
    const caretPos = mergedParsedCue
        ? mergedParsedCue.textStartOffset + leftText.length + separator.length
        : 0;

    return {
        text: nextText,
        changed: nextText !== parsed.text,
        reason: null,
        caretPos,
        mergedCueIndex: baseIndex,
        removedCueIndex: mergeIndex
    };
};

export const splitSRTCueAtCaret = (text, caretPos) => {
    const parsed = parseSRTCues(text);
    const targetCue = parsed.cues.find((cue) => caretPos > cue.textStartOffset && caretPos < cue.textEndOffset);

    if (!targetCue) {
        return {
            text: parsed.text,
            changed: false,
            reason: "caret_not_in_cue_text"
        };
    }

    const localCaretPos = caretPos - targetCue.textStartOffset;
    const rawLeftText = targetCue.text.slice(0, localCaretPos);
    const rawRightText = targetCue.text.slice(localCaretPos);
    const leftText = trimCueBoundaryEnd(rawLeftText);
    const rightText = trimCueBoundaryStart(rawRightText);

    if (!leftText.trim() || !rightText.trim()) {
        return {
            text: parsed.text,
            changed: false,
            reason: "empty_side"
        };
    }

    const durationMs = Math.max(0, targetCue.endMs - targetCue.startMs);
    if (durationMs < MIN_SPLIT_CUE_DURATION_MS * 2) {
        return {
            text: parsed.text,
            changed: false,
            reason: "cue_too_short"
        };
    }

    const leftWeight = getSplitWeight(leftText, { biasTerminalPunctuation: true });
    const rightWeight = getSplitWeight(rightText);
    const totalWeight = leftWeight + rightWeight;
    const idealSplitMs = targetCue.startMs + (durationMs * leftWeight / totalWeight);
    const snappedSplitMs = Math.round(idealSplitMs / SPLIT_SNAP_MS) * SPLIT_SNAP_MS;
    const splitMs = clamp(
        snappedSplitMs,
        targetCue.startMs + MIN_SPLIT_CUE_DURATION_MS,
        targetCue.endMs - MIN_SPLIT_CUE_DURATION_MS
    );

    const firstCue = {
        ...targetCue,
        endMs: splitMs,
        text: leftText
    };
    const secondCue = {
        ...targetCue,
        startMs: splitMs,
        text: rightText
    };

    const nextCues = [
        ...parsed.cues.slice(0, targetCue.entryIndex),
        firstCue,
        secondCue,
        ...parsed.cues.slice(targetCue.entryIndex + 1)
    ];
    const nextText = serializeSRTCues(nextCues);
    const nextParsed = parseSRTCues(nextText);
    const secondParsedCue = nextParsed.cues[targetCue.entryIndex + 1];

    return {
        text: nextText,
        changed: nextText !== parsed.text,
        reason: null,
        caretPos: secondParsedCue ? secondParsedCue.textStartOffset : 0,
        splitCueIndex: targetCue.entryIndex,
        newCueIndex: targetCue.entryIndex + 1,
        splitMs
    };
};

export class SRTCueEditController {
    constructor({
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

        this.handlePointerDown = this.handlePointerDown.bind(this);
        this.handleKeyDown = this.handleKeyDown.bind(this);

        this.editor.addEventListener("pointerdown", this.handlePointerDown, true);
        this.editor.addEventListener("keydown", this.handleKeyDown, true);
    }

    dispose() {
        this.editor.removeEventListener("pointerdown", this.handlePointerDown, true);
        this.editor.removeEventListener("keydown", this.handleKeyDown, true);
    }

    applySRTEdit(nextText, caretPos, notificationMessage) {
        const finalText = nextText ?? this.getPlainText();
        const finalCaretPos = Math.max(0, Math.min(caretPos ?? 0, finalText.length));

        this.setEditorText(finalText);
        this.state.addToHistory(finalText, finalCaretPos);
        this.state.saveToLocalStorage(this.storageKey);
        this.widget.value = finalText;
        this.widget.callback?.(finalText);
        this.historyStatus.textContent = this.state.getHistoryStatus();

        setTimeout(() => {
            this.editor.focus();
            this.setCaretPos(finalCaretPos);
        }, 0);

        if (notificationMessage) {
            this.showNotification(notificationMessage, 1800);
        }
    }

    handlePointerDown(event) {
        if (event.button !== 0 || !event.altKey) {
            return;
        }

        const cueNumber = event.target.closest(".string-multiline-tag-editor-srt-number");
        if (!cueNumber || !this.editor.contains(cueNumber)) {
            return;
        }

        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation?.();

        const entryIndex = Number(cueNumber.dataset.srtEntryIndex);
        if (!Number.isInteger(entryIndex)) {
            return;
        }

        const direction = event.shiftKey ? "previous" : "next";
        const result = mergeSRTCues(this.getPlainText(), entryIndex, direction);
        if (!result.changed) {
            const message =
                result.reason === "no_previous_cue" ? "No previous cue to merge into." :
                result.reason === "no_next_cue" ? "No next cue to merge with." :
                "Need at least two SRT cues to merge.";
            this.showNotification(message, 1800);
            return;
        }

        this.applySRTEdit(
            result.text,
            result.caretPos,
            `Merged cue ${result.removedCueIndex + 1} into cue ${result.mergedCueIndex + 1}`
        );
    }

    handleKeyDown(event) {
        const isSplitShortcut =
            event.key === "Enter" &&
            event.shiftKey &&
            !event.altKey &&
            (event.ctrlKey || event.metaKey);

        if (!isSplitShortcut || document.activeElement !== this.editor) {
            return;
        }

        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation?.();

        const result = splitSRTCueAtCaret(this.getPlainText(), this.getCaretPos());
        if (!result.changed) {
            const message =
                result.reason === "cue_too_short" ? "Cue is too short to split safely." :
                result.reason === "empty_side" ? "Move the caret so both split halves contain text." :
                "Place the caret inside subtitle text to split the cue.";
            this.showNotification(message, 1800);
            return;
        }

        this.applySRTEdit(
            result.text,
            result.caretPos,
            `Split cue ${result.splitCueIndex + 1} into cues ${result.splitCueIndex + 1} and ${result.newCueIndex + 1}`
        );
    }
}
