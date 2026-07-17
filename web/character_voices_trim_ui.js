// Interaction design inspired by WhatDreamsCost LoadAudioUI:
// https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI
// Reimplemented for TTS Audio Suite Character Voices; no source code is copied here.

import { createWaveformRenderer } from "./character_voices_waveform.js";

const COMPACT_WIDTH = 320;

function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
}

function formatTime(seconds) {
    const value = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
    if (value < 60) {
        return `${value.toFixed(2)}s`;
    }
    const minutes = Math.floor(value / 60);
    return `${minutes}:${(value % 60).toFixed(1).padStart(4, "0")}`;
}

export function hideNativeWidget(widget) {
    if (!widget) return;
    widget.hidden = true;
    widget.options = widget.options || {};
    widget.options.hidden = true;
    widget.computeSize = () => [0, -4];
    widget.draw = () => {};
    if (widget.element) {
        widget.element.style.display = "none";
    }
}

function createNumberInput(labelText) {
    const wrapper = document.createElement("label");
    Object.assign(wrapper.style, {
        display: "flex",
        alignItems: "center",
        gap: "6px",
        color: "#aaa",
        fontSize: "11px",
    });

    const label = document.createElement("span");
    label.textContent = labelText;
    const input = document.createElement("input");
    input.type = "number";
    input.min = "0";
    input.step = "0.01";
    Object.assign(input.style, {
        width: "78px",
        padding: "3px 5px",
        color: "#ddd",
        background: "#181818",
        border: "1px solid #444",
        borderRadius: "4px",
    });
    wrapper.append(label, input);
    return { wrapper, input };
}

export function createCharacterVoiceTrimUI(onRangeChange) {
    const root = document.createElement("div");
    Object.assign(root.style, {
        position: "relative",
        display: "flex",
        flexDirection: "column",
        gap: "8px",
        width: "100%",
        minWidth: "0",
        padding: "10px",
        boxSizing: "border-box",
        borderRadius: "6px",
        color: "#ddd",
        background: "rgba(24, 24, 24, 0.94)",
        fontFamily: "sans-serif",
    });

    const header = document.createElement("div");
    Object.assign(header.style, {
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        gap: "8px",
        minWidth: "0",
    });
    const title = document.createElement("span");
    Object.assign(title.style, {
        flex: "1 1 auto",
        minWidth: "0",
        overflow: "hidden",
        color: "#aaa",
        fontSize: "11px",
        textOverflow: "ellipsis",
        whiteSpace: "nowrap",
    });
    title.textContent = "No library voice selected";
    const badge = document.createElement("span");
    Object.assign(badge.style, {
        flexShrink: "0",
        padding: "2px 6px",
        borderRadius: "999px",
        color: "#9ca3af",
        background: "rgba(156, 163, 175, 0.12)",
        fontSize: "10px",
        fontWeight: "600",
    });
    badge.textContent = "NO VOICE";
    header.append(title, badge);

    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "metadata";
    audio.classList.add("comfy-audio");
    Object.assign(audio.style, {
        width: "100%",
        minWidth: "0",
        height: "40px",
        outline: "none",
    });

    const trimArea = document.createElement("div");
    Object.assign(trimArea.style, {
        display: "flex",
        flexDirection: "column",
        minWidth: "0",
        gap: "6px",
        padding: "9px",
        border: "1px solid rgba(255,255,255,0.07)",
        borderRadius: "6px",
        background: "rgba(0,0,0,0.3)",
    });

    const ruler = document.createElement("div");
    Object.assign(ruler.style, {
        display: "flex",
        justifyContent: "space-between",
        color: "#777",
        fontSize: "10px",
        userSelect: "none",
    });
    const rulerStart = document.createElement("span");
    const rulerMiddle = document.createElement("span");
    const rulerEnd = document.createElement("span");
    ruler.append(rulerStart, rulerMiddle, rulerEnd);

    const track = document.createElement("div");
    Object.assign(track.style, {
        position: "relative",
        width: "100%",
        height: "42px",
        overflow: "hidden",
        border: "1px solid rgba(148, 163, 184, 0.12)",
        borderRadius: "6px",
        background: "linear-gradient(180deg, rgba(30, 41, 59, 0.42), rgba(2, 6, 23, 0.72))",
        boxShadow: "inset 0 1px 4px rgba(0,0,0,0.55), 0 1px 0 rgba(255,255,255,0.025)",
        cursor: "pointer",
        touchAction: "none",
        userSelect: "none",
    });
    const waveformCanvas = document.createElement("canvas");
    Object.assign(waveformCanvas.style, {
        position: "absolute",
        inset: "0",
        width: "100%",
        height: "100%",
        opacity: "0.9",
        pointerEvents: "none",
    });
    const selection = document.createElement("div");
    Object.assign(selection.style, {
        position: "absolute",
        top: "0",
        height: "100%",
        borderRadius: "3px",
        background: "rgba(14, 165, 233, 0.18)",
        boxShadow: "inset 0 1px 0 rgba(125, 211, 252, 0.24), inset 0 -1px 0 rgba(14, 165, 233, 0.18)",
        pointerEvents: "none",
    });

    const makeHandle = () => {
        const handle = document.createElement("div");
        Object.assign(handle.style, {
            position: "absolute",
            top: "0",
            width: "6px",
            height: "100%",
            borderRadius: "3px",
            background: "linear-gradient(180deg, #7dd3fc, #0ea5e9)",
            boxShadow: "0 0 0 1px rgba(2, 6, 23, 0.55), 0 0 8px rgba(14, 165, 233, 0.38)",
            transform: "translateX(-50%)",
            pointerEvents: "none",
        });
        return handle;
    };
    const startHandle = makeHandle();
    const endHandle = makeHandle();
    const playhead = document.createElement("div");
    Object.assign(playhead.style, {
        position: "absolute",
        top: "3px",
        bottom: "3px",
        width: "2px",
        borderRadius: "2px",
        background: "#e0f2fe",
        boxShadow: "0 0 0 1px rgba(14, 165, 233, 0.45), 0 0 7px rgba(125, 211, 252, 0.8)",
        opacity: "0",
        transform: "translateX(-50%)",
        pointerEvents: "none",
    });
    track.append(waveformCanvas, selection, playhead, startHandle, endHandle);
    const waveform = createWaveformRenderer(waveformCanvas);

    const controls = document.createElement("div");
    Object.assign(controls.style, {
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        gap: "8px",
        flexWrap: "wrap",
    });
    const startControl = createNumberInput("Start");
    const endControl = createNumberInput("End");
    const length = document.createElement("span");
    Object.assign(length.style, { color: "#38bdf8", fontSize: "11px", fontWeight: "600" });
    controls.append(startControl.wrapper, endControl.wrapper, length);
    const trimWarning = document.createElement("div");
    Object.assign(trimWarning.style, {
        position: "absolute",
        right: "20px",
        bottom: "3px",
        left: "20px",
        overflow: "hidden",
        color: "#fbbf24",
        fontSize: "10px",
        lineHeight: "1.25",
        opacity: "0",
        pointerEvents: "none",
        textOverflow: "ellipsis",
        transition: "opacity 120ms ease",
        visibility: "hidden",
        whiteSpace: "nowrap",
    });
    trimWarning.textContent = "Trimmed reference: make sure the transcription matches the selected speech.";
    trimWarning.title = "Review the transcription so it matches only the selected speech.";
    const expandHint = document.createElement("div");
    expandHint.setAttribute("aria-hidden", "true");
    Object.assign(expandHint.style, {
        position: "absolute",
        right: "3px",
        bottom: "3px",
        width: "12px",
        height: "12px",
        display: "none",
        opacity: "0.72",
        background: "repeating-linear-gradient(135deg, transparent 0 2px, #38bdf8 2px 3px)",
        clipPath: "polygon(100% 0, 100% 100%, 0 100%)",
        pointerEvents: "none",
    });
    trimArea.append(ruler, track, controls);
    root.append(header, audio, trimArea, trimWarning, expandHint);

    let compact = false;
    const setCompact = (value) => {
        if (compact === value) return;
        compact = value;
        root.style.gap = compact ? "0" : "8px";
        root.style.padding = compact ? "4px" : "10px";
        header.style.display = compact ? "none" : "flex";
        audio.style.height = compact ? "32px" : "40px";
        trimArea.style.display = compact ? "none" : "flex";
        trimWarning.style.display = compact ? "none" : "block";
        expandHint.style.display = compact ? "block" : "none";
        root.title = compact ? "Widen this node to show waveform and trimming controls." : "";
    };
    const resizeObserver = typeof ResizeObserver === "undefined"
        ? null
        : new ResizeObserver((entries) => {
            const width = entries[0]?.contentRect?.width || root.clientWidth;
            if (width > 0) setCompact(width < COMPACT_WIDTH);
        });
    resizeObserver?.observe(root);

    let duration = 0;
    let start = 0;
    let end = 0;
    let dragMode = null;
    let dragOffset = 0;
    let dragWidth = 0;
    let suppressInputs = false;
    let playbackFrame = null;

    const renderPlayback = () => {
        const current = clamp(Number(audio.currentTime) || 0, 0, duration);
        waveform.setPlayback(current, start, end, duration);
        playhead.style.left = `${duration ? (current / duration) * 100 : 0}%`;
        playhead.style.opacity = duration > 0 ? "1" : "0";
    };

    const playbackTick = () => {
        playbackFrame = null;
        renderPlayback();
        if (!audio.paused && !audio.ended) {
            playbackFrame = requestAnimationFrame(playbackTick);
        }
    };

    const startPlaybackAnimation = () => {
        if (playbackFrame === null) playbackFrame = requestAnimationFrame(playbackTick);
    };

    const stopPlaybackAnimation = () => {
        if (playbackFrame !== null) cancelAnimationFrame(playbackFrame);
        playbackFrame = null;
        renderPlayback();
    };

    const syncPausedPlayback = () => {
        if (audio.paused) renderPlayback();
    };

    audio.addEventListener("play", startPlaybackAnimation);
    audio.addEventListener("pause", stopPlaybackAnimation);
    audio.addEventListener("ended", stopPlaybackAnimation);
    audio.addEventListener("seeked", renderPlayback);
    audio.addEventListener("timeupdate", syncPausedPlayback);

    const render = () => {
        const safeDuration = Math.max(duration, 0);
        const startPercent = safeDuration ? (start / safeDuration) * 100 : 0;
        const endPercent = safeDuration ? (end / safeDuration) * 100 : 100;
        startHandle.style.left = `${startPercent}%`;
        endHandle.style.left = `${endPercent}%`;
        selection.style.left = `${startPercent}%`;
        selection.style.width = `${Math.max(0, endPercent - startPercent)}%`;
        rulerStart.textContent = "0.00s";
        rulerMiddle.textContent = formatTime(safeDuration / 2);
        rulerEnd.textContent = formatTime(safeDuration);
        length.textContent = `Selected: ${formatTime(Math.max(0, end - start))}`;
        suppressInputs = true;
        startControl.input.value = start.toFixed(2);
        endControl.input.value = end.toFixed(2);
        startControl.input.max = String(safeDuration || 100000);
        endControl.input.max = String(safeDuration || 100000);
        suppressInputs = false;
        renderPlayback();
    };

    const setRange = (nextStart, nextEnd, notify = false) => {
        const max = Math.max(duration, 0);
        start = clamp(Number(nextStart) || 0, 0, max);
        end = clamp(Number(nextEnd) || max, start, max);
        render();
        if (notify) onRangeChange(start, end);
    };

    const updateFromPointer = (event) => {
        if (!duration || !dragMode) return;
        const rect = track.getBoundingClientRect();
        const time = clamp((event.clientX - rect.left) / rect.width, 0, 1) * duration;
        if (dragMode === "start") {
            setRange(Math.min(time, end), end, true);
        } else if (dragMode === "end") {
            setRange(start, Math.max(time, start), true);
        } else {
            let nextStart = time - dragOffset;
            let nextEnd = nextStart + dragWidth;
            if (nextStart < 0) {
                nextStart = 0;
                nextEnd = dragWidth;
            } else if (nextEnd > duration) {
                nextEnd = duration;
                nextStart = duration - dragWidth;
            }
            setRange(nextStart, nextEnd, true);
        }
    };

    track.addEventListener("pointerdown", (event) => {
        if (!duration) return;
        const rect = track.getBoundingClientRect();
        const time = clamp((event.clientX - rect.left) / rect.width, 0, 1) * duration;
        const tolerance = (10 / Math.max(rect.width, 1)) * duration;
        if (time > start + tolerance && time < end - tolerance) {
            dragMode = "center";
            dragOffset = time - start;
            dragWidth = end - start;
        } else {
            dragMode = Math.abs(time - start) <= Math.abs(time - end) ? "start" : "end";
        }
        track.setPointerCapture(event.pointerId);
        updateFromPointer(event);
    });
    track.addEventListener("pointermove", updateFromPointer);
    const stopDragging = (event) => {
        dragMode = null;
        if (track.hasPointerCapture(event.pointerId)) track.releasePointerCapture(event.pointerId);
    };
    track.addEventListener("pointerup", stopDragging);
    track.addEventListener("pointercancel", stopDragging);

    const onNumericInput = () => {
        if (suppressInputs) return;
        setRange(Number(startControl.input.value), Number(endControl.input.value), true);
    };
    startControl.input.addEventListener("change", onNumericInput);
    endControl.input.addEventListener("change", onNumericInput);

    render();
    return {
        element: root,
        audio,
        setDuration(value) {
            duration = Number.isFinite(value) ? Math.max(0, value) : 0;
            setRange(start, end || duration, false);
        },
        setRange,
        getDuration: () => duration,
        getRange: () => ({ start, end }),
        setTitle(value) {
            title.textContent = value || "No library voice selected";
        },
        setCustomState(isCustom, sourceName = "") {
            title.textContent = isCustom && sourceName
                ? `Custom — based on ${sourceName}`
                : sourceName || "No library voice selected";
            badge.textContent = isCustom ? "CUSTOM" : sourceName ? "LIBRARY" : "NO VOICE";
            badge.style.color = isCustom ? "#fbbf24" : sourceName ? "#86efac" : "#9ca3af";
            badge.style.background = isCustom
                ? "rgba(251, 191, 36, 0.14)"
                : sourceName
                    ? "rgba(134, 239, 172, 0.12)"
                    : "rgba(156, 163, 175, 0.12)";
        },
        setTrimWarning(visible) {
            trimWarning.style.opacity = visible ? "1" : "0";
            trimWarning.style.visibility = visible ? "visible" : "hidden";
        },
        loadWaveform(url) {
            waveform.load(url);
        },
        clearWaveform() {
            waveform.clear();
            playhead.style.opacity = "0";
        },
        destroy() {
            if (playbackFrame !== null) cancelAnimationFrame(playbackFrame);
            resizeObserver?.disconnect();
            audio.removeEventListener("play", startPlaybackAnimation);
            audio.removeEventListener("pause", stopPlaybackAnimation);
            audio.removeEventListener("ended", stopPlaybackAnimation);
            audio.removeEventListener("seeked", renderPlayback);
            audio.removeEventListener("timeupdate", syncPausedPlayback);
            waveform.destroy();
        },
    };
}
