import { app } from "../../scripts/app.js";

const PANEL_MIN_WIDTH = 420;
const PANEL_MIN_HEIGHT = 320;
const PANEL_WIDGET_MIN_HEIGHT = 220;
const PANEL_HORIZONTAL_PADDING = 16;
const PANEL_VERTICAL_PADDING = 44;

const VISUAL_PRESET_VALUES = {
    "Netflix-Standard": {
        srt_max_chars_per_line: 42,
        srt_max_lines: 2,
        srt_max_duration: 7.0,
        srt_min_duration: 0.85,
        srt_min_gap: 0.2,
        srt_max_cps: 17.0,
        tts_ready_mode: false,
        tts_ready_paragraph_mode: false,
    },
    "Broadcast": {
        srt_max_chars_per_line: 42,
        srt_max_lines: 2,
        srt_max_duration: 6.0,
        srt_min_duration: 1.0,
        srt_min_gap: 0.6,
        srt_max_cps: 17.0,
        tts_ready_mode: false,
        tts_ready_paragraph_mode: false,
    },
    "Fast speech": {
        srt_max_chars_per_line: 42,
        srt_max_lines: 2,
        srt_max_duration: 6.0,
        srt_min_duration: 0.8,
        srt_min_gap: 0.4,
        srt_max_cps: 20.0,
        tts_ready_mode: false,
        tts_ready_paragraph_mode: false,
    },
    "Mobile": {
        srt_max_chars_per_line: 32,
        srt_max_lines: 2,
        srt_max_duration: 5.0,
        srt_min_duration: 1.0,
        srt_min_gap: 0.6,
        srt_max_cps: 17.0,
        tts_ready_mode: false,
        tts_ready_paragraph_mode: false,
    },
    "TTS-Ready": {
        srt_max_chars_per_line: 240,
        srt_max_lines: 1,
        srt_max_duration: 12.0,
        srt_min_duration: 1.0,
        srt_min_gap: 0.8,
        srt_max_cps: 17.0,
        tts_ready_mode: true,
        tts_ready_paragraph_mode: false,
    },
    "TTS-Ready (Paragraphs)": {
        srt_max_chars_per_line: 320,
        srt_max_lines: 1,
        srt_max_duration: 24.0,
        srt_min_duration: 1.2,
        srt_min_gap: 1.0,
        srt_max_cps: 15.0,
        tts_ready_mode: true,
        tts_ready_paragraph_mode: true,
    },
};

const SECTION_DEFS = [
    {
        id: "core",
        title: "Core Mode",
        accent: false,
        expanded: true,
        fields: ["srt_mode", "heuristic_language_profile"],
    },
    {
        id: "tts",
        title: "TTS-Ready Settings",
        accent: true,
        expanded: true,
        fields: ["tts_ready_mode", "tts_ready_paragraph_mode", "srt_max_lines"],
    },
    {
        id: "timing",
        title: "Readability & Timing",
        accent: false,
        expanded: true,
        fields: ["srt_max_chars_per_line", "srt_max_duration", "srt_min_duration", "srt_min_gap", "srt_max_cps"],
    },
    {
        id: "merge",
        title: "Merge & Segmentation Rules",
        accent: false,
        expanded: false,
        fields: [
            "min_words_per_segment",
            "min_segment_seconds",
            "merge_trailing_punct_word",
            "merge_trailing_punct_max_gap",
            "merge_leading_short_phrase",
            "merge_leading_short_max_words",
            "merge_leading_short_max_gap",
            "merge_dangling_tail",
            "merge_dangling_tail_max_words",
            "merge_dangling_tail_max_gap",
            "merge_dangling_tail_allowlist",
            "merge_leading_short_no_punct",
            "merge_leading_short_no_punct_max_words",
            "merge_leading_short_no_punct_max_gap",
            "merge_incomplete_sentence",
            "merge_incomplete_max_gap",
            "merge_incomplete_keywords",
            "merge_incomplete_split_next",
            "merge_allow_overlong",
        ],
    },
    {
        id: "cleanup",
        title: "Cleanup & Reliability",
        accent: false,
        expanded: false,
        fields: [
            "dedupe_overlaps",
            "dedupe_window_ms",
            "dedupe_min_words",
            "dedupe_overlap_ratio",
            "punctuation_grace_chars",
            "normalize_cue_end_punctuation",
        ],
    },
];

const FIELD_DEFS = {
    srt_mode: {
        label: "srt_mode",
        kind: "select",
        options: [
            { value: "smart", label: "smart" },
            { value: "engine_segments", label: "engine_segments" },
            { value: "words", label: "words" },
        ],
    },
    heuristic_language_profile: {
        label: "heuristic_language_profile",
        kind: "select",
        options: [
            "Auto",
            "English",
            "Portuguese (Brazil)",
            "Spanish",
            "French",
            "Italian",
            "German",
            "Dutch",
            "Russian",
            "Romanian",
            "Indonesian",
            "Malay",
            "Turkish",
            "Polish",
            "Czech",
            "Swedish",
            "Danish",
            "Finnish",
            "Greek",
            "Custom",
        ],
    },
    tts_ready_mode: { label: "TTS-Ready Mode", kind: "toggle" },
    tts_ready_paragraph_mode: { label: "Paragraph Mode", kind: "toggle" },
    srt_max_lines: { label: "srt_max_lines", kind: "locked-number", min: 1, max: 3, step: 1 },
    srt_max_chars_per_line: { label: "srt_max_chars_per_line", kind: "slider", integer: true, min: 10, max: 10000, uiMax: 400, step: 1 },
    srt_max_duration: { label: "srt_max_duration", kind: "slider", integer: false, min: 0.2, max: 9999.0, uiMax: 40.0, step: 0.1 },
    srt_min_duration: { label: "srt_min_duration", kind: "slider", integer: false, min: 0.0, max: 9999.0, uiMax: 10.0, step: 0.1 },
    srt_min_gap: { label: "srt_min_gap", kind: "slider", integer: false, min: 0.0, max: 9999.0, uiMax: 5.0, step: 0.1 },
    srt_max_cps: { label: "srt_max_cps", kind: "slider", integer: false, min: 0.1, max: 9999.0, uiMax: 30.0, step: 0.5 },
    dedupe_overlaps: { label: "dedupe_overlaps", kind: "toggle" },
    dedupe_window_ms: { label: "dedupe_window_ms", kind: "slider", integer: true, min: 0, max: 10000, uiMax: 3000, step: 50 },
    dedupe_min_words: { label: "dedupe_min_words", kind: "slider", integer: true, min: 1, max: 10, step: 1 },
    dedupe_overlap_ratio: { label: "dedupe_overlap_ratio", kind: "slider", integer: false, min: 0.1, max: 1.0, step: 0.05 },
    punctuation_grace_chars: { label: "punctuation_grace_chars", kind: "slider", integer: true, min: 0, max: 100, uiMax: 30, step: 1 },
    min_words_per_segment: { label: "min_words_per_segment", kind: "slider", integer: true, min: 1, max: 10, step: 1 },
    min_segment_seconds: { label: "min_segment_seconds", kind: "slider", integer: false, min: 0.0, max: 5.0, step: 0.05 },
    merge_trailing_punct_word: { label: "merge_trailing_punct_word", kind: "toggle" },
    merge_trailing_punct_max_gap: { label: "merge_trailing_punct_max_gap", kind: "slider", integer: false, min: 0.0, max: 5.0, step: 0.05 },
    merge_leading_short_phrase: { label: "merge_leading_short_phrase", kind: "toggle" },
    merge_leading_short_max_words: { label: "merge_leading_short_max_words", kind: "slider", integer: true, min: 1, max: 6, step: 1 },
    merge_leading_short_max_gap: { label: "merge_leading_short_max_gap", kind: "slider", integer: false, min: 0.0, max: 5.0, step: 0.05 },
    merge_dangling_tail: { label: "merge_dangling_tail", kind: "toggle" },
    merge_dangling_tail_max_words: { label: "merge_dangling_tail_max_words", kind: "slider", integer: true, min: 1, max: 8, step: 1 },
    merge_dangling_tail_max_gap: { label: "merge_dangling_tail_max_gap", kind: "slider", integer: false, min: 0.0, max: 6.0, step: 0.05 },
    merge_dangling_tail_allowlist: { label: "merge_dangling_tail_allowlist", kind: "text", rows: 2 },
    merge_leading_short_no_punct: { label: "merge_leading_short_no_punct", kind: "toggle" },
    merge_leading_short_no_punct_max_words: { label: "merge_leading_short_no_punct_max_words", kind: "slider", integer: true, min: 1, max: 6, step: 1 },
    merge_leading_short_no_punct_max_gap: { label: "merge_leading_short_no_punct_max_gap", kind: "slider", integer: false, min: 0.0, max: 5.0, step: 0.05 },
    merge_incomplete_sentence: { label: "merge_incomplete_sentence", kind: "toggle" },
    merge_incomplete_max_gap: { label: "merge_incomplete_max_gap", kind: "slider", integer: false, min: 0.0, max: 5.0, step: 0.05 },
    merge_incomplete_keywords: { label: "merge_incomplete_keywords", kind: "text", rows: 2 },
    merge_incomplete_split_next: { label: "merge_incomplete_split_next", kind: "toggle" },
    merge_allow_overlong: { label: "merge_allow_overlong", kind: "toggle" },
    normalize_cue_end_punctuation: { label: "normalize_cue_end_punctuation", kind: "toggle" },
};

function findWidgetByName(node, name) {
    return node.widgets ? node.widgets.find((widget) => widget.name === name) : null;
}

function isSrtOptionsNode(node) {
    return node && node.comfyClass === "SRTAdvancedOptionsNode";
}

function hideWidget(widget) {
    if (!widget) {
        return;
    }
    if (widget.__srtOriginalType === undefined) {
        widget.__srtOriginalType = widget.type;
    }
    widget.type = "hidden";
    widget.hidden = true;
    widget.computeSize = () => [0, -4];
    widget.draw = () => {};
    if (widget.element) {
        widget.element.style.display = "none";
    }
}

function bindWidgetValueHandler(widget, onChange) {
    if (!widget) {
        return;
    }

    if (!widget.__ttsAudioSuiteValueHandlers) {
        widget.__ttsAudioSuiteValueHandlers = [];
    }
    if (!widget.__ttsAudioSuiteValueHandlers.includes(onChange)) {
        widget.__ttsAudioSuiteValueHandlers.push(onChange);
    }

    if (widget.__ttsAudioSuiteValueBound) {
        return;
    }

    let widgetValue = widget.value;
    let originalDescriptor = Object.getOwnPropertyDescriptor(widget, "value") ||
        Object.getOwnPropertyDescriptor(Object.getPrototypeOf(widget), "value");
    if (!originalDescriptor) {
        originalDescriptor = Object.getOwnPropertyDescriptor(widget.constructor.prototype, "value");
    }

    Object.defineProperty(widget, "value", {
        get() {
            return originalDescriptor && originalDescriptor.get
                ? originalDescriptor.get.call(widget)
                : widgetValue;
        },
        set(newVal) {
            if (originalDescriptor && originalDescriptor.set) {
                originalDescriptor.set.call(widget, newVal);
            } else {
                widgetValue = newVal;
            }
            for (const handler of widget.__ttsAudioSuiteValueHandlers || []) {
                try {
                    handler(newVal);
                } catch (error) {
                    console.warn("SRT panel handler error:", error);
                }
            }
        }
    });

    widget.__ttsAudioSuiteValueBound = true;
}

function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
}

function parseNumeric(raw, def) {
    const parsed = def.integer ? parseInt(raw, 10) : parseFloat(raw);
    if (Number.isNaN(parsed)) {
        return null;
    }
    return def.integer ? Math.round(clamp(parsed, def.min, def.max)) : clamp(parsed, def.min, def.max);
}

function formatNumeric(value, def) {
    if (value === undefined || value === null || value === "") {
        return def.integer ? String(def.min) : String(def.min ?? 0);
    }
    const numeric = Number(value);
    if (Number.isNaN(numeric)) {
        return def.integer ? String(def.min) : String(def.min ?? 0);
    }
    return def.integer ? String(Math.round(numeric)) : String(Number(numeric.toFixed(2)));
}

function getWidgetNumericValue(widget, def) {
    const candidates = [
        widget?.value,
        widget?.options?.default,
        widget?.defaultValue,
        def?.defaultValue,
        def?.min,
    ];
    for (const candidate of candidates) {
        const numeric = Number(candidate);
        if (!Number.isNaN(numeric)) {
            return clamp(numeric, Number(def.min), Number(def.max));
        }
    }
    return Number(def.min);
}

function createEl(tag, className, text) {
    const el = document.createElement(tag);
    if (className) {
        el.className = className;
    }
    if (text !== undefined) {
        el.textContent = text;
    }
    return el;
}

function ensureStyles(panel) {
    if (panel.__srtStylesInjected) {
        return;
    }
    panel.__srtStylesInjected = true;

    const style = document.createElement("style");
    style.textContent = `
        .srt-advanced-options-panel {
            background: transparent;
            border: 0;
            border-radius: 0;
            width: 100%;
            height: 100%;
            max-width: none;
            margin: 0;
            padding: 10px;
            box-shadow: none;
            color: #d1d5db;
            font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            font-size: 12px;
            box-sizing: border-box;
            overflow: visible;
        }
        .srt-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 10px;
            padding: 0 2px;
        }
        .srt-header-title {
            display: flex;
            align-items: center;
            gap: 8px;
            color: white;
            font-size: 14px;
            font-weight: 500;
            min-width: 0;
        }
        .srt-header-title h1 {
            margin: 0;
            font-size: 14px;
            line-height: 1.2;
            font-weight: 600;
            white-space: nowrap;
        }
        .srt-header-title .icon {
            color: #8a8a8a;
            font-size: 13px;
            line-height: 1;
        }
        .srt-preset-chip {
            background-color: #172a1e;
            color: #4ade80;
            font-size: 10px;
            padding: 2px 8px;
            border-radius: 999px;
            border: 1px solid #23402e;
            white-space: nowrap;
        }
        .srt-core-panel {
            background-color: #1e1e1e;
            border: 1px solid #2d2d2d;
            border-radius: 12px;
            padding: 14px;
            box-sizing: border-box;
        }
        .srt-presets {
            margin-bottom: 14px;
            padding: 0 2px;
        }
        .srt-label {
            display: block;
            font-size: 10px;
            color: #9ca3af;
            margin-bottom: 4px;
            margin-left: 2px;
        }
        .srt-select-row {
            background-color: #262626;
            border: 1px solid #3a3a3a;
            border-radius: 10px;
            padding: 6px 12px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            min-height: 30px;
            gap: 8px;
        }
        .srt-select-row select {
            width: 100%;
            border: 0;
            background: transparent;
            color: #f3f4f6;
            outline: none;
            font-size: 12px;
            appearance: none;
            cursor: pointer;
            color-scheme: dark;
        }
        .srt-select-row select option {
            background-color: #262626;
            color: #f3f4f6;
        }
        .srt-inline-select {
            background: transparent;
            border: 0;
            color: #fff;
            font-size: 11px;
            outline: none;
            min-width: 110px;
            color-scheme: dark;
        }
        .srt-inline-select option {
            background-color: #262626;
            color: #f3f4f6;
        }
        .srt-select-row .caret {
            color: #6b7280;
            font-size: 11px;
            flex: 0 0 auto;
        }
        .srt-section {
            background-color: #242424;
            border: 1px solid #333333;
            border-radius: 8px;
            margin-bottom: 10px;
            overflow: hidden;
        }
        .srt-section.active-blue {
            border: 1px solid #30568c;
            background-color: #242c38;
        }
        .srt-section-header {
            padding: 8px 12px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            cursor: pointer;
            user-select: none;
        }
        .srt-section-header-left {
            display: flex;
            align-items: center;
            gap: 8px;
            min-width: 0;
        }
        .srt-section-header h2 {
            font-size: 12px;
            font-weight: 500;
            color: #f3f4f6;
            margin: 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .srt-section-body {
            padding: 0 10px 10px 10px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .srt-section-body.blue {
            padding: 8px 10px 10px 10px;
        }
        .srt-note {
            font-size: 10px;
            color: #a3a3a3;
            padding: 0 2px;
            line-height: 1.35;
        }
        .srt-note.compact {
            padding: 2px 2px 0 2px;
            color: #7f8a96;
        }
        .srt-note.blue {
            color: #9cc1ff;
            background: rgba(62, 108, 208, 0.12);
            border: 1px solid rgba(103, 153, 255, 0.18);
            border-radius: 10px;
            padding: 8px 10px;
        }
        .srt-input-row {
            background-color: #1e1e1e;
            border: 1px solid #2a2a2a;
            border-radius: 20px;
            padding: 4px 12px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            min-height: 28px;
            gap: 10px;
        }
        .srt-input-row.dimmed {
            opacity: 0.45;
        }
        .srt-row-left {
            display: flex;
            align-items: center;
            gap: 6px;
            min-width: 0;
            flex: 1 1 auto;
        }
        .srt-row-label {
            font-size: 11px;
            color: #9ca3af;
            line-height: 1.2;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .srt-row-help {
            font-size: 9px;
            color: #6b7280;
            line-height: 1;
            flex: 0 0 auto;
        }
        .srt-row-right {
            display: flex;
            align-items: center;
            gap: 8px;
            min-width: 0;
            flex: 0 0 auto;
        }
        .srt-mod-tag {
            display: flex;
            align-items: center;
            gap: 4px;
            font-size: 9px;
            color: #f97316;
            line-height: 1;
            white-space: nowrap;
        }
        .srt-mod-tag .dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #f97316;
        }
        .srt-value-badge {
            background-color: #262626;
            border: 1px solid #3a3a3a;
            border-radius: 10px;
            padding: 2px 10px;
            font-size: 12px;
            color: #ffffff;
            min-width: 48px;
            text-align: center;
            white-space: nowrap;
            flex: 0 0 auto;
        }
        .srt-value-badge.dimmed {
            background-color: #2a2a2a;
            border-color: #333;
            color: #666;
            min-width: 54px;
        }
        .srt-lock {
            color: #6b7280;
            font-size: 10px;
        }
        .srt-switch {
            position: relative;
            display: inline-block;
            width: 34px;
            height: 18px;
        }
        .srt-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        .srt-switch-slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #4a4a4a;
            transition: .2s;
            border-radius: 20px;
        }
        .srt-switch-slider:before {
            position: absolute;
            content: '';
            height: 14px;
            width: 14px;
            left: 2px;
            bottom: 2px;
            background-color: white;
            transition: .2s;
            border-radius: 50%;
        }
        .srt-switch input:checked + .srt-switch-slider {
            background-color: #3b82f6;
        }
        .srt-switch input:checked + .srt-switch-slider:before {
            transform: translateX(16px);
        }
        .srt-range-row {
            background-color: #1e1e1e;
            border: 1px solid #2a2a2a;
            border-radius: 6px;
            padding: 6px 8px 8px 10px;
            margin-bottom: 4px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .srt-range-head {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 10px;
        }
        .srt-range-main {
            display: flex;
            align-items: center;
            gap: 12px;
            min-width: 0;
        }
        .srt-range-label {
            font-size: 11px;
            color: #9ca3af;
            line-height: 1.2;
            min-width: 0;
        }
        .srt-range-track {
            position: relative;
            height: 2px;
            background: #3a3a3a;
            border-radius: 1px;
            flex: 1 1 auto;
        }
        .srt-range-fill {
            position: absolute;
            height: 100%;
            background: #3d85c6;
            z-index: 1;
            border-radius: 1px;
            pointer-events: none;
            left: 0;
            top: 0;
        }
        input[type=range].srt-range {
            -webkit-appearance: none;
            appearance: none;
            width: 100%;
            background: transparent;
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            z-index: 2;
            margin: 0;
        }
        input[type=range].srt-range::-webkit-slider-runnable-track {
            width: 100%;
            height: 2px;
            background: transparent;
        }
        input[type=range].srt-range::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            height: 10px;
            width: 10px;
            border-radius: 50%;
            background: #8e8e8e;
            cursor: pointer;
            margin-top: -4px;
            border: none;
            box-shadow: 0 0 2px rgba(0,0,0,0.5);
        }
        input[type=range].srt-range::-moz-range-track {
            height: 2px;
            background: transparent;
            border: none;
        }
        input[type=range].srt-range::-moz-range-thumb {
            height: 10px;
            width: 10px;
            border: none;
            border-radius: 50%;
            background: #8e8e8e;
            cursor: pointer;
            box-shadow: 0 0 2px rgba(0,0,0,0.5);
        }
        .srt-textarea {
            width: 100%;
            min-height: 54px;
            padding: 8px 9px;
            font-size: 12px;
            line-height: 1.35;
            background: #272727;
            color: #ececec;
            border: 1px solid #434343;
            border-radius: 10px;
            outline: none;
            box-sizing: border-box;
            resize: vertical;
            font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
        }
        .srt-collapse-caret {
            color: #6b7280;
            font-size: 10px;
            width: 12px;
            text-align: center;
            flex: 0 0 auto;
        }
        .srt-section.active-blue .srt-section-header {
            background-color: #30568c;
            padding-top: 6px;
            padding-bottom: 6px;
        }
        .srt-section.active-blue .srt-section-header h2,
        .srt-section.active-blue .srt-collapse-caret {
            color: #ffffff;
        }
    `;
    panel.appendChild(style);
}

function setSelectOptions(select, options) {
    select.innerHTML = "";
    for (const option of options) {
        const opt = document.createElement("option");
        if (typeof option === "string") {
            opt.value = option;
            opt.textContent = option;
        } else {
            opt.value = option.value;
            opt.textContent = option.label;
        }
        select.appendChild(opt);
    }
}

function buildPresetSelect(node) {
    const presetWidget = findWidgetByName(node, "srt_preset");
    const row = createEl("div", "srt-presets");

    const label = createEl("label", "srt-label", "SRT Preset");
    row.appendChild(label);

    const box = createEl("div", "srt-select-row");
    const select = document.createElement("select");
    select.className = "srt-inline-select";
    setSelectOptions(select, ["Custom", ...Object.keys(VISUAL_PRESET_VALUES)]);
    select.value = presetWidget ? presetWidget.value : "Custom";
    select.addEventListener("change", () => {
        if (presetWidget) {
            presetWidget.value = select.value;
        }
    });

    const caret = createEl("span", "caret", "▾");
    box.appendChild(select);
    box.appendChild(caret);
    row.appendChild(box);

    return { element: row, select };
}

function isPresetField(fieldName) {
    return Object.prototype.hasOwnProperty.call(VISUAL_PRESET_VALUES["Broadcast"], fieldName);
}

function fieldDiffersFromPreset(node, fieldName, presetName) {
    const values = VISUAL_PRESET_VALUES[presetName];
    if (!values || !Object.prototype.hasOwnProperty.call(values, fieldName)) {
        return false;
    }
    const widget = findWidgetByName(node, fieldName);
    if (!widget) {
        return false;
    }
    return widget.value !== values[fieldName];
}

function createSection(node, ui, sectionDef) {
    const section = createEl("div", `srt-section${sectionDef.accent ? " active-blue" : ""}`);
    const header = createEl("div", "srt-section-header");
    const left = createEl("div", "srt-section-header-left");
    const caret = createEl("span", "srt-collapse-caret", ui.sectionState[sectionDef.id] ? "▾" : "▸");
    const titleWrap = createEl("div");
    const title = createEl("h2", "", sectionDef.title);
    titleWrap.appendChild(title);
    left.appendChild(caret);
    left.appendChild(titleWrap);
    header.appendChild(left);
    const body = createEl("div", `srt-section-body${sectionDef.accent ? " blue" : ""}`);
    body.style.display = ui.sectionState[sectionDef.id] ? "flex" : "none";
    header.addEventListener("click", () => {
        ui.sectionState[sectionDef.id] = !ui.sectionState[sectionDef.id];
        body.style.display = ui.sectionState[sectionDef.id] ? "flex" : "none";
        caret.textContent = ui.sectionState[sectionDef.id] ? "▾" : "▸";
        if (typeof ui.onLayoutChanged === "function") {
            ui.onLayoutChanged();
        }
        if (node.graph && node.graph.setDirtyCanvas) {
            node.graph.setDirtyCanvas(true, true);
        }
    });
    section.appendChild(header);
    section.appendChild(body);
    return { section, body, caret };
}

function summarizeHeuristicField(value, limit) {
    const parts = String(value || "")
        .split(",")
        .map((part) => part.trim())
        .filter(Boolean);
    if (!parts.length) {
        return "none";
    }
    const preview = parts.slice(0, limit).join(", ");
    return parts.length > limit ? `${preview}...` : preview;
}

function updateRangeFill(range, fill, def) {
    const min = Number(def.min);
    const max = Number(def.uiMax ?? def.max);
    const value = Number(range.value);
    const pct = max > min ? ((value - min) / (max - min)) * 100 : 0;
    fill.style.width = `${clamp(pct, 0, 100)}%`;
}

function createSelectRow(node, fieldName, def, ui) {
    const widget = findWidgetByName(node, fieldName);
    const wrap = createEl("div");
    const row = createEl("div", "srt-input-row");
    const left = createEl("div", "srt-row-left");
    const label = createEl("div", "srt-row-label", def.label);
    left.appendChild(label);
    row.appendChild(left);

    const right = createEl("div", "srt-row-right");
    const select = document.createElement("select");
    select.className = "srt-inline-select";
    setSelectOptions(select, def.options);
    select.value = widget ? widget.value : (typeof def.options[0] === "string" ? def.options[0] : def.options[0].value);
    select.addEventListener("change", () => {
        if (widget) {
            widget.value = select.value;
        }
    });
    const caret = createEl("span", "srt-row-help", "▾");
    right.appendChild(select);
    right.appendChild(caret);
    row.appendChild(right);
    wrap.appendChild(row);

    const heuristicPreview = fieldName === "heuristic_language_profile"
        ? createEl("div", "srt-note compact")
        : null;
    if (heuristicPreview) {
        wrap.appendChild(heuristicPreview);
    }

    const sync = () => {
        if (!widget) {
            return;
        }
        select.value = widget.value;
        row.classList.toggle("dimmed", Boolean(widget.disabled));
        select.disabled = Boolean(widget.disabled);
        const presetName = findWidgetByName(node, "srt_preset")?.value || "Custom";
        const showCustom = fieldName === "heuristic_language_profile" && select.value === "Custom";
        const showModified = !showCustom && presetName !== "Custom" && isPresetField(fieldName) && fieldDiffersFromPreset(node, fieldName, presetName);
        right.querySelector(".srt-mod-tag")?.remove();
        if (showCustom || showModified) {
            const mod = createEl("div", "srt-mod-tag");
            mod.appendChild(createEl("span", "", showCustom ? "CUSTOM" : "modified"));
            mod.appendChild(createEl("div", "dot"));
            right.prepend(mod);
        }
        if (heuristicPreview) {
            const allowlist = findWidgetByName(node, "merge_dangling_tail_allowlist")?.value;
            const keywords = findWidgetByName(node, "merge_incomplete_keywords")?.value;
            heuristicPreview.textContent =
                `Tail words: ${summarizeHeuristicField(allowlist, 5)} | Keywords: ${summarizeHeuristicField(keywords, 4)}`;
        }
    };

    return { row: wrap, sync };
}

function createToggleRow(node, fieldName, def, ui) {
    const widget = findWidgetByName(node, fieldName);
    const row = createEl("div", "srt-input-row");
    const left = createEl("div", "srt-row-left");
    const label = createEl("div", "srt-row-label", def.label);
    left.appendChild(label);
    row.appendChild(left);

    const right = createEl("div", "srt-row-right");
    const switchLabel = createEl("label", "srt-switch");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = Boolean(widget?.value);
    const slider = createEl("span", "srt-switch-slider");
    switchLabel.appendChild(checkbox);
    switchLabel.appendChild(slider);
    right.appendChild(switchLabel);
    row.appendChild(right);

    checkbox.addEventListener("change", () => {
        if (widget) {
            widget.value = checkbox.checked;
        }
    });

    const sync = () => {
        if (!widget) {
            return;
        }
        checkbox.checked = Boolean(widget.value);
        checkbox.disabled = Boolean(widget.disabled);
        row.classList.toggle("dimmed", Boolean(widget.disabled));
        row.title = def.label;
    };

    return { row, sync };
}

function createSliderRow(node, fieldName, def) {
    const widget = findWidgetByName(node, fieldName);
    const row = createEl("div", "srt-range-row");
    const head = createEl("div", "srt-range-head");
    const label = createEl("div", "srt-range-label", def.label);
    const tags = createEl("div");
    head.appendChild(label);
    head.appendChild(tags);
    const track = createEl("div", "srt-range-track");
    const fill = createEl("div", "srt-range-fill");
    const main = createEl("div", "srt-range-main");
    const range = document.createElement("input");
    range.type = "range";
    range.className = "srt-range";
    range.min = String(def.min);
    range.max = String(def.uiMax ?? def.max);
    range.step = String(def.step);
    const valueBadge = createEl("div", "srt-value-badge", formatNumeric(getWidgetNumericValue(widget, def), def));
    track.appendChild(fill);
    track.appendChild(range);
    row.appendChild(head);
    main.appendChild(track);
    main.appendChild(valueBadge);
    row.appendChild(main);

    range.addEventListener("input", () => {
        if (!widget) {
            return;
        }
        const numeric = parseNumeric(range.value, def);
        if (numeric === null) {
            return;
        }
        widget.value = def.integer ? Math.round(numeric) : numeric;
        updateRangeFill(range, fill, def);
        valueBadge.textContent = formatNumeric(widget.value, def);
    });

    const sync = () => {
        if (!widget) {
            return;
        }
        const widgetNumericValue = getWidgetNumericValue(widget, def);
        const visibleValue = clamp(widgetNumericValue, Number(def.min), Number(def.uiMax ?? def.max));
        range.value = String(visibleValue);
        updateRangeFill(range, fill, def);
        valueBadge.textContent = formatNumeric(widgetNumericValue, def);
        const presetName = findWidgetByName(node, "srt_preset")?.value || "Custom";
        const showModified = presetName !== "Custom" && isPresetField(fieldName) && fieldDiffersFromPreset(node, fieldName, presetName);
        tags.replaceChildren();
        if (showModified) {
            const mod = createEl("div", "srt-mod-tag");
            mod.appendChild(createEl("span", "", "modified"));
            mod.appendChild(createEl("div", "dot"));
            tags.appendChild(mod);
        }
        row.classList.toggle("dimmed", Boolean(widget.disabled));
        range.disabled = Boolean(widget.disabled);
        valueBadge.classList.toggle("dimmed", Boolean(widget.disabled));
    };

    return { row, sync };
}

function createLockedNumberRow(node, fieldName, def) {
    const widget = findWidgetByName(node, fieldName);
    const row = createEl("div", "srt-input-row dimmed");
    const left = createEl("div", "srt-row-left");
    const help = createEl("span", "srt-row-help", "ℹ");
    const label = createEl("div", "srt-row-label", def.label);
    left.appendChild(label);
    left.appendChild(help);
    row.appendChild(left);

    const right = createEl("div", "srt-row-right");
    const lock = createEl("span", "srt-lock", "🔒");
    const badge = createEl("div", "srt-value-badge dimmed", formatNumeric(widget?.value ?? 1, def));
    right.appendChild(lock);
    right.appendChild(badge);
    row.appendChild(right);

    const note = createEl("div", "srt-note", "Locked in TTS-ready mode");
    note.style.display = "none";

    const sync = () => {
        if (!widget) {
            return;
        }
        const ttsReady = Boolean(findWidgetByName(node, "tts_ready_mode")?.value);
        const paragraphMode = Boolean(findWidgetByName(node, "tts_ready_paragraph_mode")?.value);
        const locked = ttsReady;
        row.classList.toggle("dimmed", locked);
        widget.disabled = locked;
        badge.textContent = formatNumeric(widget.value, def);
        note.textContent = locked
            ? (paragraphMode ? "Locked in Paragraph Mode" : "Locked in TTS-ready mode")
            : "";
        note.style.display = locked ? "block" : "none";
    };

    return { row, sync, note };
}

function createTextRow(node, fieldName, def) {
    const widget = findWidgetByName(node, fieldName);
    const row = createEl("div", "srt-range-row");
    const label = createEl("div", "srt-range-label", def.label);
    const textarea = document.createElement("textarea");
    textarea.className = "srt-textarea";
    textarea.rows = def.rows || 2;
    textarea.value = widget?.value || "";
    textarea.addEventListener("input", () => {
        if (widget) {
            widget.value = textarea.value;
        }
    });
    row.appendChild(label);
    row.appendChild(textarea);

    const sync = () => {
        if (!widget) {
            return;
        }
        textarea.value = widget.value || "";
        textarea.disabled = Boolean(widget.disabled);
        row.classList.toggle("dimmed", Boolean(widget.disabled));
    };

    return { row, sync };
}

function createFieldControl(node, fieldName) {
    const def = FIELD_DEFS[fieldName];
    if (!def) {
        return null;
    }

    if (def.kind === "select") {
        return createSelectRow(node, fieldName, def);
    }
    if (def.kind === "toggle") {
        return createToggleRow(node, fieldName, def);
    }
    if (def.kind === "locked-number") {
        return createLockedNumberRow(node, fieldName, def);
    }
    if (def.kind === "slider") {
        return createSliderRow(node, fieldName, def);
    }
    if (def.kind === "text") {
        return createTextRow(node, fieldName, def);
    }
    return null;
}

function applyPresetVisibility(node, ui) {
    const preset = findWidgetByName(node, "srt_preset")?.value || "Custom";

    if (ui.presetChip) {
        ui.presetChip.textContent = `Preset: ${preset}`;
        ui.presetChip.style.backgroundColor = preset === "Custom" ? "#2a2a2a" : preset.startsWith("TTS-Ready") ? "#1b314b" : "#172a1e";
        ui.presetChip.style.color = preset === "Custom" ? "#d1d5db" : preset.startsWith("TTS-Ready") ? "#9cc1ff" : "#4ade80";
        ui.presetChip.style.borderColor = preset === "Custom" ? "#3a3a3a" : preset.startsWith("TTS-Ready") ? "#30568c" : "#23402e";
    }
    if (ui.presetSelect) {
        ui.presetSelect.value = preset;
    }

    for (const [fieldName, control] of Object.entries(ui.controls)) {
        if (control && typeof control.sync === "function") {
            control.sync();
        }
        if (fieldName === "srt_max_lines") {
            const note = control?.note;
            if (note && note.style) {
                const locked = Boolean(findWidgetByName(node, "tts_ready_mode")?.value);
                note.style.display = locked ? "block" : "none";
            }
        }
    }

    for (const sectionDef of SECTION_DEFS) {
        const sectionUi = ui.sections[sectionDef.id];
        if (!sectionUi) {
            continue;
        }
        const expanded = Boolean(ui.sectionState[sectionDef.id]);
        sectionUi.body.style.display = expanded ? "flex" : "none";
        sectionUi.caret.textContent = expanded ? "▾" : "▸";
    }
}

function createPanel(node) {
    if (!isSrtOptionsNode(node)) {
        return false;
    }

    const nodeWidgets = node.widgets || [];
    const relevantWidgets = nodeWidgets.filter((widget) => widget.name === "srt_preset" || Object.prototype.hasOwnProperty.call(FIELD_DEFS, widget.name));
    if (!relevantWidgets.length) {
        return false;
    }
    if (typeof node.addDOMWidget !== "function") {
        return false;
    }

    if (node.__srtCompactPanelUi) {
        return true;
    }

    for (const widget of nodeWidgets) {
        hideWidget(widget);
    }

    const panel = createEl("div", "srt-advanced-options-panel");
    ensureStyles(panel);
    const corePanel = createEl("div", "srt-core-panel");
    panel.appendChild(corePanel);

    const header = createEl("header", "srt-header");
    const title = createEl("div", "srt-header-title");
    title.appendChild(createEl("span", "icon", "🔧"));
    title.appendChild(createEl("h1", "", "SRT Advanced Options"));
    header.appendChild(title);

    const presetChip = createEl("div", "srt-preset-chip", "Preset: Custom");
    header.appendChild(presetChip);
    corePanel.appendChild(header);

    const presetBlock = buildPresetSelect(node);
    corePanel.appendChild(presetBlock.element);

    const ui = {
        panel,
        presetChip,
        presetSelect: presetBlock.select,
        sections: {},
        controls: {},
        onLayoutChanged: null,
        sectionState: {
            core: true,
            tts: true,
            timing: true,
            merge: false,
            cleanup: false,
        },
    };

    for (const sectionDef of SECTION_DEFS) {
        const sectionUi = createSection(node, ui, sectionDef);
        ui.sections[sectionDef.id] = sectionUi;
        corePanel.appendChild(sectionUi.section);
    }

    for (const sectionDef of SECTION_DEFS) {
        const sectionUi = ui.sections[sectionDef.id];
        if (!sectionUi) {
            continue;
        }
        for (const fieldName of sectionDef.fields) {
            const control = createFieldControl(node, fieldName);
            if (!control) {
                continue;
            }
            ui.controls[fieldName] = control;

            sectionUi.body.appendChild(control.row);
            if (fieldName === "srt_max_lines" && control.note) {
                sectionUi.body.appendChild(control.note);
            }
        }
    }

    const panelWidget = node.addDOMWidget("srt_advanced_options_panel", "div", panel, {
        serialize: false,
        hideOnZoom: false,
    });
    node.__srtPanelWidgetHeight = Math.max(PANEL_WIDGET_MIN_HEIGHT, (node.size?.[1] || PANEL_MIN_HEIGHT) - PANEL_VERTICAL_PADDING);
    panelWidget.computeSize = (width) => [Math.max(PANEL_MIN_WIDTH, width || PANEL_MIN_WIDTH), node.__srtPanelWidgetHeight || PANEL_WIDGET_MIN_HEIGHT];
    panelWidget.getHeight = () => node.__srtPanelWidgetHeight || PANEL_WIDGET_MIN_HEIGHT;
    panelWidget.height = node.__srtPanelWidgetHeight || PANEL_WIDGET_MIN_HEIGHT;

    node.__srtCompactPanelUi = ui;
    node.__srtCompactPanelWidget = panelWidget;
    if (typeof node.setSize === "function") {
        const width = Math.max(node.size?.[0] || PANEL_MIN_WIDTH, PANEL_MIN_WIDTH);
        const height = Math.max(node.size?.[1] || PANEL_MIN_HEIGHT, PANEL_MIN_HEIGHT);
        node.setSize([width, height]);
    }

    const syncPanelBounds = (size = node.size) => {
        const width = Math.max((size?.[0] || PANEL_MIN_WIDTH) - PANEL_HORIZONTAL_PADDING, 260);
        panel.style.width = `${width}px`;
        panel.style.height = "auto";
    };

    const resizeToContent = () => {
        if (node.__srtPanelSizing) {
            return;
        }
        node.__srtPanelSizing = true;
        requestAnimationFrame(() => {
            try {
                syncPanelBounds(node.size);
                const contentHeight = Math.max(Math.ceil(panel.scrollHeight), PANEL_WIDGET_MIN_HEIGHT);
                node.__srtPanelWidgetHeight = contentHeight;
                panelWidget.height = contentHeight;
                panelWidget.computedHeight = contentHeight;
                if (panelWidget.element) {
                    panelWidget.element.style.height = `${contentHeight}px`;
                    panelWidget.element.style.minHeight = `${contentHeight}px`;
                    panelWidget.element.style.display = "block";
                    panelWidget.element.style.position = "relative";
                    panelWidget.element.style.boxSizing = "border-box";
                    panelWidget.element.style.width = "100%";
                }
                if (typeof node.computeSize === "function" && typeof node.setSize === "function") {
                    const computedSize = node.computeSize();
                    const targetWidth = Math.max(computedSize?.[0] || 0, node.size?.[0] || 0, PANEL_MIN_WIDTH);
                    const targetHeight = Math.max(computedSize?.[1] || 0, PANEL_MIN_HEIGHT, contentHeight + PANEL_VERTICAL_PADDING);
                    node.setSize([targetWidth, targetHeight]);
                }
                if (node.graph && node.graph.setDirtyCanvas) {
                    node.graph.setDirtyCanvas(true, true);
                }
            } finally {
                node.__srtPanelSizing = false;
            }
        });
    };

    const refresh = () => {
        applyPresetVisibility(node, ui);
        resizeToContent();
        setTimeout(resizeToContent, 0);
    };
    ui.onLayoutChanged = resizeToContent;

    const originalOnConfigure = node.onConfigure?.bind(node);
    node.onConfigure = function (info) {
        const result = originalOnConfigure ? originalOnConfigure(info) : undefined;
        syncPanelBounds(this.size);
        refresh();
        setTimeout(refresh, 0);
        setTimeout(refresh, 50);
        return result;
    };

    const originalOnResize = node.onResize?.bind(node);
    node.onResize = function (size) {
        const result = originalOnResize ? originalOnResize(size) : undefined;
        syncPanelBounds(size || this.size);
        panelWidget.height = node.__srtPanelWidgetHeight || PANEL_WIDGET_MIN_HEIGHT;
        return result;
    };

    const originalOnRemoved = node.onRemoved;
    node.onRemoved = function () {
        delete node.__srtCompactPanelUi;
        delete node.__srtCompactPanelWidget;
        if (originalOnRemoved) {
            return originalOnRemoved.apply(this, arguments);
        }
    };

    bindWidgetValueHandler(findWidgetByName(node, "srt_preset"), refresh);
    bindWidgetValueHandler(findWidgetByName(node, "heuristic_language_profile"), refresh);
    bindWidgetValueHandler(findWidgetByName(node, "tts_ready_mode"), refresh);
    bindWidgetValueHandler(findWidgetByName(node, "tts_ready_paragraph_mode"), refresh);

    for (const fieldName of Object.keys(FIELD_DEFS)) {
        bindWidgetValueHandler(findWidgetByName(node, fieldName), refresh);
    }

    syncPanelBounds(node.size);
    refresh();
    setTimeout(refresh, 0);
    if (app.graph && app.graph.setDirtyCanvas) {
        app.graph.setDirtyCanvas(true, true);
    }
    return true;
}

export function setupSrtAdvancedOptionsPanel(node) {
    return createPanel(node);
}
