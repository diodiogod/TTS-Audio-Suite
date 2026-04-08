import { app } from "../../scripts/app.js";

const PANEL_MIN_WIDTH = 420;
const PANEL_MIN_HEIGHT = 320;
const PANEL_WIDGET_MIN_HEIGHT = 220;
const PANEL_HORIZONTAL_PADDING = 4;
const PANEL_VERTICAL_PADDING = 28;
const PANEL_DOM_WIDGET_HEIGHT = 8;

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
        ],
    },
    tts_ready_mode: { label: "TTS-Ready Mode", kind: "toggle" },
    tts_ready_paragraph_mode: { label: "Paragraph Mode", kind: "toggle" },
    srt_max_lines: { label: "srt_max_lines", kind: "adaptive-number", min: 1, max: 3, step: 1 },
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

const TOOLTIP_TEXT = {
    srt_preset: `Readability preset for subtitle building.
Choose a preset to seed the knobs below with recommended values, then edit them as needed.
If you change a preset-derived knob, the UI will switch to Custom automatically.

Examples:
• Broadcast: conservative timing, safe desktop readability
• Netflix-Standard: similar readability with longer max duration
• Fast speech: denser subtitles for rapid speech
• Mobile: shorter lines for smaller screens
• TTS-Ready: single-line cues that stop by meaning instead of display wrapping
• TTS-Ready (Paragraphs): same TTS-ready behavior, but tuned for longer paragraph-sized cues`,
    srt_mode: `How subtitle cues are grouped before final display wrapping:
• smart: rebuild cues from timed words using this node's gap, duration, CPS, and merge rules. Best default for final subtitles.
• engine_segments: keep the incoming ASR/engine segments as the base chunks, then only split later for display if needed. Use this when the source segments are already good.
• words: one timed word per cue. This is mainly for debugging alignment or inspecting bad timing data.

Use smart for almost everything.`,
    tts_ready_mode: `Build cues for downstream TTS instead of on-screen subtitles.
This disables multi-line display wrapping pressure, keeps each cue on one line, and prefers semantic stopping points over character-count stops.`,
    tts_ready_paragraph_mode: `Only used when TTS-ready is enabled.
Prefer one cue per paragraph and only split if a paragraph is genuinely too long for clean TTS playback.`,
    heuristic_language_profile: `Language profile for heuristic defaults.
Pick a language to seed the dangling-tail and incomplete-sentence lists.
Auto uses the ASR timing language when one is available, then falls back to English.
This is a seed only. You can still edit the text fields manually after selection.`,
    srt_max_chars_per_line: `Maximum characters per subtitle line.
Lower = shorter lines, more splits.
Typical values: 32 mobile, 42 desktop/broadcast.`,
    srt_max_lines: `Maximum lines per subtitle cue.
2 is the normal default. 3 is denser but harder to read.`,
    srt_max_duration: `Maximum on-screen duration for a subtitle cue in seconds.
Higher = fewer splits; too high feels laggy.`,
    srt_min_duration: `Minimum on-screen duration in seconds.
Higher = fewer flash cues; lower = tighter sync.`,
    srt_min_gap: `Pause length that forces a new subtitle cue.
Higher = more merging across short pauses.`,
    srt_max_cps: `Maximum reading speed in characters per second.
Lower = easier reading, more splits.
Higher = denser subtitles.`,
    dedupe_overlaps: `Remove overlapping duplicate phrases from bad word timing data.
Useful for alignment glitches.
Can also remove real repetitions like choruses.`,
    dedupe_window_ms: `Time window used to detect overlapping duplicates in milliseconds.
Higher = more aggressive dedupe.`,
    dedupe_min_words: `Minimum matching word count before a repeated phrase is considered a duplicate.
Higher = safer.`,
    dedupe_overlap_ratio: `Required timing overlap ratio before duplicate text is removed.
Higher = stricter dedupe.`,
    punctuation_grace_chars: `Allow a sentence-ending punctuation mark to exceed the max line length by this many chars.
Helps avoid ugly breaks right before punctuation.`,
    min_words_per_segment: `Merge very tiny subtitle segments into neighbors.
Higher = fewer one-word cues.`,
    min_segment_seconds: `Merge subtitle cues shorter than this duration.
Higher = fewer micro-cues.`,
    merge_trailing_punct_word: `Keep a trailing word with punctuation attached to the previous subtitle when possible.
Fixes splits like "beautiful / world."`,
    merge_trailing_punct_max_gap: `Maximum pause allowed when bridging that trailing punctuation word.
Higher = more aggressive bridging.`,
    merge_leading_short_phrase: `Merge a very short phrase into the previous cue when it follows punctuation.
Fixes splits like "I'm a / riddle."`,
    merge_leading_short_max_words: `Maximum word count for that short leading phrase.
Higher = more aggressive merging.`,
    merge_leading_short_max_gap: `Maximum pause allowed when merging a short leading phrase.
Higher = more merging across pauses.`,
    merge_dangling_tail: `Merge a short dangling ending into the next subtitle when it ends on a connector word.
Useful for incomplete fragments.`,
    merge_dangling_tail_max_words: `Maximum words allowed in that dangling ending.
Higher = more aggressive merging.`,
    merge_dangling_tail_max_gap: `Maximum pause allowed when merging a dangling tail.
Higher = more aggressive merging.`,
    merge_dangling_tail_allowlist: `Comma-separated connector words treated as dangling tails.
Example: a, the, to, of, and, I'm`,
    merge_leading_short_no_punct: `Merge a very short follow-up into the previous subtitle even without punctuation.
Useful for awkward mid-thought splits.`,
    merge_leading_short_no_punct_max_words: `Maximum words in that short follow-up.
Higher = more aggressive merging.`,
    merge_leading_short_no_punct_max_gap: `Maximum pause allowed when merging that follow-up.
Higher = more aggressive merging.`,
    merge_incomplete_sentence: `Merge short continuations when the previous subtitle clearly looks incomplete.
Useful for broken questions and sentence fragments.`,
    merge_incomplete_max_gap: `Maximum pause allowed when merging an incomplete sentence.
Higher = more aggressive merging.`,
    merge_incomplete_keywords: `Comma-separated keywords that suggest the previous subtitle is incomplete.
Example: what, why, how, where`,
    merge_incomplete_split_next: `If the next subtitle contains multiple sentences, split it and only merge the first sentence.
Helps keep merged subtitles readable.`,
    merge_allow_overlong: `Allow merges even if the final subtitle exceeds max duration.
Good for songs and slow speech. Disable for strict timing limits.`,
    normalize_cue_end_punctuation: `Optional subtitle-style cleanup.
When enabled, removes trailing commas, periods, semicolons, and colons at the visual end of a subtitle cue.
If a cue is cleaned this way, the next cue start is uppercased to keep the subtitle flow visually coherent.

Question marks, exclamation points, and ellipses are preserved.
This is a style transform, not grammatical truth, so it stays disabled by default.`,
};

function findWidgetByName(node, name) {
    return node.widgets ? node.widgets.find((widget) => widget.name === name) : null;
}

function setWidgetValue(widget, value) {
    if (!widget) {
        return;
    }
    widget.value = value;
    widget.callback?.(widget.value);
}

function setWidgetHeightSafe(widget, height) {
    if (!widget) {
        return;
    }
    try {
        widget.height = height;
    } catch {
        // Some ComfyUI builds expose BaseWidget.height as getter-only.
    }
    try {
        widget.computedHeight = height;
    } catch {
        // Ignore readonly implementations.
    }
}

function getWidgetTooltip(widget) {
    const candidates = [
        widget?.options?.tooltip,
        widget?.tooltip,
        TOOLTIP_TEXT[widget?.name],
    ];
    for (const candidate of candidates) {
        const text = String(candidate ?? "").trim();
        if (text) {
            return text;
        }
    }
    return "";
}

function createModifiedTag(labelText = "modified") {
    const mod = createEl("div", "srt-mod-tag");
    mod.appendChild(createEl("span", "", labelText));
    mod.appendChild(createEl("div", "dot"));
    return mod;
}

function applyTooltip(targets, tooltip) {
    if (!tooltip) {
        return;
    }
    for (const target of targets) {
        if (target) {
            target.title = tooltip;
        }
    }
}

function openSelectDropdown(select) {
    if (!select || select.disabled) {
        return;
    }
    if (typeof select.showPicker === "function") {
        try {
            select.showPicker();
            return;
        } catch {
            // Fallback below.
        }
    }
    select.focus();
    try {
        select.dispatchEvent(new MouseEvent("mousedown", { bubbles: true, cancelable: true, view: window }));
    } catch {
        select.click?.();
    }
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

    const originalCallback = widget.callback;
    widget.callback = function (...args) {
        const result = originalCallback ? originalCallback.apply(this, args) : undefined;
        for (const handler of widget.__ttsAudioSuiteValueHandlers || []) {
            try {
                handler(widget.value);
            } catch (error) {
                console.warn("SRT panel handler error:", error);
            }
        }
        return result;
    };

    widget.__ttsAudioSuiteValueBound = true;
}

function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
}

function normalizeFieldValue(fieldName, rawValue) {
    const def = FIELD_DEFS[fieldName];
    if (!def) {
        return rawValue;
    }
    if (def.kind === "toggle") {
        return Boolean(rawValue);
    }
    if (def.kind === "slider" || def.kind === "locked-number") {
        const numeric = Number(rawValue);
        if (Number.isNaN(numeric)) {
            return null;
        }
        return def.integer ? Math.round(numeric) : numeric;
    }
    return rawValue == null ? "" : String(rawValue);
}

function fieldValuesMatch(fieldName, actualValue, expectedValue) {
    const actual = normalizeFieldValue(fieldName, actualValue);
    const expected = normalizeFieldValue(fieldName, expectedValue);
    if (actual === null || expected === null) {
        return actual === expected;
    }
    if (typeof actual === "number" && typeof expected === "number") {
        return Math.abs(actual - expected) < 1e-9;
    }
    return actual === expected;
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

function applyWheelStep(widget, def, event) {
    if (!widget || widget.disabled) {
        return false;
    }
    event.preventDefault();
    const current = getWidgetNumericValue(widget, def);
    const delta = event.deltaY < 0 ? Number(def.step) : -Number(def.step);
    const next = clamp(current + delta, Number(def.min), Number(def.max));
    const normalized = def.integer ? Math.round(next) : Number(next.toFixed(6));
    setWidgetValue(widget, normalized);
    return true;
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
            height: auto;
            max-width: none;
            margin: 0;
            padding: 6px 2px 0 2px;
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
            background: rgba(27, 29, 31, 0.88);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 10px;
            padding: 8px 8px 4px 8px;
            box-sizing: border-box;
        }
        .srt-presets {
            margin-bottom: 14px;
            padding: 0 2px;
        }
        .srt-presets-top {
            display: flex;
            justify-content: flex-end;
            margin-bottom: 8px;
        }
        .srt-label {
            display: block;
            font-size: 10px;
            color: #9ca3af;
            margin-bottom: 4px;
            margin-left: 2px;
        }
        .srt-label.preset-label {
            font-size: 18px;
            font-weight: 700;
            color: #eef2f7;
            margin-bottom: 8px;
            letter-spacing: 0.01em;
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
        .srt-preset-select {
            font-size: 14px;
            font-weight: 600;
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
        .srt-core-panel > .srt-section:last-child {
            margin-bottom: 0;
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
        .srt-value-input {
            background-color: #262626;
            border: 1px solid #3a3a3a;
            border-radius: 10px;
            padding: 2px 10px;
            font-size: 12px;
            color: #ffffff;
            min-width: 56px;
            width: 56px;
            text-align: center;
            white-space: nowrap;
            flex: 0 0 auto;
            outline: none;
            box-sizing: border-box;
            appearance: textfield;
            -moz-appearance: textfield;
        }
        .srt-value-input::-webkit-outer-spin-button,
        .srt-value-input::-webkit-inner-spin-button {
            -webkit-appearance: none;
            margin: 0;
        }
        .srt-value-input.dimmed {
            background-color: #2a2a2a;
            border-color: #333;
            color: #666;
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
        .srt-range-knob {
            position: absolute;
            top: 50%;
            width: 10px;
            height: 10px;
            margin-left: -5px;
            transform: translateY(-50%);
            border-radius: 50%;
            background: #a8adb4;
            box-shadow: 0 0 2px rgba(0,0,0,0.45);
            pointer-events: none;
            z-index: 3;
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
            opacity: 0;
            height: 16px;
        }
        input[type=range].srt-range::-webkit-slider-runnable-track {
            width: 100%;
            height: 2px;
            background: transparent;
        }
        input[type=range].srt-range::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            height: 16px;
            width: 16px;
            border-radius: 50%;
            background: transparent;
            cursor: pointer;
            margin-top: -7px;
            border: none;
            box-shadow: none;
        }
        input[type=range].srt-range::-moz-range-track {
            height: 2px;
            background: transparent;
            border: none;
        }
        input[type=range].srt-range::-moz-range-thumb {
            height: 16px;
            width: 16px;
            border: none;
            border-radius: 50%;
            background: transparent;
            cursor: pointer;
            box-shadow: none;
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
    const tooltip = getWidgetTooltip(presetWidget);
    const row = createEl("div", "srt-presets");
    const top = createEl("div", "srt-presets-top");
    const chip = createEl("div", "srt-preset-chip", "Preset: Custom");
    top.appendChild(chip);
    row.appendChild(top);

    const label = createEl("label", "srt-label preset-label", "SRT Preset");
    row.appendChild(label);

    const box = createEl("div", "srt-select-row");
    const select = document.createElement("select");
    select.className = "srt-inline-select srt-preset-select";
    setSelectOptions(select, ["Custom", ...Object.keys(VISUAL_PRESET_VALUES)]);
    select.value = presetWidget ? presetWidget.value : "Custom";
    select.addEventListener("change", () => {
        applyVisualPreset(node, select.value);
        node.__srtPanelRequestRefresh?.();
        queueMicrotask(() => node.__srtPanelRequestRefresh?.());
        requestAnimationFrame(() => node.__srtPanelRequestRefresh?.());
    });
    box.addEventListener("click", (event) => {
        if (event.target instanceof Element && event.target.closest("select")) {
            return;
        }
        openSelectDropdown(select);
    });

    const caret = createEl("span", "caret", "▾");
    box.appendChild(select);
    box.appendChild(caret);
    row.appendChild(box);
    applyTooltip([row, label, box, select, caret], tooltip);

    return { element: row, select, chip };
}

function isPresetField(fieldName) {
    return Object.prototype.hasOwnProperty.call(VISUAL_PRESET_VALUES["Broadcast"], fieldName);
}

function getEffectivePresetBaseline(node) {
    const activePreset = findWidgetByName(node, "srt_preset")?.value;
    if (activePreset && activePreset !== "Custom") {
        return activePreset;
    }
    return node.__srtLastPresetBaseline || null;
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
    return !fieldValuesMatch(fieldName, widget.value, values[fieldName]);
}

function applyVisualPreset(node, preset) {
    const presetWidget = findWidgetByName(node, "srt_preset");
    node.__srtLastPresetBaseline = preset && preset !== "Custom" ? preset : null;
    setWidgetValue(presetWidget, preset);

    const values = VISUAL_PRESET_VALUES[preset];
    if (!values) {
        return;
    }

    node.__applyingSrtPreset = true;
    try {
        for (const [fieldName, value] of Object.entries(values)) {
            setWidgetValue(findWidgetByName(node, fieldName), value);
        }
    } finally {
        node.__applyingSrtPreset = false;
    }

    const ttsReady = Boolean(values.tts_ready_mode);
    const paragraphWidget = findWidgetByName(node, "tts_ready_paragraph_mode");
    const maxLinesWidget = findWidgetByName(node, "srt_max_lines");
    if (paragraphWidget) {
        paragraphWidget.disabled = !ttsReady;
    }
    if (maxLinesWidget) {
        maxLinesWidget.disabled = ttsReady;
        if (ttsReady) {
            setWidgetValue(maxLinesWidget, 1);
        }
    }

    if (node.__srtCompactPanelUi) {
        applyPresetVisibility(node, node.__srtCompactPanelUi);
    }
    node.__srtPanelRequestRefresh?.();
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
    const clampedPct = clamp(pct, 0, 100);
    fill.style.width = `${clampedPct}%`;
    return clampedPct;
}

function createSelectRow(node, fieldName, def, ui) {
    const widget = findWidgetByName(node, fieldName);
    const tooltip = getWidgetTooltip(widget);
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
        setWidgetValue(widget, select.value);
    });
    row.addEventListener("click", (event) => {
        if (event.target instanceof Element && event.target.closest("select")) {
            return;
        }
        openSelectDropdown(select);
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
    applyTooltip([wrap, row, label, select, caret], tooltip);
    if (heuristicPreview) {
        applyTooltip([heuristicPreview], tooltip);
    }

    const sync = () => {
        if (!widget) {
            return;
        }
        select.value = widget.value;
        row.classList.toggle("dimmed", Boolean(widget.disabled));
        select.disabled = Boolean(widget.disabled);
        const presetName = getEffectivePresetBaseline(node);
        const showModified = Boolean(presetName) && isPresetField(fieldName) && fieldDiffersFromPreset(node, fieldName, presetName);
        right.querySelector(".srt-mod-tag")?.remove();
        if (showModified) {
            right.prepend(createModifiedTag());
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
    const tooltip = getWidgetTooltip(widget);
    const row = createEl("div", "srt-input-row");
    const left = createEl("div", "srt-row-left");
    const label = createEl("div", "srt-row-label", def.label);
    left.appendChild(label);
    row.appendChild(left);

    const right = createEl("div", "srt-row-right");
    const tags = createEl("div");
    const switchLabel = createEl("label", "srt-switch");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = Boolean(widget?.value);
    const slider = createEl("span", "srt-switch-slider");
    switchLabel.appendChild(checkbox);
    switchLabel.appendChild(slider);
    right.appendChild(tags);
    right.appendChild(switchLabel);
    row.appendChild(right);

    checkbox.addEventListener("change", () => {
        setWidgetValue(widget, checkbox.checked);
    });

    const sync = () => {
        if (!widget) {
            return;
        }
        checkbox.checked = Boolean(widget.value);
        checkbox.disabled = Boolean(widget.disabled);
        row.classList.toggle("dimmed", Boolean(widget.disabled));
        const presetName = getEffectivePresetBaseline(node);
        const showModified = Boolean(presetName) && isPresetField(fieldName) && fieldDiffersFromPreset(node, fieldName, presetName);
        tags.replaceChildren();
        if (showModified) {
            tags.appendChild(createModifiedTag());
        }
    };

    applyTooltip([row, label, switchLabel, checkbox, slider], tooltip);

    return { row, sync };
}

function createSliderRow(node, fieldName, def) {
    const widget = findWidgetByName(node, fieldName);
    const tooltip = getWidgetTooltip(widget);
    const row = createEl("div", "srt-range-row");
    const head = createEl("div", "srt-range-head");
    const label = createEl("div", "srt-range-label", def.label);
    const tags = createEl("div");
    head.appendChild(label);
    head.appendChild(tags);
    const track = createEl("div", "srt-range-track");
    const fill = createEl("div", "srt-range-fill");
    const knob = createEl("div", "srt-range-knob");
    const main = createEl("div", "srt-range-main");
    const range = document.createElement("input");
    range.type = "range";
    range.className = "srt-range";
    range.min = String(def.min);
    range.max = String(def.uiMax ?? def.max);
    range.step = String(def.step);
    const valueInput = document.createElement("input");
    valueInput.type = "text";
    valueInput.className = "srt-value-input";
    valueInput.inputMode = def.integer ? "numeric" : "decimal";
    valueInput.value = formatNumeric(getWidgetNumericValue(widget, def), def);
    track.appendChild(fill);
    track.appendChild(knob);
    track.appendChild(range);
    row.appendChild(head);
    main.appendChild(track);
    main.appendChild(valueInput);
    row.appendChild(main);
    applyTooltip([row, label, range, valueInput, track, fill, knob], tooltip);

    const commitTypedValue = () => {
        if (!widget) {
            return;
        }
        const numeric = parseNumeric(valueInput.value, def);
        if (numeric === null) {
            valueInput.value = formatNumeric(getWidgetNumericValue(widget, def), def);
            return;
        }
        setWidgetValue(widget, def.integer ? Math.round(numeric) : numeric);
        const widgetNumericValue = getWidgetNumericValue(widget, def);
        const visibleValue = clamp(widgetNumericValue, Number(def.min), Number(def.uiMax ?? def.max));
        range.value = String(visibleValue);
        const pct = updateRangeFill(range, fill, def);
        knob.style.left = `${pct}%`;
        valueInput.value = formatNumeric(widgetNumericValue, def);
    };

    range.addEventListener("input", () => {
        if (!widget) {
            return;
        }
        const numeric = parseNumeric(range.value, def);
        if (numeric === null) {
            return;
        }
        setWidgetValue(widget, def.integer ? Math.round(numeric) : numeric);
        const pct = updateRangeFill(range, fill, def);
        knob.style.left = `${pct}%`;
        valueInput.value = formatNumeric(widget.value, def);
    });

    valueInput.addEventListener("blur", commitTypedValue);
    valueInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            event.preventDefault();
            commitTypedValue();
            valueInput.blur();
        }
    });
    const syncNumericUi = () => {
        const widgetNumericValue = getWidgetNumericValue(widget, def);
        const visibleValue = clamp(widgetNumericValue, Number(def.min), Number(def.uiMax ?? def.max));
        range.value = String(visibleValue);
        const pct = updateRangeFill(range, fill, def);
        knob.style.left = `${pct}%`;
        valueInput.value = formatNumeric(widgetNumericValue, def);
    };
    valueInput.addEventListener("wheel", (event) => {
        if (!applyWheelStep(widget, def, event)) {
            return;
        }
        syncNumericUi();
    }, { passive: false });
    for (const target of [track, fill, knob, range]) {
        target.addEventListener("wheel", (event) => {
            if (!applyWheelStep(widget, def, event)) {
                return;
            }
            syncNumericUi();
        }, { passive: false });
    }

    const sync = () => {
        if (!widget) {
            return;
        }
        const widgetNumericValue = getWidgetNumericValue(widget, def);
        const visibleValue = clamp(widgetNumericValue, Number(def.min), Number(def.uiMax ?? def.max));
        range.value = String(visibleValue);
        const pct = updateRangeFill(range, fill, def);
        knob.style.left = `${pct}%`;
        if (document.activeElement !== valueInput) {
            valueInput.value = formatNumeric(widgetNumericValue, def);
        }
        const presetName = getEffectivePresetBaseline(node);
        const showModified = Boolean(presetName) && isPresetField(fieldName) && fieldDiffersFromPreset(node, fieldName, presetName);
        tags.replaceChildren();
        if (showModified) {
            tags.appendChild(createModifiedTag());
        }
        row.classList.toggle("dimmed", Boolean(widget.disabled));
        range.disabled = Boolean(widget.disabled);
        valueInput.disabled = Boolean(widget.disabled);
        valueInput.classList.toggle("dimmed", Boolean(widget.disabled));
    };

    return { row, sync };
}

function createAdaptiveNumberRow(node, fieldName, def) {
    const widget = findWidgetByName(node, fieldName);
    const tooltip = getWidgetTooltip(widget);
    const row = createEl("div", "srt-input-row");
    const left = createEl("div", "srt-row-left");
    const help = createEl("span", "srt-row-help", "ℹ");
    const label = createEl("div", "srt-row-label", def.label);
    left.appendChild(label);
    left.appendChild(help);
    row.appendChild(left);

    const right = createEl("div", "srt-row-right");
    const lock = createEl("span", "srt-lock", "🔒");
    const tags = createEl("div");
    const valueInput = document.createElement("input");
    valueInput.type = "text";
    valueInput.className = "srt-value-input";
    valueInput.inputMode = "numeric";
    valueInput.value = formatNumeric(widget?.value ?? def.min, def);
    right.appendChild(tags);
    right.appendChild(lock);
    right.appendChild(valueInput);
    row.appendChild(right);

    const note = createEl("div", "srt-note", "Locked in TTS-ready mode");
    note.style.display = "none";
    applyTooltip([row, label, help, valueInput, lock, note], tooltip);

    const commitTypedValue = () => {
        if (!widget || widget.disabled) {
            valueInput.value = formatNumeric(widget?.value ?? def.min, def);
            return;
        }
        const numeric = parseNumeric(valueInput.value, def);
        if (numeric === null) {
            valueInput.value = formatNumeric(widget?.value ?? def.min, def);
            return;
        }
        setWidgetValue(widget, Math.round(numeric));
        valueInput.value = formatNumeric(widget.value, def);
    };

    valueInput.addEventListener("blur", commitTypedValue);
    valueInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
            event.preventDefault();
            commitTypedValue();
            valueInput.blur();
        }
    });
    valueInput.addEventListener("wheel", (event) => {
        if (!applyWheelStep(widget, def, event)) {
            return;
        }
        valueInput.value = formatNumeric(widget.value, def);
    }, { passive: false });

    const sync = () => {
        if (!widget) {
            return;
        }
        const ttsReady = Boolean(findWidgetByName(node, "tts_ready_mode")?.value);
        const paragraphMode = Boolean(findWidgetByName(node, "tts_ready_paragraph_mode")?.value);
        const locked = ttsReady;
        row.classList.toggle("dimmed", locked);
        widget.disabled = locked;
        lock.style.display = locked ? "" : "none";
        valueInput.disabled = locked;
        valueInput.classList.toggle("dimmed", locked);
        if (document.activeElement !== valueInput || locked) {
            valueInput.value = formatNumeric(widget.value, def);
        }
        note.textContent = locked
            ? (paragraphMode ? "Locked in Paragraph Mode" : "Locked in TTS-ready mode")
            : "";
        note.style.display = locked ? "block" : "none";
        const presetName = getEffectivePresetBaseline(node);
        const showModified = Boolean(presetName) && isPresetField(fieldName) && fieldDiffersFromPreset(node, fieldName, presetName);
        tags.replaceChildren();
        if (showModified) {
            tags.appendChild(createModifiedTag());
        }
    };

    return { row, sync, note };
}

function createTextRow(node, fieldName, def) {
    const widget = findWidgetByName(node, fieldName);
    const tooltip = getWidgetTooltip(widget);
    const row = createEl("div", "srt-range-row");
    const label = createEl("div", "srt-range-label", def.label);
    const textarea = document.createElement("textarea");
    textarea.className = "srt-textarea";
    textarea.rows = def.rows || 2;
    textarea.value = widget?.value || "";
    textarea.addEventListener("input", () => {
        setWidgetValue(widget, textarea.value);
    });
    row.appendChild(label);
    row.appendChild(textarea);
    applyTooltip([row, label, textarea], tooltip);

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
    if (def.kind === "adaptive-number") {
        return createAdaptiveNumberRow(node, fieldName, def);
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

    const presetBlock = buildPresetSelect(node);
    corePanel.appendChild(presetBlock.element);

    const ui = {
        panel,
        presetChip: presetBlock.chip,
        presetSelect: presetBlock.select,
        sections: {},
        controls: {},
        onLayoutChanged: null,
        requestRefresh: null,
        sectionState: {
            core: true,
            tts: true,
            timing: true,
            merge: false,
            cleanup: false,
        },
    };

    const heuristicWidget = findWidgetByName(node, "heuristic_language_profile");
    if (heuristicWidget && heuristicWidget.value === "Custom") {
        setWidgetValue(heuristicWidget, "Auto");
    }

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

    node.__srtPanelWidgetHeight = Math.max(PANEL_WIDGET_MIN_HEIGHT, (node.size?.[1] || PANEL_MIN_HEIGHT) - PANEL_VERTICAL_PADDING);
    const panelWidget = node.addDOMWidget("srt_advanced_options_panel", "div", panel, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight() {
            return PANEL_DOM_WIDGET_HEIGHT;
        },
        getHeight() {
            return PANEL_DOM_WIDGET_HEIGHT;
        },
    });
    panelWidget.computeSize = (inputWidth) => {
        const width = Array.isArray(inputWidth) ? inputWidth[0] : inputWidth;
        return [Math.max(PANEL_MIN_WIDTH, width || PANEL_MIN_WIDTH), PANEL_DOM_WIDGET_HEIGHT];
    };
    panelWidget.getHeight = () => PANEL_DOM_WIDGET_HEIGHT;
    panelWidget.computeLayoutSize = () => ({
        minWidth: PANEL_MIN_WIDTH,
        minHeight: PANEL_DOM_WIDGET_HEIGHT,
    });
    setWidgetHeightSafe(panelWidget, PANEL_DOM_WIDGET_HEIGHT);

    const spacerWidget = {
        type: "srt_panel_spacer",
        name: "srt_panel_spacer",
        value: "",
        serialize: false,
        computedHeight: node.__srtPanelWidgetHeight || PANEL_WIDGET_MIN_HEIGHT,
        height: node.__srtPanelWidgetHeight || PANEL_WIDGET_MIN_HEIGHT,
        computeSize(width) {
            const safeWidth = Array.isArray(width) ? width[0] : width;
            return [Math.max(PANEL_MIN_WIDTH, safeWidth || PANEL_MIN_WIDTH), node.__srtPanelWidgetHeight || PANEL_WIDGET_MIN_HEIGHT];
        },
        getHeight() {
            return node.__srtPanelWidgetHeight || PANEL_WIDGET_MIN_HEIGHT;
        },
        draw() {},
        mouse() {
            return false;
        },
    };

    if (node.widgets) {
        const panelWidgetIndex = node.widgets.indexOf(panelWidget);
        if (panelWidgetIndex >= 0) {
            node.widgets.splice(panelWidgetIndex + 1, 0, spacerWidget);
        } else {
            node.widgets.push(spacerWidget);
        }
    }

    node.__srtCompactPanelUi = ui;
    node.__srtCompactPanelWidget = panelWidget;
    node.__srtCompactPanelSpacerWidget = spacerWidget;

    if (typeof node.setSize === "function") {
        const width = Math.max(node.size?.[0] || PANEL_MIN_WIDTH, PANEL_MIN_WIDTH);
        const height = Math.max(node.size?.[1] || PANEL_MIN_HEIGHT, PANEL_MIN_HEIGHT);
        node.setSize([width, height]);
    }

    const syncPanelBounds = (size = node.size) => {
        panel.style.width = "100%";
        panel.style.maxWidth = "100%";
        panel.style.height = "auto";
        if (panelWidget.element) {
            panelWidget.element.style.width = "100%";
            panelWidget.element.style.maxWidth = "100%";
            panelWidget.element.style.overflow = "visible";
        }
    };

    const applyMeasuredHeight = () => {
        syncPanelBounds(node.size);
        if (panelWidget.element) {
            panelWidget.element.style.height = "auto";
            panelWidget.element.style.minHeight = "0";
        }
        const measuredHeight = Math.max(
            PANEL_WIDGET_MIN_HEIGHT,
            Math.ceil(Math.max(
                panel.scrollHeight,
                corePanel.scrollHeight,
                panel.getBoundingClientRect().height,
                corePanel.getBoundingClientRect().height,
            ) + 8)
        );
        node.__srtPanelWidgetHeight = measuredHeight;
        setWidgetHeightSafe(panelWidget, PANEL_DOM_WIDGET_HEIGHT);
        setWidgetHeightSafe(spacerWidget, measuredHeight);
        panel.style.setProperty("--comfy-widget-height", `${measuredHeight}px`);
        panel.style.setProperty("--comfy-widget-min-height", `${measuredHeight}px`);
        if (panelWidget.element) {
            panelWidget.element.style.height = `${measuredHeight}px`;
            panelWidget.element.style.minHeight = `${measuredHeight}px`;
            panelWidget.element.style.display = "block";
            panelWidget.element.style.position = "relative";
            panelWidget.element.style.boxSizing = "border-box";
            panelWidget.element.style.width = "100%";
            panelWidget.element.style.setProperty("--comfy-widget-height", `${measuredHeight}px`);
            panelWidget.element.style.setProperty("--comfy-widget-min-height", `${measuredHeight}px`);
        }
        if (typeof node.computeSize === "function" && typeof node.setSize === "function") {
            const computed = node.computeSize();
            const currentWidth = Math.max(node.size?.[0] || PANEL_MIN_WIDTH, PANEL_MIN_WIDTH);
            node.setSize([
                currentWidth,
                Math.max(PANEL_MIN_HEIGHT, computed?.[1] || PANEL_MIN_HEIGHT),
            ]);
        }
        if (node.graph && node.graph.setDirtyCanvas) {
            node.graph.setDirtyCanvas(true, true);
        }
    };

    const resizeToContent = () => {
        if (node.__srtPanelSizing || node.__srtPanelRelayoutQueued) {
            return;
        }
        node.__srtPanelRelayoutQueued = true;
        requestAnimationFrame(() => {
            try {
                node.__srtPanelSizing = true;
                applyMeasuredHeight();
                requestAnimationFrame(() => {
                    try {
                        node.__srtPanelSizing = true;
                        applyMeasuredHeight();
                    } finally {
                        node.__srtPanelSizing = false;
                    }
                });
            } finally {
                node.__srtPanelRelayoutQueued = false;
            }
        });
    };

    const refresh = () => {
        applyPresetVisibility(node, ui);
        resizeToContent();
        setTimeout(resizeToContent, 0);
    };
    ui.requestRefresh = refresh;
    node.__srtPanelRequestRefresh = refresh;
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
        setWidgetHeightSafe(panelWidget, PANEL_DOM_WIDGET_HEIGHT);
        setWidgetHeightSafe(spacerWidget, node.__srtPanelWidgetHeight || PANEL_WIDGET_MIN_HEIGHT);
        requestAnimationFrame(() => node.__srtPanelRequestRefresh?.());
        return result;
    };

    const originalOnRemoved = node.onRemoved;
    node.onRemoved = function () {
        delete node.__srtCompactPanelUi;
        delete node.__srtCompactPanelWidget;
        delete node.__srtCompactPanelSpacerWidget;
        delete node.__srtPanelRequestRefresh;
        if (this.widgets) {
            const spacerIndex = this.widgets.indexOf(spacerWidget);
            if (spacerIndex >= 0) {
                this.widgets.splice(spacerIndex, 1);
            }
        }
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
