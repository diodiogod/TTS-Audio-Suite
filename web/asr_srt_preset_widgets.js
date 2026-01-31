import { app } from "../../scripts/app.js";

// ASR Transcribe Node UI Control
// Lock/unlock SRT fields based on preset

const PRESET_VALUES = {
    "Netflix-Standard": {
        srt_max_chars_per_line: 42,
        srt_max_lines: 2,
        srt_max_duration: 7.0,
        srt_min_duration: 0.85,
        srt_min_gap: 0.2,
        srt_max_cps: 17.0,
    },
    "Broadcast": {
        srt_max_chars_per_line: 42,
        srt_max_lines: 2,
        srt_max_duration: 6.0,
        srt_min_duration: 1.0,
        srt_min_gap: 0.6,
        srt_max_cps: 17.0,
    },
    "Fast speech": {
        srt_max_chars_per_line: 42,
        srt_max_lines: 2,
        srt_max_duration: 6.0,
        srt_min_duration: 0.8,
        srt_min_gap: 0.4,
        srt_max_cps: 20.0,
    },
    "Mobile": {
        srt_max_chars_per_line: 32,
        srt_max_lines: 2,
        srt_max_duration: 5.0,
        srt_min_duration: 1.0,
        srt_min_gap: 0.6,
        srt_max_cps: 17.0,
    },
};

const SRT_FIELDS = [
    "srt_max_chars_per_line",
    "srt_max_lines",
    "srt_max_duration",
    "srt_min_duration",
    "srt_min_gap",
    "srt_max_cps",
];

function findWidgetByName(node, name) {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
}

function applyPreset(node, preset) {
    if (!preset || preset === "Custom") {
        return;
    }
    const values = PRESET_VALUES[preset];
    if (!values) return;

    for (const field of SRT_FIELDS) {
        const w = findWidgetByName(node, field);
        if (!w) continue;
        if (values[field] !== undefined) {
            w.value = values[field];
        }
    }
}

function lockFields(node, shouldLock) {
    for (const field of SRT_FIELDS) {
        const w = findWidgetByName(node, field);
        if (!w) continue;
        w.disabled = shouldLock;
    }
}

function asrPresetHandler(node) {
    if (node.comfyClass !== "UnifiedASRTranscribeNode") {
        return;
    }

    const presetWidget = findWidgetByName(node, "srt_preset");
    if (!presetWidget) return;

    const preset = presetWidget.value;
    const shouldLock = preset !== "Custom";

    if (!node.__asrPresetCache) {
        node.__asrPresetCache = {};
    }

    if (shouldLock) {
        // Cache current custom values once before overwriting
        for (const field of SRT_FIELDS) {
            const w = findWidgetByName(node, field);
            if (!w) continue;
            if (node.__asrPresetCache[field] === undefined) {
                node.__asrPresetCache[field] = w.value;
            }
        }
        applyPreset(node, preset);
    } else {
        // Restore cached custom values
        for (const field of SRT_FIELDS) {
            const w = findWidgetByName(node, field);
            if (!w) continue;
            if (node.__asrPresetCache[field] !== undefined) {
                w.value = node.__asrPresetCache[field];
            }
        }
        node.__asrPresetCache = {};
    }
    lockFields(node, shouldLock);
}

app.registerExtension({
    name: "tts-audio-suite.asr-srt-preset.widgets",
    nodeCreated(node) {
        if (node.comfyClass !== "UnifiedASRTranscribeNode") {
            return;
        }

        asrPresetHandler(node);

        const presetWidget = findWidgetByName(node, "srt_preset");
        if (!presetWidget) return;

        let widgetValue = presetWidget.value;
        let originalDescriptor = Object.getOwnPropertyDescriptor(presetWidget, "value") ||
            Object.getOwnPropertyDescriptor(Object.getPrototypeOf(presetWidget), "value");
        if (!originalDescriptor) {
            originalDescriptor = Object.getOwnPropertyDescriptor(presetWidget.constructor.prototype, "value");
        }

        Object.defineProperty(presetWidget, "value", {
            get() {
                return originalDescriptor && originalDescriptor.get
                    ? originalDescriptor.get.call(presetWidget)
                    : widgetValue;
            },
            set(newVal) {
                if (originalDescriptor && originalDescriptor.set) {
                    originalDescriptor.set.call(presetWidget, newVal);
                } else {
                    widgetValue = newVal;
                }
                asrPresetHandler(node);
            }
        });
    }
});
