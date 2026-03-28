import { app } from "../../scripts/app.js";

// SRT Advanced Options UI Control
// Presets lock core timing fields.
// Heuristic language profiles seed editable text fields and restore custom values.

const PRESET_VALUES = {
    "Netflix-Standard": {
        srt_max_chars_per_line: 42,
        srt_max_lines: 2,
        srt_max_duration: 7.0,
        srt_min_duration: 0.85,
        srt_min_gap: 0.2,
        srt_max_cps: 17.0,
        tts_ready_mode: false,
    },
    "Broadcast": {
        srt_max_chars_per_line: 42,
        srt_max_lines: 2,
        srt_max_duration: 6.0,
        srt_min_duration: 1.0,
        srt_min_gap: 0.6,
        srt_max_cps: 17.0,
        tts_ready_mode: false,
    },
    "Fast speech": {
        srt_max_chars_per_line: 42,
        srt_max_lines: 2,
        srt_max_duration: 6.0,
        srt_min_duration: 0.8,
        srt_min_gap: 0.4,
        srt_max_cps: 20.0,
        tts_ready_mode: false,
    },
    "Mobile": {
        srt_max_chars_per_line: 32,
        srt_max_lines: 2,
        srt_max_duration: 5.0,
        srt_min_duration: 1.0,
        srt_min_gap: 0.6,
        srt_max_cps: 17.0,
        tts_ready_mode: false,
    },
    "TTS-Ready": {
        srt_max_chars_per_line: 240,
        srt_max_lines: 1,
        srt_max_duration: 12.0,
        srt_min_duration: 1.0,
        srt_min_gap: 0.8,
        srt_max_cps: 17.0,
        tts_ready_mode: true,
    },
};

const HEURISTIC_PROFILE_VALUES = {
    "Auto": {
        merge_dangling_tail_allowlist: "a,an,the,to,of,and,or,im,i'm,you,you're,we,they,he,she,it",
        merge_incomplete_keywords: "what,why,how,where,who,which,when",
    },
    "English": {
        merge_dangling_tail_allowlist: "a,an,the,to,of,and,or,im,i'm,you,you're,we,they,he,she,it",
        merge_incomplete_keywords: "what,why,how,where,who,which,when",
    },
    "Portuguese (Brazil)": {
        merge_dangling_tail_allowlist: "o,a,os,as,um,uma,uns,umas,de,do,da,dos,das,e,ou,mas,se,que,como,quando,onde,quem,para,pra,por,com,sem,em,no,na,nos,nas,ao,aos,pelo,pela,pelos,pelas",
        merge_incomplete_keywords: "o que,por que,porque,como,onde,quem,qual,quais,quando",
    },
};

const SRT_FIELDS = [
    "srt_max_chars_per_line",
    "srt_max_lines",
    "srt_max_duration",
    "srt_min_duration",
    "srt_min_gap",
    "srt_max_cps",
    "tts_ready_mode",
];

const HEURISTIC_FIELDS = [
    "merge_dangling_tail_allowlist",
    "merge_incomplete_keywords",
];

const TTS_READY_DISABLED_FIELDS = [
    "srt_max_lines",
];

function isSrtOptionsNode(node) {
    return node.comfyClass === "SRTAdvancedOptionsNode";
}

function findWidgetByName(node, name) {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
}

function bindWidgetValueHandler(widget, onChange) {
    if (!widget || widget.__ttsAudioSuiteValueBound) {
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
            onChange(newVal);
        }
    });

    widget.__ttsAudioSuiteValueBound = true;
}

function captureFieldValues(node, fieldNames) {
    const snapshot = {};
    for (const field of fieldNames) {
        const widget = findWidgetByName(node, field);
        if (!widget) {
            continue;
        }
        snapshot[field] = widget.value;
    }
    return snapshot;
}

function restoreFieldValues(node, fieldNames, snapshot) {
    for (const field of fieldNames) {
        const widget = findWidgetByName(node, field);
        if (!widget || snapshot[field] === undefined) {
            continue;
        }
        widget.value = snapshot[field];
    }
}

function applyPreset(node, preset) {
    if (!preset || preset === "Custom") {
        return;
    }

    const values = PRESET_VALUES[preset];
    if (!values) {
        return;
    }

    node.__applyingSrtPreset = true;
    try {
        for (const field of SRT_FIELDS) {
            const widget = findWidgetByName(node, field);
            if (!widget || values[field] === undefined) {
                continue;
            }
            widget.value = values[field];
        }
    } finally {
        node.__applyingSrtPreset = false;
    }
}

function lockFields(node, shouldLock) {
    for (const field of SRT_FIELDS) {
        const widget = findWidgetByName(node, field);
        if (!widget) {
            continue;
        }
        widget.disabled = shouldLock;
    }
}

function applyTtsReadyFieldState(node) {
    const ttsReadyWidget = findWidgetByName(node, "tts_ready_mode");
    const isTtsReady = Boolean(ttsReadyWidget && ttsReadyWidget.value);

    for (const field of TTS_READY_DISABLED_FIELDS) {
        const widget = findWidgetByName(node, field);
        if (!widget) {
            continue;
        }
        widget.disabled = isTtsReady;
    }

    const presetWidget = findWidgetByName(node, "srt_preset");
    const presetLocked = Boolean(presetWidget && presetWidget.value !== "Custom");
    if (presetLocked) {
        lockFields(node, true);
    }
}

function srtPresetHandler(node) {
    if (!isSrtOptionsNode(node)) {
        return;
    }

    const presetWidget = findWidgetByName(node, "srt_preset");
    if (!presetWidget) {
        return;
    }

    const preset = presetWidget.value;
    const shouldLock = preset !== "Custom";

    if (!node.__srtPresetCache) {
        node.__srtPresetCache = {};
    }

    if (shouldLock) {
        for (const field of SRT_FIELDS) {
            const widget = findWidgetByName(node, field);
            if (!widget) {
                continue;
            }
            if (node.__srtPresetCache[field] === undefined) {
                node.__srtPresetCache[field] = widget.value;
            }
        }
        applyPreset(node, preset);
    } else {
        restoreFieldValues(node, SRT_FIELDS, node.__srtPresetCache);
        node.__srtPresetCache = {};
    }

    lockFields(node, shouldLock);
    applyTtsReadyFieldState(node);
}

function applyHeuristicProfile(node, profile) {
    if (!profile || profile === "Custom") {
        return;
    }

    const values = HEURISTIC_PROFILE_VALUES[profile] || HEURISTIC_PROFILE_VALUES["English"];
    node.__applyingHeuristicProfile = true;
    try {
        for (const field of HEURISTIC_FIELDS) {
            const widget = findWidgetByName(node, field);
            if (!widget || values[field] === undefined) {
                continue;
            }
            widget.value = values[field];
        }
    } finally {
        node.__applyingHeuristicProfile = false;
    }
}

function heuristicProfileHandler(node) {
    if (!isSrtOptionsNode(node)) {
        return;
    }

    const profileWidget = findWidgetByName(node, "heuristic_language_profile");
    if (!profileWidget) {
        return;
    }

    const profile = profileWidget.value;

    if (!node.__heuristicProfileCache) {
        node.__heuristicProfileCache = {};
    }

    if (profile !== "Custom") {
        if (Object.keys(node.__heuristicProfileCache).length === 0) {
            node.__heuristicProfileCache = captureFieldValues(node, HEURISTIC_FIELDS);
        }
        applyHeuristicProfile(node, profile);
    } else if (Object.keys(node.__heuristicProfileCache).length > 0) {
        restoreFieldValues(node, HEURISTIC_FIELDS, node.__heuristicProfileCache);
    }
}

function heuristicFieldEdited(node) {
    if (!isSrtOptionsNode(node) || node.__applyingHeuristicProfile) {
        return;
    }

    const profileWidget = findWidgetByName(node, "heuristic_language_profile");
    if (!profileWidget || profileWidget.value === "Custom") {
        return;
    }

    node.__heuristicProfileCache = captureFieldValues(node, HEURISTIC_FIELDS);
    profileWidget.value = "Custom";
}

app.registerExtension({
    name: "tts-audio-suite.srt-preset.widgets",
    nodeCreated(node) {
        if (!isSrtOptionsNode(node)) {
            return;
        }

        srtPresetHandler(node);
        heuristicProfileHandler(node);
        applyTtsReadyFieldState(node);

        bindWidgetValueHandler(findWidgetByName(node, "srt_preset"), () => srtPresetHandler(node));
        bindWidgetValueHandler(findWidgetByName(node, "heuristic_language_profile"), () => heuristicProfileHandler(node));
        bindWidgetValueHandler(findWidgetByName(node, "tts_ready_mode"), () => applyTtsReadyFieldState(node));

        for (const field of HEURISTIC_FIELDS) {
            bindWidgetValueHandler(findWidgetByName(node, field), () => heuristicFieldEdited(node));
        }
    }
});
