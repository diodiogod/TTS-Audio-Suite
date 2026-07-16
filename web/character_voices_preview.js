import { api } from "../../scripts/api.js";
import { app } from "../../scripts/app.js";
import { createCharacterVoiceTrimUI, hideNativeWidget } from "./character_voices_trim_ui.js";

function findWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name) || null;
}

function ensureState(node) {
    if (!node.__ttsCharacterVoiceState) {
        node.__ttsCharacterVoiceState = {
            editor: null,
            fallbackAudio: null,
            canonicalText: null,
            duration: 0,
            suppressWidgetCallbacks: false,
            configuring: false,
            resetTrimOnMetadata: false,
            metadataRequest: 0,
        };
    }
    return node.__ttsCharacterVoiceState;
}

async function refreshVoiceLibrary(node, forceRefresh = false) {
    try {
        const suffix = forceRefresh ? "?refresh=1" : "";
        const response = await api.fetchApi(`/api/tts-audio-suite/voice-library${suffix}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        const voiceWidget = findWidget(node, "voice_name");
        const voices = Array.isArray(data.voices) ? data.voices : [];
        if (!voiceWidget) return;
        voiceWidget.options = voiceWidget.options || {};
        voiceWidget.options.values = voices;
        if (!voices.includes(voiceWidget.value)) {
            setWidgetValue(node, voiceWidget, "none");
            resetForVoiceSelection(node, "none");
        }
        app.graph?.setDirtyCanvas(true, true);
    } catch (error) {
        console.warn("Character Voices library refresh failed:", error);
    }
}

function setWidgetValue(node, widget, value) {
    if (!widget || widget.value === value) return;
    const state = ensureState(node);
    state.suppressWidgetCallbacks = true;
    widget.value = value;
    state.suppressWidgetCallbacks = false;
    app.graph?.setDirtyCanvas(true, false);
}

function buildVoiceUrl(route, voiceName) {
    const params = new URLSearchParams({ voice_name: voiceName });
    return api.apiURL(`/api/tts-audio-suite/${route}?${params.toString()}`);
}

function buildVoiceRoute(route, voiceName) {
    const params = new URLSearchParams({ voice_name: voiceName });
    return `/api/tts-audio-suite/${route}?${params.toString()}`;
}

function selectedRange(node) {
    const state = ensureState(node);
    const start = Math.max(0, Number(findWidget(node, "trim_start")?.value) || 0);
    const rawEnd = Number(findWidget(node, "trim_end")?.value) || 0;
    const end = rawEnd > 0 ? rawEnd : state.duration;
    return { start, end };
}

function computeCustomState(node) {
    const state = ensureState(node);
    const voiceName = findWidget(node, "voice_name")?.value;
    if (!voiceName || voiceName === "none") {
        return Boolean(findWidget(node, "reference_text")?.value?.trim());
    }

    const { start, end } = selectedRange(node);
    const trimChanged = state.duration > 0 && (start > 0.005 || end < state.duration - 0.005);
    const currentText = String(findWidget(node, "reference_text")?.value || "");
    const textChanged = state.canonicalText !== null && currentText !== state.canonicalText;
    return trimChanged || textChanged;
}

function setCustomState(node, isCustom) {
    const state = ensureState(node);
    const voiceName = findWidget(node, "voice_name")?.value;
    setWidgetValue(node, findWidget(node, "customized"), Boolean(isCustom));
    state.editor?.setCustomState(
        Boolean(isCustom),
        voiceName && voiceName !== "none" ? voiceName : "",
    );
    const { start, end } = selectedRange(node);
    const isTrimmed = state.duration > 0 && (start > 0.005 || end < state.duration - 0.005);
    state.editor?.setTrimWarning(isTrimmed);
}

function refreshCustomState(node) {
    setCustomState(node, computeCustomState(node));
}

function setTrimRange(node, start, end, userInitiated = false) {
    const state = ensureState(node);
    setWidgetValue(node, findWidget(node, "trim_start"), Number(start.toFixed(2)));
    setWidgetValue(node, findWidget(node, "trim_end"), Number(end.toFixed(2)));
    state.editor?.setRange(start, end, false);
    if (userInitiated) {
        const audio = state.editor?.audio;
        if (audio && Number.isFinite(audio.duration)) {
            audio.currentTime = start;
        }
        refreshCustomState(node);
    }
}

function clearPlayer(node) {
    const state = ensureState(node);
    const audio = state.editor?.audio || state.fallbackAudio;
    if (audio) {
        audio.pause();
        audio.removeAttribute("src");
        audio.load?.();
    }
    state.duration = 0;
    state.editor?.setDuration(0);
    state.editor?.setTitle("No library voice selected");
    state.editor?.setCustomState(false, "");
    state.editor?.setTrimWarning(false);
    state.editor?.clearWaveform();
}

function loadPlayer(node, voiceName) {
    const state = ensureState(node);
    const audio = state.editor?.audio || state.fallbackAudio;
    if (!audio || !voiceName || voiceName === "none") {
        clearPlayer(node);
        return;
    }
    audio.pause();
    const previewUrl = buildVoiceUrl("voice-preview", voiceName);
    audio.src = previewUrl;
    audio.load();
    state.editor?.loadWaveform(previewUrl);
    state.editor?.setTitle(voiceName);
}

async function loadVoiceMetadata(node, voiceName, applyCanonicalText) {
    const state = ensureState(node);
    const requestId = ++state.metadataRequest;
    if (!voiceName || voiceName === "none") {
        state.canonicalText = "";
        return;
    }

    try {
        const response = await api.fetchApi(buildVoiceRoute("voice-info", voiceName));
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        if (requestId !== state.metadataRequest) return;
        state.canonicalText = String(data.reference_text || "");
        if (applyCanonicalText && !findWidget(node, "customized")?.value) {
            setWidgetValue(node, findWidget(node, "reference_text"), state.canonicalText);
        }
        refreshCustomState(node);
    } catch (error) {
        console.warn(`Character Voices metadata failed to load for ${voiceName}:`, error);
    }
}

function resetForVoiceSelection(node, voiceName) {
    const state = ensureState(node);
    state.metadataRequest += 1;
    state.canonicalText = null;
    state.resetTrimOnMetadata = true;
    setWidgetValue(node, findWidget(node, "customized"), false);
    setWidgetValue(node, findWidget(node, "trim_start"), 0);
    setWidgetValue(node, findWidget(node, "trim_end"), 0);

    if (!voiceName || voiceName === "none") {
        setWidgetValue(node, findWidget(node, "reference_text"), "");
        clearPlayer(node);
        return;
    }
    loadPlayer(node, voiceName);
    loadVoiceMetadata(node, voiceName, true);
}

function restoreSerializedWidgets(node, info) {
    if (!Array.isArray(info?.widgets_values) || !node.widgets) return;
    const names = new Set(["voice_name", "reference_text", "trim_start", "trim_end", "customized"]);
    node.widgets.forEach((widget, index) => {
        if (names.has(widget.name) && index < info.widgets_values.length) {
            const savedValue = info.widgets_values[index];
            if (savedValue !== undefined) widget.value = savedValue;
        }
    });
}

function migrateLegacySerializedWidgets(node, info) {
    if (!Array.isArray(info?.widgets_values) || !node.widgets) return false;

    const customizedIndex = node.widgets.findIndex((widget) => widget.name === "customized");
    if (customizedIndex < 0 || customizedIndex < info.widgets_values.length) return false;

    // Workflows saved before editable references had no trim/customized fields.
    // Treat their selected library voice as canonical, regardless of stale
    // positional widget values restored by ComfyUI.
    const voiceName = findWidget(node, "voice_name")?.value;
    if (!voiceName || voiceName === "none") return false;

    setWidgetValue(node, findWidget(node, "reference_text"), "");
    setWidgetValue(node, findWidget(node, "trim_start"), 0);
    setWidgetValue(node, findWidget(node, "trim_end"), 0);
    setWidgetValue(node, findWidget(node, "customized"), false);
    return true;
}

function syncRestoredVoice(node) {
    const voiceName = findWidget(node, "voice_name")?.value;
    if (!voiceName || voiceName === "none") {
        clearPlayer(node);
        return;
    }
    loadPlayer(node, voiceName);
    const preserveCustomText = Boolean(findWidget(node, "customized")?.value);
    loadVoiceMetadata(node, voiceName, !preserveCustomText);
}

function wrapWidgetCallbacks(node) {
    const state = ensureState(node);
    const voiceWidget = findWidget(node, "voice_name");
    const referenceWidget = findWidget(node, "reference_text");

    const originalVoiceCallback = voiceWidget?.callback;
    if (voiceWidget) {
        voiceWidget.callback = function () {
            const result = originalVoiceCallback?.apply(this, arguments);
            if (!state.suppressWidgetCallbacks && !state.configuring) {
                resetForVoiceSelection(node, voiceWidget.value);
            }
            return result;
        };
    }

    const originalReferenceCallback = referenceWidget?.callback;
    if (referenceWidget) {
        referenceWidget.callback = function () {
            const result = originalReferenceCallback?.apply(this, arguments);
            if (!state.suppressWidgetCallbacks && !state.configuring) {
                if (state.canonicalText === null) setCustomState(node, true);
                else refreshCustomState(node);
            }
            return result;
        };
    }
}

function setupDomEditor(node) {
    const state = ensureState(node);
    if (typeof node.addDOMWidget !== "function") return false;

    const editor = createCharacterVoiceTrimUI((start, end) => {
        setTrimRange(node, start, end, true);
    });
    const audio = editor.audio;
    state.editor = editor;

    audio.addEventListener("loadedmetadata", () => {
        state.duration = Number.isFinite(audio.duration) ? audio.duration : 0;
        editor.setDuration(state.duration);
        if (state.resetTrimOnMetadata) {
            setTrimRange(node, 0, state.duration, false);
            state.resetTrimOnMetadata = false;
        } else {
            const range = selectedRange(node);
            setTrimRange(
                node,
                Math.min(range.start, state.duration),
                Math.min(range.end || state.duration, state.duration),
                false,
            );
        }
        refreshCustomState(node);
    });
    audio.addEventListener("play", () => {
        const { start, end } = selectedRange(node);
        if (audio.currentTime < start || audio.currentTime >= end) audio.currentTime = start;
    });
    audio.addEventListener("timeupdate", () => {
        const { start, end } = selectedRange(node);
        if (end > start && audio.currentTime >= end) {
            audio.pause();
            audio.currentTime = start;
        }
    });

    const domWidget = node.addDOMWidget("voice_reference_editor", "audioUI", editor.element, {
        serialize: false,
        hideOnZoom: true,
    });
    domWidget.computeSize = (width) => [Math.max(320, (node.size?.[0] || width || 430) - 20), 232];
    hideNativeWidget(findWidget(node, "trim_start"));
    hideNativeWidget(findWidget(node, "trim_end"));
    hideNativeWidget(findWidget(node, "customized"));
    node.setSize([Math.max(node.size?.[0] || 0, 430), node.computeSize()[1]]);
    return true;
}

function setupFallbackPlayer(node) {
    const state = ensureState(node);
    state.fallbackAudio = new Audio();
    node.addWidget("button", "▶ Play/Pause Voice", "", () => {
        const voiceName = findWidget(node, "voice_name")?.value;
        if (!voiceName || voiceName === "none") return;
        if (!state.fallbackAudio.src) loadPlayer(node, voiceName);
        if (state.fallbackAudio.paused) state.fallbackAudio.play();
        else state.fallbackAudio.pause();
    }, { serialize: false });
}

function setupCharacterVoices(node) {
    if (node.__ttsCharacterVoicesSetup) return;
    node.__ttsCharacterVoicesSetup = true;
    const state = ensureState(node);
    if (!setupDomEditor(node)) setupFallbackPlayer(node);
    wrapWidgetCallbacks(node);

    const originalOnConfigure = node.onConfigure?.bind(node);
    node.onConfigure = function (info) {
        state.configuring = true;
        const result = originalOnConfigure ? originalOnConfigure(info) : undefined;
        restoreSerializedWidgets(node, info);
        migrateLegacySerializedWidgets(node, info);
        state.configuring = false;
        setTimeout(() => syncRestoredVoice(node), 0);
        return result;
    };

    const originalOnRemoved = node.onRemoved;
    node.onRemoved = function () {
        clearPlayer(node);
        state.editor?.destroy();
        state.metadataRequest += 1;
        if (originalOnRemoved) return originalOnRemoved.apply(this, arguments);
    };

    const voiceName = findWidget(node, "voice_name")?.value;
    if (voiceName && voiceName !== "none") syncRestoredVoice(node);
    refreshVoiceLibrary(node, false);
}

app.registerExtension({
    name: "tts-audio-suite.character-voices.editor",
    async setup() {
        api.addEventListener("tts-audio-suite.voice-library-changed", () => {
            for (const node of app.graph?._nodes || []) {
                if (node.comfyClass === "CharacterVoicesNode") refreshVoiceLibrary(node, true);
            }
        });
    },
    nodeCreated(node) {
        if (node.comfyClass === "CharacterVoicesNode") setupCharacterVoices(node);
    },
});
