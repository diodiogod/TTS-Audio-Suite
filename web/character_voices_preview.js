import { api } from "../../scripts/api.js";
import { app } from "../../scripts/app.js";

function findWidgetByName(node, name) {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
}

function ensurePreviewState(node) {
    if (!node.__ttsVoicePreviewState) {
        node.__ttsVoicePreviewState = {
            audio: null, // Fallback Audio() when addDOMWidget is unavailable
            audioElement: null, // DOM <audio> player for seek/scrub/volume controls
            audioWidget: null,
            isPlaying: false,
            currentVoice: null,
            playWidget: null,
            hasDomPlayer: false,
        };
    }
    return node.__ttsVoicePreviewState;
}

function updatePlayButtonLabel(node) {
    const state = ensurePreviewState(node);
    if (!state.playWidget) {
        return;
    }
    state.playWidget.name = state.isPlaying ? "⏹ Stop Voice" : "▶ Play Voice";
}

function stopVoicePreview(node) {
    const state = ensurePreviewState(node);
    if (state.audioElement) {
        try {
            state.audioElement.pause();
            state.audioElement.currentTime = 0;
        } catch (e) {
            console.warn("Character Voices DOM preview pause failed:", e);
        }
    }

    if (state.audio) {
        try {
            state.audio.pause();
        } catch (e) {
            console.warn("Character Voices preview pause failed:", e);
        }
        state.audio.src = "";
        state.audio = null;
    }
    state.isPlaying = false;
    state.currentVoice = null;
    updatePlayButtonLabel(node);
}

function buildVoicePreviewUrl(voiceName) {
    const params = new URLSearchParams({
        voice_name: voiceName,
    });
    return api.apiURL(`/api/tts-audio-suite/voice-preview?${params.toString()}`);
}

function loadVoiceIntoDomPlayer(node, voiceName) {
    const state = ensurePreviewState(node);
    if (!state.audioElement || !voiceName || voiceName === "none") {
        return false;
    }

    if (state.currentVoice === voiceName && state.audioElement.src) {
        return true;
    }

    state.audioElement.src = buildVoicePreviewUrl(voiceName);
    state.audioElement.load();
    state.currentVoice = voiceName;
    state.isPlaying = false;
    updatePlayButtonLabel(node);
    return true;
}

function clearDomPlayer(node) {
    const state = ensurePreviewState(node);
    if (!state.audioElement) {
        return;
    }
    try {
        state.audioElement.pause();
        state.audioElement.removeAttribute("src");
        state.audioElement.load();
    } catch (e) {
        console.warn("Character Voices DOM preview clear failed:", e);
    }
    state.currentVoice = null;
    state.isPlaying = false;
    updatePlayButtonLabel(node);
}

function playVoicePreview(node, voiceName) {
    const state = ensurePreviewState(node);

    // Preferred path: use in-node DOM audio player with scrub/seek/volume controls.
    if (state.audioElement) {
        const loaded = loadVoiceIntoDomPlayer(node, voiceName);
        if (!loaded) {
            return;
        }

        if (!state.audioElement.paused) {
            state.audioElement.pause();
            state.isPlaying = false;
            updatePlayButtonLabel(node);
            return;
        }

        const playPromise = state.audioElement.play();
        if (playPromise && typeof playPromise.catch === "function") {
            playPromise.catch((e) => {
                console.warn(`Character Voices DOM preview playback failed: ${voiceName}`, e);
                state.isPlaying = false;
                updatePlayButtonLabel(node);
            });
        }
        return;
    }

    // Fallback path: browser Audio() without embedded controls.
    stopVoicePreview(node);

    const audio = new Audio();
    audio.src = buildVoicePreviewUrl(voiceName);
    audio.preload = "auto";

    audio.addEventListener("ended", () => {
        state.isPlaying = false;
        state.currentVoice = null;
        updatePlayButtonLabel(node);
    });

    audio.addEventListener("error", () => {
        console.warn(`Character Voices preview failed to load: ${voiceName}`);
        state.isPlaying = false;
        state.currentVoice = null;
        updatePlayButtonLabel(node);
    });

    const playPromise = audio.play();
    if (playPromise && typeof playPromise.catch === "function") {
        playPromise.catch((e) => {
            console.warn(`Character Voices preview playback failed: ${voiceName}`, e);
            state.isPlaying = false;
            state.currentVoice = null;
            updatePlayButtonLabel(node);
        });
    }

    state.audio = audio;
    state.isPlaying = true;
    state.currentVoice = voiceName;
    updatePlayButtonLabel(node);
}

function setupCharacterVoicesPreview(node) {
    if (node.__ttsCharacterVoicesPreviewSetup) {
        return;
    }
    node.__ttsCharacterVoicesPreviewSetup = true;

    const voiceWidget = findWidgetByName(node, "voice_name");
    if (!voiceWidget) {
        return;
    }

    const state = ensurePreviewState(node);

    // Add a native audio player inside the node when DOM widgets are available.
    if (typeof node.addDOMWidget === "function") {
        const audioElement = document.createElement("audio");
        audioElement.controls = true;
        audioElement.classList.add("comfy-audio");
        audioElement.style.width = "100%";
        audioElement.style.maxWidth = "100%";

        try {
            const audioWidget = node.addDOMWidget(
                "voice_preview_player",
                "audioUI",
                audioElement,
                {
                    serialize: false,
                    hideOnZoom: false,
                }
            );
            state.audioElement = audioElement;
            state.audioWidget = audioWidget;
            state.hasDomPlayer = true;

            audioElement.addEventListener("play", () => {
                state.isPlaying = true;
                updatePlayButtonLabel(node);
            });
            audioElement.addEventListener("pause", () => {
                state.isPlaying = false;
                updatePlayButtonLabel(node);
            });
            audioElement.addEventListener("ended", () => {
                state.isPlaying = false;
                updatePlayButtonLabel(node);
            });
            audioElement.addEventListener("error", () => {
                state.isPlaying = false;
                updatePlayButtonLabel(node);
            });
        } catch (e) {
            console.warn("Character Voices preview: failed to create DOM audio widget, using fallback audio only.", e);
            state.hasDomPlayer = false;
            state.audioElement = null;
            state.audioWidget = null;
        }
    }

    // Only add explicit play/stop button in fallback mode.
    // When DOM audio controls are available, the built-in player handles play/pause/seek/volume.
    if (!state.hasDomPlayer) {
        state.playWidget = node.addWidget(
            "button",
            "▶ Play/Pause Voice",
            "",
            () => {
                const selectedVoice = voiceWidget.value;
                if (!selectedVoice || selectedVoice === "none") {
                    stopVoicePreview(node);
                    clearDomPlayer(node);
                    return;
                }

                playVoicePreview(node, selectedVoice);
            },
            { serialize: false }
        );
        updatePlayButtonLabel(node);
    } else {
        state.playWidget = null;
    }

    // Stop currently playing preview whenever dropdown value changes.
    const originalCallback = voiceWidget.callback;
    voiceWidget.callback = function () {
        stopVoicePreview(node);
        const selectedVoice = voiceWidget.value;
        if (selectedVoice && selectedVoice !== "none") {
            loadVoiceIntoDomPlayer(node, selectedVoice);
        } else {
            clearDomPlayer(node);
        }
        if (originalCallback) {
            return originalCallback.apply(this, arguments);
        }
    };

    // Preload currently selected voice into the player (if any), matching Load Audio UX.
    if (voiceWidget.value && voiceWidget.value !== "none") {
        loadVoiceIntoDomPlayer(node, voiceWidget.value);
    }

    // Stop on node removal.
    const originalOnRemoved = node.onRemoved;
    node.onRemoved = function () {
        stopVoicePreview(node);
        clearDomPlayer(node);
        if (originalOnRemoved) {
            return originalOnRemoved.apply(this, arguments);
        }
    };
}

app.registerExtension({
    name: "tts-audio-suite.character-voices.preview",
    nodeCreated(node) {
        if (node.comfyClass !== "CharacterVoicesNode") {
            return;
        }
        setupCharacterVoicesPreview(node);
    },
});
