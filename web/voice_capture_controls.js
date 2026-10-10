import { api } from "../../scripts/api.js";
import { app } from "../../scripts/app.js";
import { createAudioTrimUI, hideNativeWidget } from "./character_voices_trim_ui.js";

const ROUTE = "/api/tts-audio-suite/voice-capture";

function widget(node, name) {
    return node.widgets?.find((item) => item.name === name);
}

async function request(route, data) {
    const response = await api.fetchApi(route, data === undefined ? {} : {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
    return result;
}

export function setupVoiceCaptureControls(node) {
    if (node.__ttsVoiceCaptureControls || typeof node.addDOMWidget !== "function") return;
    const state = { session: null, timer: null, removed: false, stopping: false, duration: 0 };
    node.__ttsVoiceCaptureControls = state;

    const setValue = (name, value) => {
        const target = widget(node, name);
        if (target) target.value = value;
        app.graph?.setDirtyCanvas(true, true);
    };
    const range = () => ({
        start: Math.max(0, Number(widget(node, "trim_start")?.value) || 0),
        end: Number(widget(node, "trim_end")?.value) || state.duration,
    });
    const editor = createAudioTrimUI((start, end) => {
        setValue("trim_start", Number(start.toFixed(2)));
        setValue("trim_end", Number(end.toFixed(2)));
        editor.audio.currentTime = start;
    }, { emptyTitle: "No recording", emptyBadge: "EMPTY", trimWarningText: "" });
    state.editor = editor;

    const container = document.createElement("div");
    Object.assign(container.style, { display: "flex", flexDirection: "column", gap: "8px", width: "100%", height: "auto" });
    const buttons = document.createElement("div");
    Object.assign(buttons.style, { display: "flex", gap: "8px", flexShrink: "0" });
    editor.element.style.flexShrink = "0";
    const recordButton = document.createElement("button");
    const clearButton = document.createElement("button");
    for (const button of [recordButton, clearButton]) {
        Object.assign(button.style, { padding: "7px 10px", borderRadius: "5px", border: "1px solid #555",
            background: "#303030", color: "#eee", cursor: "pointer" });
    }
    recordButton.style.flex = "1";
    recordButton.textContent = "🎙️ Start Recording";
    clearButton.textContent = "Clear";
    buttons.append(recordButton, clearButton);
    container.append(buttons, editor.element);

    function showError(error) {
        editor.setTitle(error);
        editor.setStatus("ERROR", "#f87171");
        recordButton.title = error;
    }

    function updateButtons() {
        recordButton.textContent = state.session ? "⏹ Stop Recording" : "🎙️ Start Recording";
        recordButton.disabled = state.stopping;
        clearButton.disabled = Boolean(state.session) || !widget(node, "recording")?.value;
    }

    function loadRecording() {
        const recording = widget(node, "recording")?.value || "";
        editor.audio.pause();
        state.duration = 0;
        if (!/^[a-f0-9]{32}$/.test(recording)) {
            editor.audio.removeAttribute("src");
            editor.audio.load();
            editor.setDuration(0);
            editor.clearWaveform();
            editor.setTitle("No recording");
            editor.setStatus("EMPTY");
        } else {
            const params = new URLSearchParams({ filename: `${recording}.wav`, subfolder: "voice_capture", type: "input" });
            const url = api.apiURL(`/view?${params}`);
            editor.audio.src = url;
            editor.audio.load();
            editor.loadWaveform(url);
            editor.setTitle("Recorded audio");
            editor.setStatus("READY");
        }
        updateButtons();
    }
    state.loadRecording = loadRecording;

    editor.audio.addEventListener("loadedmetadata", () => {
        state.duration = Number.isFinite(editor.audio.duration) ? editor.audio.duration : 0;
        editor.setDuration(state.duration);
        const selected = range();
        editor.setRange(selected.start, selected.end, false);
    });
    editor.audio.addEventListener("play", () => {
        const { start, end } = range();
        if (editor.audio.currentTime < start || editor.audio.currentTime >= end) editor.audio.currentTime = start;
    });
    editor.audio.addEventListener("timeupdate", () => {
        const { start, end } = range();
        if (end > start && editor.audio.currentTime >= end) {
            editor.audio.pause();
            editor.audio.currentTime = start;
        }
    });
    editor.audio.addEventListener("error", () => {
        if (widget(node, "recording")?.value) {
            showError("Saved recording could not be loaded. Record a new clip.");
            editor.setStatus("MISSING", "#f87171");
        }
    });

    function finish(error = null) {
        state.session = null;
        state.stopping = false;
        updateButtons();
        if (error) showError(error);
        else recordButton.title = "";
    }

    async function poll() {
        if (state.removed || !state.session) return;
        try {
            const result = await request(`${ROUTE}/status?recording=${state.session}`);
            if (state.removed) return;
            if (result.state === "ready") {
                setValue("recording", result.recording);
                setValue("trim_start", 0);
                setValue("trim_end", 0);
                loadRecording();
                finish();
                return;
            }
            if (result.state === "error") {
                finish(result.error || "Recording failed.");
                return;
            }
            editor.setStatus(`${state.stopping ? "STOPPING" : result.state.toUpperCase()} ${Number(result.duration).toFixed(1)}s`, "#f87171");
            state.timer = setTimeout(poll, 300);
        } catch (error) {
            // Release the microphone if status communication fails.
            request(`${ROUTE}/stop`, { recording: state.session }).catch(() => {});
            finish(error.message);
        }
    }

    recordButton.addEventListener("click", async () => {
        recordButton.disabled = true;
        try {
            if (state.session) {
                state.stopping = true;
                await request(`${ROUTE}/stop`, { recording: state.session });
                return;
            }
            editor.audio.pause();
            const settings = {};
            for (const name of ["voice_device", "voice_sample_rate", "voice_max_recording_time", "voice_volume_gain",
                "voice_silence_threshold", "voice_silence_duration", "voice_auto_normalize"]) {
                settings[name] = widget(node, name)?.value;
            }
            const result = await request(`${ROUTE}/start`, settings);
            if (state.removed) {
                request(`${ROUTE}/stop`, { recording: result.recording }).catch(() => {});
                return;
            }
            state.session = result.recording;
            updateButtons();
            poll();
        } catch (error) {
            finish(error.message);
        }
    });
    clearButton.addEventListener("click", () => {
        setValue("recording", "");
        setValue("trim_start", 0);
        setValue("trim_end", 0);
        loadRecording();
        recordButton.title = "";
    });

    const domWidget = node.addDOMWidget("recording_preview", "audioUI", container, { serialize: false, hideOnZoom: true });
    domWidget.computeSize = (width) => {
        const available = Math.max(160, (node.size?.[0] || width || 430) - 20);
        // DOM widgets add a 4px gap and have a 10px margin on each side.
        const contentHeight = container.offsetHeight || (available < 320 ? 80 : 244);
        return [available, contentHeight + 16];
    };
    const resizeObserver = new ResizeObserver(() => app.graph?.setDirtyCanvas(true, true));
    resizeObserver.observe(container);
    for (const name of ["voice_trigger", "recording", "trim_start", "trim_end"]) hideNativeWidget(widget(node, name));

    const onSerialize = node.onSerialize;
    node.onSerialize = function (info) {
        onSerialize?.call(this, info);
        info.properties = { ...info.properties, voice_capture_version: 1 };
    };
    const onConfigure = node.onConfigure;
    node.onConfigure = function (info) {
        onConfigure?.call(this, info);
        if (!info.properties?.voice_capture_version) {
            // Legacy workflows contain no saved clip; preserve their capture settings.
            setValue("recording", "");
            setValue("trim_start", 0);
            setValue("trim_end", 0);
        }
        setTimeout(loadRecording, 0);
    };
    const onRemoved = node.onRemoved;
    node.onRemoved = function () {
        state.removed = true;
        resizeObserver.disconnect();
        clearTimeout(state.timer);
        if (state.session) request(`${ROUTE}/stop`, { recording: state.session }).catch(() => {});
        editor.audio.pause();
        editor.destroy();
        return onRemoved?.apply(this, arguments);
    };
    loadRecording();
    node.setSize([Math.max(node.size?.[0] || 0, 430), node.computeSize()[1]]);
}
