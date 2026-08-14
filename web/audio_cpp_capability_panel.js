import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET = "AudioCppEngineNode";
const ENDPOINT = "/api/tts-audio-suite/audio-cpp-capabilities";
const STATUS_ENDPOINT = "/api/tts-audio-suite/audio-cpp-status";
const PANEL_HEIGHT = 210;
const PANEL_MIN_WIDTH = 360;
const PANEL_BOTTOM_PADDING = 14;
const PANEL_LAYOUT_HEIGHT = PANEL_HEIGHT + PANEL_BOTTOM_PADDING;
const REQUEST_ADVANCED_WIDGETS = [
    "temperature", "top_p", "top_k", "repetition_penalty",
    "max_tokens", "max_steps", "num_inference_steps", "guidance_scale",
    "advanced_json",
];
const OWNED_ADVANCED_WIDGETS = [
    "auto_download_runtime", "auto_download_model", "show_server_console",
];
let manifestPromise;

function manifest() {
    manifestPromise ??= api.fetchApi(ENDPOINT).then((response) => {
        if (!response.ok) throw new Error(`Capability request failed (${response.status})`);
        return response.json();
    });
    return manifestPromise;
}

function widget(node, name) {
    return (node.widgets || []).find((item) => item.name === name);
}

function hideWidget(item) {
    if (!item || item.__ttsAudioCppHidden) return;
    item.__ttsAudioCppOriginalType = item.type;
    item.__ttsAudioCppOriginalComputeSize = item.computeSize;
    item.type = "hidden";
    item.hidden = true;
    item.computeSize = () => [0, -4];
    if (item.element) item.element.style.display = "none";
    item.__ttsAudioCppHidden = true;
}

function showWidget(item) {
    if (!item || !item.__ttsAudioCppHidden) return;
    item.type = item.__ttsAudioCppOriginalType;
    item.computeSize = item.__ttsAudioCppOriginalComputeSize;
    item.hidden = false;
    if (item.element) item.element.style.display = "";
    item.__ttsAudioCppHidden = false;
}

function setWidgetVisible(node, name, visible) {
    const item = widget(node, name);
    if (visible) showWidget(item);
    else hideWidget(item);
}

function resizeNodeToContent(node) {
    if (node.__ttsAudioCppResizeFrame) cancelAnimationFrame(node.__ttsAudioCppResizeFrame);
    node.__ttsAudioCppResizeFrame = requestAnimationFrame(() => {
        node.__ttsAudioCppResizeFrame = 0;
        const computed = node.computeSize();
        const width = Math.max(Number(node.size?.[0]) || 0, PANEL_MIN_WIDTH, Number(computed?.[0]) || 0);
        const height = Number(computed?.[1]) || Number(node.size?.[1]) || PANEL_LAYOUT_HEIGHT;
        if (Math.abs(width - node.size[0]) > 0.5 || Math.abs(height - node.size[1]) > 0.5) {
            node.setSize([width, height]);
        }
        app.graph?.setDirtyCanvas(true, true);
    });
}

function applyWidgetVisibility(node, capability) {
    if (!capability) return;
    const advanced = Boolean(node.__ttsAudioCppAdvancedOpen);
    const mode = String(widget(node, "connection_mode")?.value || "auto");
    const backend = String(widget(node, "backend")?.value || "auto");
    const external = mode === "external_server";
    const existingBinary = mode === "existing_binary";
    const suiteTasks = new Set(capability.suite_tasks || []);
    const upstreamTasks = new Set(capability.upstream_tasks || []);

    setWidgetVisible(node, "package_id", !external);
    setWidgetVisible(node, "task", external || upstreamTasks.size > 1 || advanced);
    setWidgetVisible(node, "backend", !external);
    setWidgetVisible(node, "device", !external && backend !== "cpu" && advanced);
    setWidgetVisible(node, "threads", !external && (backend === "cpu" || advanced));
    setWidgetVisible(node, "language", suiteTasks.has("tts") || suiteTasks.has("asr"));
    setWidgetVisible(node, "server_url", external);
    setWidgetVisible(node, "binary_path", existingBinary || (!external && advanced));
    setWidgetVisible(node, "model_path", existingBinary || (!external && advanced));
    setWidgetVisible(node, "model_id", external);
    setWidgetVisible(node, "voice_id", Boolean(capability.built_in_voices));
    setWidgetVisible(node, "instruct", Boolean(capability.voice_design));
    for (const name of REQUEST_ADVANCED_WIDGETS) setWidgetVisible(node, name, advanced);
    for (const name of OWNED_ADVANCED_WIDGETS) setWidgetVisible(node, name, !external && advanced);

    const toggle = widget(node, "audio_cpp_advanced_toggle");
    if (toggle) {
        toggle.label = advanced ? "▾ Hide advanced settings" : "▸ Show advanced settings";
    }
    resizeNodeToContent(node);
}

function addAdvancedToggle(node) {
    const toggle = node.addWidget("button", "▸ Show advanced settings", null, () => {
        node.__ttsAudioCppAdvancedOpen = !node.__ttsAudioCppAdvancedOpen;
        applyWidgetVisibility(node, node.__ttsAudioCppCapability);
    });
    toggle.name = "audio_cpp_advanced_toggle";
    toggle.label = "▸ Show advanced settings";
    toggle.options ??= {};
    toggle.options.tooltip = "Show uncommon runtime paths, device tuning, sampling overrides, download controls, server debugging, and raw request JSON.";
    toggle.options.serialize = false;
    toggle.tooltip = toggle.options.tooltip;
    toggle.serialize = false;
    toggle.serializeValue = () => undefined;
    return toggle;
}

function formatBytes(bytes) {
    const value = Number(bytes);
    if (!Number.isFinite(value) || value <= 0) return "unavailable";
    if (value >= 1073741824) return `${(value / 1073741824).toFixed(2)} GB`;
    return `${(value / 1048576).toFixed(1)} MB`;
}

function syncPackageChoices(node, data, capability) {
    const packageWidget = widget(node, "package_id");
    if (!packageWidget || !capability) return null;
    const allowed = ["auto", ...(capability.packages || [])];
    packageWidget.options ??= {};
    packageWidget.options.values = allowed;
    if (!allowed.includes(String(packageWidget.value))) packageWidget.value = "auto";
    const resolvedId = packageWidget.value === "auto"
        ? capability.recommended_package_id
        : String(packageWidget.value);
    return data.packages?.[resolvedId] || null;
}

function syncTaskChoices(node, capability) {
    const taskWidget = widget(node, "task");
    if (!taskWidget || !capability) return;
    const allowed = ["auto", ...(capability.upstream_tasks || [])];
    taskWidget.options ??= {};
    taskWidget.options.values = [...new Set(allowed)];
    if (!taskWidget.options.values.includes(String(taskWidget.value))) taskWidget.value = "auto";
}

function normalizedUrl(value) {
    return String(value || "").trim().replace(/\/$/, "");
}

function matchingStatus(node, sessions) {
    const family = String(widget(node, "family")?.value || "");
    const modelId = String(widget(node, "model_id")?.value || "").trim();
    const mode = String(widget(node, "connection_mode")?.value || "auto");
    const endpoint = normalizedUrl(widget(node, "server_url")?.value);
    return (sessions || []).find((session) => {
        if (modelId && session.model_id !== modelId) return false;
        if (mode === "external_server" && endpoint) {
            return !session.owned && normalizedUrl(session.endpoint) === endpoint;
        }
        return session.family === family && (mode === "external_server" ? !session.owned : true);
    });
}

function renderStatus(node, panel, sessions, failed = false) {
    const status = matchingStatus(node, sessions);
    const light = panel.querySelector(".tts-acpp-light");
    const label = panel.querySelector(".tts-acpp-status-text");
    const state = failed ? "error" : (status?.state || "unchecked");
    const labels = {
        unchecked: "Not checked",
        configured: "Server stopped",
        server_ready: "Server connected · model not confirmed",
        model_ready: "Server connected · model ready",
        error: "Status unavailable",
    };
    light.dataset.state = state;
    label.textContent = labels[state] || state;
    const memory = panel.querySelector(".tts-acpp-memory");
    const stop = panel.querySelector(".tts-acpp-stop");
    node.__ttsAudioCppSessionId = status?.session_id || "";
    if (status?.pid) {
        const privateGb = status.private_bytes ? ` · ${(status.private_bytes / 1073741824).toFixed(1)} GB private` : "";
        memory.textContent = `PID ${status.pid}${privateGb}`;
        memory.hidden = false;
    } else {
        memory.hidden = true;
    }
    stop.hidden = !(status?.owned && ["server_ready", "model_ready"].includes(status.state));
}

async function refreshStatus(node, panel) {
    try {
        const response = await api.fetchApi(STATUS_ENDPOINT);
        if (!response.ok) throw new Error(`Status request failed (${response.status})`);
        const data = await response.json();
        renderStatus(node, panel, data.sessions);
    } catch (_error) {
        renderStatus(node, panel, [], true);
    }
}

function inputIsSpeaker(input) {
    return /^speaker\d+$/.test(String(input?.name || ""));
}

function removeUnusedSpeakerInputs(node, maximum) {
    for (let index = (node.inputs || []).length - 1; index >= 0; index -= 1) {
        const input = node.inputs[index];
        const number = Number(String(input?.name || "").replace("speaker", ""));
        if (inputIsSpeaker(input) && input.link == null && number > maximum) node.removeInput(index);
    }
}

function syncSpeakers(node, capability) {
    const native = capability?.native_multi_speaker || {};
    const maximum = native.supported ? Number(native.max_speakers || 1) : 1;
    removeUnusedSpeakerInputs(node, maximum);
    const speakers = (node.inputs || []).filter(inputIsSpeaker);
    if (maximum <= 1) {
        for (let index = (node.inputs || []).length - 1; index >= 0; index -= 1) {
            if (inputIsSpeaker(node.inputs[index]) && node.inputs[index].link == null) node.removeInput(index);
        }
        return;
    }
    speakers.forEach((input, index) => {
        input.name = `speaker${index + 2}`;
        input.label = `Speaker ${index + 2}`;
    });
    const last = speakers[speakers.length - 1];
    if (speakers.length < maximum - 1 && (!last || last.link != null)) {
        node.addInput(`speaker${speakers.length + 2}`, "*");
    }
}

function pill(text, tone = "normal", title = "") {
    const element = document.createElement("span");
    element.textContent = text;
    element.className = `tts-acpp-pill ${tone}`;
    if (title) element.title = title;
    return element;
}

function appendAsrFeaturePills(container, capability) {
    const features = capability.asr_features || {};
    if (features.diarization === "native") {
        container.append(pill(
            "Diarization",
            "special",
            "Native speaker-attributed turns are available from this ASR family.",
        ));
    } else {
        container.append(pill(
            "No diarization",
            "muted",
            "This ASR family does not return speaker identities.",
        ));
    }

    const timing = features.timing || "none";
    if (timing === "native_word") {
        container.append(pill(
            "Word timestamps",
            "info",
            "Native word or token timestamps are available without a separate aligner.",
        ));
    } else if (timing === "native_segment") {
        container.append(pill(
            "Segment timestamps",
            "info",
            "Native timed transcript or speaker segments are available.",
        ));
    } else if (timing === "optional_forced_aligner") {
        container.append(pill(
            "Optional forced aligner",
            "warn",
            "Word timestamps require the separate Qwen3 Forced Aligner model.",
        ));
    } else {
        container.append(pill(
            "No timestamps",
            "muted",
            "This ASR family currently returns transcription text without timing alignment.",
        ));
    }
}

function setWidgetHeight(widget, height) {
    try { widget.height = height; } catch (_error) { /* getter-only on some builds */ }
    try { widget.computedHeight = height; } catch (_error) { /* getter-only on some builds */ }
}

function makePanel(node) {
    const panel = document.createElement("div");
    panel.className = "tts-acpp-panel";
    panel.innerHTML = `<style>
      .tts-acpp-panel{box-sizing:border-box;width:100%;max-width:100%;height:${PANEL_HEIGHT}px;overflow:hidden;margin:0;padding:9px 10px;border:1px solid var(--border-color,#454545);border-radius:7px;background:color-mix(in srgb,var(--comfy-menu-bg,#202020) 90%,#4d78a8 10%);color:var(--input-text,#ddd);font:12px/1.35 sans-serif}
      .tts-acpp-head{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:5px 8px;margin-bottom:5px}.tts-acpp-title{font-weight:650;font-size:13px}.tts-acpp-status{display:flex;align-items:center;gap:5px;color:var(--descrip-text,#aaa);font-size:11px;white-space:nowrap}
      .tts-acpp-light{width:8px;height:8px;border-radius:50%;background:#777;box-shadow:0 0 0 2px color-mix(in srgb,#777 25%,transparent)}.tts-acpp-light[data-state="configured"]{background:#d49a36}.tts-acpp-light[data-state="server_ready"]{background:#4d9bea}.tts-acpp-light[data-state="model_ready"]{background:#48bf78;box-shadow:0 0 5px #48bf78}.tts-acpp-light[data-state="error"]{background:#d85b5b}
      .tts-acpp-runtime{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:4px 8px;margin:-1px 0 6px;color:var(--descrip-text,#aaa);font-size:11px}.tts-acpp-stop{border:1px solid #6d4b4b;border-radius:5px;background:#3d2929;color:#efc5c5;padding:2px 6px;cursor:pointer}.tts-acpp-stop:hover{background:#543232}
      .tts-acpp-pills{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:6px}
      .tts-acpp-pill{padding:2px 7px;border:1px solid transparent;border-radius:999px;background:#3a4652;color:#dcecff;font-size:11px;line-height:1.35;white-space:nowrap}.tts-acpp-pill.warn{border-color:#806635;background:#5b4929;color:#ffe0a3}.tts-acpp-pill.good{border-color:#38684e;background:#294d3b;color:#bdebd2}.tts-acpp-pill.info{border-color:#365f7d;background:#29485f;color:#c8e7ff}.tts-acpp-pill.special{border-color:#685485;background:#46385d;color:#e8d9ff}.tts-acpp-pill.muted{border-color:#4d565f;background:#343b42;color:#b9c1c9}
      .tts-acpp-detail{color:var(--descrip-text,#b8b8b8);margin-top:2px}.tts-acpp-summary{margin-top:6px;color:var(--input-text,#ddd)}
    </style><div class="tts-acpp-head"><div class="tts-acpp-title">audio.cpp capabilities</div><div class="tts-acpp-status"><span class="tts-acpp-light" data-state="unchecked"></span><span class="tts-acpp-status-text">Not checked</span></div></div><div class="tts-acpp-runtime"><span class="tts-acpp-memory" hidden></span><button class="tts-acpp-stop" type="button" hidden>Stop owned server</button></div><div class="tts-acpp-pills"></div><div class="tts-acpp-details"></div><div class="tts-acpp-summary"></div>`;
    panel.querySelector(".tts-acpp-stop").addEventListener("click", async () => {
        const sessionId = node.__ttsAudioCppSessionId;
        if (!sessionId) return;
        const response = await api.fetchApi("/api/tts-audio-suite/audio-cpp-stop", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sessionId }),
        });
        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            throw new Error(data.error || `Stop failed (${response.status})`);
        }
        refreshStatus(node, panel);
    });
    const panelWidget = node.addDOMWidget("audio_cpp_capabilities", "div", panel, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => PANEL_LAYOUT_HEIGHT,
        getHeight: () => PANEL_LAYOUT_HEIGHT,
    });
    panelWidget.computeSize = (inputWidth) => {
        const width = Array.isArray(inputWidth) ? inputWidth[0] : inputWidth;
        return [Math.max(PANEL_MIN_WIDTH, Number(width) || PANEL_MIN_WIDTH), PANEL_LAYOUT_HEIGHT];
    };
    panelWidget.getHeight = () => PANEL_LAYOUT_HEIGHT;
    panelWidget.computeLayoutSize = () => ({
        minWidth: PANEL_MIN_WIDTH,
        minHeight: PANEL_LAYOUT_HEIGHT,
    });
    panelWidget.options ??= {};
    panelWidget.options.minNodeSize = [PANEL_MIN_WIDTH, PANEL_LAYOUT_HEIGHT];
    setWidgetHeight(panelWidget, PANEL_LAYOUT_HEIGHT);
    if (panelWidget.element) {
        panelWidget.element.style.boxSizing = "border-box";
        panelWidget.element.style.width = "100%";
        panelWidget.element.style.maxWidth = "100%";
        panelWidget.element.style.height = `${PANEL_HEIGHT}px`;
        panelWidget.element.style.minHeight = `${PANEL_HEIGHT}px`;
        panelWidget.element.style.overflow = "hidden";
    }
    return panel;
}

function render(node, panel, data, capability) {
    if (!capability) return;
    node.__ttsAudioCppCapability = capability;
    const selectedPackage = syncPackageChoices(node, data, capability);
    syncTaskChoices(node, capability);
    panel.querySelector(".tts-acpp-title").textContent = capability.display_name;
    const pills = panel.querySelector(".tts-acpp-pills");
    pills.replaceChildren();
    const suiteTasks = new Set(capability.suite_tasks || []);
    if (suiteTasks.has("tts")) pills.append(pill("TTS", "good", "Text-to-speech is wired to the Suite's Unified Text and SRT nodes."));
    if (suiteTasks.has("asr")) pills.append(pill("ASR", "good", "Speech recognition is wired to the Suite's Unified ASR node."));
    if (suiteTasks.has("voice_conversion")) pills.append(pill("Voice conversion", "good", "Voice conversion is wired to the Suite's Unified Voice Changer node."));
    if (suiteTasks.has("asr")) appendAsrFeaturePills(pills, capability);
    if (suiteTasks.has("tts") && capability.reference_audio !== "none") pills.append(pill("Voice clone", "normal", "This family accepts reference audio for voice cloning or conditioning."));
    if (["required", "required_per_speaker"].includes(capability.reference_audio)) {
        pills.append(pill("Reference required", "warn", "Generation requires reference audio."));
    }
    if (capability.reference_transcript === "required") {
        pills.append(pill("Transcript required", "warn", "The transcript matching the reference audio is required."));
    }
    if (capability.built_in_voices) pills.append(pill("Built-in voices", "info", "This family includes model-provided voices."));
    if (capability.voice_design) pills.append(pill("Voice design", "normal", "This family can synthesize from a written voice description."));
    if (capability.inline_controls) pills.append(pill("Inline controls", "normal", "Suite-standard inline controls are translated for this family."));
    if (capability.native_multi_speaker?.supported) {
        const status = capability.native_multi_speaker.suite_status === "supported" ? "good" : "warn";
        pills.append(pill(`Up to ${capability.native_multi_speaker.max_speakers} speakers`, status));
    }
    const transcript = capability.reference_transcript;
    let taskDetails = "";
    if (suiteTasks.has("tts")) {
        taskDetails =
            `<div class="tts-acpp-detail">Reference audio: ${capability.reference_audio.replaceAll("_", " ")}</div>` +
            `<div class="tts-acpp-detail">Reference transcript: ${transcript}</div>`;
    } else if (suiteTasks.has("asr")) {
        const languages = capability.languages || [];
        const languageSummary = languages.length > 8 ? `${languages.length} declared languages` : languages.join(", ");
        taskDetails = `<div class="tts-acpp-detail">Languages: ${languageSummary || "model-defined"}</div>`;
    } else if (suiteTasks.has("voice_conversion")) {
        taskDetails = `<div class="tts-acpp-detail">Inputs: source audio + target reference audio</div>`;
    }
    panel.querySelector(".tts-acpp-details").innerHTML =
        `<div class="tts-acpp-detail">Package: ${selectedPackage?.display_name || "external server / unresolved"}</div>` +
        `<div class="tts-acpp-detail">Estimated download: ${formatBytes(selectedPackage?.estimated_download_bytes)}</div>` +
        taskDetails +
        `<div class="tts-acpp-detail">Suite: ${(capability.suite_tasks || []).join(" · ")}</div>`;
    panel.querySelector(".tts-acpp-summary").textContent = capability.summary || capability.description;
    syncSpeakers(node, capability);
    applyWidgetVisibility(node, capability);
}

function hookWidgetCallback(node, name, callback) {
    const item = widget(node, name);
    if (!item || item.__ttsAudioCppCallbackHooked) return;
    item.__ttsAudioCppCallbackHooked = true;
    const original = item.callback;
    item.callback = (...args) => {
        const result = original?.apply(item, args);
        callback();
        return result;
    };
}

app.registerExtension({
    name: "TTS_Audio_Suite.AudioCppCapabilities",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if ((nodeData?.name || nodeData?.comfyClass) !== TARGET) return;
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = original?.apply(this, arguments);
            addAdvancedToggle(this);
            const panel = makePanel(this);
            const family = widget(this, "family");
            const update = () => manifest().then((data) => render(this, panel, data, data.families?.[family?.value])).catch((error) => {
                panel.querySelector(".tts-acpp-summary").textContent = error.message;
            });
            hookWidgetCallback(this, "family", update);
            hookWidgetCallback(this, "package_id", update);
            hookWidgetCallback(this, "task", update);
            hookWidgetCallback(this, "connection_mode", () => applyWidgetVisibility(this, this.__ttsAudioCppCapability));
            hookWidgetCallback(this, "backend", () => applyWidgetVisibility(this, this.__ttsAudioCppCapability));
            const refresh = () => refreshStatus(this, panel);
            this.__ttsAudioCppStatusRefresh = refresh;
            setTimeout(() => { update(); refresh(); }, 0);
            this.__ttsAudioCppStatusTimer = setInterval(refresh, 5000);
            return result;
        };
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function() {
            clearInterval(this.__ttsAudioCppStatusTimer);
            if (this.__ttsAudioCppResizeFrame) cancelAnimationFrame(this.__ttsAudioCppResizeFrame);
            return removed?.apply(this, arguments);
        };
        const connectionChanged = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function(type, index) {
            const result = connectionChanged?.apply(this, arguments);
            if (inputIsSpeaker(this.inputs?.[index])) {
                const family = widget(this, "family")?.value;
                manifest().then((data) => syncSpeakers(this, data.families?.[family]));
            }
            return result;
        };
    },
    setup() {
        for (const eventName of ["executing", "executed", "execution_error"]) {
            api.addEventListener(eventName, () => {
                for (const node of app.graph?._nodes || []) node.__ttsAudioCppStatusRefresh?.();
            });
        }
    },
});
