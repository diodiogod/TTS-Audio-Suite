// ChatterBox Voice Capture Extension

import { app } from "../../scripts/app.js";

const VOICE_CAPTURE_CLASSES = new Set(["ChatterBoxVoiceCaptureDiogod", "ChatterBoxVoiceCapture"]);
const DEFAULT_DEVICE_LABEL = "System Default Input Device";
const LOADING_DEVICE_LABEL = "Loading input devices...";

function isVoiceCaptureNode(nodeOrData) {
    const comfyClass = nodeOrData?.comfyClass || nodeOrData?.name;
    return VOICE_CAPTURE_CLASSES.has(comfyClass);
}

function findWidgetByName(node, name) {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
}

function hideWidget(widget) {
    if (!widget) return;
    widget.type = "hidden";
    widget.computeSize = () => [0, -4];
}

function backendValueToLabel(value) {
    return value && value.trim() ? value : DEFAULT_DEVICE_LABEL;
}

function labelToBackendValue(value) {
    return value === DEFAULT_DEVICE_LABEL ? "" : value;
}

function ensureDeviceWidgets(node) {
    if (node.__ttsVoiceDeviceWidgetsInitialized) {
        return;
    }
    node.__ttsVoiceDeviceWidgetsInitialized = true;

    const backendWidget = findWidgetByName(node, "voice_device");
    if (!backendWidget) {
        return;
    }

    hideWidget(backendWidget);

    const initialLabel = backendValueToLabel(backendWidget.value);
    const comboWidget = node.addWidget(
        "combo",
        "Input Device",
        initialLabel,
        (value) => {
            backendWidget.value = labelToBackendValue(value);
            if (backendWidget.callback) {
                backendWidget.callback(backendWidget.value);
            }
        },
        {
            values: [DEFAULT_DEVICE_LABEL],
            serialize: false,
        }
    );

    const refreshWidget = node.addWidget(
        "button",
        "Refresh Input Devices",
        "",
        () => refreshInputDevices(node),
        { serialize: false }
    );

    node.__ttsVoiceDeviceBackendWidget = backendWidget;
    node.__ttsVoiceDeviceComboWidget = comboWidget;
    node.__ttsVoiceDeviceRefreshWidget = refreshWidget;

    refreshInputDevices(node, { auto: true });
}

function applyDeviceList(node, devices, errorMessage = "") {
    const backendWidget = node.__ttsVoiceDeviceBackendWidget;
    const comboWidget = node.__ttsVoiceDeviceComboWidget;
    const refreshWidget = node.__ttsVoiceDeviceRefreshWidget;
    if (!backendWidget || !comboWidget || !refreshWidget) {
        return;
    }

    const currentBackendValue = (backendWidget.value || "").trim();
    const options = [DEFAULT_DEVICE_LABEL];
    const seen = new Set(options);

    for (const device of devices || []) {
        const normalized = String(device || "").trim();
        if (!normalized || seen.has(normalized)) {
            continue;
        }
        seen.add(normalized);
        options.push(normalized);
    }

    if (currentBackendValue && !seen.has(currentBackendValue)) {
        options.push(currentBackendValue);
    }

    comboWidget.options.values = options;
    comboWidget.value = backendValueToLabel(currentBackendValue);
    refreshWidget.name = errorMessage ? "Refresh Input Devices (retry)" : "Refresh Input Devices";

    if (errorMessage) {
        console.warn("ChatterBox Voice Capture: input device refresh failed:", errorMessage);
    }

    app.graph.setDirtyCanvas(true);
}

async function refreshInputDevices(node, { auto = false } = {}) {
    const comboWidget = node.__ttsVoiceDeviceComboWidget;
    const refreshWidget = node.__ttsVoiceDeviceRefreshWidget;
    if (!comboWidget || !refreshWidget) {
        return;
    }

    if (node.__ttsVoiceDeviceRefreshInFlight) {
        return;
    }

    node.__ttsVoiceDeviceRefreshInFlight = true;
    const previousLabel = comboWidget.value;
    comboWidget.options.values = [LOADING_DEVICE_LABEL];
    comboWidget.value = LOADING_DEVICE_LABEL;
    refreshWidget.name = auto ? "Refreshing Input Devices..." : "Refreshing...";
    app.graph.setDirtyCanvas(true);

    try {
        const response = await fetch("/api/tts-audio-suite/voice-input-devices");
        const result = await response.json();
        const devices = Array.isArray(result.devices) ? result.devices : [];
        if (!response.ok) {
            throw new Error(result.error || `HTTP ${response.status}`);
        }
        applyDeviceList(node, devices);
    } catch (error) {
        comboWidget.options.values = [previousLabel || DEFAULT_DEVICE_LABEL];
        comboWidget.value = previousLabel || DEFAULT_DEVICE_LABEL;
        refreshWidget.name = "Refresh Input Devices (retry)";
        console.warn("ChatterBox Voice Capture: unable to load input devices.", error);
        app.graph.setDirtyCanvas(true);
    } finally {
        node.__ttsVoiceDeviceRefreshInFlight = false;
    }
}

app.registerExtension({
    name: "ChatterBoxVoiceCapture.UI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!isVoiceCaptureNode(nodeData)) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            this.setSize([420, 340]);
            this.isRecording = false;
            this.recordingTimeout = null;

            setTimeout(() => {
                hideWidget(findWidgetByName(this, "voice_trigger"));
                ensureDeviceWidgets(this);
                app.graph.setDirtyCanvas(true);
            }, 100);

            return result;
        };

        nodeType.prototype.onDrawForeground = function(ctx) {
            const size = this.size;
            const w = size[0];
            const h = size[1];

            const buttonX = 20;
            const buttonY = h - 70;
            const buttonW = w - 40;
            const buttonH = 40;

            ctx.fillStyle = this.isRecording ? "#ff4444" : "#44aa44";
            ctx.fillRect(buttonX, buttonY, buttonW, buttonH);

            ctx.strokeStyle = this.isRecording ? "#ff0000" : "#00aa00";
            ctx.lineWidth = 2;
            ctx.strokeRect(buttonX, buttonY, buttonW, buttonH);

            ctx.fillStyle = "#ffffff";
            ctx.font = "bold 14px Arial";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";

            const text = this.isRecording ? "🔴 RECORDING..." : "🎙️ START RECORDING";
            ctx.fillText(text, w / 2, buttonY + buttonH / 2);

            this.buttonArea = [buttonX, buttonY, buttonW, buttonH];
        };

        nodeType.prototype.onMouseDown = function(event, localPos) {
            if (!this.buttonArea) return false;

            const [x, y, w, h] = this.buttonArea;
            if (localPos[0] < x || localPos[0] > x + w || localPos[1] < y || localPos[1] > y + h) {
                return false;
            }

            if (!this.isRecording) {
                this.isRecording = true;

                this.recordingTimeout = setTimeout(() => {
                    this.isRecording = false;
                    app.graph.setDirtyCanvas(true);
                }, 10000);

                const triggerWidget = findWidgetByName(this, "voice_trigger");
                if (triggerWidget) {
                    triggerWidget.value = (triggerWidget.value || 0) + 1;
                }

                app.queuePrompt();
            } else {
                this.isRecording = false;
                if (this.recordingTimeout) {
                    clearTimeout(this.recordingTimeout);
                }
            }

            app.graph.setDirtyCanvas(true);
            return true;
        };
    }
});

console.log("🎙️ ChatterBox: Extension loaded");
