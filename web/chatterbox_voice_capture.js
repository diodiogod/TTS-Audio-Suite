// ChatterBox Voice Capture Extension

import { app } from "../../scripts/app.js";
import { setupVoiceCaptureControls } from "./voice_capture_controls.js";

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
    nodeCreated(node) {
        if (!isVoiceCaptureNode(node)) return;
        ensureDeviceWidgets(node);
        setupVoiceCaptureControls(node);
    },
});
