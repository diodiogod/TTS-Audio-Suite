import { app } from "../../scripts/app.js";

function findWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function setWidgetEnabled(widget, enabled) {
    if (!widget) return;
    widget.disabled = !enabled;
    if (widget.element) {
        widget.element.style.opacity = enabled ? "1" : "0.55";
        widget.element.style.pointerEvents = enabled ? "auto" : "none";
    }
}

function refreshOmniVoiceWidgets(node) {
    if (node.comfyClass !== "OmniVoiceEngineNode") return;
    const modeWidget = findWidget(node, "mode");
    setWidgetEnabled(findWidget(node, "instruct"), modeWidget?.value !== "Voice Design");
    node.graph?.setDirtyCanvas?.(true, true);
}

function hookMode(node) {
    const widget = findWidget(node, "mode");
    if (!widget || widget.__ttsOmniModeHooked) return;
    widget.__ttsOmniModeHooked = true;
    let storedValue = widget.value;
    const descriptor = Object.getOwnPropertyDescriptor(widget, "value")
        || Object.getOwnPropertyDescriptor(Object.getPrototypeOf(widget), "value")
        || Object.getOwnPropertyDescriptor(widget.constructor?.prototype || {}, "value");
    Object.defineProperty(widget, "value", {
        get() {
            return descriptor?.get ? descriptor.get.call(widget) : storedValue;
        },
        set(value) {
            if (descriptor?.set) descriptor.set.call(widget, value);
            else storedValue = value;
            refreshOmniVoiceWidgets(node);
        },
    });
}

function setupNode(node) {
    if (node.comfyClass !== "OmniVoiceEngineNode") return;
    hookMode(node);
    refreshOmniVoiceWidgets(node);
}

app.registerExtension({
    name: "tts-audio-suite.omnivoice.widgets",
    nodeCreated(node) {
        if (node.comfyClass === "OmniVoiceEngineNode") setTimeout(() => setupNode(node), 0);
    },
    loadedGraphNode(node) {
        if (node.comfyClass === "OmniVoiceEngineNode") setTimeout(() => setupNode(node), 0);
    },
});
