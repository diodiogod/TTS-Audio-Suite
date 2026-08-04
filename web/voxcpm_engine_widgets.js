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

function isVoxCPM2(value) {
    const model = String(value || "").replace(/^local:/, "");
    return model === "VoxCPM2" || !["VoxCPM1.5", "VoxCPM-0.5B"].includes(model);
}

function refreshVoxCPMWidgets(node) {
    if (node.comfyClass !== "VoxCPMEngineNode") return;

    const modelWidget = findWidget(node, "model_variant");
    const modeWidget = findWidget(node, "mode");
    const voiceInstruction = findWidget(node, "voice_instruction");
    const supportsVoiceDesign = isVoxCPM2(modelWidget?.value);

    if (!supportsVoiceDesign && modeWidget?.value === "Voice Design") {
        modeWidget.value = "Text to Speech";
    }

    setWidgetEnabled(modeWidget, supportsVoiceDesign);
    setWidgetEnabled(
        voiceInstruction,
        supportsVoiceDesign && modeWidget?.value !== "Voice Design",
    );
    node.graph?.setDirtyCanvas?.(true, true);
}

function hookWidget(node, widget) {
    if (!widget || widget.__ttsVoxCPMHooked) return;
    widget.__ttsVoxCPMHooked = true;

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
            refreshVoxCPMWidgets(node);
        },
    });
}

function setupNode(node) {
    if (node.comfyClass !== "VoxCPMEngineNode") return;
    hookWidget(node, findWidget(node, "model_variant"));
    hookWidget(node, findWidget(node, "mode"));
    refreshVoxCPMWidgets(node);
}

app.registerExtension({
    name: "tts-audio-suite.voxcpm.widgets",
    nodeCreated(node) {
        if (node.comfyClass === "VoxCPMEngineNode") {
            setTimeout(() => setupNode(node), 0);
        }
    },
    loadedGraphNode(node) {
        if (node.comfyClass === "VoxCPMEngineNode") {
            setTimeout(() => setupNode(node), 0);
        }
    },
});
