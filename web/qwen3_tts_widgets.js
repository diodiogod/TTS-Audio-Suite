import { app } from "../../scripts/app.js";

const LEGACY_MODEL_VALUES = new Set(["1.7B", "0.6B"]);

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

function modelKind(value) {
    const text = String(value || "");
    if (text.includes("VoiceDesign") || text.includes("Voice Design")) return "VoiceDesign";
    if (text.includes("CustomVoice")) return "CustomVoice";
    if (text.includes("Base")) return "Base";
    return "";
}

function modelSize(value) {
    return String(value || "").includes("0.6B") ? "0.6B" : "1.7B";
}

function migrateLegacyModel(node, modelWidget, presetWidget) {
    if (!LEGACY_MODEL_VALUES.has(modelWidget.value)) return;
    const size = modelWidget.value;
    const kind = presetWidget?.value && presetWidget.value !== "None (Zero-shot / Custom)"
        ? "CustomVoice"
        : "Base";
    const options = Array.isArray(modelWidget.options?.values) ? modelWidget.options.values : [];
    const target = options.find((value) => {
        const text = String(value);
        return !LEGACY_MODEL_VALUES.has(text) && modelKind(text) === kind && modelSize(text) === size;
    });
    if (target) modelWidget.value = target;
}

function refreshQwenWidgets(node) {
    if (node.comfyClass !== "Qwen3TTSEngineNode" || node.__ttsQwenRefreshing) return;
    node.__ttsQwenRefreshing = true;
    try {
        const modelWidget = findWidget(node, "model_variant");
        const presetWidget = findWidget(node, "voice_preset");
        const instructWidget = findWidget(node, "instruct");
        const xVectorWidget = findWidget(node, "x_vector_only_mode");
        if (!modelWidget) return;

        migrateLegacyModel(node, modelWidget, presetWidget);
        if (Array.isArray(modelWidget.options?.values)) {
            modelWidget.options.values = modelWidget.options.values.filter(
                (value) => !LEGACY_MODEL_VALUES.has(String(value)),
            );
        }

        modelWidget.label = "model";
        const kind = modelKind(modelWidget.value);
        const size = modelSize(modelWidget.value);

        if (kind === "CustomVoice") {
            setWidgetEnabled(presetWidget, true);
            setWidgetEnabled(xVectorWidget, false);
            if (presetWidget?.value === "None (Zero-shot / Custom)") presetWidget.value = "Vivian";
            setWidgetEnabled(instructWidget, size === "1.7B");
        } else if (kind === "VoiceDesign") {
            setWidgetEnabled(presetWidget, false);
            setWidgetEnabled(xVectorWidget, false);
            setWidgetEnabled(instructWidget, false);
        } else {
            setWidgetEnabled(presetWidget, false);
            setWidgetEnabled(xVectorWidget, true);
            setWidgetEnabled(instructWidget, false);
        }

        node.setDirtyCanvas?.(true, true);
        node.graph?.setDirtyCanvas?.(true, true);
    } finally {
        node.__ttsQwenRefreshing = false;
    }
}

function hookWidget(node, widget) {
    if (!widget || widget.__ttsQwenHooked) return;
    widget.__ttsQwenHooked = true;
    let storedValue = widget.value;
    let descriptor = Object.getOwnPropertyDescriptor(widget, "value")
        || Object.getOwnPropertyDescriptor(Object.getPrototypeOf(widget), "value")
        || Object.getOwnPropertyDescriptor(widget.constructor?.prototype || {}, "value");
    Object.defineProperty(widget, "value", {
        get() {
            return descriptor?.get ? descriptor.get.call(widget) : storedValue;
        },
        set(value) {
            if (descriptor?.set) descriptor.set.call(widget, value);
            else storedValue = value;
            if (!node.__ttsQwenRefreshing) refreshQwenWidgets(node);
        },
    });
}

function setupNode(node) {
    if (node.comfyClass !== "Qwen3TTSEngineNode") return;
    hookWidget(node, findWidget(node, "model_variant"));
    hookWidget(node, findWidget(node, "voice_preset"));
    refreshQwenWidgets(node);
}

app.registerExtension({
    name: "tts-audio-suite.qwen3-tts.widgets",
    nodeCreated(node) {
        if (node.comfyClass === "Qwen3TTSEngineNode") setTimeout(() => setupNode(node), 0);
    },
    loadedGraphNode(node) {
        if (node.comfyClass === "Qwen3TTSEngineNode") setTimeout(() => setupNode(node), 0);
    },
});
