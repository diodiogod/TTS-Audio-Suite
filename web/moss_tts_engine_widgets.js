import { app } from "../../scripts/app.js";

const STANDARD_MODEL_OPTIONS = ["Small 1.7B (Local)", "8B (Delay)"];
const MODEL_LABEL = "standard model";
const MODE_LABEL = "speaker mode";
const NATIVE_MODEL_LABEL = "native model (locked)";
const NATIVE_MODEL_VALUE = "Native 8B Dialogue (MOSS-TTSD-v1.0)";
const LOCAL_N_VQ_WIDGET = "n_vq_for_inference";

function findWidgetByName(node, name) {
    return node.widgets ? node.widgets.find((widget) => widget.name === name) : null;
}

function hideWidget(widget) {
    if (!widget) {
        return;
    }
    if (!widget.__ttsOriginalType) {
        widget.__ttsOriginalType = widget.type;
    }
    if (!widget.__ttsOriginalComputeSize) {
        widget.__ttsOriginalComputeSize = widget.computeSize;
    }
    widget.type = "hidden";
    widget.computeSize = () => [0, -4];
    widget.disabled = true;
    if (widget.element) {
        widget.element.style.display = "none";
    }
}

function showWidget(widget) {
    if (!widget) {
        return;
    }
    if (widget.__ttsOriginalType) {
        widget.type = widget.__ttsOriginalType;
    }
    if (widget.__ttsOriginalComputeSize) {
        widget.computeSize = widget.__ttsOriginalComputeSize;
    }
    widget.disabled = false;
    if (widget.element) {
        widget.element.style.display = "";
    }
}

function setWidgetEnabled(widget, enabled) {
    if (!widget) {
        return;
    }
    widget.disabled = !enabled;
    if (widget.element) {
        widget.element.style.opacity = enabled ? "1" : "0.55";
        widget.element.style.pointerEvents = enabled ? "auto" : "none";
    }
}

function resizeNode(node) {
    if (typeof node?.computeSize === "function" && typeof node?.setSize === "function") {
        node.setSize(node.computeSize());
    }
}

function toStandardModelValue(modelValue) {
    if (STANDARD_MODEL_OPTIONS.includes(modelValue)) {
        return modelValue;
    }
    if (typeof modelValue !== "string") {
        return STANDARD_MODEL_OPTIONS[0];
    }
    if (modelValue === "MOSS-TTS" || modelValue === "8B (Delay)") {
        return "8B (Delay)";
    }
    if (modelValue === NATIVE_MODEL_VALUE || modelValue === "MOSS-TTSD-v1.0") {
        return STANDARD_MODEL_OPTIONS[0];
    }
    return "Small 1.7B (Local)";
}

function refreshMossWidgets(node) {
    if (node.comfyClass !== "MossTTSEngineNode") {
        return;
    }
    if (node.__ttsMossRefreshing) {
        return;
    }
    node.__ttsMossRefreshing = true;

    try {
        const modelWidget = findWidgetByName(node, "model_variant");
        const modeWidget = findWidgetByName(node, "multi_speaker_mode");
        const nVqWidget = findWidgetByName(node, LOCAL_N_VQ_WIDGET);

        if (!modelWidget || !modeWidget) {
            return;
        }

        modelWidget.label = MODEL_LABEL;
        modeWidget.label = MODE_LABEL;

        const modeValue = modeWidget.value;
        const isNativeDialogue = modeValue === "Native Multi-Speaker Dialogue";

        if (isNativeDialogue) {
            const currentStandard = toStandardModelValue(modelWidget.value);
            node.__ttsMossLastStandardModel = currentStandard;
            modelWidget.options = modelWidget.options || {};
            modelWidget.options.values = [NATIVE_MODEL_VALUE];
            if (modelWidget.value !== NATIVE_MODEL_VALUE) {
                modelWidget.value = NATIVE_MODEL_VALUE;
            }
            modelWidget.label = NATIVE_MODEL_LABEL;
            showWidget(modelWidget);
            setWidgetEnabled(modelWidget, false);
            hideWidget(nVqWidget);
            resizeNode(node);
            return;
        }

        showWidget(modelWidget);
        setWidgetEnabled(modelWidget, true);
        modelWidget.label = MODEL_LABEL;
        modelWidget.options = modelWidget.options || {};
        modelWidget.options.values = [...STANDARD_MODEL_OPTIONS];

        const restoredStandard =
            node.__ttsMossLastStandardModel
            || toStandardModelValue(modelWidget.value);

        if (!STANDARD_MODEL_OPTIONS.includes(modelWidget.value)) {
            modelWidget.value = restoredStandard;
        }

        const selectedStandard = toStandardModelValue(modelWidget.value);
        node.__ttsMossLastStandardModel = selectedStandard;

        if (selectedStandard === "Small 1.7B (Local)") {
            showWidget(nVqWidget);
        } else {
            hideWidget(nVqWidget);
        }

        resizeNode(node);
    } finally {
        node.__ttsMossRefreshing = false;
    }
}

function hookWidgetValue(node, widget, callback) {
    if (!widget || widget.__ttsMossHooked) {
        return;
    }
    widget.__ttsMossHooked = true;

    let widgetValue = widget.value;
    let originalDescriptor = Object.getOwnPropertyDescriptor(widget, "value")
        || Object.getOwnPropertyDescriptor(Object.getPrototypeOf(widget), "value");
    if (!originalDescriptor) {
        originalDescriptor = Object.getOwnPropertyDescriptor(widget.constructor?.prototype || {}, "value");
    }

    Object.defineProperty(widget, "value", {
        get() {
            if (originalDescriptor?.get) {
                return originalDescriptor.get.call(widget);
            }
            return widgetValue;
        },
        set(newValue) {
            if (originalDescriptor?.set) {
                originalDescriptor.set.call(widget, newValue);
            } else {
                widgetValue = newValue;
            }
            if (!node.__ttsMossRefreshing) {
                callback(node);
            }
        },
    });
}

app.registerExtension({
    name: "tts-audio-suite.moss-tts.widgets",
    nodeCreated(node) {
        if (node.comfyClass !== "MossTTSEngineNode") {
            return;
        }

        refreshMossWidgets(node);

        hookWidgetValue(node, findWidgetByName(node, "multi_speaker_mode"), refreshMossWidgets);
        hookWidgetValue(node, findWidgetByName(node, "model_variant"), refreshMossWidgets);
    },
});
