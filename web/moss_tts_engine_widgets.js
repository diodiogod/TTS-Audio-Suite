import { app } from "../../scripts/app.js";

const MODEL_LABEL = "standard model";
const MODE_LABEL = "speaker mode";
const NATIVE_MODEL_LABEL = "native model (locked)";
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
        const computed = node.computeSize();
        const current = Array.isArray(node.size) ? node.size : computed;
        const nextSize = [
            Math.max(current[0] || 0, computed[0] || 0),
            Math.max(current[1] || 0, computed[1] || 0),
        ];
        if (nextSize[0] !== current[0] || nextSize[1] !== current[1]) {
            node.setSize(nextSize);
        }
    }
    if (node?.graph?.setDirtyCanvas) {
        node.graph.setDirtyCanvas(true, true);
    } else if (app.graph?.setDirtyCanvas) {
        app.graph.setDirtyCanvas(true, true);
    }
}

function isNativeModelOption(value) {
    return typeof value === "string" && value.includes("TTSD");
}

function getStoredStandardOptions(node, modelWidget) {
    if (!node.__ttsMossStandardModelOptions) {
        const values = Array.isArray(modelWidget?.options?.values) ? [...modelWidget.options.values] : [];
        node.__ttsMossStandardModelOptions = values.filter((value) => !isNativeModelOption(value));
    }
    return node.__ttsMossStandardModelOptions || [];
}

function getStoredNativeOption(node, modelWidget) {
    if (!node.__ttsMossNativeModelOption) {
        const values = Array.isArray(modelWidget?.options?.values) ? [...modelWidget.options.values] : [];
        node.__ttsMossNativeModelOption = values.find((value) => isNativeModelOption(value)) || "MOSS-TTSD-v1.0";
    }
    return node.__ttsMossNativeModelOption;
}

function toStandardModelValue(node, modelWidget, modelValue) {
    const standardOptions = getStoredStandardOptions(node, modelWidget);
    if (standardOptions.includes(modelValue)) {
        return modelValue;
    }
    return standardOptions[0];
}

function isLocalSmallModel(value) {
    return typeof value === "string" && value.includes("Local-Transformer");
}

function isSoundEffectModel(value) {
    return typeof value === "string"
        && (value.includes("MOSS-SoundEffect") || value.includes("Sound Effects"));
}

function isVoiceDesignModel(value) {
    return typeof value === "string"
        && (value.includes("MOSS-VoiceGenerator") || value.includes("Voice Design"));
}

const SPEECH_ONLY_WIDGETS = [
    "multi_speaker_mode", "language", "duration_tokens", "chunk_minutes",
    "instruction", "quality", "sound_event", "ambient_sound",
    "speaker2_voice", "speaker3_voice", "speaker4_voice", "speaker5_voice",
    "local_lora_adapter", "lora_adapter_override",
];

function updateSoundEffectState(node, soundEffectSelected) {
    for (const name of SPEECH_ONLY_WIDGETS) {
        setWidgetEnabled(findWidgetByName(node, name), !soundEffectSelected);
    }
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

        const standardOptions = getStoredStandardOptions(node, modelWidget);
        const nativeOption = getStoredNativeOption(node, modelWidget);
        if (!standardOptions.length) {
            return;
        }

        modelWidget.label = MODEL_LABEL;
        modeWidget.label = MODE_LABEL;

        const modeValue = modeWidget.value;
        const isNativeDialogue = modeValue === "Native Multi-Speaker Dialogue";

        if (isNativeDialogue) {
            const currentStandard = toStandardModelValue(node, modelWidget, modelWidget.value);
            node.__ttsMossLastStandardModel = currentStandard;
            modelWidget.options = modelWidget.options || {};
            modelWidget.options.values = [nativeOption];
            if (modelWidget.value !== nativeOption) {
                modelWidget.value = nativeOption;
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
        modelWidget.options.values = [...standardOptions];

        const restoredStandard =
            node.__ttsMossLastStandardModel
            || toStandardModelValue(node, modelWidget, modelWidget.value);

        if (!standardOptions.includes(modelWidget.value)) {
            modelWidget.value = restoredStandard;
        }

        const selectedStandard = toStandardModelValue(node, modelWidget, modelWidget.value);
        node.__ttsMossLastStandardModel = selectedStandard;
        const soundEffectSelected = isSoundEffectModel(selectedStandard);
        updateSoundEffectState(node, soundEffectSelected);
        if (!soundEffectSelected) {
            setWidgetEnabled(
                findWidgetByName(node, "instruction"),
                !isVoiceDesignModel(selectedStandard),
            );
        }

        if (isLocalSmallModel(selectedStandard) || selectedStandard === "1.7B") {
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
