import { app } from "../../scripts/app.js";

// Qwen3-TTS Engine Node UI Control
// Lock/unlock instruct field based on voice_preset and model_size

function qwen3TTSHandler(node) {
    if (node.comfyClass !== "Qwen3TTSEngineNode") {
        return;
    }

    // Find widgets
    const voicePresetWidget = findWidgetByName(node, "voice_preset");
    const modelSizeWidget = findWidgetByName(node, "model_size");
    const instructWidget = findWidgetByName(node, "instruct");

    if (!voicePresetWidget || !modelSizeWidget || !instructWidget) return;

    const voicePreset = voicePresetWidget.value;
    const modelSize = modelSizeWidget.value;

    // Lock/unlock logic:
    // LOCKED if:
    //   - voice_preset = "None (Zero-shot / Custom)" (Base model has no instruction)
    //   - model_size = "0.6B" + preset selected (0.6B CustomVoice has no instruction)
    // UNLOCKED if:
    //   - model_size = "1.7B" + preset selected (1.7B CustomVoice supports instruction)

    let shouldLock = false;
    let lockReason = "";

    if (voicePreset === "None (Zero-shot / Custom)") {
        shouldLock = true;
        lockReason = "Base model (zero-shot) does not support instruction parameter";
    } else if (modelSize === "0.6B") {
        shouldLock = true;
        lockReason = "0.6B CustomVoice does not support instruction parameter (requires 1.7B)";
    }

    // Lock/unlock the instruct field (preserve value when locked)
    instructWidget.disabled = shouldLock;

    // Log to console for debugging
    if (shouldLock) {
        console.log(`⚠️ Qwen3-TTS: Instruction field locked - ${lockReason}`);
    }
}

const findWidgetByName = (node, name) => {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
};

app.registerExtension({
    name: "tts-audio-suite.qwen3-tts.widgets",
    nodeCreated(node) {
        if (node.comfyClass !== "Qwen3TTSEngineNode") {
            return;
        }

        // Initial setup
        qwen3TTSHandler(node);

        // Intercept both voice_preset and model_size widgets
        const voicePresetWidget = findWidgetByName(node, "voice_preset");
        const modelSizeWidget = findWidgetByName(node, "model_size");

        // Setup voice_preset interceptor
        if (voicePresetWidget) {
            let widgetValue = voicePresetWidget.value;

            // Store the original descriptor
            let originalDescriptor = Object.getOwnPropertyDescriptor(voicePresetWidget, 'value') ||
                Object.getOwnPropertyDescriptor(Object.getPrototypeOf(voicePresetWidget), 'value');
            if (!originalDescriptor) {
                originalDescriptor = Object.getOwnPropertyDescriptor(voicePresetWidget.constructor.prototype, 'value');
            }

            Object.defineProperty(voicePresetWidget, 'value', {
                get() {
                    let valueToReturn = originalDescriptor && originalDescriptor.get
                        ? originalDescriptor.get.call(voicePresetWidget)
                        : widgetValue;
                    return valueToReturn;
                },
                set(newVal) {
                    if (originalDescriptor && originalDescriptor.set) {
                        originalDescriptor.set.call(voicePresetWidget, newVal);
                    } else {
                        widgetValue = newVal;
                    }
                    qwen3TTSHandler(node); // Re-evaluate lock state
                }
            });
        }

        // Setup model_size interceptor
        if (modelSizeWidget) {
            let widgetValue = modelSizeWidget.value;

            // Store the original descriptor
            let originalDescriptor = Object.getOwnPropertyDescriptor(modelSizeWidget, 'value') ||
                Object.getOwnPropertyDescriptor(Object.getPrototypeOf(modelSizeWidget), 'value');
            if (!originalDescriptor) {
                originalDescriptor = Object.getOwnPropertyDescriptor(modelSizeWidget.constructor.prototype, 'value');
            }

            Object.defineProperty(modelSizeWidget, 'value', {
                get() {
                    let valueToReturn = originalDescriptor && originalDescriptor.get
                        ? originalDescriptor.get.call(modelSizeWidget)
                        : widgetValue;
                    return valueToReturn;
                },
                set(newVal) {
                    if (originalDescriptor && originalDescriptor.set) {
                        originalDescriptor.set.call(modelSizeWidget, newVal);
                    } else {
                        widgetValue = newVal;
                    }
                    qwen3TTSHandler(node); // Re-evaluate lock state
                }
            });
        }
    }
});
