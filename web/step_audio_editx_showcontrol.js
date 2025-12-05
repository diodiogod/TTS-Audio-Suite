import { app } from "../../scripts/app.js";

// Step Audio EditX Audio Editor Node UI Control
// Grey out options that are not relevant for the selected edit_type

function stepAudioEditXHandler(node) {
    if (node.comfyClass !== "StepAudioEditXAudioEditorNode") {
        return;
    }

    // Find widgets
    const editTypeWidget = findWidgetByName(node, "edit_type");
    const emotionWidget = findWidgetByName(node, "emotion");
    const styleWidget = findWidgetByName(node, "style");
    const speedWidget = findWidgetByName(node, "speed");
    const nIterationsWidget = findWidgetByName(node, "n_edit_iterations");

    if (!editTypeWidget) return;

    const editType = editTypeWidget.value;

    // Initially hide all conditional widgets
    toggleWidget(node, emotionWidget, false);
    toggleWidget(node, styleWidget, false);
    toggleWidget(node, speedWidget, false);

    // Show relevant widgets based on edit_type
    if (editType === "emotion") {
        toggleWidget(node, emotionWidget, true);
    } else if (editType === "style") {
        toggleWidget(node, styleWidget, true);
    } else if (editType === "speed") {
        toggleWidget(node, speedWidget, true);
    }
    // For paralinguistic, denoise, vad: no dropdowns needed

    // n_edit_iterations is relevant for all edit types
    // Keep it enabled for all
    if (nIterationsWidget) {
        toggleWidget(node, nIterationsWidget, true);
    }
}

const findWidgetByName = (node, name) => {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
};

// Toggle Widget disabled state (greys out)
function toggleWidget(node, widget, show = false) {
    if (!widget) return;
    widget.disabled = !show;
}

app.registerExtension({
    name: "tts-audio-suite.step-audio-editx.showcontrol",
    nodeCreated(node) {
        if (node.comfyClass !== "StepAudioEditXAudioEditorNode") {
            return;
        }

        // Initial setup
        stepAudioEditXHandler(node);

        // Only intercept edit_type widget for efficiency (like chatterbox pattern)
        const editTypeWidget = findWidgetByName(node, "edit_type");
        if (editTypeWidget) {
            let widgetValue = editTypeWidget.value;

            // Store the original descriptor if it exists
            let originalDescriptor = Object.getOwnPropertyDescriptor(editTypeWidget, 'value') ||
                Object.getOwnPropertyDescriptor(Object.getPrototypeOf(editTypeWidget), 'value');
            if (!originalDescriptor) {
                originalDescriptor = Object.getOwnPropertyDescriptor(editTypeWidget.constructor.prototype, 'value');
            }

            Object.defineProperty(editTypeWidget, 'value', {
                get() {
                    let valueToReturn = originalDescriptor && originalDescriptor.get
                        ? originalDescriptor.get.call(editTypeWidget)
                        : widgetValue;
                    return valueToReturn;
                },
                set(newVal) {
                    if (originalDescriptor && originalDescriptor.set) {
                        originalDescriptor.set.call(editTypeWidget, newVal);
                    } else {
                        widgetValue = newVal;
                    }
                    stepAudioEditXHandler(node); // Re-evaluate visibility
                }
            });
        }
    }
});