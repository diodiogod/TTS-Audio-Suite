import { app } from "../../scripts/app.js";

function chatterboxSRTShowControl(node) {
    if (node.comfyClass === "ChatterboxSRTTTSNode") {
        const timingModeWidget = findWidgetByName(node, "timing_mode");
        if (!timingModeWidget) return;

        const fadeDurationWidget = findWidgetByName(node, "fade_duration");
        const maxStretchRatioWidget = findWidgetByName(node, "max_stretch_ratio");
        const minStretchRatioWidget = findWidgetByName(node, "min_stretch_ratio");
        const timingToleranceWidget = findWidgetByName(node, "timing_tolerance");

        // Hide all potentially conditional widgets first
        toggleWidget(node, fadeDurationWidget, false);
        toggleWidget(node, maxStretchRatioWidget, false);
        toggleWidget(node, minStretchRatioWidget, false);
        toggleWidget(node, timingToleranceWidget, false);

        // Apply visibility based on timing_mode
        if (timingModeWidget.value === "stretch_to_fit") {
            toggleWidget(node, fadeDurationWidget, true);
        } else if (timingModeWidget.value === "smart_natural") {
            toggleWidget(node, maxStretchRatioWidget, true);
            toggleWidget(node, minStretchRatioWidget, true);
            toggleWidget(node, timingToleranceWidget, true);
        }
    }
}

const findWidgetByName = (node, name) => {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
};

// Toggle Widget + change size
function toggleWidget(node, widget, show = false) {
    if (!widget) return;
    widget.disabled = !show;
    // ComfyUI widgets often have an 'element' property for their DOM representation
    if (widget.element) {
        widget.element.style.display = show ? "block" : "none";
    }
}

app.registerExtension({
    name: "chatterbox-srt.showcontrol",
    nodeCreated(node) {
        if (node.comfyClass === "ChatterboxSRTTTSNode") {
            chatterboxSRTShowControl(node);

            // Intercept value changes to update visibility dynamically
            for (const w of node.widgets || []) {
                // Only apply to the timing_mode widget for efficiency
                if (w.name === "timing_mode") {
                    let widgetValue = w.value;

                    // Store the original descriptor if it exists
                    let originalDescriptor = Object.getOwnPropertyDescriptor(w, 'value') ||
                        Object.getOwnPropertyDescriptor(Object.getPrototypeOf(w), 'value');
                    if (!originalDescriptor) {
                        originalDescriptor = Object.getOwnPropertyDescriptor(w.constructor.prototype, 'value');
                    }

                    Object.defineProperty(w, 'value', {
                        get() {
                            let valueToReturn = originalDescriptor && originalDescriptor.get
                                ? originalDescriptor.get.call(w)
                                : widgetValue;
                            return valueToReturn;
                        },
                        set(newVal) {
                            if (originalDescriptor && originalDescriptor.set) {
                                originalDescriptor.set.call(w, newVal);
                            } else {
                                widgetValue = newVal;
                            }
                            chatterboxSRTShowControl(node); // Re-evaluate visibility
                        }
                    });
                }
            }
        }
    }
});