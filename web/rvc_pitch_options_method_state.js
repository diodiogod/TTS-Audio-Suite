import { app } from "../../scripts/app.js";

const TARGET_CLASS = "RVCPitchOptionsNode";

function findWidget(node, name) {
    return node.widgets ? node.widgets.find((widget) => widget.name === name) : null;
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

function applyPitchMethodState(node) {
    if (!node || node.comfyClass !== TARGET_CLASS) {
        return;
    }

    const methodWidget = findWidget(node, "pitch_detection");
    if (!methodWidget) {
        return;
    }

    const method = String(methodWidget.value || "rmvpe").toLowerCase();
    const isCrepeFamily = method.includes("crepe");
    const isHarvest = method === "harvest";

    setWidgetEnabled(findWidget(node, "crepe_hop_length"), isCrepeFamily);
    setWidgetEnabled(findWidget(node, "batch_size"), isCrepeFamily);
    setWidgetEnabled(findWidget(node, "filter_radius"), isHarvest);

    node.setDirtyCanvas?.(true, true);
    node.graph?.setDirtyCanvas?.(true, true);
}

function hookWidgetValue(node, widget, callback) {
    if (!widget || widget.__ttsHooked) {
        return;
    }
    widget.__ttsHooked = true;

    let widgetValue = widget.value;
    const originalDescriptor = Object.getOwnPropertyDescriptor(widget, "value")
        || Object.getOwnPropertyDescriptor(Object.getPrototypeOf(widget), "value")
        || Object.getOwnPropertyDescriptor(widget.constructor?.prototype || {}, "value");

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
            callback(node);
        },
    });
}

app.registerExtension({
    name: "TTS_Audio_Suite.RVCPitchOptionsMethodState",

    nodeCreated(node) {
        if (node.comfyClass !== TARGET_CLASS) {
            return;
        }

        const methodWidget = findWidget(node, "pitch_detection");
        if (methodWidget) {
            hookWidgetValue(node, methodWidget, applyPitchMethodState);
        }

        applyPitchMethodState(node);
    },
});
