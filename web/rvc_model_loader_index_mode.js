import { app } from "../../scripts/app.js";

const TARGET_CLASS = "LoadRVCModelNode";

function findWidget(node, name) {
    return node.widgets ? node.widgets.find((widget) => widget.name === name) : null;
}

function applyIndexModeState(node) {
    if (!node || node.comfyClass !== TARGET_CLASS) {
        return;
    }

    const modeWidget = findWidget(node, "index_mode");
    const indexWidget = findWidget(node, "index_file");
    if (!modeWidget || !indexWidget) {
        return;
    }

    const customEnabled = String(modeWidget.value || "auto") === "custom";
    indexWidget.disabled = !customEnabled;

    if (indexWidget.element) {
        indexWidget.element.style.opacity = customEnabled ? "1" : "0.55";
        indexWidget.element.style.pointerEvents = customEnabled ? "auto" : "none";
    }

    node.setDirtyCanvas?.(true, true);
    node.graph?.setDirtyCanvas?.(true, true);
}

function hookWidgetValue(node, widget, callback) {
    if (!widget || widget.__ttsHooked) {
        return;
    }
    widget.__ttsHooked = true;

    let widgetValue = widget.value;
    let originalDescriptor = Object.getOwnPropertyDescriptor(widget, "value")
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
    name: "TTS_Audio_Suite.RVCModelLoaderIndexMode",

    nodeCreated(node) {
        if (node.comfyClass !== TARGET_CLASS) {
            return;
        }

        const modeWidget = findWidget(node, "index_mode");
        if (modeWidget) {
            hookWidgetValue(node, modeWidget, applyIndexModeState);
        }

        applyIndexModeState(node);
    },
});
