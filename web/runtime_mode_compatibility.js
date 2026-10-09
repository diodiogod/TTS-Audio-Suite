import { app } from "../../scripts/app.js";

const ENGINE_NODES = new Set(["Qwen3TTSEngineNode", "StepAudioEditXEngineNode"]);
const LEGACY_DEDICATED_VALUES = new Set([
    "⚠️ Dedicated Runtime", "Dedicated Runtime", "dedicated_runtime",
]);

function migrateRuntimeWidget(node) {
    if (!ENGINE_NODES.has(node.comfyClass)) return;
    const widget = node.widgets?.find((candidate) => candidate.name === "runtime_mode");
    if (!widget || !LEGACY_DEDICATED_VALUES.has(widget.value)) return;
    widget.value = "⚠️ Shared Runtime";
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "tts-audio-suite.runtime-mode.compatibility",
    loadedGraphNode(node) {
        migrateRuntimeWidget(node);
    },
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!ENGINE_NODES.has(nodeData.name)) return;
        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalConfigure?.apply(this, arguments);
            migrateRuntimeWidget(this);
            return result;
        };
    },
});
