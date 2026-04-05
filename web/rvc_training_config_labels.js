import { app } from "../../scripts/app.js";

const TARGET_CLASS = "RVCTrainingConfigNode";

function relabelWidgets(node) {
    const widgets = node?.widgets || [];
    for (const widget of widgets) {
        if (!widget?.name) {
            continue;
        }
        if (widget.name === "save_every_epoch") {
            widget.label = "checkpoint every N epochs";
        } else if (widget.name === "max_checkpoints") {
            widget.label = "keep max checkpoints";
        } else if (widget.name === "save_every_weights") {
            widget.label = "export extra weights on each save";
        }
    }
}

app.registerExtension({
    name: "TTS_Audio_Suite.RVCTrainingConfigLabels",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== TARGET_CLASS) {
            return;
        }

        const originalNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = originalNodeCreated ? originalNodeCreated.apply(this, arguments) : undefined;
            relabelWidgets(this);
            return result;
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function() {
            const result = originalOnConfigure ? originalOnConfigure.apply(this, arguments) : undefined;
            relabelWidgets(this);
            return result;
        };
    },
});
