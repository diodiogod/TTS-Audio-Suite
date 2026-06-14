import { app } from "../../scripts/app.js";

const LABEL_OVERRIDES = {
    RVCTrainingConfigNode: {
        save_every_epoch: "checkpoint every N epochs",
        max_checkpoints: "keep max checkpoints",
        save_every_weights: "export extra weights on each save",
    },
};

function relabelWidgets(node) {
    const overrides = LABEL_OVERRIDES[node?.comfyClass];
    if (!overrides) {
        return;
    }

    const widgets = node?.widgets || [];
    for (const widget of widgets) {
        if (!widget?.name) {
            continue;
        }
        const overrideLabel = overrides[widget.name];
        if (overrideLabel) {
            widget.label = overrideLabel;
        }
    }
}

app.registerExtension({
    name: "TTS_Audio_Suite.RVCTrainingConfigLabels",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!LABEL_OVERRIDES[nodeData.name]) {
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
