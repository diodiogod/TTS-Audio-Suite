import { app } from "../../scripts/app.js";

const TARGET_CLASSES = new Set([
    "MossClipStagingNode",
]);

const AUDIO_INPUT_PREFIX = "opt_audio";
const AUDIO_INPUT_TYPE = "AUDIO";

function isTargetNode(nodeOrData) {
    const comfyClass = nodeOrData?.comfyClass || nodeOrData?.name;
    return TARGET_CLASSES.has(comfyClass);
}

function isDynamicAudioInput(input) {
    return Boolean(input?.name && String(input.name).startsWith(AUDIO_INPUT_PREFIX));
}

function getDynamicAudioInputs(node) {
    return (node.inputs || []).filter(isDynamicAudioInput);
}

function renumberDynamicAudioInputs(node) {
    let nextIndex = 1;
    for (const input of node.inputs || []) {
        if (!isDynamicAudioInput(input)) {
            continue;
        }
        input.name = `${AUDIO_INPUT_PREFIX}${nextIndex}`;
        input.label = input.name;
        nextIndex += 1;
    }
}

function ensureTrailingDynamicAudioInput(node) {
    const audioInputs = getDynamicAudioInputs(node);
    if (!audioInputs.length || audioInputs[audioInputs.length - 1].link != null) {
        const newIndex = audioInputs.length + 1;
        node.addInput(`${AUDIO_INPUT_PREFIX}${newIndex}`, AUDIO_INPUT_TYPE);
    }
}

function syncDynamicAudioInputs(node) {
    if (!node || node.__ttsMossSyncingAudioInputs) {
        return;
    }

    node.__ttsMossSyncingAudioInputs = true;
    try {
        for (let index = (node.inputs || []).length - 1; index >= 0; index -= 1) {
            const input = node.inputs[index];
            if (isDynamicAudioInput(input) && input.link == null) {
                node.removeInput(index);
            }
        }

        renumberDynamicAudioInputs(node);
        ensureTrailingDynamicAudioInput(node);
        renumberDynamicAudioInputs(node);

        if (typeof node.computeSize === "function" && typeof node.setSize === "function") {
            const computed = node.computeSize();
            const currentSize = Array.isArray(node.size) ? node.size : null;
            if (!currentSize) {
                node.setSize(computed);
            } else {
                const nextWidth = Math.max(currentSize[0] || 0, computed[0] || 0);
                const nextHeight = Math.max(currentSize[1] || 0, computed[1] || 0);
                if (nextWidth !== currentSize[0] || nextHeight !== currentSize[1]) {
                    node.setSize([nextWidth, nextHeight]);
                }
            }
        }
        app.graph.setDirtyCanvas(true, true);
    } finally {
        node.__ttsMossSyncingAudioInputs = false;
    }
}

app.registerExtension({
    name: "TTS_Audio_Suite.MossTrainingAudioInputs",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!isTargetNode(nodeData)) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            setTimeout(() => {
                syncDynamicAudioInputs(this);
                app.graph.setDirtyCanvas(true);
            }, 100);
            return result;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function(type, index, connected, linkInfo) {
            const result = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
            const input = this.inputs?.[index];
            if (type === 2 || !isDynamicAudioInput(input)) {
                return result;
            }

            const stackTrace = new Error().stack || "";
            if (stackTrace.includes("loadGraphData") || stackTrace.includes("pasteFromClipboard")) {
                setTimeout(() => syncDynamicAudioInputs(this), 0);
                return result;
            }

            if (!linkInfo && !connected) {
                setTimeout(() => syncDynamicAudioInputs(this), 0);
                return result;
            }

            syncDynamicAudioInputs(this);
            return result;
        };
    },
});
