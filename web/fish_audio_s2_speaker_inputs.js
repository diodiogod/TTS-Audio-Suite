import { app } from "../../scripts/app.js";

const TARGET_CLASS = "FishAudioS2EngineNode";
const PREFIX = "speaker";

function isSpeakerInput(input) {
    return Boolean(input?.name && new RegExp(`^${PREFIX}\\d+$`).test(String(input.name)));
}

function syncSpeakerInputs(node) {
    if (!node || node.__ttsFishSyncingSpeakers) return;
    node.__ttsFishSyncingSpeakers = true;
    try {
        for (let index = (node.inputs || []).length - 1; index >= 0; index -= 1) {
            const input = node.inputs[index];
            if (isSpeakerInput(input) && input.link == null) node.removeInput(index);
        }
        const speakers = (node.inputs || []).filter(isSpeakerInput);
        speakers.forEach((input, index) => {
            input.name = `${PREFIX}${index + 2}`;
            input.label = `Speaker ${index + 2}`;
        });
        if (!speakers.length || speakers[speakers.length - 1].link != null) {
            node.addInput(`${PREFIX}${speakers.length + 2}`, "*");
        }
        app.graph.setDirtyCanvas(true, true);
    } finally {
        node.__ttsFishSyncingSpeakers = false;
    }
}

app.registerExtension({
    name: "TTS_Audio_Suite.FishAudioS2SpeakerInputs",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if ((nodeData?.name || nodeData?.comfyClass) !== TARGET_CLASS) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            setTimeout(() => syncSpeakerInputs(this), 100);
            return result;
        };
        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function(type, index, connected, linkInfo) {
            const result = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
            const input = this.inputs?.[index];
            if (type !== 2 && isSpeakerInput(input)) setTimeout(() => syncSpeakerInputs(this), 0);
            return result;
        };
    },
});
