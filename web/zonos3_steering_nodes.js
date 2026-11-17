import { app } from "../../scripts/app.js";
import { createEmotionRadarCanvasWidget, INDEX_TTS_EMOTION_VISUALS } from "./emotion_radar_canvas_widget.js";
import { installStyle, addPanel, element, numberRow, hideBackendWidget, setValue, connectedEngines, engineValue } from "./zonos3_steering_ui.js";
import { createEffectsPanel } from "./zonos3_steering_effects.js";
import { createEnvelopePanel } from "./zonos3_steering_envelope.js";

const VOICE = "Zonos3MiniVoiceCharacteristicsNode";
const DELIVERY = "Zonos3MiniDeliveryNode";
// Opposite placement is visual; every native effect remains independent.
const RADAR_ORDER = ["happy", "angry", "confused", "sad", "calm", "scared"];
const SHARED_EMOTIONS = { happy: "Happy", angry: "Angry", sad: "Sad", calm: "Calm", scared: "Afraid", confused: "Surprised" };
const EMOTION_COLORS = Object.fromEntries(Object.entries(SHARED_EMOTIONS).map(([name, shared]) => [
    name, INDEX_TTS_EMOTION_VISUALS.find(emotion => emotion.name === shared).color,
]));

function setup(node, nodeData) {
    if (node.zonos3UI) return;
    installStyle();
    const isVoice = nodeData.name === VOICE;
    // ComfyUI upgrades widget objects after node creation. Resolve the current
    // object whenever a control reads/writes, rather than retaining the originals.
    const widgetNames = new Set(Object.keys(nodeData.input.optional));
    const widgets = {
        get: name => node.widgets.find(widget => widget.name === name),
        values: () => node.widgets.filter(widget => widgetNames.has(widget.name)),
    };
    const definitions = Object.entries(nodeData.input.optional).filter(([, spec]) => spec[1]?.zonos3)
        .map(([name, spec]) => [name, spec[1]]);
    for (const widget of widgets.values()) hideBackendWidget(widget);
    node.resizable = false;
    const controls = addPanel(node, "zonos3_steering_controls");
    controls.body.append(element("h3", isVoice ? "Voice characteristics" : "Emotion & delivery"));
    const modeNotice = element("p");
    controls.body.appendChild(modeNotice);
    const rows = new Map();
    const groups = new Map();
    for (const [name, definition] of definitions) {
        const groupName = definition.zonos3.group;
        if (!groups.has(groupName)) {
            const group = element("details");
            group.open = isVoice ? ["Identity", "Pitch & intonation", "Voice texture"].includes(groupName) : groupName === "Emotions";
            group.append(element("summary", groupName));
            groups.set(groupName, group);
            controls.body.appendChild(group);
        }
        rows.set(name, numberRow(groups.get(groupName), definition.zonos3.label, {
            ...definition, onChange: value => setValue(node, widgets.get(name), value),
        }));
    }
    const catalog = nodeData.input.optional.additional_effects?.[1]?.zonos3_catalog;
    let effects, envelope;
    if (!isVoice) {
        const axes = RADAR_ORDER.map(name => [name, nodeData.input.optional[name][1]]).map(([name, def], index, all) => ({
            name, label: def.zonos3.label, min: def.min, max: def.max, step: def.step,
            angle: index * 2 * Math.PI / all.length, color: EMOTION_COLORS[name],
        }));
        node.widgets.push(createEmotionRadarCanvasWidget(node, {
            emotions: axes, centerX: 200, modernControls: true, showNativeValues: true,
            exportFilename: "zonos3_emotions.json",
        }));
        envelope = createEnvelopePanel(node, widgets, catalog.release, nodeData.input.optional);
        effects = createEffectsPanel(node, widgets, catalog.effects);
    }
    const panels = [controls, envelope, effects].filter(Boolean);
    function sync() {
        const engines = connectedEngines(node);
        const referenceMode = engines.some(engine => engineValue(engine, "voice_mode") === "reference");
        effects?.sync();
        const wrongSocket = engines.some(engine => (engine.inputs ?? []).some(input => {
            const link = node.graph.links?.get?.(input.link) ?? node.graph.links?.[input.link];
            return link?.origin_id === node.id && input.name !== (isVoice ? "voice_characteristics" : "delivery");
        }));
        modeNotice.className = wrongSocket || (isVoice && referenceMode) ? "zonos3-notice" : "";
        modeNotice.textContent = wrongSocket ? `Connect this node to the engine's ${isVoice ? "voice_characteristics" : "delivery"} input.`
            : isVoice ? (referenceMode ? "Voice changes require embedding mode. Select embedding, or disconnect this node before generating with reference mode."
                : "Changes the voice itself. Requires embedding voice mode. Embedding changes stay constant; Mandarin also adds a timed delivery accent.")
            : "Native delivery effects work in both voice modes. Zero leaves an effect off. The radar uses each emotion's own scale.";
        for (const [name, definition] of definitions) {
            rows.get(name).sync(widgets.get(name).value, isVoice && referenceMode);
        }
        let reason = referenceMode ? "Unavailable in reference voice mode. Normal steering still works; use guidance 1." : "";
        if (engines.some(engine => engineValue(engine, "laci"))) reason = "LACI requires guidance 1. Normal steering still works.";
        if (engines.some(engine => Number(engineValue(engine, "speaker_cfg_scale")) !== 1)) reason = "Speaker guidance and control guidance cannot be combined. Use control guidance 1.";
        envelope?.sync(reason);
        node.setDirtyCanvas?.(true, true);
    }
    node.zonos3UI = { sync, dispose: () => panels.forEach(panel => panel.dispose()) };
    for (const widget of widgets.values()) {
        const original = widget.callback;
        widget.callback = function(...args) { const result = original?.apply(this, args); sync(); return result; };
    }
    sync();
    requestAnimationFrame(() => panels.forEach(panel => panel.resize()));
}

function refreshEngineInputs(engine) {
    for (const input of engine.inputs ?? []) {
        const link = engine.graph?.links?.get?.(input.link) ?? engine.graph?.links?.[input.link];
        engine.graph?.getNodeById(link?.origin_id)?.zonos3UI?.sync();
    }
}

app.registerExtension({
    name: "TTS_Audio_Suite.Zonos3Steering",
    beforeRegisterNodeDef(nodeType, nodeData) {
        const steering = [VOICE, DELIVERY].includes(nodeData.name);
        if (!steering && nodeData.name !== "Zonos3MiniEngineNode") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function(...args) {
            const result = created?.apply(this, args);
            if (steering) setup(this, nodeData);
            else for (const widget of this.widgets ?? []) {
                if (!["voice_mode", "laci", "speaker_cfg_scale"].includes(widget.name)) continue;
                const callback = widget.callback;
                const thisNode = this;
                widget.callback = function(...values) { const result = callback?.apply(this, values); refreshEngineInputs(thisNode); return result; };
            }
            return result;
        };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(...args) {
            const result = configured?.apply(this, args);
            if (steering) {
                this.zonos3UI?.sync();
                requestAnimationFrame(() => this.setSize([400, this.computeSize()[1]]));
            }
            return result;
        };
        const connections = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function(...args) {
            const result = connections?.apply(this, args);
            queueMicrotask(() => steering ? this.zonos3UI?.sync() : refreshEngineInputs(this));
            return result;
        };
        if (steering) {
            const removed = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function(...args) { this.zonos3UI?.dispose(); return removed?.apply(this, args); };
        }
    },
});
