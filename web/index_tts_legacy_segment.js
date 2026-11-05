import { app } from "../../scripts/app.js";

const LEGACY_WIDGETS = new Set(["more_segment_before"]);

function hideLegacySegmentWidget(node) {
    if (node.comfyClass !== "IndexTTSEngineNode") return;

    // Leave the widgets in place so saved values for subsequent widgets keep
    // their positions. The backend ignores the legacy value.
    for (const widget of node.widgets || []) {
        if (!LEGACY_WIDGETS.has(widget.name)) continue;
        widget.type = "hidden";
        widget.hidden = true;
        widget.disabled = true;
        widget.options = widget.options || {};
        widget.options.hidden = true;
        widget.computeSize = () => [0, -4];
        widget.draw = () => {};
        if (widget.element) widget.element.style.display = "none";
    }
    node.setDirtyCanvas?.(true, true);
}

function hideLegacyInputsFromPreview(definitions) {
    for (const key in definitions) {
        const definition = definitions[key];
        if (definition?.name !== "IndexTTSEngineNode" || !definition.input?.optional) continue;

        // Change the preview catalog only; the registered node retains the
        // original inputs so old workflow values keep their widget positions.
        const optional = { ...definition.input.optional };
        for (const name of LEGACY_WIDGETS) delete optional[name];
        definitions[key] = { ...definition, input: { ...definition.input, optional } };
    }
}

app.registerExtension({
    name: "tts-audio-suite.index-tts.legacy-segment",
    beforeRegisterVueAppNodeDefs: hideLegacyInputsFromPreview,
    nodeCreated: hideLegacySegmentWidget,
    loadedGraphNode: hideLegacySegmentWidget,
});
