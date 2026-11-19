/** Shared layout and controls for the ZONOS3 configuration panels. */
const STYLE_ID = "tts-zonos3-steering-style";

export function installStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
    .zonos3-panel {box-sizing:border-box; width:100%; color:#e0e5ef; background:#19212c;
        border:1px solid #3b4b60; border-radius:8px; padding:12px; font:12px Arial,sans-serif;}
    .zonos3-panel * {box-sizing:border-box;}
    .zonos3-panel h3 {font-size:13px; margin:0 0 8px; color:#f1e8ff;}
    .zonos3-panel p {font-size:11px; line-height:1.5; color:#aebbd0; margin:4px 0 10px;}
    .zonos3-panel details {border-top:1px solid #344154; padding-top:8px; margin-top:8px;}
    .zonos3-panel summary {cursor:pointer; color:#b6c9ed; font-weight:bold; margin-bottom:8px;}
    .zonos3-row {display:grid; grid-template-columns:135px minmax(30px,1fr) 64px; gap:6px; margin:7px 0; align-items:center;}
    .zonos3-row label {align-self:center; color:#d1dcee;}
    .zonos3-row input[type=number] {grid-column:3; grid-row:1;}
    .zonos3-row input[type=range] {grid-column:2; grid-row:1; width:100%; accent-color:#b993ef; height:16px;}
    .zonos3-number-only {grid-template-columns:minmax(0,1fr) 76px;}
    .zonos3-number-only input[type=number] {grid-column:2;}
    .zonos3-panel input[type=number], .zonos3-panel select {min-width:0; width:100%; color:#e0e5ef;
        background:#101722; border:1px solid #4b5b72; border-radius:4px; padding:4px; font:12px Arial;}
    .zonos3-panel button {color:#dce5f4; background:#2b384c; border:1px solid #536582;
        border-radius:4px; padding:5px 8px; cursor:pointer;}
    .zonos3-panel input:disabled {opacity:.4;}
    .zonos3-panel .zonos3-notice {color:#edc280;}
    .zonos3-effect {border-top:1px solid #344154; padding-top:8px; margin-top:8px;}
    .zonos3-effect header {display:flex; justify-content:space-between; gap:8px; align-items:center;}
    .zonos3-effect header span {flex:1;}
    .zonos3-effect input {width:90px!important;}
    .zonos3-panel svg {display:block; width:100%; height:auto; background:#101722; border-radius:5px;}
    .zonos3-defaults {display:flex; align-items:center; gap:6px; margin:8px 0;}
    `;
    document.head.appendChild(style);
}

export function element(tag, text, className) {
    const el = document.createElement(tag);
    if (text) el.textContent = text;
    if (className) el.className = className;
    return el;
}

export function setValue(node, widget, value) {
    if (!widget || widget.value === value) return;
    node.graph?.beforeChange?.();
    widget.value = value;
    widget.callback?.(value);
    node.graph?.afterChange?.();
    node.graph?.change?.();
    node.setDirtyCanvas?.(true, true);
}

export function hideBackendWidget(widget) {
    widget.type = "hidden";
    widget.disabled = true;
    widget.draw = () => {};
    widget.mouse = () => false;
    widget.computeSize = () => [0, -4];
    if (widget.element) widget.element.style.display = "none";
}

export function addPanel(node, name) {
    const holder = element("div");
    const body = element("div", null, "zonos3-panel");
    holder.appendChild(body);
    let height = 80;
    const widget = node.addDOMWidget(name, "zonos3_panel", holder, {
        serialize: false, hideOnZoom: true,
        getMinHeight: () => height, getMaxHeight: () => height,
    });
    widget.computeSize = () => [400, height];
    const resize = () => {
        const measured = Math.ceil(body.getBoundingClientRect().height / (node.graph?.list_of_graphcanvas?.[0]?.ds?.scale || 1));
        // scrollHeight is independent of the canvas zoom transform.
        const next = body.scrollHeight || measured;
        if (!next || next + 4 === height) return;
        height = next + 4;
        node.setSize([400, node.computeSize()[1]]);
        node.setDirtyCanvas?.(true, true);
    };
    const observer = new ResizeObserver(resize);
    observer.observe(body);
    return { body, resize, dispose: () => observer.disconnect() };
}

export function numberRow(parent, label, { min, max, step = 0.1, showRange = true, tooltip = "", onChange }) {
    const row = element("div", null, `zonos3-row${showRange ? "" : " zonos3-number-only"}`);
    row.title = tooltip;
    const caption = element("label", label);
    const number = document.createElement("input");
    number.type = "number";
    number.step = step;
    number.setAttribute("aria-label", label);
    if (min !== undefined) number.min = min;
    if (max !== undefined) number.max = max;
    row.append(caption, number);
    let range;
    let currentValue;
    if (showRange && min !== undefined && max !== undefined) {
        range = document.createElement("input");
        Object.assign(range, { type: "range", min, max, step });
        range.setAttribute("aria-label", `${label} slider`);
        row.appendChild(range);
        range.addEventListener("input", () => commit(range.value));
    }
    function commit(raw) {
        if (raw === "") return;
        let value = Number(raw);
        if (!Number.isFinite(value)) return;
        value = Number((Math.round(value / step) * step).toFixed(6));
        if (min !== undefined) value = Math.max(min, value);
        if (max !== undefined) value = Math.min(max, value);
        onChange(value);
    }
    number.addEventListener("change", () => commit(number.value));
    number.addEventListener("input", () => commit(number.value));
    number.addEventListener("blur", () => { number.value = currentValue; });
    parent.appendChild(row);
    return {
        row,
        sync(value, disabled = false) {
            currentValue = value;
            if (document.activeElement !== number) number.value = value;
            number.disabled = disabled;
            if (range) { range.value = value; range.disabled = disabled; }
        },
    };
}

export function connectedEngines(node) {
    const graph = node.graph;
    if (!graph) return [];
    return (node.outputs ?? []).flatMap(output => output.links ?? []).map(id => {
        const link = graph.links?.get?.(id) ?? graph.links?.[id];
        return graph.getNodeById(link?.target_id);
    }).filter(target => target?.comfyClass === "Zonos3MiniEngineNode");
}

export function engineValue(engine, name) {
    return engine.widgets?.find(widget => widget.name === name)?.value;
}
