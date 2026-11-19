import { addPanel, element, numberRow, setValue } from "./zonos3_steering_ui.js";

export function readEnvelope(widgets, defaults) {
    if (widgets.get("use_model_defaults").value) {
        return { full: defaults.onset_seconds, ramp: defaults.ramp_seconds, remaining: defaults.hold };
    }
    return {
        full: Number(widgets.get("full_strength_seconds").value),
        ramp: Number(widgets.get("reduction_seconds").value),
        remaining: Number(widgets.get("remaining_strength_percent").value) / 100,
    };
}

export function envelopeViewSeconds({ full, ramp }) {
    return Math.max(10, Math.ceil((full + ramp) * 1.3));
}

export function envelopeDragValues(handle, point, values, seconds) {
    const time = Math.round(Math.max(0, (point.x - 44) / 294 * seconds) * 10) / 10;
    if (handle === "full") return { ...values, full: Math.min(600, time) };
    return {
        ...values,
        ramp: Number(Math.max(0, Math.min(600, time - values.full)).toFixed(1)),
        remaining: Math.round(Math.max(0, Math.min(1, (126 - point.y) / 94)) * 100) / 100,
    };
}

export function envelopeSvg({ full, ramp, remaining }, end = envelopeViewSeconds({ full, ramp })) {
    const x = t => 44 + t / end * 294;
    const y = strength => 126 - strength * 94;
    const path = `M44,32 H${x(full)} L${x(full + ramp)},${y(remaining)} H338`;
    return `<svg viewBox="0 0 360 170" role="img" aria-label="Effect strength over generated speech time">
        <g fill="none" stroke="#293a50"><path d="M44 32H338 M44 79H338 M44 126H338"/></g>
        <path d="M44 24V126H338" fill="none" stroke="#8497b2"/>
        <g fill="#aebbd0" font-family="Arial" font-size="10" text-anchor="end">
        <text x="38" y="35">100%</text><text x="38" y="82">50%</text><text x="38" y="129">0%</text></g>
        <g fill="#aebbd0" font-family="Arial" font-size="10" text-anchor="middle">
        <text x="44" y="142">0</text><text x="${x(end / 2)}" y="142">${Number((end / 2).toFixed(1))}</text>
        <text x="338" y="142">${Number(end.toFixed(1))}</text><text x="190" y="160">Generated speech (seconds)</text></g>
        <path d="${path} L338,126 H44 Z" fill="#b993ef" opacity=".12"/>
        <path d="${path}" fill="none" stroke="#c7a2fa" stroke-width="3"/>
        <g fill="#e1cbff" stroke="#47315e" stroke-width="2">
        <circle data-envelope-handle="full" cx="${x(full)}" cy="32" r="7" tabindex="0"
            role="slider" aria-label="Full-strength duration" aria-valuemin="0" aria-valuemax="600" aria-valuenow="${full}"
            style="cursor:ew-resize"><title>Drag left/right to set the initial full-strength duration. Initial strength stays at 100%.</title></circle>
        <circle data-envelope-handle="remaining" cx="${x(full + ramp)}" cy="${y(remaining)}" r="7" tabindex="0"
            role="slider" aria-label="Remaining effect strength" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round(remaining * 100)}"
            style="cursor:move"><title>Drag up/down for remaining strength; left/right for reduction duration.</title></circle></g>
        </svg>`;
}

export function createEnvelopePanel(node, widgets, defaults, definitions) {
    const panel = addPanel(node, "zonos3_effect_timing");
    const { body } = panel;
    const heading = element("h3", "Effect strength over time ");
    const help = element("span", "ⓘ");
    help.title = "Drag the first dot left/right to set how long full strength lasts.\n"
        + "Drag the second dot up/down for remaining strength, left/right for fade duration.\n"
        + "Scroll to change the time scale. Double-click to fit.\n\n"
        + "Timing restarts for each chunk or subtitle.\n"
        + "Pace, whisper and Voice Characteristics stay constant; Mandarin's delivery component follows the timing.\n"
        + "The graph shows effect settings, not how strong an emotion will sound.";
    help.style.cursor = "help";
    help.tabIndex = 0;
    help.setAttribute("role", "img");
    help.setAttribute("aria-label", help.title);
    heading.appendChild(help);
    body.appendChild(heading);
    const label = element("label", null, "zonos3-defaults");
    label.title = definitions.use_model_defaults[1].tooltip;
    const useDefaults = document.createElement("input");
    useDefaults.type = "checkbox";
    label.append(useDefaults, document.createTextNode("Use model defaults"));
    body.appendChild(label);
    const fields = [
        ["full_strength_seconds", "Initial full strength (seconds)", 0, 600, 0.1],
        ["reduction_seconds", "Strength reduction (seconds)", 0, 600, 0.1],
        ["remaining_strength_percent", "Remaining strength (%)", 0, 100, 1],
    ];
    const rows = fields.map(([key, caption, min, max, step]) => numberRow(body, caption, {
        min, max, step, showRange: false, tooltip: definitions[key][1].tooltip,
        onChange: value => setValue(node, widgets.get(key), value),
    }));
    useDefaults.addEventListener("change", () => {
        setValue(node, widgets.get("use_model_defaults"), useDefaults.checked);
    });
    const graph = element("div");
    graph.style.touchAction = "none";
    graph.style.userSelect = "none";
    const description = element("p");
    body.append(graph, description);
    const advanced = element("details");
    advanced.append(element("summary", "Advanced: control guidance"));
    const guidance = numberRow(advanced, "Control guidance", {
        min: 0, max: 5, step: 0.1, tooltip: definitions.control_guidance[1].tooltip,
        onChange: value => setValue(node, widgets.get("control_guidance"), value),
    });
    const guidanceHint = element("p");
    const reset = element("button", "Use normal guidance (1)");
    reset.onclick = () => setValue(node, widgets.get("control_guidance"), 1);
    advanced.append(guidanceHint, reset);
    body.appendChild(advanced);
    let drag = null;
    let viewSeconds = null;
    let lastSvg = "";
    function drawGraph(values) {
        let end = drag?.seconds ?? viewSeconds ?? envelopeViewSeconds(values);
        if (!drag && values.full + values.ramp > end) end = envelopeViewSeconds(values);
        if (viewSeconds !== null && !drag) viewSeconds = end;
        const svg = envelopeSvg(values, end);
        if (svg !== lastSvg) {
            const focus = graph.contains(document.activeElement) ? document.activeElement.dataset?.envelopeHandle : null;
            graph.innerHTML = svg;
            if (focus) graph.querySelector(`[data-envelope-handle="${focus}"]`)?.focus({ preventScroll: true });
            lastSvg = svg;
        }
    }
    function applyValues(values) {
        // Editing a default envelope starts from the values shown in the graph.
        setValue(node, widgets.get("full_strength_seconds"), values.full);
        setValue(node, widgets.get("reduction_seconds"), values.ramp);
        setValue(node, widgets.get("remaining_strength_percent"), Math.round(values.remaining * 100));
        setValue(node, widgets.get("use_model_defaults"), false);
        drawGraph(readEnvelope(widgets, defaults));
    }
    function graphPoint(event) {
        const svg = graph.querySelector("svg");
        const matrix = svg?.getScreenCTM();
        if (!matrix) return null;
        const point = svg.createSVGPoint();
        point.x = event.clientX;
        point.y = event.clientY;
        return point.matrixTransform(matrix.inverse());
    }
    graph.addEventListener("pointerdown", event => {
        const handle = event.target.closest?.("[data-envelope-handle]")?.dataset.envelopeHandle;
        if (!handle || event.button !== 0) return;
        event.preventDefault();
        event.stopPropagation();
        drag = { handle, pointerId: event.pointerId, seconds: viewSeconds ?? envelopeViewSeconds(readEnvelope(widgets, defaults)) };
        graph.setPointerCapture(event.pointerId);
    });
    graph.addEventListener("pointermove", event => {
        if (!drag || event.pointerId !== drag.pointerId) return;
        const point = graphPoint(event);
        if (!point) return;
        event.preventDefault();
        event.stopPropagation();
        applyValues(envelopeDragValues(drag.handle, point, readEnvelope(widgets, defaults), drag.seconds));
    });
    function endDrag(event) {
        if (!drag || event.pointerId !== drag.pointerId) return;
        viewSeconds = Math.max(drag.seconds, readEnvelope(widgets, defaults).full + readEnvelope(widgets, defaults).ramp);
        drag = null;
        if (graph.hasPointerCapture(event.pointerId)) graph.releasePointerCapture(event.pointerId);
        drawGraph(readEnvelope(widgets, defaults));
    }
    graph.addEventListener("pointerup", endDrag);
    graph.addEventListener("pointercancel", endDrag);
    graph.addEventListener("lostpointercapture", endDrag);
    graph.addEventListener("wheel", event => {
        event.preventDefault();
        event.stopPropagation();
        if (drag) return;
        const values = readEnvelope(widgets, defaults);
        const current = viewSeconds ?? envelopeViewSeconds(values);
        const factor = event.deltaY > 0 ? 1.2 : 1 / 1.2;
        viewSeconds = Math.max(1, values.full + values.ramp, Math.min(1560, current * factor));
        drawGraph(values);
    }, { passive: false });
    graph.addEventListener("dblclick", event => {
        event.preventDefault();
        event.stopPropagation();
        viewSeconds = null;
        drawGraph(readEnvelope(widgets, defaults));
    });
    graph.addEventListener("keydown", event => {
        const handle = event.target.dataset?.envelopeHandle;
        if (!handle || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) return;
        event.preventDefault();
        event.stopPropagation();
        const values = readEnvelope(widgets, defaults);
        if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
            const key = handle === "full" ? "full" : "ramp";
            values[key] = Number(Math.max(0, Math.min(600, values[key] + (event.key === "ArrowRight" ? 0.1 : -0.1))).toFixed(1));
        } else if (handle === "remaining") {
            values.remaining = Math.max(0, Math.min(1, values.remaining + (event.key === "ArrowUp" ? 0.01 : -0.01)));
        } else return;
        applyValues(values);
    });
    return {
        ...panel,
        sync(guidanceReason = "") {
            const values = readEnvelope(widgets, defaults);
            const modelDefaults = widgets.get("use_model_defaults").value;
            useDefaults.checked = modelDefaults;
            [values.full, values.ramp, values.remaining * 100].forEach((value, i) => rows[i].sync(Number(value.toFixed(2)), modelDefaults));
            drawGraph(values);
            description.textContent = `100% for ${values.full}s → reduce over ${values.ramp}s → keep ${Math.round(values.remaining * 100)}%.`;
            guidance.sync(widgets.get("control_guidance").value, Boolean(guidanceReason));
            guidanceHint.textContent = guidanceReason || "1 keeps normal steering active. Extra guidance needs active delivery effects in embedding mode and costs an extra model pass per frame.";
            guidanceHint.className = guidanceReason ? "zonos3-notice" : "";
        },
    };
}
