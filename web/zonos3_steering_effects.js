import { addPanel, element, setValue } from "./zonos3_steering_ui.js";

export function parseEffects(text) {
    const values = JSON.parse(text);
    if (!values || typeof values !== "object" || Array.isArray(values)) throw new Error("Additional effects must be an object of numeric strengths.");
    for (const value of Object.values(values)) {
        if (typeof value !== "number" || !Number.isFinite(value)) throw new Error("Each additional effect needs a numeric strength.");
    }
    return new Map(Object.entries(values));
}

export function createEffectsPanel(node, widgets, catalog) {
    const widget = () => widgets.get("additional_effects");
    const panel = addPanel(node, "zonos3_additional_effects");
    const { body } = panel;
    body.append(element("h3", "Additional effects"));
    body.append(element("p", "Choose an effect and set its strength. Each effect applies to the generated segment."));
    const select = document.createElement("select");
    select.setAttribute("aria-label", "Add native effect");
    const list = element("div");
    const error = element("p", null, "zonos3-notice");
    body.append(select, error, list);
    let effects = new Map();
    let lastText;
    let renderedIds = "";
    const inputs = new Map();
    const available = catalog.filter(effect => !effect.slider && !effect.embedding_only);
    const byId = new Map(available.map(effect => [effect.id, effect]));
    const write = () => setValue(node, widget(), JSON.stringify(Object.fromEntries(effects)));

    function render() {
        list.replaceChildren();
        inputs.clear();
        renderedIds = JSON.stringify([...effects.keys()]);
        for (const [id, value] of effects) {
            const effect = byId.get(id);
            const row = element("div", null, "zonos3-effect");
            const header = element("header");
            const label = effect ? `${effect.group} · ${effect.label}` : `Unavailable effect: ${id}`;
            header.append(element("span", label));
            const number = document.createElement("input");
            Object.assign(number, { type: "number", step: effect?.step ?? 0.1, value });
            number.setAttribute("aria-label", `${label} strength`);
            number.disabled = !effect;
            number.title = effect?.hint ?? "This effect does not exist in the bundled model.";
            inputs.set(id, number);
            number.oninput = number.onchange = () => {
                const next = Number(number.value);
                if (number.value !== "" && Number.isFinite(next)) { effects.set(id, next); write(); }
            };
            number.onblur = () => { number.value = effects.get(id); };
            const remove = element("button", "×");
            remove.setAttribute("aria-label", `Remove ${label}`);
            remove.onclick = () => { effects.delete(id); write(); };
            header.append(number, remove);
            row.appendChild(header);
            if (effect?.hint) row.append(element("p", effect.hint));
            list.appendChild(row);
        }
        select.replaceChildren(new Option("+ Add effect…", ""));
        const groups = new Map();
        for (const effect of available) {
            if (effects.has(effect.id)) continue;
            if (!groups.has(effect.group)) {
                const group = document.createElement("optgroup");
                group.label = effect.group;
                groups.set(effect.group, group);
                select.appendChild(group);
            }
            const option = new Option(effect.label, effect.id);
            option.title = effect.hint;
            groups.get(effect.group).appendChild(option);
        }
    }
    select.onchange = () => {
        if (!select.value) return;
        effects.set(select.value, 0);
        write();
    };
    return {
        ...panel,
        sync() {
            if (lastText === widget().value) return;
            try {
                effects = parseEffects(widget().value);
                lastText = widget().value;
                error.textContent = "";
                if (renderedIds !== JSON.stringify([...effects.keys()]) || !select.options.length) render();
                else for (const [id, input] of inputs) {
                    if (document.activeElement !== input) input.value = effects.get(id);
                }
            } catch (err) {
                error.textContent = err.message;
            }
        },
    };
}
