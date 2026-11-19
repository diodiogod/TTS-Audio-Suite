import { bindEmotionRadialPicker } from "./emotion-tag-radial-picker.js";

/** Build numeric tag cards from the engine's parser metadata. */
export function buildNumericTagSection(parameters, { title, color, hint, axes }, ui, preferences, save) {
    const panel = document.createElement("div");
    ui.stylePanelContainer(panel);
    const select = ui.createSelect([], "Select a control...");
    select.setAttribute("aria-label", title);
    const groups = new Map();
    for (const parameter of parameters) {
        let parent = select;
        if (parameter.subgroup && parameter.subgroup !== parameter.group) {
            if (!groups.has(parameter.subgroup)) {
                const group = document.createElement("optgroup");
                group.label = parameter.subgroup;
                select.appendChild(group);
                groups.set(parameter.subgroup, group);
            }
            parent = groups.get(parameter.subgroup);
        }
        parent.appendChild(new Option(parameter.label, parameter.tag));
    }
    const operation = ui.createSelect([
        { value: "set", label: "Set strength" },
        { value: "+", label: "Increase (+)" },
        { value: "-", label: "Decrease (−)" },
    ], "Choose an operation...", preferences.operation || "set");
    operation.options[0].disabled = true;
    operation.setAttribute("aria-label", `${title} operation`);

    const slider = document.createElement("input");
    slider.type = "range";
    Object.assign(slider.style, { width: "100%", boxSizing: "border-box", margin: "8px 0" });
    slider.setAttribute("aria-label", `${title} slider`);
    const value = document.createElement("input");
    value.type = "number";
    value.setAttribute("aria-label", `${title} value`);
    Object.assign(value.style, {
        width: "100%", boxSizing: "border-box", padding: "3px", marginBottom: "4px",
        fontSize: "10px", background: "#2a2a2a", color: "#eee", border: "1px solid #444",
    });
    const addButton = ui.createButton(`Add ${title}`, "Set this control in the tag at the cursor, or insert a new tag");
    const parameter = () => parameters.find(item => item.tag === select.value);
    const update = () => {
        const current = parameter();
        operation.hidden = !current?.relative;
        slider.hidden = !current?.slider;
        value.disabled = addButton.disabled = !current;
        select.title = slider.title = value.title = current?.description ?? "";
        if (!current) return;
        const initial = preferences.values?.[current.tag] ?? current.default;
        for (const input of [slider, value]) {
            Object.assign(input, { min: current.min, max: current.max, step: current.step, value: initial });
        }
        slider.dataset.fillColor = current.color || color;
        ui.updateRangeFill(slider);
    };
    select.value = parameters.some(item => item.tag === preferences.tag) ? preferences.tag : parameters[0].tag;
    select.onchange = () => { preferences.tag = select.value; update(); save(); };
    operation.onchange = () => { preferences.operation = operation.value; save(); };
    slider.oninput = () => { value.value = slider.value; };
    value.oninput = () => { slider.value = value.value; ui.updateRangeFill(slider); };
    const saveValue = () => {
        const current = parameter();
        if (!current || value.value === "" || Number(value.value) < current.min || Number(value.value) > current.max) return;
        (preferences.values ??= {})[current.tag] = value.value;
        save();
    };
    value.onchange = slider.onchange = saveValue;
    update();
    panel.appendChild(ui.createPanelLabel(title, color, hint));
    if (axes?.length > 1) {
        const wheel = ui.createButton("Press + drag to pick emotion", "Drag toward an emotion. Distance sets its strength.");
        wheel.style.marginBottom = "4px";
        bindEmotionRadialPicker(wheel, axes, (name, magnitude, isFinal) => {
            if (select.value !== name) {
                select.value = name;
                update();
            }
            slider.value = magnitude.toFixed(2);
            value.value = Number(slider.value);
            ui.updateRangeFill(slider);
            if (isFinal) {
                preferences.tag = name;
                saveValue();
            }
        });
        panel.appendChild(wheel);
    }
    panel.append(select, operation, slider, value, addButton);
    return {
        panel, addButton, parameter,
        value() {
            if (!parameter()?.relative) return value.value;
            if (value.value === "" || Number(value.value) < 0) return "";
            return operation.value === "set" ? value.value : `${operation.value}${value.value}`;
        },
    };
}

export function buildNumericEnginePanel({ parameters, vector, sections = {} }, ui, preferences, save) {
    const panel = document.createElement("div");
    Object.assign(panel.style, { flexDirection: "column", gap: "8px" });
    let addVectorBtn = null;
    if (vector) {
        const vectorSection = document.createElement("div");
        ui.stylePanelContainer(vectorSection);
        addVectorBtn = ui.createButton("Edit Emotion Vector", "Set or adjust the emotions and save vector presets");
        vectorSection.append(ui.createPanelLabel("Full Emotion Vector", "#45b7d1"), addVectorBtn);
        panel.appendChild(vectorSection);
    }
    const colors = new Map(vector?.axes.map(axis => [axis.name, axis.color]));
    const groups = new Map();
    for (const parameter of parameters) {
        if (parameter.group === "Generation") continue;
        if (!groups.has(parameter.group)) groups.set(parameter.group, []);
        groups.get(parameter.group).push({ ...parameter, color: colors.get(parameter.tag) });
    }
    const cards = [];
    for (const group of new Set([...Object.keys(sections), ...groups.keys()])) {
        if (!groups.has(group)) continue;
        const controls = groups.get(group);
        const axes = vector?.axes.filter(axis => controls.some(control => control.tag === axis.name && control.slider));
        const card = buildNumericTagSection(controls, { title: group, color: "#d0d0d0", axes, ...sections[group] },
            ui, preferences[group] ??= {}, save);
        cards.push(card);
        panel.appendChild(card.panel);
    }
    return { panel, controls: { cards, addVectorBtn } };
}
