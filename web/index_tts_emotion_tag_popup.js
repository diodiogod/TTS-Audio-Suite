import { api } from "/scripts/api.js";
import { createEmotionRadarCanvasWidget, INDEX_TTS_EMOTION_VISUALS } from "./emotion_radar_canvas_widget.js";

const ORDER = ["Happy", "Angry", "Sad", "Afraid", "Disgusted", "Melancholic", "Surprised", "Calm"];
const ENDPOINT = "/api/tts-audio-suite/index-tts-emotion-presets";
const COLORS = Object.fromEntries(INDEX_TTS_EMOTION_VISUALS.map(({ name, color }) => [name, color]));

const number = (value) => Math.max(0, Math.min(1.2, Number(value) || 0));
const format = (value) => Number(value).toFixed(2).replace(/0+$/, "").replace(/\.$/, "") || "0";
const quoteEmotionText = text => {
    const cleaned = text.trim();
    if (!cleaned.includes('"')) return `[emotion:"${cleaned}"]`;
    if (!cleaned.includes("'")) return `[emotion:'${cleaned}']`;
    return `[emotion:"${cleaned.replaceAll('"', '”')}"]`;
};

function parseVectorTag(tag) {
    const full = tag.match(/^\[vector:([^\]]+)\]$/i);
    if (full) {
        const raw = full[1].split(",").map(v => v.trim());
        if (raw.length !== 8) return null;
        const relative = raw.every(v => /^[+-]/.test(v));
        if (!relative && raw.some(v => /^[+-]/.test(v))) return null;
        return { values: raw.map(v => number(Math.abs(Number(v)))), signs: raw.map(v => v.startsWith("-") ? -1 : 1), relative, named: false };
    }
    const parts = tag.slice(1, -1).split("|");
    const values = Array(8).fill(0);
    const signs = Array(8).fill(1);
    let found = false;
    let relative = false;
    for (const part of parts) {
        const match = part.trim().match(/^([a-z]+):([+-]?\d*\.?\d+)$/i);
        if (!match) return null;
        const index = ORDER.findIndex(name => name.toLowerCase() === match[1].toLowerCase());
        if (index < 0) return null;
        found = true;
        relative ||= /^[+-]/.test(match[2]);
        signs[index] = match[2].startsWith("-") ? -1 : 1;
        values[index] = number(Math.abs(Number(match[2])));
    }
    return found ? { values, signs, relative, named: true } : null;
}

const isEmotionTextTag = tag => /^\[emotion:(?:[A-Za-z0-9_-]+|["'][\s\S]*["'])\]$/i.test(tag);

function serializeVector(values, signs, relative, named) {
    const encoded = values.map((value, index) => {
        const magnitude = format(value);
        return relative ? `${signs[index] < 0 ? "-" : "+"}${magnitude}` : magnitude;
    });
    if (named) {
        const entries = encoded.map((value, index) => ({ value, index })).filter(({ index }) => values[index] !== 0);
        if (entries.length === ORDER.length) return `[vector:${encoded.join(",")}]`;
        if (entries.length) return `[${entries.map(({ value, index }) => `${ORDER[index].toLowerCase()}:${value}`).join("|")}]`;
    }
    return `[vector:${encoded.join(",")}]`;
}

function button(label) {
    const element = document.createElement("button");
    element.type = "button";
    element.textContent = label;
    Object.assign(element.style, { padding: "7px 12px", color: "#eee", background: "#343434", border: "1px solid #666", borderRadius: "4px", cursor: "pointer" });
    return element;
}

export async function openIndexTTSEmotionEditor({
    tag = null,
    onApply,
    onPreview,
    onCommit,
    onCancel,
    anchorRect = null,
    showPresets = !anchorRect,
}) {
    const parsed = tag ? parseVectorTag(tag) : null;
    const state = parsed || { values: Array(8).fill(0), signs: Array(8).fill(1), relative: false, named: false };
    let editSign = 1;
    let closed = false;
    let previewFrame = null;
    let presets = {};
    try {
        const response = await api.fetchApi(ENDPOINT);
        presets = (await response.json()).presets || {};
    } catch (error) {
        console.warn("Could not load IndexTTS emotion presets", error);
    }

    const overlay = document.createElement("div");
    Object.assign(overlay.style, {
        position: "fixed", inset: "0", zIndex: "100000",
        background: anchorRect ? "transparent" : "rgba(0,0,0,.72)",
        display: anchorRect ? "block" : "grid",
        placeItems: anchorRect ? "initial" : "center"
    });
    const panel = document.createElement("div");
    Object.assign(panel.style, {
        width: anchorRect ? "min(320px,94vw)" : "min(760px,92vw)",
        maxHeight: anchorRect ? "min(680px,85vh)" : "92vh",
        overflow: "auto", padding: anchorRect ? "10px" : "18px", color: "#eee",
        background: "#202020", border: anchorRect ? "1px solid #45b7d1" : "1px solid #666",
        borderRadius: "10px", boxShadow: anchorRect ? "0 10px 38px rgba(0,0,0,.9), 0 0 14px rgba(69,183,209,.28)" : "0 18px 70px #000"
    });
    const title = document.createElement("h2");
    title.textContent = "IndexTTS-2 Emotion Tag";
    title.style.margin = "0 0 12px";
    title.style.fontSize = anchorRect ? "15px" : "24px";
    title.style.textAlign = anchorRect ? "center" : "left";

    const modeRow = document.createElement("div");
    Object.assign(modeRow.style, { display: "flex", flexWrap: "wrap", gap: anchorRect ? "5px" : "8px", alignItems: "center", justifyContent: anchorRect ? "center" : "flex-start", marginBottom: "8px" });
    const absolute = button("Absolute");
    const delta = button("Delta");
    const sign = button("Delta sign: +");
    const hint = document.createElement("span");
    hint.style.color = "#aaa";
    if (anchorRect) Object.assign(hint.style, { flexBasis: "100%", textAlign: "center", fontSize: "11px", lineHeight: "1.1" });
    modeRow.append(absolute, delta, sign, hint);

    if (anchorRect) {
        [absolute, delta, sign].forEach(control => Object.assign(control.style, {
            padding: "4px 8px", fontSize: "11px", lineHeight: "1.15", minHeight: "26px"
        }));
    } else {
        const modeSwitch = document.createElement("div");
        Object.assign(modeSwitch.style, { display: "inline-flex", padding: "3px", gap: "2px", border: "1px solid #4a4d56", borderRadius: "8px", background: "#111217" });
        [absolute, delta].forEach(control => Object.assign(control.style, { padding: "6px 13px", minWidth: "78px", border: "none", borderRadius: "5px", fontSize: "11px" }));
        Object.assign(sign.style, { padding: "6px 10px", fontSize: "10px", borderRadius: "7px" });
        Object.assign(hint.style, { flexBasis: "100%", fontSize: "10px", color: "#868994", marginTop: "1px" });
        modeSwitch.append(absolute, delta);
        modeRow.replaceChildren(modeSwitch, sign, hint);
    }

    const widgets = ORDER.map((name, index) => ({
        name,
        value: state.values[index],
        callback(value) {
            state.values[index] = value;
            if (state.relative) state.signs[index] = editSign;
            schedulePreview();
            render();
        }
    }));
    const node = { widgets, graph: { setDirtyCanvas() { render(); } }, setDirtyCanvas() { render(); } };
    const radar = createEmotionRadarCanvasWidget(node, {
        modernControls: !anchorRect,
        getEmotionSign(emotionName) {
            if (!state.relative) return 1;
            const index = ORDER.indexOf(emotionName);
            return index >= 0 ? state.signs[index] : 1;
        }
    });
    const canvas = document.createElement("canvas");
    canvas.width = 320;
    canvas.height = 350;
    Object.assign(canvas.style, { width: "100%", maxWidth: `${canvas.width}px`, height: "350px", display: "block", margin: "0 auto", borderRadius: "6px" });
    const valueGrid = document.createElement("div");
    Object.assign(valueGrid.style, { display: "grid", gridTemplateColumns: anchorRect ? "repeat(4,minmax(60px,1fr))" : "repeat(4,minmax(120px,1fr))", gap: anchorRect ? "5px" : "7px", margin: "8px 0" });
    const valueInputs = ORDER.map((name, index) => {
        const label = document.createElement("label");
        Object.assign(label.style, { display: "grid", gap: "3px", color: "#bbb", fontSize: anchorRect ? "10px" : "12px" });
        const nameElement = document.createElement("span");
        nameElement.textContent = name;
        if (!anchorRect) {
            const color = COLORS[name] || "#666";
            const dot = document.createElement("span");
            Object.assign(dot.style, { width: "9px", height: "9px", borderRadius: "50%", background: color, boxShadow: `0 0 7px ${color}88`, flex: "0 0 auto" });
            const nameRow = document.createElement("span");
            Object.assign(nameRow.style, { display: "flex", alignItems: "center", gap: "6px", minWidth: "0" });
            nameRow.append(dot, nameElement);
            Object.assign(label.style, {
                gridTemplateColumns: "minmax(0,1fr) 72px", alignItems: "center", padding: "7px 9px",
                border: `1px solid ${color}55`, borderRadius: "7px", background: `${color}0d`,
                color: "#e7e7e7", fontWeight: "600"
            });
            label.appendChild(nameRow);
        } else {
            label.appendChild(nameElement);
        }
        const input = document.createElement("input");
        input.type = anchorRect ? "text" : "number";
        input.inputMode = "decimal";
        if (!anchorRect) {
            input.min = "-1.2";
            input.max = "1.2";
            input.step = "0.01";
        }
        Object.assign(input.style, { width: "100%", boxSizing: "border-box", padding: anchorRect ? "4px" : "5px", fontSize: anchorRect ? "11px" : "inherit", color: "#eee", background: "#151515", border: "1px solid #555", borderRadius: "4px" });
        input.addEventListener("input", () => {
            const rawText = input.value.trim();
            if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$/.test(rawText)) return;
            const raw = Number(rawText) || 0;
            state.values[index] = number(Math.abs(raw));
            state.signs[index] = rawText.startsWith("-") ? -1 : 1;
            widgets[index].value = state.values[index];
            schedulePreview();
            render();
        });
        label.appendChild(input);
        valueGrid.appendChild(label);
        return input;
    });
    const currentVectorTag = () => serializeVector(state.values, state.signs, state.relative, state.named);
    const schedulePreview = () => {
        if (!anchorRect || !onPreview || closed) return;
        if (previewFrame !== null) cancelAnimationFrame(previewFrame);
        previewFrame = requestAnimationFrame(() => {
            previewFrame = null;
            onPreview(currentVectorTag());
        });
    };
    const render = () => {
        if (anchorRect) {
            absolute.style.boxShadow = state.relative ? "none" : "0 0 10px #45b7d1";
            delta.style.boxShadow = state.relative ? `0 0 12px ${state.signs.some(v => v < 0) ? "#ff5b65" : "#ffd166"}` : "none";
        } else {
            absolute.style.boxShadow = "none";
            delta.style.boxShadow = "none";
            absolute.style.background = state.relative ? "transparent" : "#573677";
            delta.style.background = state.relative ? "#573677" : "transparent";
            absolute.style.color = state.relative ? "#9a9da6" : "#fff";
            delta.style.color = state.relative ? "#fff" : "#9a9da6";
        }
        sign.style.display = state.relative ? "inline-block" : "none";
        sign.textContent = `New edits: ${editSign < 0 ? "negative −" : "positive +"}`;
        sign.style.background = editSign < 0 ? "#682e36" : "#66551e";
        hint.textContent = state.relative
            ? (anchorRect ? "Adjust connected vector" : "Adjusts the connected vector; click the sign to invert edits.")
            : (anchorRect ? "Replace vector" : "Replaces the connected vector.");
        valueInputs.forEach((input, index) => {
            const magnitude = format(state.values[index]);
            input.value = state.relative
                ? `${state.signs[index] < 0 ? "-" : (anchorRect ? "+" : "")}${magnitude}`
                : magnitude;
            input.style.borderColor = state.relative && state.signs[index] < 0 ? "#c8505b" : "#555";
        });
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        radar.draw(ctx, node, canvas.width, 0, canvas.height);
    };
    const pointer = event => {
        const rect = canvas.getBoundingClientRect();
        const pos = [(event.clientX - rect.left) * canvas.width / rect.width, (event.clientY - rect.top) * canvas.height / rect.height];
        radar.mouse(event, pos, node);
        render();
    };
    ["pointerdown", "pointermove", "pointerup"].forEach(type => canvas.addEventListener(type, pointer));
    absolute.onclick = () => { state.relative = false; schedulePreview(); render(); };
    delta.onclick = () => { state.relative = true; schedulePreview(); render(); };
    sign.onclick = () => { editSign *= -1; render(); };

    const presetBox = document.createElement("div");
    presetBox.style.marginTop = "14px";
    presetBox.style.display = showPresets ? "block" : "none";
    const presetTitle = document.createElement("h3");
    presetTitle.textContent = "Emotion presets";
    const presetName = document.createElement("input");
    presetName.placeholder = "preset_name";
    const presetDescription = document.createElement("textarea");
    presetDescription.rows = 3;
    presetDescription.placeholder = "Text emotion description, optionally containing {seg}";
    [presetName, presetDescription].forEach(input => Object.assign(input.style, { padding: "7px", margin: "3px", background: "#151515", color: "#eee", border: "1px solid #555", minWidth: input === presetDescription ? "360px" : "150px", boxSizing: "border-box" }));
    const presetSelect = document.createElement("select");
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Select preset…";
    presetSelect.appendChild(placeholder);
    Object.keys(presets).sort().forEach(name => {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = `${name} (${presets[name]?.type || "unknown"})`;
        presetSelect.appendChild(option);
    });
    const savePreset = button("Save text preset");
    const saveVectorPreset = button("Save current vector preset");
    const applyQuotedText = button("Apply quoted text tag");
    const insertPreset = button("Insert preset tag");
    const deletePreset = button("Delete preset");
    const persist = async () => {
        const response = await api.fetchApi(ENDPOINT, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ presets }) });
        if (!response.ok) throw new Error(`Could not save IndexTTS emotion presets (${response.status})`);
        window.dispatchEvent(new CustomEvent("tts-audio-suite:index-tts-presets-changed", {
            detail: { presets: structuredClone(presets) }
        }));
        return response;
    };
    presetSelect.onchange = () => {
        const preset = presets[presetSelect.value];
        presetName.value = presetSelect.value;
        presetDescription.value = preset?.description || "";
        if (preset?.type === "vector" && Array.isArray(preset.values) && preset.values.length === 8) {
            preset.values.forEach((value, index) => { state.values[index] = number(value); widgets[index].value = state.values[index]; });
            state.relative = false;
            render();
        }
    };
    const validName = name => /^[A-Za-z0-9_-]+$/.test(name);
    const reopen = () => openIndexTTSEmotionEditor({ tag, onApply, anchorRect, showPresets });
    savePreset.onclick = async () => { const name = presetName.value.trim(); const description = presetDescription.value.trim(); if (!validName(name) || !description) return; presets[name] = { type: "text", description }; await persist(); overlay.remove(); reopen(); };
    saveVectorPreset.onclick = async () => {
        const name = presetName.value.trim();
        if (!validName(name)) return;
        presets[name] = {
            type: "vector",
            values: state.values.map(value => Number(Number(value).toFixed(2)))
        };
        await persist();
        overlay.remove();
        reopen();
    };
    applyQuotedText.onclick = () => { const description = presetDescription.value.trim(); if (!description) return; onApply(quoteEmotionText(description)); overlay.remove(); };
    deletePreset.onclick = async () => { if (!presetSelect.value) return; delete presets[presetSelect.value]; await persist(); overlay.remove(); reopen(); };
    insertPreset.onclick = () => {
        const preset = presets[presetSelect.value];
        if (!preset) return;
        onApply(preset.type === "vector" ? serializeVector(preset.values, Array(8).fill(1), false, false) : `[emotion:${presetSelect.value}]`);
        overlay.remove();
    };
    presetBox.append(presetTitle, presetSelect, presetName, presetDescription, applyQuotedText, savePreset, saveVectorPreset, insertPreset, deletePreset);

    if (!anchorRect && showPresets) {
        let activePresetType = tag?.startsWith("[emotion:") ? "text" : "vector";
        Object.assign(presetBox.style, {
            margin: "0", padding: "0", display: "flex", flexDirection: "column",
            minHeight: "0", overflow: "hidden"
        });
        presetBox.replaceChildren();
        presetSelect.style.display = "none";

        const presetTabs = document.createElement("div");
        Object.assign(presetTabs.style, { display: "flex", gap: "6px", padding: "10px", borderBottom: "1px solid #38383d" });
        const vectorTab = button("Vector Presets");
        const textTab = button("Text / Presets");
        [vectorTab, textTab].forEach(tab => Object.assign(tab.style, { padding: "6px 11px", fontSize: "11px" }));
        presetTabs.append(vectorTab, textTab);

        const presetContent = document.createElement("div");
        Object.assign(presetContent.style, { padding: "11px", overflowY: "auto", minHeight: "0" });
        const presetList = document.createElement("div");
        Object.assign(presetList.style, { display: "grid", gap: "6px", marginBottom: "12px", maxHeight: "250px", overflowY: "auto" });

        const editorCard = document.createElement("div");
        Object.assign(editorCard.style, { display: "grid", gap: "7px", padding: "10px", background: "#19191c", border: "1px solid #3a3a40", borderRadius: "8px" });
        const editorHeading = document.createElement("div");
        Object.assign(editorHeading.style, { fontWeight: "700", fontSize: "12px", color: "#eee" });
        const nameLabel = document.createElement("label");
        nameLabel.textContent = "Preset name";
        Object.assign(nameLabel.style, { display: "grid", gap: "4px", fontSize: "10px", color: "#999" });
        Object.assign(presetName.style, { width: "100%", margin: "0", minWidth: "0", borderRadius: "5px" });
        nameLabel.appendChild(presetName);
        const descriptionLabel = document.createElement("label");
        descriptionLabel.textContent = "Emotion text · supports {seg}";
        Object.assign(descriptionLabel.style, { display: "grid", gap: "4px", fontSize: "10px", color: "#999" });
        Object.assign(presetDescription.style, { width: "100%", margin: "0", minWidth: "0", resize: "vertical", borderRadius: "5px", fontFamily: "inherit" });
        descriptionLabel.appendChild(presetDescription);

        const presetActions = document.createElement("div");
        Object.assign(presetActions.style, { display: "flex", flexWrap: "wrap", gap: "6px", alignItems: "center" });
        [savePreset, saveVectorPreset, applyQuotedText, insertPreset, deletePreset].forEach(action => Object.assign(action.style, { padding: "6px 9px", fontSize: "10px" }));
        Object.assign(deletePreset.style, { marginLeft: "auto", color: "#ff8c92", borderColor: "#8a353b", background: "#351b1e" });
        presetActions.append(savePreset, saveVectorPreset, applyQuotedText, insertPreset, deletePreset);
        editorCard.append(editorHeading, nameLabel, descriptionLabel, presetActions, presetSelect);
        presetContent.append(presetList, editorCard);
        presetBox.append(presetTabs, presetContent);

        const renderPresetWorkspace = () => {
            const isText = activePresetType === "text";
            vectorTab.style.background = isText ? "#29292d" : "#493063";
            vectorTab.style.borderColor = isText ? "#555" : "#9966c7";
            textTab.style.background = isText ? "#493063" : "#29292d";
            textTab.style.borderColor = isText ? "#9966c7" : "#555";
            editorHeading.textContent = isText ? "Text preset editor" : "Vector preset editor";
            descriptionLabel.style.display = isText ? "grid" : "none";
            savePreset.style.display = isText ? "inline-block" : "none";
            applyQuotedText.style.display = isText ? "inline-block" : "none";
            saveVectorPreset.style.display = isText ? "none" : "inline-block";

            presetList.replaceChildren();
            const entries = Object.entries(presets)
                .filter(([, preset]) => preset?.type === activePresetType)
                .sort(([a], [b]) => a.localeCompare(b));
            if (!entries.length) {
                const empty = document.createElement("div");
                empty.textContent = `No ${activePresetType} presets yet`;
                Object.assign(empty.style, { padding: "18px", textAlign: "center", color: "#777", border: "1px dashed #444", borderRadius: "7px" });
                presetList.appendChild(empty);
            }
            entries.forEach(([name, preset]) => {
                const card = document.createElement("button");
                card.type = "button";
                Object.assign(card.style, {
                    display: "grid", gridTemplateColumns: "1fr auto", gap: "8px", alignItems: "center",
                    width: "100%", padding: "8px 10px", textAlign: "left", color: "#eee",
                    background: presetSelect.value === name ? "#2d2340" : "#222226",
                    border: presetSelect.value === name ? "1px solid #9d67cb" : "1px solid #3d3d42",
                    borderRadius: "7px", cursor: "pointer"
                });
                const copy = document.createElement("div");
                const cardName = document.createElement("div");
                cardName.textContent = name;
                cardName.style.fontWeight = "650";
                const summary = document.createElement("div");
                summary.textContent = preset.type === "text"
                    ? String(preset.description || "").slice(0, 72)
                    : `${(preset.values || []).filter(value => Number(value) !== 0).length} active dimensions`;
                Object.assign(summary.style, { marginTop: "2px", color: "#888", fontSize: "9px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" });
                copy.append(cardName, summary);
                const badge = document.createElement("span");
                badge.textContent = preset.type.toUpperCase();
                Object.assign(badge.style, { padding: "2px 6px", borderRadius: "999px", fontSize: "8px", color: preset.type === "text" ? "#9ee6b2" : "#9fd2ff", background: preset.type === "text" ? "#1d4a2b" : "#183f5a" });
                card.append(copy, badge);
                card.onclick = () => {
                    presetSelect.value = name;
                    presetSelect.onchange();
                    activePresetType = preset.type;
                    renderPresetWorkspace();
                };
                presetList.appendChild(card);
            });
        };
        vectorTab.onclick = () => { activePresetType = "vector"; presetSelect.value = ""; presetName.value = ""; renderPresetWorkspace(); };
        textTab.onclick = () => { activePresetType = "text"; presetSelect.value = ""; presetName.value = ""; presetDescription.value = ""; renderPresetWorkspace(); };
        renderPresetWorkspace();
    }

    const actions = document.createElement("div");
    Object.assign(actions.style, { display: "flex", justifyContent: anchorRect ? "center" : "flex-end", gap: anchorRect ? "6px" : "8px", marginTop: anchorRect ? "9px" : "15px" });
    const cancel = button("Cancel");
    const apply = button(anchorRect ? "Apply" : "Apply vector tag");
    if (anchorRect) [cancel, apply].forEach(control => Object.assign(control.style, { padding: "5px 10px", fontSize: "11px" }));
    const finishContextualEdit = (commit) => {
        if (closed) return;
        closed = true;
        if (previewFrame !== null) {
            cancelAnimationFrame(previewFrame);
            previewFrame = null;
        }
        if (commit) onCommit?.(currentVectorTag());
        else onCancel?.();
        overlay.remove();
    };
    cancel.onclick = () => anchorRect ? finishContextualEdit(false) : overlay.remove();
    apply.onclick = () => { onApply(serializeVector(state.values, state.signs, state.relative, state.named)); overlay.remove(); };
    actions.append(cancel);
    if (!anchorRect) actions.append(apply);
    if (!anchorRect && showPresets) {
        Object.assign(panel.style, {
            width: "min(1080px,95vw)", padding: "0", background: "#111216",
            border: "1px solid #353840", borderRadius: "12px", overflow: "hidden"
        });
        const header = document.createElement("div");
        Object.assign(header.style, { display: "grid", gridTemplateColumns: "1fr auto", padding: "16px 18px 13px", borderBottom: "1px solid #30323a", background: "linear-gradient(180deg,#17181d,#131419)" });
        const headingCopy = document.createElement("div");
        title.style.margin = "0";
        title.style.fontSize = "21px";
        const subtitle = document.createElement("div");
        subtitle.textContent = "Shape emotional tone with an 8-axis vector or reusable presets.";
        Object.assign(subtitle.style, { marginTop: "4px", color: "#858892", fontSize: "11px" });
        headingCopy.append(title, subtitle);
        const close = button("×");
        Object.assign(close.style, { alignSelf: "start", padding: "0 5px", fontSize: "20px", color: "#999", background: "transparent", border: "none" });
        close.onclick = () => overlay.remove();
        header.append(headingCopy, close);

        const workspace = document.createElement("div");
        Object.assign(workspace.style, { display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(min(440px,100%),1fr))", gap: "10px", padding: "10px", minHeight: "0" });
        const vectorPane = document.createElement("section");
        Object.assign(vectorPane.style, { display: "flex", flexDirection: "column", minWidth: "0", padding: "11px", background: "#18191e", border: "1px solid #34363d", borderRadius: "9px" });
        Object.assign(modeRow.style, { paddingBottom: "8px", borderBottom: "1px solid #303239" });
        canvas.style.background = "#141519";
        valueGrid.style.gridTemplateColumns = "repeat(auto-fit,minmax(190px,1fr))";
        vectorPane.append(modeRow, canvas, valueGrid);

        const presetPane = document.createElement("section");
        Object.assign(presetPane.style, { minWidth: "0", background: "#18191e", border: "1px solid #34363d", borderRadius: "9px", overflow: "hidden" });
        presetPane.appendChild(presetBox);
        workspace.append(vectorPane, presetPane);

        const footer = document.createElement("div");
        Object.assign(footer.style, { display: "flex", justifyContent: "flex-end", padding: "11px 14px", borderTop: "1px solid #30323a", background: "#15161a" });
        Object.assign(apply.style, { background: "#6739a0", borderColor: "#925fd0", boxShadow: "0 0 14px rgba(146,95,208,.22)" });
        footer.append(actions);
        panel.append(header, workspace, footer);
    } else {
        panel.append(title, modeRow, canvas, valueGrid, presetBox, actions);
    }
    overlay.append(panel);
    overlay.addEventListener("pointerdown", event => {
        if (event.target === overlay) anchorRect ? finishContextualEdit(true) : overlay.remove();
    });
    document.body.append(overlay);
    if (anchorRect) {
        const margin = 8;
        const measured = panel.getBoundingClientRect();
        const panelWidth = measured.width;
        const panelHeight = measured.height;
        const left = Math.max(margin, Math.min(anchorRect.left, window.innerWidth - panelWidth - margin));
        const above = anchorRect.top - panelHeight - margin;
        const top = above >= margin ? above : Math.min(window.innerHeight - panelHeight - margin, anchorRect.bottom + margin);
        Object.assign(panel.style, { position: "fixed", left: `${left}px`, top: `${Math.max(margin, top)}px` });
    }
    overlay.tabIndex = -1;
    overlay.addEventListener("keydown", event => {
        if (event.key === "Escape") anchorRect ? finishContextualEdit(false) : overlay.remove();
    });
    overlay.focus({ preventScroll: true });
    const presetMatch = tag?.match(/^\[emotion:([A-Za-z0-9_-]+)\]$/i);
    if (presetMatch && presets[presetMatch[1]]) {
        presetSelect.value = presetMatch[1];
        presetSelect.onchange();
    }
    const quotedMatch = tag?.match(/^\[emotion:(["'])([\s\S]*)\1\]$/i);
    if (quotedMatch) presetDescription.value = quotedMatch[2];
    render();
}

export async function openIndexTTSEmotionPresetPicker({ presetName, anchorRect, onSelect }) {
    let presets = {};
    try {
        const response = await api.fetchApi(ENDPOINT);
        presets = (await response.json()).presets || {};
    } catch (error) {
        console.warn("Could not load IndexTTS emotion presets", error);
    }

    const names = Object.entries(presets)
        .filter(([, preset]) => preset?.type === "text")
        .map(([name]) => name)
        .sort((a, b) => a.localeCompare(b));

    const overlay = document.createElement("div");
    Object.assign(overlay.style, { position: "fixed", inset: "0", zIndex: "100000", background: "transparent" });
    const panel = document.createElement("div");
    Object.assign(panel.style, {
        position: "fixed", width: "min(260px,90vw)", padding: "9px", color: "#eee",
        background: "#202022", border: "1px solid #d78cff", borderRadius: "8px",
        boxShadow: "0 9px 30px rgba(0,0,0,.85), 0 0 12px rgba(215,140,255,.22)"
    });
    const label = document.createElement("div");
    label.textContent = "Swap emotion preset";
    Object.assign(label.style, { marginBottom: "6px", fontSize: "11px", fontWeight: "bold", color: "#d78cff" });
    const list = document.createElement("div");
    Object.assign(list.style, { display: "grid", gap: "4px", maxHeight: "240px", overflowY: "auto" });
    if (!names.length) {
        const empty = document.createElement("div");
        empty.textContent = "No text presets saved";
        Object.assign(empty.style, { padding: "10px", color: "#777", textAlign: "center" });
        list.appendChild(empty);
    } else {
        names.forEach(name => {
            const item = document.createElement("button");
            item.type = "button";
            item.textContent = name;
            Object.assign(item.style, {
                width: "100%", padding: "7px 9px", color: name === presetName ? "#fff" : "#d0d0d4",
                textAlign: "left", background: name === presetName ? "#493063" : "#29292d",
                border: name === presetName ? "1px solid #a06bd0" : "1px solid #44464d",
                borderRadius: "5px", cursor: "pointer", fontSize: "11px"
            });
            item.addEventListener("pointerenter", () => { if (name !== presetName) item.style.background = "#35323d"; });
            item.addEventListener("pointerleave", () => { if (name !== presetName) item.style.background = "#29292d"; });
            item.onclick = () => {
                onSelect(name);
                overlay.remove();
            };
            list.appendChild(item);
        });
    }
    panel.append(label, list);
    overlay.appendChild(panel);
    document.body.appendChild(overlay);

    const measured = panel.getBoundingClientRect();
    const margin = 8;
    const left = Math.max(margin, Math.min(anchorRect.left, window.innerWidth - measured.width - margin));
    const above = anchorRect.top - measured.height - margin;
    const top = above >= margin ? above : Math.min(window.innerHeight - measured.height - margin, anchorRect.bottom + margin);
    Object.assign(panel.style, { left: `${left}px`, top: `${Math.max(margin, top)}px` });

    overlay.addEventListener("pointerdown", event => {
        if (event.target === overlay) overlay.remove();
    });
    overlay.tabIndex = -1;
    overlay.addEventListener("keydown", event => {
        if (event.key === "Escape") overlay.remove();
    });
    overlay.focus({ preventScroll: true });
}

export { parseVectorTag, isEmotionTextTag };
