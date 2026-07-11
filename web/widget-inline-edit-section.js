/**
 * 🏷️ Widget Inline Tags Section
 * Builds an engine-aware inline tag panel for Step Audio EditX, Higgs Audio v3,
 * CosyVoice3, and OmniVoice.
 */

import { api } from "/scripts/api.js";
import { INDEX_TTS_EMOTION_VISUALS } from "./emotion_radar_canvas_widget.js";

const INDEX_TTS_EMOTIONS = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"];
const INDEX_TTS_RADIAL_EMOTIONS = ["happy", "surprised", "angry", "disgusted", "sad", "afraid", "calm", "melancholic"];
const INDEX_TTS_EMOTION_COLORS = Object.fromEntries(
    INDEX_TTS_EMOTION_VISUALS.map(({ name, color }) => [name.toLowerCase(), color])
);

const HIGGS_TAGS = {
    emotion: [
        "elation", "amusement", "enthusiasm", "determination", "pride", "contentment", "affection",
        "relief", "contemplation", "confusion", "surprise", "awe", "longing", "arousal", "anger",
        "fear", "disgust", "bitterness", "sadness", "shame", "helplessness"
    ],
    style: ["singing", "shouting", "whispering"],
    prosody: [
        "speed_very_slow", "speed_slow", "speed_fast", "speed_very_fast",
        "pitch_low", "pitch_high", "pause", "long_pause", "expressive_high", "expressive_low"
    ],
    sfx: ["cough", "laughter", "crying", "screaming", "burping", "humming", "sigh", "sniff", "sneeze"],
};

const COSY_SINGLE_TAGS = [
    "breath", "quick_breath", "laughter", "cough", "sigh", "gasp", "noise",
    "hissing", "vocalized-noise", "lipsmack", "mn", "clucking", "accent"
];

const COSY_WRAPPER_TAGS = ["laughing", "strong"];

const OMNIVOICE_NON_VERBAL_TAGS = [
    "laughter", "sigh", "confirmation-en", "question-en", "question-ah",
    "question-oh", "question-ei", "question-yi", "surprise-ah", "surprise-oh",
    "surprise-wa", "surprise-yo", "dissatisfaction-hnn"
];

function stylePanelContainer(element, { separated = true } = {}) {
    element.style.marginBottom = "8px";
    if (separated) {
        element.style.paddingBottom = "8px";
        element.style.borderBottom = "1px solid #444";
    }
}

function createPanelLabel(text, color, title = "") {
    const label = document.createElement("div");
    label.textContent = text;
    label.style.fontWeight = "bold";
    label.style.marginBottom = "5px";
    label.style.fontSize = "11px";
    label.style.color = color;
    if (title) {
        label.title = title;
    }
    return label;
}

function createSelect(options, placeholder, value = "") {
    const select = document.createElement("select");
    select.style.width = "100%";
    select.style.marginBottom = "4px";
    select.style.padding = "3px";
    select.style.fontSize = "10px";
    select.style.background = "#2a2a2a";
    select.style.color = "#eee";
    select.style.border = "1px solid #444";
    select.innerHTML = `<option value="">${placeholder}</option>${options.map(({ value: optionValue, label }) => `<option value="${optionValue}">${label}</option>`).join("")}`;
    if (value) {
        select.value = value;
    }
    return select;
}

function createInfoText(text, title = "") {
    const label = document.createElement("div");
    label.textContent = text;
    label.style.fontSize = "9px";
    label.style.marginBottom = "4px";
    label.style.color = "#999";
    if (title) {
        label.title = title;
    }
    return label;
}

function createRangeControl(state, storageKey, {
    stateValueKey,
    labelPrefix,
    min = "1",
    max = "5",
    title = "",
}) {
    const value = state[stateValueKey] || "1";
    const label = createInfoText(`${labelPrefix}: ${value}`, title);
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = min;
    slider.max = max;
    slider.value = value;
    slider.style.width = "100%";
    slider.style.marginBottom = "4px";
    if (title) {
        slider.title = title;
    }
    slider.addEventListener("input", () => {
        label.textContent = `${labelPrefix}: ${slider.value}`;
        state[stateValueKey] = slider.value;
        state.saveToLocalStorage(storageKey);
    });
    return { label, slider };
}

function createButton(text, title) {
    const button = document.createElement("button");
    button.textContent = text;
    button.title = title;
    button.style.width = "100%";
    button.style.padding = "4px";
    button.style.cursor = "pointer";
    button.style.fontSize = "10px";
    button.style.background = "#3a3a3a";
    button.style.color = "#eee";
    button.style.border = "1px solid #555";
    button.style.borderRadius = "2px";
    return button;
}

function buildStepSection(state, storageKey) {
    const section = document.createElement("div");
    section.style.display = "flex";
    section.style.flexDirection = "column";
    section.style.gap = "8px";

    const paraSection = document.createElement("div");
    stylePanelContainer(paraSection);
    const paraSelect = createSelect([
        { value: "Laughter", label: "Laughter" },
        { value: "Breathing", label: "Breathing" },
        { value: "Sigh", label: "Sigh" },
        { value: "Uhm", label: "Uhm" },
        { value: "Surprise-oh", label: "Surprise (oh)" },
        { value: "Surprise-ah", label: "Surprise (ah)" },
        { value: "Surprise-wa", label: "Surprise (wa)" },
        { value: "Confirmation-en", label: "Confirmation (en)" },
        { value: "Question-ei", label: "Question (ei)" },
        { value: "Dissatisfaction-hnn", label: "Dissatisfaction (hnn)" },
    ], "Select sound...", state.lastParalinguisticType || "");
    paraSelect.addEventListener("change", () => {
        state.lastParalinguisticType = paraSelect.value;
        state.saveToLocalStorage(storageKey);
    });
    const { label: paraIterLabel, slider: paraIterSlider } = createRangeControl(state, storageKey, {
        stateValueKey: "lastParalinguisticIter",
        labelPrefix: "Iterations",
    });
    const addParaBtn = createButton("Add Paralinguistic", "Insert paralinguistic tag at cursor");
    paraSection.append(
        createPanelLabel("Paralinguistic", "#00ffff"),
        paraSelect,
        paraIterLabel,
        paraIterSlider,
        addParaBtn
    );

    const emotionSection = document.createElement("div");
    stylePanelContainer(emotionSection);
    const emotionSelect = createSelect([
        "happy", "sad", "angry", "excited", "calm", "fearful", "surprised",
        "disgusted", "confusion", "empathy", "embarrass", "depressed", "coldness", "admiration"
    ].map((value) => ({ value, label: value.charAt(0).toUpperCase() + value.slice(1) })), "Select emotion...", state.lastEmotionType || "");
    emotionSelect.addEventListener("change", () => {
        state.lastEmotionType = emotionSelect.value;
        state.saveToLocalStorage(storageKey);
    });
    const { label: emotionIterLabel, slider: emotionIterSlider } = createRangeControl(state, storageKey, {
        stateValueKey: "lastEmotionIter",
        labelPrefix: "Iterations",
    });
    const addEmotionBtn = createButton("Add Emotion", "Insert emotion tag at cursor");
    emotionSection.append(
        createPanelLabel("Emotion", "#ffaa00"),
        emotionSelect,
        emotionIterLabel,
        emotionIterSlider,
        addEmotionBtn
    );

    const styleSection = document.createElement("div");
    stylePanelContainer(styleSection);
    const styleSelect = createSelect([
        "whisper", "serious", "child", "older", "girl", "pure", "sister", "sweet",
        "exaggerated", "ethereal", "generous", "recite", "act_coy", "warm", "shy",
        "comfort", "authority", "chat", "radio", "soulful", "gentle", "story", "vivid",
        "program", "news", "advertising", "roar", "murmur", "shout", "deeply", "loudly",
        "arrogant", "friendly"
    ].map((value) => ({ value, label: value.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase()) })), "Select style...", state.lastStyleType || "");
    styleSelect.addEventListener("change", () => {
        state.lastStyleType = styleSelect.value;
        state.saveToLocalStorage(storageKey);
    });
    const { label: styleIterLabel, slider: styleIterSlider } = createRangeControl(state, storageKey, {
        stateValueKey: "lastStyleIter",
        labelPrefix: "Iterations",
    });
    const addStyleBtn = createButton("Add Style", "Insert style tag at cursor");
    styleSection.append(
        createPanelLabel("Style", "#ff5555"),
        styleSelect,
        styleIterLabel,
        styleIterSlider,
        addStyleBtn
    );

    const speedSection = document.createElement("div");
    stylePanelContainer(speedSection);
    const speedSelect = createSelect([
        { value: "faster", label: "Faster" },
        { value: "slower", label: "Slower" },
        { value: "more_faster", label: "More Faster" },
        { value: "more_slower", label: "More Slower" },
    ], "Select speed...", state.lastSpeedType || "");
    speedSelect.addEventListener("change", () => {
        state.lastSpeedType = speedSelect.value;
        state.saveToLocalStorage(storageKey);
    });
    const { label: speedIterLabel, slider: speedIterSlider } = createRangeControl(state, storageKey, {
        stateValueKey: "lastSpeedIter",
        labelPrefix: "Iterations",
    });
    const addSpeedBtn = createButton("Add Speed", "Insert speed tag at cursor");
    speedSection.append(
        createPanelLabel("Speed", "#66ff66"),
        speedSelect,
        speedIterLabel,
        speedIterSlider,
        addSpeedBtn
    );

    const restoreSection = document.createElement("div");
    stylePanelContainer(restoreSection, { separated: false });
    const { label: restorePassLabel, slider: restorePassSlider } = createRangeControl(state, storageKey, {
        stateValueKey: "lastRestorePasses",
        labelPrefix: "VC Passes",
        title: "How many restoration passes to run. More passes push harder toward the chosen reference voice.",
    });
    const restoreRefLabel = createInfoText(
        "Reference Iteration (optional)",
        "Optional edit-step reference. In <restore:1@2>, the @2 points to edit step 2 from earlier inline edits."
    );
    const restoreRefInput = document.createElement("input");
    restoreRefInput.type = "number";
    restoreRefInput.placeholder = "Leave empty for original";
    restoreRefInput.min = "1";
    restoreRefInput.max = "10";
    restoreRefInput.value = state.lastRestoreRefIter || "";
    restoreRefInput.style.width = "100%";
    restoreRefInput.style.padding = "3px";
    restoreRefInput.style.fontSize = "10px";
    restoreRefInput.style.background = "#2a2a2a";
    restoreRefInput.style.color = "#eee";
    restoreRefInput.style.border = "1px solid #444";
    restoreRefInput.style.marginBottom = "4px";
    restoreRefInput.title = "Leave empty to use the original clean pre-edit audio. Enter an edit step number to use that earlier edited snapshot as the reference voice.";
    restoreRefInput.addEventListener("change", () => {
        state.lastRestoreRefIter = restoreRefInput.value;
        state.saveToLocalStorage(storageKey);
    });
    const addRestoreBtn = createButton("Add Restore", "Insert <restore>, <restore:2>, or <restore:1@2>.");
    restoreSection.append(
        createPanelLabel("Voice Restoration", "#ffcc33", "Restore runs after all other inline edits."),
        restorePassLabel,
        restorePassSlider,
        restoreRefLabel,
        restoreRefInput,
        addRestoreBtn
    );

    section.append(paraSection, emotionSection, styleSection, speedSection, restoreSection);

    return {
        panel: section,
        controls: {
            paraSelect, paraIterSlider, addParaBtn,
            emotionSelect, emotionIterSlider, addEmotionBtn,
            styleSelect, styleIterSlider, addStyleBtn,
            speedSelect, speedIterSlider, addSpeedBtn,
            restorePassSlider, restoreRefInput, addRestoreBtn,
        },
    };
}

function buildHiggsSection(state, storageKey) {
    const section = document.createElement("div");
    section.style.display = "flex";
    section.style.flexDirection = "column";
    section.style.gap = "8px";

    state.lastHiggsInlineValues = state.lastHiggsInlineValues || {};

    const buildHiggsTagSubsection = ({ category, label, color, separated = true }) => {
        const subsection = document.createElement("div");
        stylePanelContainer(subsection, { separated });
        const savedValue = state.lastHiggsInlineValues[category] || HIGGS_TAGS[category][0] || "";
        const select = createSelect(
            HIGGS_TAGS[category].map((value) => ({ value, label: value })),
            `Select ${label.toLowerCase()}...`,
            savedValue
        );
        select.addEventListener("change", () => {
            state.lastHiggsInlineValues[category] = select.value;
            state.saveToLocalStorage(storageKey);
        });
        const addBtn = createButton(`Add ${label}`, `Insert canonical Higgs ${label.toLowerCase()} tag at cursor`);
        subsection.append(
            createPanelLabel(label, color),
            select,
            addBtn
        );
        return { subsection, select, addBtn };
    };

    const emotionSection = buildHiggsTagSubsection({
        category: "emotion",
        label: "Emotion",
        color: "#64d7ff",
    });
    const styleSection = buildHiggsTagSubsection({
        category: "style",
        label: "Style",
        color: "#8fcbff",
    });
    const prosodySection = buildHiggsTagSubsection({
        category: "prosody",
        label: "Prosody",
        color: "#b5a3ff",
    });
    const sfxSection = buildHiggsTagSubsection({
        category: "sfx",
        label: "SFX",
        color: "#ff9e7a",
        separated: false,
    });

    section.append(
        emotionSection.subsection,
        styleSection.subsection,
        prosodySection.subsection,
        sfxSection.subsection
    );

    return {
        panel: section,
        controls: {
            emotionSelect: emotionSection.select,
            addEmotionBtn: emotionSection.addBtn,
            styleSelect: styleSection.select,
            addStyleBtn: styleSection.addBtn,
            prosodySelect: prosodySection.select,
            addProsodyBtn: prosodySection.addBtn,
            sfxSelect: sfxSection.select,
            addSfxBtn: sfxSection.addBtn,
        },
    };
}

function buildCosySection(state, storageKey) {
    const section = document.createElement("div");
    section.style.display = "flex";
    section.style.flexDirection = "column";
    section.style.gap = "8px";

    const singleSection = document.createElement("div");
    stylePanelContainer(singleSection);
    const singleTagSelect = createSelect(
        COSY_SINGLE_TAGS.map((value) => ({ value, label: value })),
        "Select single tag...",
        state.lastCosyInlineSingleTag || "breath"
    );
    singleTagSelect.addEventListener("change", () => {
        state.lastCosyInlineSingleTag = singleTagSelect.value;
        state.saveToLocalStorage(storageKey);
    });
    const addSingleTagBtn = createButton("Add Single Tag", "Insert native CosyVoice3 single tag like <breath>");
    singleSection.append(
        createPanelLabel("Single Tags", "#8cff9d"),
        singleTagSelect,
        addSingleTagBtn
    );

    const wrapperSection = document.createElement("div");
    stylePanelContainer(wrapperSection, { separated: false });
    const wrapperTagSelect = createSelect(
        COSY_WRAPPER_TAGS.map((value) => ({ value, label: value })),
        "Select wrapper tag...",
        state.lastCosyInlineWrapperTag || "laughing"
    );
    wrapperTagSelect.addEventListener("change", () => {
        state.lastCosyInlineWrapperTag = wrapperTagSelect.value;
        state.saveToLocalStorage(storageKey);
    });
    const wrapperHint = createInfoText("If text is selected, the wrapper will surround it. Otherwise it inserts an example pair.");
    const addWrapperTagBtn = createButton("Add Wrapper Tag", "Insert wrapper tag pair like <laughing>text</laughing>");
    wrapperSection.append(
        createPanelLabel("Wrapper Tags", "#f6c95b"),
        wrapperTagSelect,
        wrapperHint,
        addWrapperTagBtn
    );

    section.append(singleSection, wrapperSection);

    return {
        panel: section,
        controls: {
            singleTagSelect,
            addSingleTagBtn,
            wrapperTagSelect,
            addWrapperTagBtn,
        },
    };
}

function buildOmniVoiceSection(state, storageKey) {
    const section = document.createElement("div");
    section.style.display = "flex";
    section.style.flexDirection = "column";
    section.style.gap = "8px";

    const tagSection = document.createElement("div");
    stylePanelContainer(tagSection, { separated: false });
    const tagSelect = createSelect(
        OMNIVOICE_NON_VERBAL_TAGS.map((value) => ({ value, label: value })),
        "Select OmniVoice tag...",
        state.lastOmniVoiceInlineTag || "laughter"
    );
    tagSelect.addEventListener("change", () => {
        state.lastOmniVoiceInlineTag = tagSelect.value;
        state.saveToLocalStorage(storageKey);
    });
    const tagHint = createInfoText(
        "Editor inserts suite-default aliases like <laughter>. The OmniVoice processor converts them to official [laughter] tags at generation time.",
        "OmniVoice also supports bracketed CMU pronunciation overrides like [B EY1 S], but those are typed manually."
    );
    const addTagBtn = createButton("Add OmniVoice Tag", "Insert suite-default OmniVoice tag like <laughter>");
    tagSection.append(
        createPanelLabel("Native Non-Verbal Tags", "#82d4ff"),
        tagSelect,
        tagHint,
        addTagBtn
    );

    section.append(tagSection);

    return {
        panel: section,
        controls: {
            tagSelect,
            addTagBtn,
        },
    };
}

function createIndexTTSRadialPicker(onPick) {
    const button = createButton("Press + drag to pick emotion", "Drag toward an emotion; distance sets its magnitude");
    button.style.background = `linear-gradient(90deg, ${INDEX_TTS_RADIAL_EMOTIONS.map(emotion => INDEX_TTS_EMOTION_COLORS[emotion]).join(", ")})`;

    button.addEventListener("pointerdown", event => {
        if (event.button !== 0) return;
        event.preventDefault();
        event.stopPropagation();

        const size = 220;
        const center = size / 2;
        const maxRadius = 82;
        const originX = event.clientX;
        const originY = event.clientY;
        const canvas = document.createElement("canvas");
        canvas.width = size;
        canvas.height = size;
        Object.assign(canvas.style, {
            position: "fixed", left: `${originX - center}px`, top: `${originY - center}px`,
            width: `${size}px`, height: `${size}px`, zIndex: "100002", pointerEvents: "none",
            borderRadius: "50%", filter: "drop-shadow(0 8px 20px rgba(0,0,0,.85))"
        });
        document.body.appendChild(canvas);
        const ctx = canvas.getContext("2d");
        let selectedIndex = 0;
        let magnitude = 0;

        const draw = () => {
            ctx.clearRect(0, 0, size, size);
            const selectedEmotion = INDEX_TTS_RADIAL_EMOTIONS[selectedIndex];
            const selectedColor = INDEX_TTS_EMOTION_COLORS[selectedEmotion];
            ctx.fillStyle = "rgba(25,25,27,.96)";
            ctx.beginPath();
            ctx.arc(center, center, 104, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = selectedColor;
            ctx.lineWidth = 2;
            ctx.stroke();

            const selectedAngle = selectedIndex * Math.PI / 4 - Math.PI / 2;
            ctx.fillStyle = `${selectedColor}38`;
            ctx.beginPath();
            ctx.moveTo(center, center);
            ctx.arc(center, center, 100, selectedAngle - Math.PI / 8, selectedAngle + Math.PI / 8);
            ctx.closePath();
            ctx.fill();

            INDEX_TTS_RADIAL_EMOTIONS.forEach((emotion, index) => {
                const angle = index * Math.PI / 4 - Math.PI / 2;
                const x = center + Math.cos(angle) * maxRadius;
                const y = center + Math.sin(angle) * maxRadius;
                ctx.fillStyle = index === selectedIndex ? INDEX_TTS_EMOTION_COLORS[emotion] : "#aaa";
                ctx.font = index === selectedIndex ? "bold 11px Arial" : "10px Arial";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(emotion, x, y);
            });

            ctx.fillStyle = selectedColor;
            ctx.beginPath();
            ctx.arc(center, center, 28, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = selectedEmotion === "happy" ? "#201b00" : "#f5f5f5";
            ctx.font = "bold 12px Arial";
            ctx.textAlign = "center";
            ctx.fillText(magnitude.toFixed(2), center, center);
        };

        const update = pointerEvent => {
            const dx = pointerEvent.clientX - originX;
            const dy = pointerEvent.clientY - originY;
            const distance = Math.sqrt(dx * dx + dy * dy);
            let angle = Math.atan2(dy, dx) + Math.PI / 2;
            if (angle < 0) angle += Math.PI * 2;
            selectedIndex = Math.round(angle / (Math.PI / 4)) % 8;
            magnitude = Math.min(1.2, Math.max(0, distance / maxRadius * 1.2));
            onPick(INDEX_TTS_RADIAL_EMOTIONS[selectedIndex], magnitude, false);
            draw();
        };

        const finish = pointerEvent => {
            update(pointerEvent);
            window.removeEventListener("pointermove", update, true);
            window.removeEventListener("pointerup", finish, true);
            window.removeEventListener("pointercancel", cancel, true);
            canvas.remove();
            onPick(INDEX_TTS_RADIAL_EMOTIONS[selectedIndex], magnitude, true);
        };
        const cancel = () => {
            window.removeEventListener("pointermove", update, true);
            window.removeEventListener("pointerup", finish, true);
            window.removeEventListener("pointercancel", cancel, true);
            canvas.remove();
        };
        window.addEventListener("pointermove", update, true);
        window.addEventListener("pointerup", finish, true);
        window.addEventListener("pointercancel", cancel, true);
        draw();
    });
    return button;
}

function buildIndexTTSSection(state, storageKey) {
    const section = document.createElement("div");
    Object.assign(section.style, { display: "flex", flexDirection: "column", gap: "8px" });

    const vectorSection = document.createElement("div");
    stylePanelContainer(vectorSection);
    const vectorModeSelect = createSelect([
        { value: "absolute", label: "Absolute vector" },
        { value: "delta", label: "Delta vector" },
    ], "Vector mode...", state.lastIndexTTSVectorMode || "absolute");
    vectorModeSelect.addEventListener("change", () => {
        state.lastIndexTTSVectorMode = vectorModeSelect.value;
        state.saveToLocalStorage(storageKey);
    });
    const addVectorBtn = createButton("Add Vector Tag", "Insert a full vector tag; click it in the editor to open the radar");
    vectorSection.append(
        createPanelLabel("Full Vector Radar", "#45b7d1"),
        vectorModeSelect,
        createInfoText("Insert the tag, then click it in the editor for live radar editing."),
        addVectorBtn
    );

    const namedSection = document.createElement("div");
    stylePanelContainer(namedSection);
    const namedEmotionSelect = createSelect(
        INDEX_TTS_EMOTIONS.map(value => ({ value, label: value.charAt(0).toUpperCase() + value.slice(1) })),
        "Select emotion...",
        state.lastIndexTTSNamedEmotion || "happy"
    );
    const namedOperationSelect = createSelect([
        { value: "absolute", label: "Absolute value" },
        { value: "positive", label: "Positive delta (+)" },
        { value: "negative", label: "Negative delta (−)" },
    ], "Operation...", state.lastIndexTTSNamedOperation || "absolute");
    const namedValueInput = document.createElement("input");
    namedValueInput.type = "range";
    namedValueInput.min = "0";
    namedValueInput.max = "1.2";
    namedValueInput.step = "0.05";
    namedValueInput.value = state.lastIndexTTSNamedValue ?? "0.5";
    Object.assign(namedValueInput.style, { width: "100%", boxSizing: "border-box", marginBottom: "4px" });
    const namedValueLabel = createInfoText(`Magnitude: ${Number(namedValueInput.value).toFixed(2)}`);
    const updateNamedSliderFill = () => {
        const ratio = Math.max(0, Math.min(100, Number(namedValueInput.value) / 1.2 * 100));
        const color = INDEX_TTS_EMOTION_COLORS[namedEmotionSelect.value] || "#20B2AA";
        namedValueInput.dataset.fillColor = color;
        namedValueInput.style.accentColor = color;
        namedValueInput.style.background = `linear-gradient(90deg, ${color} 0%, ${color} ${ratio}%, rgba(53,53,52,.9) ${ratio}%, rgba(53,53,52,.9) 100%)`;
    };
    const saveNamedState = (persist = true) => {
        state.lastIndexTTSNamedEmotion = namedEmotionSelect.value;
        state.lastIndexTTSNamedOperation = namedOperationSelect.value;
        state.lastIndexTTSNamedValue = namedValueInput.value;
        if (persist) state.saveToLocalStorage(storageKey);
    };
    namedEmotionSelect.addEventListener("change", saveNamedState);
    namedOperationSelect.addEventListener("change", saveNamedState);
    namedValueInput.addEventListener("input", () => {
        namedValueLabel.textContent = `Magnitude: ${Number(namedValueInput.value).toFixed(2)}`;
        updateNamedSliderFill();
        saveNamedState();
    });
    const radialPickerBtn = createIndexTTSRadialPicker((emotion, magnitude, isFinal) => {
        namedEmotionSelect.value = emotion;
        namedValueInput.value = magnitude.toFixed(2);
        namedValueLabel.textContent = `Magnitude: ${magnitude.toFixed(2)} · ${emotion}`;
        updateNamedSliderFill();
        saveNamedState(isFinal);
    });
    namedEmotionSelect.addEventListener("change", updateNamedSliderFill);
    updateNamedSliderFill();
    const addNamedEmotionBtn = createButton("Add Named Emotion", "Insert [sad:0.5], [sad:+0.5], or [sad:-0.5]");
    namedSection.append(
        createPanelLabel("Named Emotion Value", "#ffcc66"),
        radialPickerBtn,
        namedEmotionSelect, namedOperationSelect, namedValueLabel, namedValueInput, addNamedEmotionBtn
    );

    const textSection = document.createElement("div");
    stylePanelContainer(textSection, { separated: false });
    const presetSelect = createSelect([], "Select saved preset...");
    const populateTextPresets = (presets = {}) => {
        const previous = presetSelect.value;
        while (presetSelect.options.length > 1) presetSelect.remove(1);
        Object.entries(presets)
            .filter(([, preset]) => preset?.type === "text" || preset?.type === "vector")
            .sort(([a], [b]) => a.localeCompare(b))
            .forEach(([name, preset]) => {
                const option = document.createElement("option");
                option.value = name;
                option.textContent = `${name} (${preset.type})`;
                option.dataset.presetType = preset.type;
                if (preset.type === "vector") option.dataset.vectorValues = JSON.stringify(preset.values || []);
                presetSelect.appendChild(option);
            });
        if ([...presetSelect.options].some(option => option.value === previous)) presetSelect.value = previous;
    };
    const refreshTextPresets = () => api.fetchApi("/api/tts-audio-suite/index-tts-emotion-presets")
        .then(response => response.json())
        .then(({ presets = {} }) => populateTextPresets(presets))
        .catch(error => console.warn("Could not load IndexTTS emotion presets", error));
    refreshTextPresets();
    presetSelect.addEventListener("focus", refreshTextPresets);
    window.addEventListener("tts-audio-suite:index-tts-presets-changed", event => {
        populateTextPresets(event.detail?.presets || {});
    });
    const addTextPresetBtn = createButton("Add Preset Tag", "Insert a saved text or vector emotion preset");
    const managePresetsBtn = createButton("Manage Emotion Presets", "Open the IndexTTS emotion preset editor");
    managePresetsBtn.style.marginBottom = "5px";
    const emotionTextInput = document.createElement("input");
    emotionTextInput.type = "text";
    emotionTextInput.placeholder = "Emotion description; {seg} makes it dynamic";
    emotionTextInput.value = state.lastIndexTTSEmotionText || "";
    Object.assign(emotionTextInput.style, { width: "100%", boxSizing: "border-box", padding: "3px", margin: "4px 0", fontSize: "10px", color: "#eee", background: "#2a2a2a", border: "1px solid #444" });
    emotionTextInput.addEventListener("change", () => {
        state.lastIndexTTSEmotionText = emotionTextInput.value;
        state.saveToLocalStorage(storageKey);
    });
    const addEmotionTextBtn = createButton("Add Quoted Text", "Insert a quoted text emotion; include {seg} for dynamic analysis");
    textSection.append(
        createPanelLabel("Text Emotion", "#d78cff"),
        managePresetsBtn,
        presetSelect, addTextPresetBtn,
        emotionTextInput, addEmotionTextBtn
    );

    section.append(vectorSection, namedSection, textSection);
    return {
        panel: section,
        controls: {
            vectorModeSelect, addVectorBtn,
            namedEmotionSelect, namedOperationSelect, namedValueInput, addNamedEmotionBtn,
            presetSelect, addTextPresetBtn, managePresetsBtn, emotionTextInput, addEmotionTextBtn,
        },
    };
}

export function buildInlineEditSection(state, storageKey) {
    const container = document.createElement("div");
    container.style.display = "flex";
    container.style.flexDirection = "column";
    container.style.gap = "8px";
    container.style.overflowY = "visible";
    container.style.flex = "0 0 auto";
    container.style.paddingRight = "0";

    const engineSection = document.createElement("div");
    stylePanelContainer(engineSection);
    const inlineEngineSelect = createSelect([
        { value: "step_audio_editx", label: "Step Audio EditX" },
        { value: "higgs_audio_v3", label: "Higgs Audio v3" },
        { value: "cosyvoice3", label: "CosyVoice3" },
        { value: "omnivoice", label: "OmniVoice" },
        { value: "index_tts", label: "IndexTTS-2" },
    ], "Select inline tag engine...", state.activeInlineTagEngine || "step_audio_editx");
    inlineEngineSelect.addEventListener("change", () => {
        state.activeInlineTagEngine = inlineEngineSelect.value;
        updateVisiblePanel();
        state.saveToLocalStorage(storageKey);
    });
    engineSection.append(
        createPanelLabel("Tag Engine", "#d0d0d0"),
        inlineEngineSelect
    );

    const stepSection = buildStepSection(state, storageKey);
    const higgsSection = buildHiggsSection(state, storageKey);
    const cosySection = buildCosySection(state, storageKey);
    const omnivoiceSection = buildOmniVoiceSection(state, storageKey);
    const indexTTSSection = buildIndexTTSSection(state, storageKey);

    const panels = {
        step_audio_editx: stepSection.panel,
        higgs_audio_v3: higgsSection.panel,
        cosyvoice3: cosySection.panel,
        omnivoice: omnivoiceSection.panel,
        index_tts: indexTTSSection.panel,
    };

    const updateVisiblePanel = () => {
        const activeEngine = inlineEngineSelect.value || "step_audio_editx";
        Object.entries(panels).forEach(([engineKey, panel]) => {
            panel.style.display = engineKey === activeEngine ? "flex" : "none";
        });
    };

    container.append(engineSection, stepSection.panel, higgsSection.panel, cosySection.panel, omnivoiceSection.panel, indexTTSSection.panel);
    updateVisiblePanel();

    return {
        inlineEditSection: container,
        inlineTagControls: {
            inlineEngineSelect,
            step: stepSection.controls,
            higgs: higgsSection.controls,
            cosy: cosySection.controls,
            omnivoice: omnivoiceSection.controls,
            indexTTS: indexTTSSection.controls,
        },
        updateInlineEnginePanelVisibility: updateVisiblePanel,
    };
}
