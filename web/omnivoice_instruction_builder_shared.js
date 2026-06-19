const NODE_CLASS = "OmniVoiceInstructionBuilderNode";
const PANEL_MIN_WIDTH = 700;
const PANEL_MIN_HEIGHT = 500;
const PANEL_WIDGET_MIN_HEIGHT = 392;
const PANEL_BOTTOM_PADDING = 12;
const RESTING_OFFSET_LIMIT = 26;
const DRAG_START_THRESHOLD = 4;
const PATH_BASE_STROKE_WIDTH = 2.2;
const TEMP_TEXT_PRESET_ID = "temp_text";
const OMNIVOICE_PRESET_ID = "omnivoice";
const OMNIVOICE_PRESET_LIBRARY_URL = "/api/tts-audio-suite/omnivoice-presets";

const EN_TO_ZH = {
    "male": "男",
    "female": "女",
    "child": "儿童",
    "teenager": "少年",
    "young adult": "青年",
    "middle-aged": "中年",
    "elderly": "老年",
    "very low pitch": "极低音调",
    "low pitch": "低音调",
    "moderate pitch": "中音调",
    "high pitch": "高音调",
    "very high pitch": "极高音调",
    "whisper": "耳语",
};

const UI_TEXT = {
    en: {
        previewLabel: "OUT >",
        emptyPreview: "Select attributes or type comma-separated tags",
        localeEn: "EN",
        localeZh: "中文",
        editPreset: "Edit",
        presetName: "Preset Name",
        addColumn: "+ Column",
        addSwitch: "+ Switch",
        addMode: "+ Mode",
        remove: "Remove",
        save: "Save",
        cancel: "Cancel",
        deletePreset: "Delete",
        columnTitle: "Column Title",
        columnType: "Column Type",
        modeTitle: "Mode Title",
        options: "Options",
        optionsHint: "One option per line. Use label=value or just value.",
        single: "Single",
        switch: "Switch",
        editPresetTitle: "Edit Preset",
        createPresetTitle: "Create Preset",
    },
    zh: {
        previewLabel: "输出 >",
        emptyPreview: "选择属性或输入逗号分隔标签",
        localeEn: "EN",
        localeZh: "中文",
        editPreset: "编辑",
        presetName: "预设名称",
        addColumn: "+ 列",
        addSwitch: "+ 切换列",
        addMode: "+ 模式",
        remove: "删除",
        save: "保存",
        cancel: "取消",
        deletePreset: "删除",
        columnTitle: "列标题",
        columnType: "列类型",
        modeTitle: "模式标题",
        options: "选项",
        optionsHint: "每行一个选项。可用 label=value 或只写 value。",
        single: "单列",
        switch: "切换列",
        editPresetTitle: "编辑预设",
        createPresetTitle: "创建预设",
    },
};

const ZH_UI_LABELS = {
    ...EN_TO_ZH,
    "american accent": "美式口音",
    "british accent": "英式口音",
    "australian accent": "澳式口音",
    "canadian accent": "加拿大口音",
    "indian accent": "印度口音",
    "chinese accent": "中式口音",
    "korean accent": "韩式口音",
    "japanese accent": "日式口音",
    "portuguese accent": "葡萄牙口音",
    "russian accent": "俄式口音",
    "河南话": "河南话",
    "陕西话": "陕西话",
    "四川话": "四川话",
    "贵州话": "贵州话",
    "云南话": "云南话",
    "桂林话": "桂林话",
    "济南话": "济南话",
    "石家庄话": "石家庄话",
    "甘肃话": "甘肃话",
    "宁夏话": "宁夏话",
    "青岛话": "青岛话",
    "东北话": "东北话",
};

const OMNIVOICE_PRESET = {
    id: OMNIVOICE_PRESET_ID,
    name: "OmniVoice",
    outputMode: "omnivoice",
    columns: [
        {
            id: "gender",
            title: "Gender",
            zhTitle: "性别",
            type: "single",
            options: [
                { id: "male", label: "Male", zhLabel: "男", value: "male", title: "male" },
                { id: "female", label: "Female", zhLabel: "女", value: "female", title: "female" },
            ],
        },
        {
            id: "age",
            title: "Age",
            zhTitle: "年龄",
            type: "single",
            options: [
                { id: "child", label: "Child", zhLabel: "儿童", value: "child", title: "child" },
                { id: "teen", label: "Teen", zhLabel: "少年", value: "teenager", title: "teenager" },
                { id: "young_adult", label: "Young Adult", zhLabel: "青年", value: "young adult", title: "young adult" },
                { id: "middle_aged", label: "Middle-aged", zhLabel: "中年", value: "middle-aged", title: "middle-aged" },
                { id: "elderly", label: "Elderly", zhLabel: "老年", value: "elderly", title: "elderly" },
            ],
        },
        {
            id: "pitch",
            title: "Pitch",
            zhTitle: "音调",
            type: "single",
            options: [
                { id: "very_low", label: "Very Low", zhLabel: "极低音调", value: "very low pitch", title: "very low pitch" },
                { id: "low", label: "Low", zhLabel: "低音调", value: "low pitch", title: "low pitch" },
                { id: "moderate", label: "Moderate", zhLabel: "中音调", value: "moderate pitch", title: "moderate pitch" },
                { id: "high", label: "High", zhLabel: "高音调", value: "high pitch", title: "high pitch" },
                { id: "very_high", label: "Very High", zhLabel: "极高音调", value: "very high pitch", title: "very high pitch" },
            ],
        },
        {
            id: "style",
            title: "Style",
            zhTitle: "风格",
            type: "single",
            options: [
                { id: "whisper", label: "Whisper", zhLabel: "耳语", value: "whisper", title: "whisper" },
            ],
        },
        {
            id: "language",
            title: "Language",
            zhTitle: "语言",
            type: "switch",
            modes: [
                {
                    id: "accent",
                    title: "Accent",
                    zhTitle: "口音",
                    options: [
                        { id: "us", label: "US", value: "american accent", title: "american accent", zhTitle: "美式口音" },
                        { id: "uk", label: "UK", value: "british accent", title: "british accent", zhTitle: "英式口音" },
                        { id: "au", label: "AU", value: "australian accent", title: "australian accent", zhTitle: "澳式口音" },
                        { id: "ca", label: "CA", value: "canadian accent", title: "canadian accent", zhTitle: "加拿大口音" },
                        { id: "in", label: "IN", value: "indian accent", title: "indian accent", zhTitle: "印度口音" },
                        { id: "cn", label: "CN", value: "chinese accent", title: "chinese accent", zhTitle: "中式口音" },
                        { id: "kr", label: "KR", value: "korean accent", title: "korean accent", zhTitle: "韩式口音" },
                        { id: "jp", label: "JP", value: "japanese accent", title: "japanese accent", zhTitle: "日式口音" },
                        { id: "pt", label: "PT", value: "portuguese accent", title: "portuguese accent", zhTitle: "葡萄牙口音" },
                        { id: "ru", label: "RU", value: "russian accent", title: "russian accent", zhTitle: "俄式口音" },
                    ],
                },
                {
                    id: "dialect",
                    title: "Dialect",
                    zhTitle: "方言",
                    options: [
                        { id: "henan", label: "Henan", value: "河南话", title: "河南话", zhLabel: "河南话" },
                        { id: "shaanxi", label: "Shaanxi", value: "陕西话", title: "陕西话", zhLabel: "陕西话" },
                        { id: "sichuan", label: "Sichuan", value: "四川话", title: "四川话", zhLabel: "四川话" },
                        { id: "guizhou", label: "Guizhou", value: "贵州话", title: "贵州话", zhLabel: "贵州话" },
                        { id: "yunnan", label: "Yunnan", value: "云南话", title: "云南话", zhLabel: "云南话" },
                        { id: "guilin", label: "Guilin", value: "桂林话", title: "桂林话", zhLabel: "桂林话" },
                        { id: "jinan", label: "Jinan", value: "济南话", title: "济南话", zhLabel: "济南话" },
                        { id: "shijiazhuang", label: "Shijiazhuang", value: "石家庄话", title: "石家庄话", zhLabel: "石家庄话" },
                        { id: "gansu", label: "Gansu", value: "甘肃话", title: "甘肃话", zhLabel: "甘肃话" },
                        { id: "ningxia", label: "Ningxia", value: "宁夏话", title: "宁夏话", zhLabel: "宁夏话" },
                        { id: "qingdao", label: "Qingdao", value: "青岛话", title: "青岛话", zhLabel: "青岛话" },
                        { id: "northeast", label: "Northeast", value: "东北话", title: "东北话", zhLabel: "东北话" },
                    ],
                },
            ],
        },
    ],
};

function cloneJson(value) {
    return value ? JSON.parse(JSON.stringify(value)) : value;
}

function slugifyId(value, fallback = "preset") {
    const slug = String(value || "")
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "_")
        .replace(/^_+|_+$/g, "");
    return slug || fallback;
}

function normalizeWidgetValue(value) {
    const text = String(value ?? "").trim();
    return !text || text === "None" ? "" : text;
}

function normalizePresetStateSnapshot(snapshot) {
    const values = {};
    const switchModes = {};

    if (snapshot?.values && typeof snapshot.values === "object") {
        for (const [key, value] of Object.entries(snapshot.values)) {
            const normalizedValue = normalizeWidgetValue(value);
            if (normalizedValue) {
                values[String(key)] = normalizedValue;
            }
        }
    }

    if (snapshot?.switchModes && typeof snapshot.switchModes === "object") {
        for (const [key, value] of Object.entries(snapshot.switchModes)) {
            const normalizedValue = normalizeWidgetValue(value);
            if (normalizedValue) {
                switchModes[String(key)] = normalizedValue;
            }
        }
    }

    return { values, switchModes };
}

function hasMeaningfulPresetState(snapshot) {
    const normalized = normalizePresetStateSnapshot(snapshot);
    return Boolean(
        Object.keys(normalized.values).length
        || Object.keys(normalized.switchModes).length
    );
}

function normalizeColumnOrder(order, presetColumnIds) {
    const filtered = Array.isArray(order)
        ? order.filter((columnId) => presetColumnIds.includes(columnId))
        : [];
    return filtered.length === presetColumnIds.length ? filtered : [...presetColumnIds];
}

function normalizeColumnOffset(value) {
    if (typeof value === "number") {
        return { x: Number(value) || 0, y: 0 };
    }
    if (value && typeof value === "object") {
        return {
            x: Number(value.x) || 0,
            y: Number(value.y) || 0,
        };
    }
    return { x: 0, y: 0 };
}

function createEl(tag, className, text) {
    const el = document.createElement(tag);
    if (className) {
        el.className = className;
    }
    if (text !== undefined) {
        el.textContent = text;
    }
    return el;
}

function isBuilderNode(node) {
    return node?.comfyClass === NODE_CLASS;
}

function findWidgetByName(node, name) {
    return node.widgets ? node.widgets.find((widget) => widget.name === name) : null;
}

function hideWidget(widget) {
    if (!widget) {
        return;
    }
    if (widget.__omnivoiceInstructionOriginalType === undefined) {
        widget.__omnivoiceInstructionOriginalType = widget.type;
    }
    widget.type = "hidden";
    widget.hidden = true;
    widget.computeSize = () => [0, -4];
    widget.draw = () => {};
    if (widget.element) {
        widget.element.style.display = "none";
    }
}

function setWidgetValue(widget, value) {
    if (!widget) {
        return;
    }
    const nextValue = value || "None";
    if (widget.value === nextValue) {
        return;
    }
    widget.value = nextValue;
    if (typeof widget.callback === "function") {
        widget.callback(nextValue);
    }
}

function getBuiltinPresetById(presetId) {
    return presetId === OMNIVOICE_PRESET_ID ? OMNIVOICE_PRESET : null;
}

function normalizePresetOption(option, index = 0) {
    const fallbackValue = `option_${index + 1}`;
    const rawValue = option?.value ?? option?.label ?? option?.title ?? fallbackValue;
    const value = normalizeWidgetValue(rawValue);
    const label = String(option?.label ?? value ?? `Option ${index + 1}`);
    return {
        id: slugifyId(option?.id || value || label, `option_${index + 1}`),
        label,
        zhLabel: option?.zhLabel ? String(option.zhLabel) : undefined,
        value,
        title: String(option?.title ?? value ?? label),
        zhTitle: option?.zhTitle ? String(option.zhTitle) : undefined,
    };
}

function normalizePresetMode(mode, index = 0) {
    const options = Array.isArray(mode?.options)
        ? mode.options.map((option, optionIndex) => normalizePresetOption(option, optionIndex)).filter((option) => option.value)
        : [];
    return {
        id: slugifyId(mode?.id || mode?.title || `mode_${index + 1}`, `mode_${index + 1}`),
        title: String(mode?.title || `Mode ${index + 1}`),
        zhTitle: mode?.zhTitle ? String(mode.zhTitle) : undefined,
        options,
    };
}

function normalizePresetColumn(column, index = 0) {
    const type = column?.type === "switch" ? "switch" : "single";
    const normalized = {
        id: slugifyId(column?.id || column?.title || `column_${index + 1}`, `column_${index + 1}`),
        title: String(column?.title || ""),
        zhTitle: column?.zhTitle ? String(column.zhTitle) : undefined,
        type,
    };
    if (type === "switch") {
        normalized.modes = Array.isArray(column?.modes)
            ? column.modes.map((mode, modeIndex) => normalizePresetMode(mode, modeIndex)).filter((mode) => mode.options.length)
            : [];
    } else {
        normalized.options = Array.isArray(column?.options)
            ? column.options.map((option, optionIndex) => normalizePresetOption(option, optionIndex)).filter((option) => option.value)
            : [];
    }
    return normalized;
}

function normalizePreset(preset, fallbackId = "custom_preset") {
    const columns = Array.isArray(preset?.columns)
        ? preset.columns.map((column, index) => normalizePresetColumn(column, index)).filter((column) => {
            if (column.type === "switch") {
                return column.modes?.length > 0;
            }
            return column.options?.length > 0;
        })
        : [];
    const presetColumnIds = columns.map((column) => column.id);
    return {
        id: slugifyId(preset?.id || preset?.name || fallbackId, fallbackId),
        name: String(preset?.name || "Preset"),
        outputMode: preset?.outputMode === "omnivoice" ? "omnivoice" : "plain",
        columns,
        columnOrder: normalizeColumnOrder(preset?.columnOrder, presetColumnIds),
        lastState: normalizePresetStateSnapshot(preset?.lastState),
    };
}

function getPresetColumnIds(preset) {
    return Array.isArray(preset?.columns) ? preset.columns.map((column) => column.id) : [];
}

function getColumnById(preset, columnId) {
    return preset?.columns?.find((column) => column.id === columnId) || null;
}

function getModeById(column, modeId) {
    return column?.modes?.find((mode) => mode.id === modeId) || column?.modes?.[0] || null;
}

function isOmniVoicePreset(preset) {
    return preset?.outputMode === "omnivoice";
}

function parsePresetOptionLines(text) {
    return String(text || "")
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean)
        .flatMap((line) => {
            const parts = line.split("=");
            if (parts.length >= 2) {
                const label = parts.shift().trim();
                const value = parts.join("=").trim();
                return [normalizePresetOption({ label, value, title: value })];
            }
            if (line.includes(",")) {
                return line
                    .split(",")
                    .map((item) => item.trim())
                    .filter(Boolean)
                    .map((item) => normalizePresetOption({ label: item, value: item, title: item }));
            }
            return [normalizePresetOption({ label: line, value: line, title: line })];
        })
        .filter((option) => option.value);
}

function createTemporaryPresetFromText(text) {
    const tokens = String(text || "")
        .split(/[,\uFF0C]/)
        .map((token) => token.trim())
        .filter(Boolean);
    if (!tokens.length) {
        return null;
    }
    return normalizePreset({
        id: TEMP_TEXT_PRESET_ID,
        name: "Temporary Text",
        outputMode: "plain",
        columns: tokens.map((token, index) => ({
            id: `tag_${index + 1}`,
            title: "",
            type: "single",
            options: [{ id: `tag_${index + 1}_option`, label: token, value: token, title: token }],
        })),
    }, TEMP_TEXT_PRESET_ID);
}

function hasTrailingSeparator(text) {
    return /[,\uFF0C]\s*$/.test(String(text || ""));
}

function createEmptyBuilderState() {
    return {
        selectedPresetId: OMNIVOICE_PRESET_ID,
        output_language: "English",
        customPresets: [],
        temporaryPreset: null,
        workflowPresetSnapshot: null,
        presetStates: {},
    };
}

function getPresetStateBucket(state, presetId) {
    state.presetStates = state.presetStates || {};
    if (!state.presetStates[presetId]) {
        state.presetStates[presetId] = {
            values: {},
            switchModes: {},
        };
    }
    state.presetStates[presetId].values = state.presetStates[presetId].values || {};
    state.presetStates[presetId].switchModes = state.presetStates[presetId].switchModes || {};
    return state.presetStates[presetId];
}

function getAvailablePresets(state) {
    const presets = [OMNIVOICE_PRESET];
    for (const preset of state.customPresets || []) {
        presets.push(normalizePreset(preset, preset?.id || "custom_preset"));
    }
    if (state.workflowPresetSnapshot?.columns?.length) {
        const workflowPreset = normalizePreset(state.workflowPresetSnapshot, state.workflowPresetSnapshot?.id || "workflow_preset");
        if (!presets.some((preset) => preset.id === workflowPreset.id)) {
            presets.push(workflowPreset);
        }
    }
    if (state.temporaryPreset?.columns?.length) {
        presets.push(normalizePreset(state.temporaryPreset, TEMP_TEXT_PRESET_ID));
    }
    return presets;
}

function getActivePreset(state) {
    const presets = getAvailablePresets(state);
    return presets.find((preset) => preset.id === state.selectedPresetId) || presets[0];
}

function getPresetOutputLocale(preset, state) {
    return isOmniVoicePreset(preset) && normalizeWidgetValue(state.output_language) === "Chinese" ? "zh" : "en";
}

function getColumnTitleForLocale(column, locale) {
    return locale === "zh" ? (column.zhTitle || column.title || "") : (column.title || "");
}

function getModeTitleForLocale(mode, locale) {
    return locale === "zh" ? (mode.zhTitle || mode.title || "") : (mode.title || "");
}

function getOptionLabelForLocale(option, locale) {
    return locale === "zh" ? (option.zhLabel || option.label || option.value) : (option.label || option.value);
}

function getOptionTitleForLocale(option, locale) {
    return locale === "zh" ? (option.zhTitle || option.zhLabel || option.title || option.value) : (option.title || option.value);
}

function buildLegacyOmniVoicePresetState(node) {
    const accent = normalizeWidgetValue(findWidgetByName(node, "accent")?.value);
    const dialect = normalizeWidgetValue(findWidgetByName(node, "dialect")?.value);
    return {
        values: {
            gender: normalizeWidgetValue(findWidgetByName(node, "gender")?.value),
            age: normalizeWidgetValue(findWidgetByName(node, "age")?.value),
            pitch: normalizeWidgetValue(findWidgetByName(node, "pitch")?.value),
            style: normalizeWidgetValue(findWidgetByName(node, "style")?.value),
            language: accent || dialect,
        },
        switchModes: {
            language: dialect ? "dialect" : "accent",
        },
    };
}

function getCachedPresetLibrary() {
    return Array.isArray(window.__ttsAudioSuiteOmniVoicePresetLibrary)
        ? cloneJson(window.__ttsAudioSuiteOmniVoicePresetLibrary)
        : [];
}

function setCachedPresetLibrary(presets) {
    window.__ttsAudioSuiteOmniVoicePresetLibrary = cloneJson(presets || []);
}

function getCachedBuiltinPresetStates() {
    return window.__ttsAudioSuiteOmniVoiceBuiltinPresetStates
        && typeof window.__ttsAudioSuiteOmniVoiceBuiltinPresetStates === "object"
        ? cloneJson(window.__ttsAudioSuiteOmniVoiceBuiltinPresetStates)
        : {};
}

function setCachedBuiltinPresetStates(states) {
    window.__ttsAudioSuiteOmniVoiceBuiltinPresetStates = cloneJson(states || {});
}

function getCachedBuiltinPresetLayouts() {
    return window.__ttsAudioSuiteOmniVoiceBuiltinPresetLayouts
        && typeof window.__ttsAudioSuiteOmniVoiceBuiltinPresetLayouts === "object"
        ? cloneJson(window.__ttsAudioSuiteOmniVoiceBuiltinPresetLayouts)
        : {};
}

function setCachedBuiltinPresetLayouts(layouts) {
    window.__ttsAudioSuiteOmniVoiceBuiltinPresetLayouts = cloneJson(layouts || {});
}

function getStoredColumnOrderForPreset(node, preset) {
    const presetColumnIds = getPresetColumnIds(preset);
    const workflowOrder = node?.properties?.omnivoiceInstructionLayout?.layouts?.[preset.id]?.columnOrder;
    if (Array.isArray(workflowOrder) && workflowOrder.length) {
        return normalizeColumnOrder(workflowOrder, presetColumnIds);
    }
    if (preset.id === OMNIVOICE_PRESET_ID) {
        const builtinOrder = node?.__omnivoiceInstructionBuiltinPresetLayouts?.[preset.id]?.columnOrder;
        return normalizeColumnOrder(builtinOrder, presetColumnIds);
    }
    return normalizeColumnOrder(preset?.columnOrder, presetColumnIds);
}

function buildPresetLibraryPayload(state, node) {
    return (state.customPresets || []).map((preset, index) => {
        const normalized = normalizePreset(preset, preset?.id || `custom_preset_${index + 1}`);
        return {
            ...normalized,
            columnOrder: getStoredColumnOrderForPreset(node, normalized),
            lastState: normalizePresetStateSnapshot(state?.presetStates?.[normalized.id] || normalized.lastState),
        };
    });
}

function buildBuiltinPresetStatePayload(state) {
    const builtinStates = {};
    for (const presetId of [OMNIVOICE_PRESET_ID]) {
        builtinStates[presetId] = normalizePresetStateSnapshot(state?.presetStates?.[presetId]);
    }
    return builtinStates;
}

function buildBuiltinPresetLayoutPayload(node) {
    const builtinLayouts = {};
    const omnivoiceOrder = getStoredColumnOrderForPreset(node, OMNIVOICE_PRESET);
    builtinLayouts[OMNIVOICE_PRESET_ID] = {
        columnOrder: [...omnivoiceOrder],
    };
    return builtinLayouts;
}

async function fetchPresetLibraryFromBackend() {
    const response = await fetch(OMNIVOICE_PRESET_LIBRARY_URL, {
        method: "GET",
        cache: "no-store",
    });
    if (!response.ok) {
        throw new Error(`Preset library fetch failed: ${response.status}`);
    }
    const data = await response.json();
    const presets = Array.isArray(data?.presets) ? data.presets : [];
    const builtinStates = data?.builtinStates && typeof data.builtinStates === "object" ? data.builtinStates : {};
    const builtinLayouts = data?.builtinLayouts && typeof data.builtinLayouts === "object" ? data.builtinLayouts : {};
    setCachedPresetLibrary(presets);
    setCachedBuiltinPresetStates(builtinStates);
    setCachedBuiltinPresetLayouts(builtinLayouts);
    return { presets, builtinStates, builtinLayouts };
}

async function savePresetLibraryToBackend(presets, builtinStates = {}, builtinLayouts = {}) {
    const response = await fetch(OMNIVOICE_PRESET_LIBRARY_URL, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ presets, builtinStates, builtinLayouts }),
    });
    if (!response.ok) {
        throw new Error(`Preset library save failed: ${response.status}`);
    }
    setCachedPresetLibrary(presets);
    setCachedBuiltinPresetStates(builtinStates);
    setCachedBuiltinPresetLayouts(builtinLayouts);
    return response.json();
}

function getWorkflowPresetSnapshotFromRaw(rawState) {
    const directSnapshot = rawState?.workflowPresetSnapshot?.columns?.length
        ? normalizePreset(rawState.workflowPresetSnapshot, rawState.workflowPresetSnapshot?.id || "workflow_preset")
        : null;
    if (directSnapshot) {
        return directSnapshot;
    }
    const selectedPresetId = rawState?.selectedPresetId;
    const legacyCustomPresets = Array.isArray(rawState?.customPresets) ? rawState.customPresets : [];
    const matchedLegacyPreset = legacyCustomPresets.find((preset) => slugifyId(preset?.id || preset?.name || "", "") === selectedPresetId);
    return matchedLegacyPreset?.columns?.length
        ? normalizePreset(matchedLegacyPreset, matchedLegacyPreset?.id || "workflow_preset")
        : null;
}

function buildWorkflowSafePresetState(state) {
    const activePreset = getActivePreset(state);
    return {
        selectedPresetId: state.selectedPresetId,
        output_language: state.output_language,
        temporaryPreset: state.temporaryPreset?.columns?.length ? cloneJson(state.temporaryPreset) : null,
        workflowPresetSnapshot: (
            activePreset
            && activePreset.id !== OMNIVOICE_PRESET_ID
            && activePreset.id !== TEMP_TEXT_PRESET_ID
            && activePreset.columns?.length
        ) ? cloneJson(activePreset) : null,
        presetStates: cloneJson(state.presetStates || {}),
    };
}

function syncStateFromWidgets(node) {
    const rawWorkflowState = cloneJson(node.properties?.omnivoiceInstructionPresetState || {});
    const state = {
        ...createEmptyBuilderState(),
        ...rawWorkflowState,
    };
    state.customPresets = Array.isArray(node.__omnivoiceInstructionLibraryPresets)
        ? node.__omnivoiceInstructionLibraryPresets.map((preset, index) => normalizePreset(preset, `custom_preset_${index + 1}`))
        : [];
    state.workflowPresetSnapshot = getWorkflowPresetSnapshotFromRaw(rawWorkflowState);
    state.temporaryPreset = state.temporaryPreset?.columns?.length
        ? normalizePreset(state.temporaryPreset, TEMP_TEXT_PRESET_ID)
        : null;
    state.presetStates = state.presetStates && typeof state.presetStates === "object" ? state.presetStates : {};
    const builtinPresetStates = node.__omnivoiceInstructionBuiltinPresetStates || {};
    for (const [presetId, snapshot] of Object.entries(builtinPresetStates)) {
        const normalizedSnapshot = normalizePresetStateSnapshot(snapshot);
        const hasWorkflowState = hasMeaningfulPresetState(state.presetStates[presetId]);
        const hasLibraryState = Object.keys(normalizedSnapshot.values).length || Object.keys(normalizedSnapshot.switchModes).length;
        if (!hasWorkflowState && hasLibraryState) {
            state.presetStates[presetId] = normalizedSnapshot;
        }
    }
    for (const preset of state.customPresets || []) {
        const snapshot = normalizePresetStateSnapshot(preset?.lastState);
        const hasWorkflowState = hasMeaningfulPresetState(state.presetStates[preset.id]);
        const hasLibraryState = Object.keys(snapshot.values).length || Object.keys(snapshot.switchModes).length;
        if (!hasWorkflowState && hasLibraryState) {
            state.presetStates[preset.id] = snapshot;
        }
    }
    state.output_language = normalizeWidgetValue(state.output_language || findWidgetByName(node, "output_language")?.value) || "English";
    if (!state.selectedPresetId) {
        state.selectedPresetId = state.temporaryPreset?.id || OMNIVOICE_PRESET_ID;
    }
    const omniBucket = getPresetStateBucket(state, OMNIVOICE_PRESET_ID);
    if (!Object.keys(omniBucket.values || {}).length) {
        state.presetStates[OMNIVOICE_PRESET_ID] = buildLegacyOmniVoicePresetState(node);
    }
    const activePreset = getActivePreset(state);
    getPresetStateBucket(state, activePreset.id);
    return state;
}

function buildPreviewFromState(state, preset = getActivePreset(state), columnOrder = getPresetColumnIds(preset)) {
    const locale = getPresetOutputLocale(preset, state);
    const bucket = getPresetStateBucket(state, preset.id);
    const ordered = [];

    for (const columnId of columnOrder) {
        const column = getColumnById(preset, columnId);
        if (!column) {
            continue;
        }
        if (column.type === "switch") {
            const mode = getModeById(column, bucket.switchModes[columnId]);
            const value = normalizeWidgetValue(bucket.values[columnId]);
            const option = mode?.options?.find((item) => item.value === value);
            if (option?.value) {
                ordered.push(option.value);
            }
            continue;
        }
        const value = normalizeWidgetValue(bucket.values[columnId]);
        if (value) {
            ordered.push(value);
        }
    }

    if (!ordered.length) {
        return "";
    }
    if (locale === "zh" && isOmniVoicePreset(preset)) {
        return ordered.map((item) => EN_TO_ZH[item] || item).join("，");
    }
    return ordered.join(", ");
}

function setPresetColumnValue(state, preset, columnId, value, toggle = true, modeId = "") {
    const bucket = getPresetStateBucket(state, preset.id);
    const column = getColumnById(preset, columnId);
    if (!column) {
        return state;
    }
    const currentValue = normalizeWidgetValue(bucket.values[columnId]);
    if (column.type === "switch") {
        const nextMode = modeId || bucket.switchModes[columnId] || column.modes?.[0]?.id || "";
        bucket.switchModes[columnId] = nextMode;
    }
    bucket.values[columnId] = toggle && currentValue === value ? "" : normalizeWidgetValue(value);
    return state;
}

function setPresetMode(state, preset, columnId, modeId) {
    const bucket = getPresetStateBucket(state, preset.id);
    bucket.switchModes[columnId] = modeId;
    bucket.values[columnId] = "";
    return state;
}

function createPresetFromDraft(draft, fallbackId = "custom_preset") {
    return normalizePreset({
        id: draft.id || slugifyId(draft.name || fallbackId, fallbackId),
        name: draft.name || "Preset",
        outputMode: draft.outputMode || "plain",
        columns: draft.columns || [],
    }, fallbackId);
}

function createPresetDraftFromPreset(preset, preserveId = false) {
    const draft = cloneJson(preset) || {};
    draft.name = draft.name || "Preset";
    draft.outputMode = preset?.outputMode === "omnivoice" ? "omnivoice" : "plain";
    draft.id = preserveId ? draft.id : "";
    draft.columns = Array.isArray(draft.columns) ? draft.columns : [];
    return draft;
}

function createEmptySingleColumn(index = 0) {
    return {
        id: `column_${index + 1}`,
        title: "",
        type: "single",
        options: [{ id: "option_1", label: "", value: "", title: "" }],
    };
}

function createEmptySwitchColumn(index = 0) {
    return {
        id: `switch_${index + 1}`,
        title: "",
        type: "switch",
        modes: [
            {
                id: "mode_1",
                title: "Mode 1",
                options: [{ id: "option_1", label: "", value: "", title: "" }],
            },
            {
                id: "mode_2",
                title: "Mode 2",
                options: [{ id: "option_1", label: "", value: "", title: "" }],
            },
        ],
    };
}

function seedBucketFromPreset(state, preset) {
    const bucket = getPresetStateBucket(state, preset.id);
    bucket.values = {};
    bucket.switchModes = {};
    for (const column of preset.columns || []) {
        if (column.type === "switch") {
            const mode = column.modes?.[0];
            if (mode?.id) {
                bucket.switchModes[column.id] = mode.id;
                if (mode.options?.[0]?.value) {
                    bucket.values[column.id] = mode.options[0].value;
                }
            }
            continue;
        }
        if (column.options?.[0]?.value) {
            bucket.values[column.id] = column.options[0].value;
        }
    }
}

function applyPreviewTextToState(state, text) {
    const temporaryPreset = createTemporaryPresetFromText(text);
    if (!temporaryPreset) {
        state.temporaryPreset = null;
        if (state.selectedPresetId === TEMP_TEXT_PRESET_ID) {
            state.selectedPresetId = OMNIVOICE_PRESET_ID;
        }
        delete state.presetStates[TEMP_TEXT_PRESET_ID];
        return state;
    }
    state.temporaryPreset = temporaryPreset;
    state.selectedPresetId = temporaryPreset.id;
    seedBucketFromPreset(state, temporaryPreset);
    return state;
}

function persistPresetState(node, state) {
    node.properties = node.properties || {};
    node.properties.omnivoiceInstructionPresetState = buildWorkflowSafePresetState(state);
}

function cachePresetLibraryOnNode(node, presets, builtinStates = null, builtinLayouts = null) {
    const normalizedPresets = cloneJson(presets || []);
    node.__omnivoiceInstructionLibraryPresets = normalizedPresets;
    node.__omnivoiceInstructionBuiltinPresetStates = cloneJson(builtinStates || getCachedBuiltinPresetStates());
    node.__omnivoiceInstructionBuiltinPresetLayouts = cloneJson(builtinLayouts || getCachedBuiltinPresetLayouts());
    node.__omnivoiceInstructionLastLibraryPayload = JSON.stringify(normalizedPresets);
    node.__omnivoiceInstructionLastBuiltinStatesPayload = JSON.stringify(node.__omnivoiceInstructionBuiltinPresetStates || {});
    node.__omnivoiceInstructionLastBuiltinLayoutsPayload = JSON.stringify(node.__omnivoiceInstructionBuiltinPresetLayouts || {});
    setCachedPresetLibrary(normalizedPresets);
    setCachedBuiltinPresetStates(node.__omnivoiceInstructionBuiltinPresetStates);
    setCachedBuiltinPresetLayouts(node.__omnivoiceInstructionBuiltinPresetLayouts);
}

async function persistPresetLibraryNow(node, state) {
    node.__omnivoiceInstructionLibraryReady = true;
    const payload = buildPresetLibraryPayload(state, node);
    const builtinStates = buildBuiltinPresetStatePayload(state);
    const builtinLayouts = buildBuiltinPresetLayoutPayload(node);
    const serialized = JSON.stringify(payload);
    const builtinSerialized = JSON.stringify(builtinStates);
    const builtinLayoutSerialized = JSON.stringify(builtinLayouts);
    if (
        serialized === (node.__omnivoiceInstructionLastLibraryPayload || "")
        && builtinSerialized === (node.__omnivoiceInstructionLastBuiltinStatesPayload || "")
        && builtinLayoutSerialized === (node.__omnivoiceInstructionLastBuiltinLayoutsPayload || "")
    ) {
        cachePresetLibraryOnNode(node, payload, builtinStates, builtinLayouts);
        return;
    }
    await savePresetLibraryToBackend(payload, builtinStates, builtinLayouts);
    cachePresetLibraryOnNode(node, payload, builtinStates, builtinLayouts);
}

function schedulePresetLibraryPersist(node, state) {
    if (!node.__omnivoiceInstructionLibraryReady) {
        return;
    }
    clearTimeout(node.__omnivoiceInstructionLibraryPersistTimer);
    node.__omnivoiceInstructionLibraryPersistTimer = window.setTimeout(async () => {
        try {
            const payload = buildPresetLibraryPayload(state, node);
            const builtinStates = buildBuiltinPresetStatePayload(state);
            const builtinLayouts = buildBuiltinPresetLayoutPayload(node);
            const serialized = JSON.stringify(payload);
            const builtinSerialized = JSON.stringify(builtinStates);
            const builtinLayoutSerialized = JSON.stringify(builtinLayouts);
            if (
                serialized === (node.__omnivoiceInstructionLastLibraryPayload || "")
                && builtinSerialized === (node.__omnivoiceInstructionLastBuiltinStatesPayload || "")
                && builtinLayoutSerialized === (node.__omnivoiceInstructionLastBuiltinLayoutsPayload || "")
            ) {
                cachePresetLibraryOnNode(node, payload, builtinStates, builtinLayouts);
                return;
            }
            await savePresetLibraryToBackend(payload, builtinStates, builtinLayouts);
            cachePresetLibraryOnNode(node, payload, builtinStates, builtinLayouts);
        } catch (error) {
            console.warn("Could not persist OmniVoice preset library state:", error);
        }
    }, 220);
}

async function loadPresetLibraryIntoNode(node) {
    try {
        const { presets, builtinStates, builtinLayouts } = await fetchPresetLibraryFromBackend();
        cachePresetLibraryOnNode(node, Array.isArray(presets) ? presets : [], builtinStates, builtinLayouts);
    } catch (error) {
        console.warn("Could not load OmniVoice preset library:", error);
        cachePresetLibraryOnNode(node, getCachedPresetLibrary(), getCachedBuiltinPresetStates(), getCachedBuiltinPresetLayouts());
    } finally {
        node.__omnivoiceInstructionLibraryReady = true;
    }
}

function writeBuilderStateToWidgets(node, state, ui = null) {
    const preset = getActivePreset(state);
    const bucket = getPresetStateBucket(state, preset.id);
    const columnOrder = ui?.columnOrder || getPresetColumnIds(preset);
    setWidgetValue(findWidgetByName(node, "output_language"), state.output_language || "English");
    setWidgetValue(findWidgetByName(node, "instruct_text"), buildPreviewFromState(state, preset, columnOrder));

    const syncOmniVoice = isOmniVoicePreset(preset) && preset.id === OMNIVOICE_PRESET_ID;
    const languageMode = bucket.switchModes.language || "accent";
    const languageValue = normalizeWidgetValue(bucket.values.language);
    setWidgetValue(findWidgetByName(node, "gender"), syncOmniVoice ? bucket.values.gender : "");
    setWidgetValue(findWidgetByName(node, "age"), syncOmniVoice ? bucket.values.age : "");
    setWidgetValue(findWidgetByName(node, "pitch"), syncOmniVoice ? bucket.values.pitch : "");
    setWidgetValue(findWidgetByName(node, "style"), syncOmniVoice ? bucket.values.style : "");
    setWidgetValue(findWidgetByName(node, "accent"), syncOmniVoice && languageMode === "accent" ? languageValue : "");
    setWidgetValue(findWidgetByName(node, "dialect"), syncOmniVoice && languageMode === "dialect" ? languageValue : "");
}

function applyStateToWidgets(node, state, ui = null) {
    persistPresetState(node, state);
    writeBuilderStateToWidgets(node, state, ui);
    schedulePresetLibraryPersist(node, state);
}

function ensureStyles(panel) {
    if (panel.__omnivoiceInstructionStylesInjected) {
        return;
    }
    panel.__omnivoiceInstructionStylesInjected = true;

    const style = document.createElement("style");
    style.textContent = `
        .omnivoice-instruction-builder-panel {
            width: 100%;
            height: 100%;
            color: #e3e2e6;
            font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            box-sizing: border-box;
            overflow: visible;
            padding: 2px 2px 0 2px;
        }
        .ovib-shell {
            display: flex;
            flex-direction: column;
            height: 100%;
            border: 1px solid rgba(152, 203, 255, 0.12);
            border-radius: 10px;
            overflow: hidden;
            background: #17191d;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.32);
        }
        .ovib-topline {
            height: 2px;
            background: linear-gradient(90deg, rgba(152, 203, 255, 0.95) 0%, rgba(0, 163, 255, 0.86) 60%, rgba(111, 251, 190, 0.72) 100%);
        }
        .ovib-header {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 9px 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            background: rgba(23, 25, 29, 0.96);
        }
        .ovib-header-preview {
            min-width: 0;
            display: flex;
            align-items: center;
            gap: 7px;
            flex: 1 1 auto;
            overflow: hidden;
        }
        .ovib-header-controls {
            display: flex;
            align-items: center;
            gap: 8px;
            flex: 0 0 auto;
        }
        .ovib-preset-select {
            min-height: 24px;
            max-width: 138px;
            padding: 4px 8px;
            border-radius: 7px;
            border: 1px solid rgba(121, 134, 155, 0.34);
            background: rgba(34, 37, 43, 0.92);
            color: #d9e8ff;
            font-size: 8px;
            font-weight: 700;
            letter-spacing: 0.03em;
            outline: none;
        }
        .ovib-icon-button {
            min-height: 24px;
            padding: 4px 8px;
            border-radius: 7px;
            border: 1px solid rgba(121, 134, 155, 0.34);
            background: rgba(34, 37, 43, 0.92);
            color: #9fb0c6;
            font-size: 8px;
            font-weight: 700;
            letter-spacing: 0.04em;
            cursor: pointer;
            transition: border-color 0.18s ease, background 0.18s ease, color 0.18s ease, box-shadow 0.18s ease;
        }
        .ovib-icon-button:hover {
            border-color: rgba(152, 203, 255, 0.42);
            color: #d9e8ff;
        }
        .ovib-locale-switch {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 6px;
            width: 84px;
            flex: 0 0 auto;
        }
        .ovib-locale-chip {
            min-height: 24px;
            padding: 4px 6px;
            border-radius: 7px;
            border: 1px solid rgba(121, 134, 155, 0.34);
            background: rgba(34, 37, 43, 0.92);
            color: #9fb0c6;
            font-size: 8px;
            font-weight: 700;
            letter-spacing: 0.04em;
            cursor: pointer;
            transition: border-color 0.18s ease, background 0.18s ease, color 0.18s ease, box-shadow 0.18s ease;
        }
        .ovib-locale-chip:hover {
            border-color: rgba(152, 203, 255, 0.42);
            color: #d9e8ff;
        }
        .ovib-locale-chip.is-active {
            border-color: rgba(152, 203, 255, 0.82);
            background: linear-gradient(180deg, rgba(40, 61, 84, 0.8) 0%, rgba(28, 43, 62, 0.78) 100%);
            color: #9ed0ff;
            box-shadow: 0 0 14px rgba(64, 167, 255, 0.18);
        }
        .ovib-body {
            position: relative;
            flex: 1 1 auto;
            min-height: 312px;
            padding: 14px 12px 10px 12px;
            background:
                linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px),
                radial-gradient(circle at top left, rgba(152, 203, 255, 0.08) 0%, transparent 34%),
                #17191d;
            background-size: 18px 18px, 18px 18px, auto, auto;
        }
        .ovib-svg {
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 3;
        }
        .ovib-grid {
            position: relative;
            z-index: 1;
            display: grid;
            grid-auto-columns: minmax(0, 1fr);
            grid-auto-flow: column;
            gap: 12px;
            align-items: start;
            height: 100%;
        }
        .ovib-column {
            display: flex;
            flex-direction: column;
            gap: 7px;
            min-width: 0;
            position: relative;
            transition: transform 0.18s ease, z-index 0.18s ease;
            will-change: transform;
        }
        .ovib-column.is-draggable .ovib-chip.is-active {
            cursor: grab;
        }
        .ovib-column.is-dragging .ovib-chip.is-active {
            cursor: grabbing;
        }
        .ovib-column.is-dragging {
            z-index: 5;
            transition: none;
        }
        .ovib-column-title,
        .ovib-subtitle {
            margin: 0;
            text-align: center;
            color: #90a0b7;
            font-size: 9px;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .ovib-column-title.is-empty,
        .ovib-subtitle.is-empty {
            min-height: 0;
            margin: 0;
            opacity: 0;
        }
        .ovib-subtitle {
            font-size: 8px;
            color: #79869b;
            margin-bottom: 2px;
        }
        .ovib-chip-list {
            display: flex;
            flex-direction: column;
            gap: 7px;
        }
        .ovib-chip {
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 30px;
            padding: 7px 8px;
            margin: 0 auto;
            border-radius: 8px;
            border: 1px solid rgba(121, 134, 155, 0.38);
            background: rgba(41, 43, 49, 0.94);
            color: #e6edf8;
            font-size: 9px;
            font-weight: 600;
            text-align: center;
            line-height: 1.2;
            letter-spacing: 0.01em;
            user-select: none;
            box-sizing: border-box;
            cursor: pointer;
            text-rendering: geometricPrecision;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            transition: border-color 0.18s ease, background 0.18s ease, color 0.18s ease, box-shadow 0.18s ease, opacity 0.18s ease, transform 0.18s ease;
        }
        .ovib-column:not(.ovib-language-column) .ovib-chip {
            width: min(100%, 84px);
            padding-left: 6px;
            padding-right: 6px;
        }
        .ovib-language-column .ovib-chip {
            width: min(100%, 108px);
        }
        .ovib-chip:hover:not(.is-disabled) {
            border-color: rgba(152, 203, 255, 0.44);
            background: rgba(50, 54, 62, 0.96);
            color: #ffffff;
            transform: translateY(-1px);
        }
        .ovib-chip.is-active {
            border-color: rgba(152, 203, 255, 0.98);
            background: linear-gradient(180deg, rgba(47, 68, 92, 0.78) 0%, rgba(35, 52, 74, 0.68) 100%);
            color: #b7ddff;
            box-shadow: 0 0 0 1px rgba(152, 203, 255, 0.16) inset, 0 0 12px rgba(64, 167, 255, 0.2);
        }
        .ovib-chip.is-disabled {
            opacity: 0.2;
            pointer-events: none;
        }
        .ovib-chip::before,
        .ovib-chip::after {
            content: "";
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            width: 3px;
            height: 3px;
            border-radius: 50%;
            background: rgba(111, 126, 147, 0.38);
            opacity: 0;
            transition: opacity 0.18s ease, background 0.18s ease, box-shadow 0.18s ease;
        }
        .ovib-chip::before {
            left: -3px;
        }
        .ovib-chip::after {
            right: -3px;
        }
        .ovib-chip.is-active::before,
        .ovib-chip.is-active::after {
            opacity: 1;
            background: #98cbff;
            box-shadow: 0 0 8px rgba(152, 203, 255, 0.7);
        }
        .ovib-column:first-child .ovib-chip::before {
            display: none;
        }
        .ovib-language-column .ovib-chip::after {
            display: none;
        }
        .ovib-language-stack {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .ovib-mode-switch {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 6px;
            margin-bottom: 2px;
        }
        .ovib-mode-chip {
            min-height: 24px;
            padding: 5px 6px;
            border-radius: 7px;
            border: 1px solid rgba(121, 134, 155, 0.34);
            background: rgba(34, 37, 43, 0.92);
            color: #9fb0c6;
            font-size: 8px;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            cursor: pointer;
            transition: border-color 0.18s ease, background 0.18s ease, color 0.18s ease, box-shadow 0.18s ease;
        }
        .ovib-mode-chip:hover {
            border-color: rgba(152, 203, 255, 0.42);
            color: #d9e8ff;
        }
        .ovib-mode-chip.is-active {
            border-color: rgba(152, 203, 255, 0.82);
            background: linear-gradient(180deg, rgba(40, 61, 84, 0.8) 0%, rgba(28, 43, 62, 0.78) 100%);
            color: #9ed0ff;
            box-shadow: 0 0 14px rgba(64, 167, 255, 0.18);
        }
        .ovib-language-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
            transition: opacity 0.2s ease;
        }
        .ovib-language-group .ovib-chip-list {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 7px;
        }
        .ovib-language-group.is-hidden {
            display: none;
        }
        .ovib-preview-label {
            color: #8492a7;
            font-size: 8px;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            white-space: nowrap;
            flex: 0 0 auto;
        }
        .ovib-preview-input {
            width: 100%;
            min-width: 0;
            border: none;
            outline: none;
            background: transparent;
            color: #98cbff;
            font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
            font-size: 9px;
            line-height: 1.35;
            padding: 0;
        }
        .ovib-preview-input::placeholder {
            color: #64748b;
        }
        .ovib-path {
            fill: none;
            stroke: url(#ovibPathGradient);
            stroke-linecap: round;
            stroke-linejoin: round;
            filter: drop-shadow(0 0 5px rgba(152, 203, 255, 0.58));
            opacity: 0.92;
        }
        .ovib-modal-backdrop {
            position: absolute;
            inset: 0;
            z-index: 20;
            display: none;
            align-items: center;
            justify-content: center;
            padding: 16px;
            background: rgba(5, 7, 10, 0.74);
            backdrop-filter: blur(8px);
        }
        .ovib-modal-backdrop.is-open {
            display: flex;
        }
        .ovib-modal {
            width: min(860px, 100%);
            max-height: min(84vh, 720px);
            overflow: auto;
            border-radius: 12px;
            border: 1px solid rgba(121, 134, 155, 0.34);
            background: #17191d;
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.42);
            padding: 14px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .ovib-modal-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
        }
        .ovib-modal-title {
            color: #f3f8ff;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.03em;
        }
        .ovib-modal-actions {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }
        .ovib-form-label {
            color: #9fb0c6;
            font-size: 9px;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }
        .ovib-form-hint {
            color: #7b8aa1;
            font-size: 8px;
            line-height: 1.4;
        }
        .ovib-form-input,
        .ovib-form-textarea,
        .ovib-form-select {
            width: 100%;
            border-radius: 8px;
            border: 1px solid rgba(121, 134, 155, 0.34);
            background: rgba(34, 37, 43, 0.92);
            color: #d9e8ff;
            font-size: 10px;
            padding: 8px 10px;
            box-sizing: border-box;
            outline: none;
        }
        .ovib-form-textarea {
            min-height: 90px;
            resize: vertical;
            font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
            line-height: 1.45;
        }
        .ovib-columns-editor {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 10px;
            align-items: start;
        }
        .ovib-column-card,
        .ovib-mode-card {
            border-radius: 10px;
            border: 1px solid rgba(121, 134, 155, 0.22);
            background: rgba(24, 27, 32, 0.94);
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .ovib-mode-card {
            background: rgba(27, 31, 36, 0.96);
        }
        .ovib-column-card-header,
        .ovib-mode-card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
        }
        .ovib-column-card-title,
        .ovib-mode-card-title {
            color: #f3f8ff;
            font-size: 10px;
            font-weight: 700;
        }
        .ovib-two-col {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        .ovib-card-topline {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .ovib-card-topline .ovib-form-select {
            flex: 0 0 110px;
            width: 110px;
        }
        .ovib-card-topline .ovib-form-input {
            flex: 1 1 auto;
        }
        .ovib-stack {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        .ovib-mode-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
    `;
    panel.appendChild(style);
}

function createChip(option, columnId, modeId = "") {
    const button = createEl("button", "ovib-chip", option.label);
    button.type = "button";
    button.dataset.columnId = columnId;
    button.dataset.modeId = modeId;
    button.dataset.value = option.value;
    button.dataset.labelEn = option.label;
    button.dataset.labelZh = option.zhLabel || ZH_UI_LABELS[option.value] || option.label;
    button.dataset.titleEn = option.title || option.value;
    button.dataset.titleZh = option.zhTitle || option.zhLabel || ZH_UI_LABELS[option.value] || option.title || option.value;
    return button;
}

function createPanelDom() {
    const panel = createEl("div", "omnivoice-instruction-builder-panel");
    ensureStyles(panel);

    const shell = createEl("div", "ovib-shell");
    panel.appendChild(shell);
    shell.appendChild(createEl("div", "ovib-topline"));

    const header = createEl("div", "ovib-header");
    const headerPreview = createEl("div", "ovib-header-preview");
    const previewLabel = createEl("span", "ovib-preview-label", UI_TEXT.en.previewLabel);
    const previewInput = createEl("input", "ovib-preview-input");
    previewInput.type = "text";
    previewInput.placeholder = UI_TEXT.en.emptyPreview;
    headerPreview.append(previewLabel, previewInput);
    header.appendChild(headerPreview);

    const headerControls = createEl("div", "ovib-header-controls");
    const presetSelect = createEl("select", "ovib-preset-select");
    const editPresetButton = createEl("button", "ovib-icon-button", UI_TEXT.en.editPreset);
    editPresetButton.type = "button";
    headerControls.append(presetSelect, editPresetButton);
    header.appendChild(headerControls);

    const localeSwitch = createEl("div", "ovib-locale-switch");
    const localeEnglishButton = createEl("button", "ovib-locale-chip is-active", UI_TEXT.en.localeEn);
    localeEnglishButton.type = "button";
    const localeChineseButton = createEl("button", "ovib-locale-chip", UI_TEXT.en.localeZh);
    localeChineseButton.type = "button";
    localeSwitch.append(localeEnglishButton, localeChineseButton);
    header.appendChild(localeSwitch);
    shell.appendChild(header);

    const body = createEl("div", "ovib-body");
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.classList.add("ovib-svg");
    svg.innerHTML = `
        <defs>
            <linearGradient id="ovibPathGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#98cbff"></stop>
                <stop offset="58%" stop-color="#00a3ff"></stop>
                <stop offset="100%" stop-color="#6ffbbe"></stop>
            </linearGradient>
        </defs>
        <path class="ovib-path" d=""></path>
    `;
    body.appendChild(svg);

    const grid = createEl("div", "ovib-grid");
    body.appendChild(grid);
    shell.appendChild(body);

    const modalBackdrop = createEl("div", "ovib-modal-backdrop");
    const modal = createEl("div", "ovib-modal");
    const modalHeader = createEl("div", "ovib-modal-header");
    const modalTitle = createEl("div", "ovib-modal-title", UI_TEXT.en.editPresetTitle);
    const modalHeaderActions = createEl("div", "ovib-modal-actions");
    const modalDeleteButton = createEl("button", "ovib-icon-button", UI_TEXT.en.deletePreset);
    modalDeleteButton.type = "button";
    const modalSaveButton = createEl("button", "ovib-icon-button", UI_TEXT.en.save);
    modalSaveButton.type = "button";
    const modalCancelButton = createEl("button", "ovib-icon-button", UI_TEXT.en.cancel);
    modalCancelButton.type = "button";
    modalHeaderActions.append(modalDeleteButton, modalSaveButton, modalCancelButton);
    modalHeader.append(modalTitle, modalHeaderActions);
    modal.appendChild(modalHeader);

    const presetNameStack = createEl("div", "ovib-stack");
    const presetNameLabel = createEl("label", "ovib-form-label", UI_TEXT.en.presetName);
    const presetNameInput = createEl("input", "ovib-form-input");
    presetNameInput.type = "text";
    presetNameStack.append(presetNameLabel, presetNameInput);
    modal.appendChild(presetNameStack);

    const columnsEditor = createEl("div", "ovib-columns-editor");
    modal.appendChild(columnsEditor);

    const modalFooter = createEl("div", "ovib-modal-actions");
    const addSingleColumnButton = createEl("button", "ovib-icon-button", UI_TEXT.en.addColumn);
    addSingleColumnButton.type = "button";
    const addSwitchColumnButton = createEl("button", "ovib-icon-button", UI_TEXT.en.addSwitch);
    addSwitchColumnButton.type = "button";
    modalFooter.append(addSingleColumnButton, addSwitchColumnButton);
    modal.appendChild(modalFooter);
    modalBackdrop.appendChild(modal);
    panel.appendChild(modalBackdrop);

    return {
        panel,
        body,
        grid,
        svg,
        path: svg.querySelector(".ovib-path"),
        previewLabel,
        previewInput,
        presetSelect,
        editPresetButton,
        localeEnglishButton,
        localeChineseButton,
        localeSwitch,
        modalBackdrop,
        modalTitle,
        presetNameLabel,
        presetNameInput,
        columnsEditor,
        modalDeleteButton,
        addSingleColumnButton,
        addSwitchColumnButton,
        modalSaveButton,
        modalCancelButton,
        chipButtons: new Map(),
        columns: new Map(),
        columnTitles: new Map(),
        switchGroups: new Map(),
        switchModeButtons: new Map(),
        columnDefsById: new Map(),
        renderedColumnIds: [],
        renderedPresetId: "",
        renderedPresetSignature: "",
        columnOrder: [],
        columnOffsets: {},
        previewCommitTimer: null,
        previewInputDirty: false,
        dragState: null,
        suppressNextClick: false,
        pendingClickToken: null,
        modalDraft: null,
        modalLocale: "en",
        modalModeSelection: {},
    };
}

function getCanvasViewportRect() {
    const canvasEl = window.app?.canvas?.canvas || window.app?.canvas?.canvasEl;
    if (canvasEl?.getBoundingClientRect) {
        return canvasEl.getBoundingClientRect();
    }
    return null;
}

function getCanvasScale() {
    const scale = Number(
        window.app?.canvas?.ds?.scale
        || window.app?.graph?.canvas?.ds?.scale
        || 1
    );
    return Number.isFinite(scale) && scale > 0 ? scale : 1;
}

function getMovementBoundsRect(ui) {
    const bodyRect = ui.body.getBoundingClientRect();
    const canvasRect = getCanvasViewportRect();
    if (!canvasRect) {
        return bodyRect;
    }
    const left = Math.max(bodyRect.left, canvasRect.left);
    const top = Math.max(bodyRect.top, canvasRect.top);
    const right = Math.min(bodyRect.right, canvasRect.right);
    const bottom = Math.min(bodyRect.bottom, canvasRect.bottom);
    if (right <= left || bottom <= top) {
        return bodyRect;
    }
    return { left, top, right, bottom };
}

function getColumnOffset(ui, columnId) {
    return normalizeColumnOffset(ui.columnOffsets?.[columnId]);
}

function setColumnOffset(ui, columnId, offset) {
    ui.columnOffsets[columnId] = normalizeColumnOffset(offset);
}

function applyColumnLayout(ui) {
    for (const [columnId, column] of ui.columns.entries()) {
        const slotIndex = ui.columnOrder.indexOf(columnId);
        if (slotIndex >= 0) {
            column.style.order = String(slotIndex);
        }
        const offset = getColumnOffset(ui, columnId);
        column.style.transform = `translate(${Math.round(offset.x)}px, ${Math.round(offset.y)}px)`;
        column.classList.toggle("is-dragging", ui.dragState?.active && ui.dragState.columnId === columnId);
    }
}

function getColumnCenterX(column) {
    const rect = column.getBoundingClientRect();
    return rect.left + rect.width / 2;
}

function clampColumnOffsetX(ui, columnId, desiredX) {
    const column = ui.columns.get(columnId);
    if (!column) {
        return desiredX;
    }
    const boundsRect = getMovementBoundsRect(ui);
    const rect = column.getBoundingClientRect();
    const currentOffset = getColumnOffset(ui, columnId);
    const scale = getCanvasScale();
    const naturalLeft = rect.left - (currentOffset.x * scale);
    const naturalRight = rect.right - (currentOffset.x * scale);
    const padding = 8;
    const minX = (boundsRect.left + padding - naturalLeft) / scale;
    const maxX = (boundsRect.right - padding - naturalRight) / scale;
    return Math.max(minX, Math.min(maxX, desiredX));
}

function clampColumnOffsetY(ui, columnId, desiredY) {
    const column = ui.columns.get(columnId);
    if (!column) {
        return desiredY;
    }
    const boundsRect = getMovementBoundsRect(ui);
    const rect = column.getBoundingClientRect();
    const currentOffset = getColumnOffset(ui, columnId);
    const scale = getCanvasScale();
    const naturalTop = rect.top - (currentOffset.y * scale);
    const naturalBottom = rect.bottom - (currentOffset.y * scale);
    const padding = 8;
    const minY = (boundsRect.top + padding - naturalTop) / scale;
    const maxY = (boundsRect.bottom - padding - naturalBottom) / scale;
    return Math.max(minY, Math.min(maxY, desiredY));
}

function queueSuppressNextClick(ui) {
    ui.suppressNextClick = true;
    setTimeout(() => {
        ui.suppressNextClick = false;
    }, 0);
}

function normalizeLayoutForPreset(stored, preset) {
    const presetColumnIds = getPresetColumnIds(preset);
    const completeOrder = normalizeColumnOrder(stored?.columnOrder, presetColumnIds);
    const columnOffsets = {};
    for (const columnId of presetColumnIds) {
        columnOffsets[columnId] = { x: 0, y: 0 };
    }
    return {
        columnOrder: completeOrder,
        columnOffsets,
    };
}

function createNodeLayoutAdapter(node) {
    return {
        load(preset) {
            const workflowLayoutStore = node.properties?.omnivoiceInstructionLayout?.layouts?.[preset.id];
            if (Array.isArray(workflowLayoutStore?.columnOrder) && workflowLayoutStore.columnOrder.length) {
                return normalizeLayoutForPreset(workflowLayoutStore, preset);
            }
            if (preset.id === OMNIVOICE_PRESET_ID) {
                const builtinLayoutStore = node.__omnivoiceInstructionBuiltinPresetLayouts?.[preset.id];
                return normalizeLayoutForPreset(builtinLayoutStore, preset);
            }
            return normalizeLayoutForPreset({ columnOrder: preset.columnOrder }, preset);
        },
        save(preset, ui) {
            node.properties = node.properties || {};
            node.properties.omnivoiceInstructionLayout = node.properties.omnivoiceInstructionLayout || { layouts: {} };
            node.properties.omnivoiceInstructionLayout.layouts[preset.id] = {
                columnOrder: [...ui.columnOrder],
            };
        },
    };
}

function createMemoryLayoutAdapter(initialLayout = {}, initialPresetId = OMNIVOICE_PRESET_ID) {
    const store = {
        layouts: {
            [initialPresetId]: {
                columnOrder: Array.isArray(initialLayout.columnOrder) ? [...initialLayout.columnOrder] : undefined,
                columnOffsets: cloneJson(initialLayout.columnOffsets || {}),
            },
        },
    };
    return {
        load(preset) {
            return normalizeLayoutForPreset(store.layouts[preset.id], preset);
        },
        save(preset, ui) {
            store.layouts[preset.id] = {
                columnOrder: [...ui.columnOrder],
            };
        },
    };
}

function syncLayoutForPreset(ui, preset) {
    if (!ui.layoutAdapter) {
        return;
    }
    const presetColumnIds = getPresetColumnIds(preset);
    const currentIds = Array.isArray(ui.renderedColumnIds) ? ui.renderedColumnIds : [];
    const sameIds = currentIds.length === presetColumnIds.length && currentIds.every((id, index) => id === presetColumnIds[index]);
    if (ui.activeLayoutPresetId === preset.id && sameIds) {
        return;
    }
    const layout = ui.layoutAdapter.load(preset);
    ui.columnOrder = layout.columnOrder;
    ui.columnOffsets = layout.columnOffsets;
    ui.activeLayoutPresetId = preset.id;
}

function persistCurrentLayout(ui, state) {
    if (!ui.layoutAdapter) {
        return;
    }
    const preset = getActivePreset(state);
    ui.layoutAdapter.save(preset, ui);
}

function populatePresetSelect(ui, state) {
    const presets = getAvailablePresets(state);
    ui.presetSelect.innerHTML = "";
    for (const preset of presets) {
        const option = document.createElement("option");
        option.value = preset.id;
        option.textContent = preset.name;
        option.selected = preset.id === state.selectedPresetId;
        ui.presetSelect.appendChild(option);
    }
}

function applyLocalizedUiText(ui, state, preset) {
    const locale = getPresetOutputLocale(preset, state);
    const text = UI_TEXT[locale];
    ui.previewLabel.textContent = text.previewLabel;
    ui.previewInput.placeholder = text.emptyPreview;
    ui.editPresetButton.textContent = text.editPreset;
    ui.presetNameLabel.textContent = text.presetName;
    ui.localeEnglishButton.textContent = text.localeEn;
    ui.localeChineseButton.textContent = text.localeZh;
    ui.addSingleColumnButton.textContent = text.addColumn;
    ui.addSwitchColumnButton.textContent = text.addSwitch;
    ui.modalSaveButton.textContent = text.save;
    ui.modalCancelButton.textContent = text.cancel;
    ui.localeEnglishButton.classList.toggle("is-active", locale === "en");
    ui.localeChineseButton.classList.toggle("is-active", locale === "zh");
    ui.localeSwitch.style.display = isOmniVoicePreset(preset) ? "grid" : "none";

    for (const [columnId, titleEl] of ui.columnTitles.entries()) {
        const column = ui.columnDefsById.get(columnId);
        const title = getColumnTitleForLocale(column, locale);
        titleEl.textContent = title || "";
        titleEl.classList.toggle("is-empty", !title);
    }
    for (const [key, group] of ui.switchGroups.entries()) {
        const [columnId, modeId] = key.split(":");
        const column = ui.columnDefsById.get(columnId);
        const mode = getModeById(column, modeId);
        if (mode) {
            const title = getModeTitleForLocale(mode, locale);
            group.subtitle.textContent = title || "";
            group.subtitle.classList.toggle("is-empty", !title);
        }
    }
    for (const [key, button] of ui.switchModeButtons.entries()) {
        const [columnId, modeId] = key.split(":");
        const column = ui.columnDefsById.get(columnId);
        const mode = getModeById(column, modeId);
        if (mode) {
            button.textContent = getModeTitleForLocale(mode, locale);
        }
    }
    for (const button of ui.chipButtons.values()) {
        button.textContent = getOptionLabelForLocale({
            label: button.dataset.labelEn,
            zhLabel: button.dataset.labelZh,
            value: button.dataset.value,
        }, locale);
        button.title = getOptionTitleForLocale({
            title: button.dataset.titleEn,
            zhTitle: button.dataset.titleZh,
            zhLabel: button.dataset.labelZh,
            value: button.dataset.value,
        }, locale);
    }
}

function rebuildPresetColumns(ui, preset, state) {
    syncLayoutForPreset(ui, preset);
    const locale = getPresetOutputLocale(preset, state);
    const presetColumnIds = getPresetColumnIds(preset);
    const presetSignature = JSON.stringify(preset.columns || []);
    const sameIds = ui.renderedPresetId === preset.id
        && ui.renderedPresetSignature === presetSignature
        && ui.renderedColumnIds.length === presetColumnIds.length
        && ui.renderedColumnIds.every((id, index) => id === presetColumnIds[index]);

    if (sameIds) {
        return;
    }

    ui.grid.innerHTML = "";
    ui.chipButtons = new Map();
    ui.columns = new Map();
    ui.columnTitles = new Map();
    ui.switchGroups = new Map();
    ui.switchModeButtons = new Map();
    ui.columnDefsById = new Map();

    for (const column of preset.columns) {
        ui.columnDefsById.set(column.id, column);
        const columnEl = createEl("div", `ovib-column${column.type === "switch" ? " ovib-language-column" : ""}`);
        columnEl.dataset.columnId = column.id;
        ui.columns.set(column.id, columnEl);

        const title = getColumnTitleForLocale(column, locale);
        const titleEl = createEl("div", "ovib-column-title", title);
        titleEl.classList.toggle("is-empty", !title);
        columnEl.appendChild(titleEl);
        ui.columnTitles.set(column.id, titleEl);

        if (column.type === "switch") {
            const stack = createEl("div", "ovib-language-stack");
            const modeSwitch = createEl("div", "ovib-mode-switch");
            for (const mode of column.modes) {
                const modeButton = createEl("button", "ovib-mode-chip", getModeTitleForLocale(mode, locale));
                modeButton.type = "button";
                modeButton.dataset.columnId = column.id;
                modeButton.dataset.modeId = mode.id;
                ui.switchModeButtons.set(`${column.id}:${mode.id}`, modeButton);
                modeSwitch.appendChild(modeButton);
            }
            stack.appendChild(modeSwitch);

            for (const mode of column.modes) {
                const group = createEl("div", "ovib-language-group");
                group.dataset.columnId = column.id;
                group.dataset.modeId = mode.id;
                const subtitle = createEl("div", "ovib-subtitle", getModeTitleForLocale(mode, locale));
                group.appendChild(subtitle);
                const list = createEl("div", "ovib-chip-list");
                for (const option of mode.options) {
                    const chip = createChip(option, column.id, mode.id);
                    ui.chipButtons.set(`${column.id}:${mode.id}:${option.value}`, chip);
                    list.appendChild(chip);
                }
                group.append(list);
                stack.appendChild(group);
                ui.switchGroups.set(`${column.id}:${mode.id}`, { element: group, subtitle });
            }
            columnEl.appendChild(stack);
        } else {
            const list = createEl("div", "ovib-chip-list");
            for (const option of column.options) {
                const chip = createChip(option, column.id);
                ui.chipButtons.set(`${column.id}:${option.value}`, chip);
                list.appendChild(chip);
            }
            columnEl.appendChild(list);
        }
        ui.grid.appendChild(columnEl);
    }

    ui.renderedPresetId = preset.id;
    ui.renderedPresetSignature = presetSignature;
    ui.renderedColumnIds = [...presetColumnIds];
    applyColumnLayout(ui);
}

function renderState(node, ui, state) {
    const preset = getActivePreset(state);
    const bucket = getPresetStateBucket(state, preset.id);
    rebuildPresetColumns(ui, preset, state);
    populatePresetSelect(ui, state);
    applyLocalizedUiText(ui, state, preset);

    for (const button of ui.chipButtons.values()) {
        const columnId = button.dataset.columnId;
        const modeId = button.dataset.modeId;
        const isActive = modeId
            ? bucket.switchModes[columnId] === modeId && normalizeWidgetValue(bucket.values[columnId]) === button.dataset.value
            : normalizeWidgetValue(bucket.values[columnId]) === button.dataset.value;
        button.classList.toggle("is-active", isActive);
        button.classList.remove("is-disabled");
    }

    for (const columnId of getPresetColumnIds(preset)) {
        const column = ui.columns.get(columnId);
        if (!column) {
            continue;
        }
        const isDraggable = Boolean(normalizeWidgetValue(bucket.values[columnId]));
        column.classList.toggle("is-draggable", isDraggable);
        if (!isDraggable && !ui.dragState?.active) {
            setColumnOffset(ui, columnId, { x: 0, y: 0 });
        }
    }

    for (const column of preset.columns) {
        if (column.type !== "switch") {
            continue;
        }
        const activeMode = getModeById(column, bucket.switchModes[column.id])?.id || column.modes?.[0]?.id || "";
        bucket.switchModes[column.id] = activeMode;
        for (const mode of column.modes) {
            const group = ui.switchGroups.get(`${column.id}:${mode.id}`);
            const modeButton = ui.switchModeButtons.get(`${column.id}:${mode.id}`);
            if (group) {
                group.element.classList.toggle("is-hidden", mode.id !== activeMode);
            }
            if (modeButton) {
                modeButton.classList.toggle("is-active", mode.id === activeMode);
            }
        }
    }

    const preview = buildPreviewFromState(state, preset, ui.columnOrder);
    if (!ui.previewInputDirty && ui.previewInput.value !== preview) {
        ui.previewInput.value = preview;
    }
}

function drawPath(ui, state) {
    const preset = getActivePreset(state);
    const bucket = getPresetStateBucket(state, preset.id);
    const scale = getCanvasScale();
    ui.path.style.strokeWidth = `${PATH_BASE_STROKE_WIDTH * scale}px`;

    const bodyRect = ui.body.getBoundingClientRect();
    const svgWidth = Math.max(1, Math.round(bodyRect.width));
    const svgHeight = Math.max(1, Math.round(bodyRect.height));
    ui.svg.setAttribute("viewBox", `0 0 ${svgWidth} ${svgHeight}`);
    ui.svg.setAttribute("width", String(svgWidth));
    ui.svg.setAttribute("height", String(svgHeight));
    ui.svg.setAttribute("preserveAspectRatio", "none");

    const selected = [];
    for (const columnId of ui.columnOrder) {
        const column = getColumnById(preset, columnId);
        if (!column) {
            continue;
        }
        if (column.type === "switch") {
            const activeMode = getModeById(column, bucket.switchModes[columnId])?.id || "";
            const value = normalizeWidgetValue(bucket.values[columnId]);
            if (activeMode && value) {
                const button = ui.chipButtons.get(`${columnId}:${activeMode}:${value}`);
                if (button?.getClientRects().length > 0) {
                    selected.push(button);
                }
            }
            continue;
        }
        const value = normalizeWidgetValue(bucket.values[columnId]);
        if (!value) {
            continue;
        }
        const button = ui.chipButtons.get(`${columnId}:${value}`);
        if (button?.getClientRects().length > 0) {
            selected.push(button);
        }
    }

    if (selected.length < 2) {
        ui.path.setAttribute("d", "");
        return;
    }

    const segments = [];
    for (let index = 0; index < selected.length - 1; index += 1) {
        const sourceRect = selected[index].getBoundingClientRect();
        const targetRect = selected[index + 1].getBoundingClientRect();
        segments.push({
            x1: sourceRect.right - bodyRect.left + 3,
            y1: sourceRect.top - bodyRect.top + sourceRect.height / 2,
            x2: targetRect.left - bodyRect.left - 3,
            y2: targetRect.top - bodyRect.top + targetRect.height / 2,
        });
    }

    let pathData = "";
    for (const segment of segments) {
        const dx = segment.x2 - segment.x1;
        const minHandle = 28 * scale;
        const maxHandle = 64 * scale;
        const handle = Math.max(minHandle, Math.min(maxHandle, Math.abs(dx) * 0.35));
        const control1X = segment.x1 + handle;
        const control2X = segment.x2 - handle;
        pathData += `M ${segment.x1} ${segment.y1} C ${control1X} ${segment.y1}, ${control2X} ${segment.y2}, ${segment.x2} ${segment.y2} `;
    }
    ui.path.setAttribute("d", pathData.trim());
}

function setWidgetHeightSafe(widget, height) {
    if (!widget) {
        return;
    }
    try {
        widget.height = height;
    } catch {
        // Some ComfyUI builds expose BaseWidget.height as getter-only.
    }
    try {
        widget.computedHeight = height;
    } catch {
        // Ignore readonly implementations.
    }
    if (widget.element) {
        widget.element.style.height = `${height}px`;
        widget.element.style.minHeight = `${height}px`;
        widget.element.style.maxHeight = `${height}px`;
    }
}

function updateBasePanelHeight(node, panelWidget, force = false) {
    if (!force && Number(node.__omnivoiceInstructionBasePanelHeight || 0) > 0) {
        return;
    }
    const nodeHeight = Math.max(PANEL_MIN_HEIGHT, Number(node?.size?.[1] || 0));
    const layoutY = Math.max(0, Number(panelWidget?.last_y || 0));
    const extraHeight = Math.max(0, nodeHeight - PANEL_MIN_HEIGHT);
    const measuredHeight = Math.max(PANEL_WIDGET_MIN_HEIGHT, Math.round(nodeHeight - layoutY - PANEL_BOTTOM_PADDING));
    const baselineHeight = Math.max(PANEL_WIDGET_MIN_HEIGHT, measuredHeight - extraHeight);
    node.__omnivoiceInstructionBasePanelHeight = baselineHeight;
}

function getRequiredNodeMinHeight(node) {
    const layoutY = Math.max(0, Number(node.__omnivoiceInstructionLastY || 0));
    return Math.max(
        PANEL_WIDGET_MIN_HEIGHT + PANEL_BOTTOM_PADDING,
        Math.round(layoutY + PANEL_WIDGET_MIN_HEIGHT + PANEL_BOTTOM_PADDING),
    );
}

function getTargetPanelHeight(node) {
    const nodeHeight = Math.max(getRequiredNodeMinHeight(node), Number(node?.size?.[1] || 0));
    const layoutY = Math.max(0, Number(node.__omnivoiceInstructionLastY || 0));
    return Math.max(PANEL_WIDGET_MIN_HEIGHT, Math.round(nodeHeight - layoutY - PANEL_BOTTOM_PADDING));
}

function resizePanel(node, ui, panelWidget) {
    if (!panelWidget) {
        return;
    }
    requestAnimationFrame(() => {
        const targetPanelHeight = getTargetPanelHeight(node);
        const panelHeightChanged = Math.abs(Number(node.__omnivoiceInstructionAppliedPanelHeight || 0) - targetPanelHeight) > 1;
        if (panelHeightChanged) {
            setWidgetHeightSafe(panelWidget, PANEL_WIDGET_MIN_HEIGHT);
        }
        if (panelWidget.element) {
            panelWidget.element.style.width = "100%";
            panelWidget.element.style.maxWidth = "100%";
            panelWidget.element.style.minWidth = "100%";
            panelWidget.element.style.height = `${targetPanelHeight}px`;
            panelWidget.element.style.minHeight = `${targetPanelHeight}px`;
            panelWidget.element.style.maxHeight = `${targetPanelHeight}px`;
            panelWidget.element.style.overflow = "visible";
            panelWidget.element.style.display = "block";
            panelWidget.element.style.position = "relative";
            panelWidget.element.style.boxSizing = "border-box";
            panelWidget.element.style.margin = "0";
            panelWidget.element.style.padding = "0";
            panelWidget.element.style.alignSelf = "stretch";
        }
        node.__omnivoiceInstructionAppliedPanelHeight = targetPanelHeight;
        if (panelHeightChanged) {
            node.graph?.setDirtyCanvas?.(true, true);
            drawPath(ui, node.__omnivoiceInstructionState || {});
        }
    });
}

function swapColumns(ui, sourceIndex, targetIndex) {
    if (sourceIndex < 0 || targetIndex < 0 || sourceIndex >= ui.columnOrder.length || targetIndex >= ui.columnOrder.length) {
        return false;
    }
    const sourceId = ui.columnOrder[sourceIndex];
    const draggedColumn = ui.columns.get(sourceId);
    if (!draggedColumn) {
        return false;
    }
    const oldCenter = getColumnCenterX(draggedColumn);
    const nextOrder = [...ui.columnOrder];
    const [moved] = nextOrder.splice(sourceIndex, 1);
    nextOrder.splice(targetIndex, 0, moved);
    ui.columnOrder = nextOrder;
    applyColumnLayout(ui);
    const newCenter = getColumnCenterX(draggedColumn);
    const offset = getColumnOffset(ui, sourceId);
    const scale = getCanvasScale();
    setColumnOffset(ui, sourceId, {
        x: clampColumnOffsetX(ui, sourceId, offset.x + ((oldCenter - newCenter) / scale)),
        y: clampColumnOffsetY(ui, sourceId, offset.y),
    });
    applyColumnLayout(ui);
    return true;
}

function beginColumnDrag(ui, event, columnId, selectionTarget = null, selectionChangedOnPointerDown = false) {
    ui.dragState = {
        active: true,
        pointerId: event.pointerId,
        columnId,
        startX: event.clientX,
        startY: event.clientY,
        moved: false,
        baseOffsetX: getColumnOffset(ui, columnId).x,
        baseOffsetY: getColumnOffset(ui, columnId).y,
        selectionTarget,
        selectionChangedOnPointerDown,
        toggleOffOnPointerEnd: false,
    };
    event.currentTarget?.setPointerCapture?.(event.pointerId);
    event.preventDefault();
}

function applyDragSelection(node, state, selectionTarget) {
    if (!selectionTarget) {
        return state;
    }
    const preset = getActivePreset(state);
    setPresetColumnValue(
        state,
        preset,
        selectionTarget.columnId,
        selectionTarget.value,
        false,
        selectionTarget.modeId,
    );
    applyStateToWidgets(node, state, node.__omnivoiceInstructionUi || null);
    node.__omnivoiceInstructionState = state;
    return state;
}

function setupColumnDragging(node, ui) {
    const handlePointerMove = (event) => {
        const dragState = ui.dragState;
        if (!dragState?.active || event.pointerId !== dragState.pointerId) {
            return;
        }

        const scale = getCanvasScale();
        const deltaX = (event.clientX - dragState.startX) / scale;
        const deltaY = (event.clientY - dragState.startY) / scale;
        if (!dragState.moved && Math.hypot(deltaX, deltaY) < DRAG_START_THRESHOLD) {
            return;
        }

        dragState.moved = true;
        const columnId = dragState.columnId;
        const column = ui.columns.get(columnId);
        if (!column) {
            return;
        }

        setColumnOffset(ui, columnId, {
            x: clampColumnOffsetX(ui, columnId, dragState.baseOffsetX + deltaX),
            y: clampColumnOffsetY(ui, columnId, dragState.baseOffsetY + deltaY),
        });
        applyColumnLayout(ui);

        const currentIndex = ui.columnOrder.indexOf(columnId);
        const currentCenter = getColumnCenterX(column);
        let swapped = false;
        if (currentIndex > 0) {
            const leftId = ui.columnOrder[currentIndex - 1];
            const leftColumn = ui.columns.get(leftId);
            if (leftColumn && currentCenter < getColumnCenterX(leftColumn)) {
                swapped = swapColumns(ui, currentIndex, currentIndex - 1);
            }
        }
        const nextIndex = ui.columnOrder.indexOf(columnId);
        if (!swapped && nextIndex < ui.columnOrder.length - 1) {
            const rightId = ui.columnOrder[nextIndex + 1];
            const rightColumn = ui.columns.get(rightId);
            if (rightColumn && currentCenter > getColumnCenterX(rightColumn)) {
                swapped = swapColumns(ui, nextIndex, nextIndex + 1);
            }
        }
        if (swapped) {
            dragState.startX = event.clientX;
            dragState.startY = event.clientY;
            dragState.baseOffsetX = getColumnOffset(ui, columnId).x;
            dragState.baseOffsetY = getColumnOffset(ui, columnId).y;
        }

        applyColumnLayout(ui);
        drawPath(ui, node.__omnivoiceInstructionState || syncStateFromWidgets(node));
    };

    const handlePointerEnd = (event) => {
        const dragState = ui.dragState;
        if (!dragState?.active || event.pointerId !== dragState.pointerId) {
            return;
        }

        const columnId = dragState.columnId;
        let state = node.__omnivoiceInstructionState || syncStateFromWidgets(node);
        if (dragState.moved) {
            state = applyDragSelection(node, state, dragState.selectionTarget);
        } else if (dragState.toggleOffOnPointerEnd && dragState.selectionTarget) {
            const preset = getActivePreset(state);
            setPresetColumnValue(
                state,
                preset,
                dragState.selectionTarget.columnId,
                dragState.selectionTarget.value,
                true,
                dragState.selectionTarget.modeId,
            );
            applyStateToWidgets(node, state, node.__omnivoiceInstructionUi || null);
            node.__omnivoiceInstructionState = state;
        }
        const preset = getActivePreset(state);
        const bucket = getPresetStateBucket(state, preset.id);
        const stillSelected = Boolean(normalizeWidgetValue(bucket.values[columnId]));
        const currentOffset = getColumnOffset(ui, columnId);
        const clampedOffset = stillSelected
            ? {
                x: clampColumnOffsetX(
                    ui,
                    columnId,
                    Math.max(-RESTING_OFFSET_LIMIT, Math.min(RESTING_OFFSET_LIMIT, currentOffset.x)),
                ),
                y: clampColumnOffsetY(ui, columnId, currentOffset.y),
            }
            : { x: 0, y: 0 };
        setColumnOffset(ui, columnId, clampedOffset);
        ui.dragState = null;
        applyColumnLayout(ui);
        drawPath(ui, state);
        if (dragState.moved || dragState.selectionChangedOnPointerDown || dragState.toggleOffOnPointerEnd) {
            queueSuppressNextClick(ui);
        }
        persistCurrentLayout(ui, state);
        node.__omnivoiceInstructionRefresh?.();
    };

    ui.handlePointerMove = handlePointerMove;
    ui.handlePointerEnd = handlePointerEnd;
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerEnd);
    window.addEventListener("pointercancel", handlePointerEnd);
}

function buildModeEditor(mode, columnIndex, modeIndex, locale) {
    const text = UI_TEXT[locale];
    const card = createEl("div", "ovib-mode-card");
    const header = createEl("div", "ovib-mode-card-header");
    header.appendChild(createEl("div", "ovib-mode-card-title", `${text.modeTitle} ${modeIndex + 1}`));
    const removeButton = createEl("button", "ovib-icon-button", text.remove);
    removeButton.type = "button";
    removeButton.dataset.action = "remove-mode";
    removeButton.dataset.columnIndex = String(columnIndex);
    removeButton.dataset.modeIndex = String(modeIndex);
    header.appendChild(removeButton);
    card.appendChild(header);

    const titleStack = createEl("div", "ovib-stack");
    titleStack.appendChild(createEl("label", "ovib-form-label", text.modeTitle));
    const titleInput = createEl("input", "ovib-form-input");
    titleInput.type = "text";
    titleInput.dataset.action = "mode-title";
    titleInput.dataset.columnIndex = String(columnIndex);
    titleInput.dataset.modeIndex = String(modeIndex);
    titleInput.value = mode.title || "";
    titleStack.appendChild(titleInput);
    card.appendChild(titleStack);

    const optionsStack = createEl("div", "ovib-stack");
    optionsStack.appendChild(createEl("label", "ovib-form-label", text.options));
    const optionsHint = createEl("div", "ovib-form-hint", text.optionsHint);
    const optionsInput = createEl("textarea", "ovib-form-textarea");
    optionsInput.dataset.action = "mode-options";
    optionsInput.dataset.columnIndex = String(columnIndex);
    optionsInput.dataset.modeIndex = String(modeIndex);
    optionsInput.value = (mode.options || []).map((option) => (
        option.label && option.value && option.label !== option.value
            ? `${option.label}=${option.value}`
            : (option.value || option.label || "")
    )).filter(Boolean).join("\n");
    optionsStack.append(optionsHint, optionsInput);
    card.appendChild(optionsStack);

    return card;
}

function getActiveModalModeIndex(ui, columnIndex, modes = []) {
    if (!modes.length) {
        return 0;
    }
    const requested = Number(ui.modalModeSelection?.[columnIndex]);
    if (Number.isInteger(requested) && requested >= 0 && requested < modes.length) {
        return requested;
    }
    return 0;
}

function renderPresetEditor(ui, state) {
    const locale = normalizeWidgetValue(state.output_language) === "Chinese" ? "zh" : "en";
    const text = UI_TEXT[locale];
    const draft = ui.modalDraft;
    ui.modalLocale = locale;
    ui.modalTitle.textContent = ui.modalEditingPresetId ? text.editPresetTitle : text.createPresetTitle;
    ui.presetNameLabel.textContent = text.presetName;
    ui.modalDeleteButton.textContent = text.deletePreset;
    ui.modalSaveButton.textContent = text.save;
    ui.modalCancelButton.textContent = text.cancel;
    ui.addSingleColumnButton.textContent = text.addColumn;
    ui.addSwitchColumnButton.textContent = text.addSwitch;
    ui.modalDeleteButton.style.display = ui.modalEditingPresetId ? "" : "none";
    ui.presetNameInput.value = draft?.name || "";
    ui.columnsEditor.innerHTML = "";

    (draft?.columns || []).forEach((column, columnIndex) => {
        const card = createEl("div", "ovib-column-card");
        const header = createEl("div", "ovib-column-card-header");
        header.appendChild(createEl("div", "ovib-column-card-title", column.title || text.columnTitle));
        const removeButton = createEl("button", "ovib-icon-button", text.remove);
        removeButton.type = "button";
        removeButton.dataset.action = "remove-column";
        removeButton.dataset.columnIndex = String(columnIndex);
        header.appendChild(removeButton);
        card.appendChild(header);

        const titleStack = createEl("div", "ovib-stack");
        const topRow = createEl("div", "ovib-card-topline");
        const typeSelect = createEl("select", "ovib-form-select");
        typeSelect.dataset.action = "column-type";
        typeSelect.dataset.columnIndex = String(columnIndex);
        const singleOption = document.createElement("option");
        singleOption.value = "single";
        singleOption.textContent = text.single;
        const switchOption = document.createElement("option");
        switchOption.value = "switch";
        switchOption.textContent = text.switch;
        typeSelect.append(singleOption, switchOption);
        typeSelect.value = column.type === "switch" ? "switch" : "single";
        const titleInput = createEl("input", "ovib-form-input");
        titleInput.type = "text";
        titleInput.dataset.action = "column-title";
        titleInput.dataset.columnIndex = String(columnIndex);
        titleInput.placeholder = text.columnTitle;
        titleInput.value = column.title || "";
        topRow.append(typeSelect, titleInput);
        titleStack.appendChild(topRow);
        card.appendChild(titleStack);

        if (column.type === "switch") {
            const modes = column.modes || [];
            const activeModeIndex = getActiveModalModeIndex(ui, columnIndex, modes);
            if (modes.length) {
                const modeSwitch = createEl("div", "ovib-mode-switch");
                modes.forEach((mode, modeIndex) => {
                    const modeButton = createEl("button", `ovib-mode-chip${modeIndex === activeModeIndex ? " is-active" : ""}`, mode.title || `${text.modeTitle} ${modeIndex + 1}`);
                    modeButton.type = "button";
                    modeButton.dataset.action = "select-mode";
                    modeButton.dataset.columnIndex = String(columnIndex);
                    modeButton.dataset.modeIndex = String(modeIndex);
                    modeSwitch.appendChild(modeButton);
                });
                card.appendChild(modeSwitch);
                card.appendChild(buildModeEditor(modes[activeModeIndex], columnIndex, activeModeIndex, locale));
            }
            const addModeButton = createEl("button", "ovib-icon-button", text.addMode);
            addModeButton.type = "button";
            addModeButton.dataset.action = "add-mode";
            addModeButton.dataset.columnIndex = String(columnIndex);
            card.appendChild(addModeButton);
        } else {
            const optionsStack = createEl("div", "ovib-stack");
            optionsStack.appendChild(createEl("label", "ovib-form-label", text.options));
            optionsStack.appendChild(createEl("div", "ovib-form-hint", text.optionsHint));
            const optionsInput = createEl("textarea", "ovib-form-textarea");
            optionsInput.dataset.action = "column-options";
            optionsInput.dataset.columnIndex = String(columnIndex);
            optionsInput.value = (column.options || []).map((option) => (
                option.label && option.value && option.label !== option.value
                    ? `${option.label}=${option.value}`
                    : (option.value || option.label || "")
            )).filter(Boolean).join("\n");
            optionsStack.appendChild(optionsInput);
            card.appendChild(optionsStack);
        }

        ui.columnsEditor.appendChild(card);
    });
}

function openPresetEditor(node, ui) {
    const state = node.__omnivoiceInstructionState || syncStateFromWidgets(node);
    const activePreset = getActivePreset(state);
    const editableExisting = activePreset.id !== OMNIVOICE_PRESET_ID && activePreset.id !== TEMP_TEXT_PRESET_ID;
    ui.modalEditingPresetId = editableExisting ? activePreset.id : "";
    ui.modalDraft = createPresetDraftFromPreset(
        activePreset,
        editableExisting,
    );
    if (!editableExisting) {
        ui.modalDraft.name = activePreset.id === TEMP_TEXT_PRESET_ID ? "Temporary Text" : `${activePreset.name} Copy`;
        ui.modalDraft.id = "";
        ui.modalDraft.outputMode = "plain";
    }
    ui.modalModeSelection = {};
    renderPresetEditor(ui, state);
    ui.modalBackdrop.classList.add("is-open");
}

function closePresetEditor(ui) {
    ui.modalBackdrop.classList.remove("is-open");
    ui.modalDraft = null;
    ui.modalEditingPresetId = "";
    ui.modalModeSelection = {};
}

function updateDraftFromInput(ui, target) {
    const draft = ui.modalDraft;
    if (!draft) {
        return;
    }
    const columnIndex = Number(target.dataset.columnIndex);
    const modeIndex = Number(target.dataset.modeIndex);

    if (target.dataset.action === "column-title" && draft.columns[columnIndex]) {
        draft.columns[columnIndex].title = target.value;
    } else if (target.dataset.action === "column-type" && draft.columns[columnIndex]) {
        const currentColumn = draft.columns[columnIndex];
        if (target.value === "switch") {
            draft.columns[columnIndex] = {
                id: currentColumn.id || slugifyId(currentColumn.title || `switch_${columnIndex + 1}`, `switch_${columnIndex + 1}`),
                title: currentColumn.title || "",
                type: "switch",
                modes: currentColumn.type === "switch" && currentColumn.modes?.length
                    ? currentColumn.modes
                    : createEmptySwitchColumn(columnIndex).modes,
            };
        } else {
            draft.columns[columnIndex] = {
                id: currentColumn.id || slugifyId(currentColumn.title || `column_${columnIndex + 1}`, `column_${columnIndex + 1}`),
                title: currentColumn.title || "",
                type: "single",
                options: currentColumn.type === "single" && currentColumn.options?.length
                    ? currentColumn.options
                    : [{ id: "option_1", label: "", value: "", title: "" }],
            };
        }
        delete ui.modalModeSelection[columnIndex];
        renderPresetEditor(ui, {
            output_language: ui.modalLocale === "zh" ? "Chinese" : "English",
        });
    } else if (target.dataset.action === "column-options" && draft.columns[columnIndex]) {
        draft.columns[columnIndex].options = parsePresetOptionLines(target.value);
    } else if (target.dataset.action === "mode-title" && draft.columns[columnIndex]?.modes?.[modeIndex]) {
        draft.columns[columnIndex].modes[modeIndex].title = target.value;
    } else if (target.dataset.action === "mode-options" && draft.columns[columnIndex]?.modes?.[modeIndex]) {
        draft.columns[columnIndex].modes[modeIndex].options = parsePresetOptionLines(target.value);
    }
}

function handleDraftButtonAction(ui, target, state) {
    const draft = ui.modalDraft;
    if (!draft) {
        return;
    }
    const columnIndex = Number(target.dataset.columnIndex);
    const modeIndex = Number(target.dataset.modeIndex);

    if (target.dataset.action === "remove-column") {
        draft.columns.splice(columnIndex, 1);
        ui.modalModeSelection = {};
        renderPresetEditor(ui, state);
    } else if (target.dataset.action === "select-mode" && draft.columns[columnIndex]?.modes?.[modeIndex]) {
        ui.modalModeSelection[columnIndex] = modeIndex;
        renderPresetEditor(ui, state);
    } else if (target.dataset.action === "add-mode" && draft.columns[columnIndex]) {
        draft.columns[columnIndex].modes = draft.columns[columnIndex].modes || [];
        draft.columns[columnIndex].modes.push({
            id: `mode_${draft.columns[columnIndex].modes.length + 1}`,
            title: `Mode ${draft.columns[columnIndex].modes.length + 1}`,
            options: [{ id: "option_1", label: "", value: "", title: "" }],
        });
        ui.modalModeSelection[columnIndex] = draft.columns[columnIndex].modes.length - 1;
        renderPresetEditor(ui, state);
    } else if (target.dataset.action === "remove-mode" && draft.columns[columnIndex]?.modes?.length) {
        draft.columns[columnIndex].modes.splice(modeIndex, 1);
        if (!draft.columns[columnIndex].modes.length) {
            draft.columns[columnIndex].modes.push({
                id: "mode_1",
                title: "Mode 1",
                options: [{ id: "option_1", label: "", value: "", title: "" }],
            });
            ui.modalModeSelection[columnIndex] = 0;
        } else {
            ui.modalModeSelection[columnIndex] = Math.max(0, Math.min(modeIndex - 1, draft.columns[columnIndex].modes.length - 1));
        }
        renderPresetEditor(ui, state);
    }
}

async function savePresetEditor(node, ui) {
    const draft = ui.modalDraft;
    if (!draft) {
        return;
    }
    const state = node.__omnivoiceInstructionState || syncStateFromWidgets(node);
    draft.name = ui.presetNameInput.value.trim() || draft.name || "Preset";
    const normalized = createPresetFromDraft(draft, ui.modalEditingPresetId || "custom_preset");
    if (!normalized.columns.length) {
        window.alert("Preset needs at least one valid column.");
        return;
    }

    const previousId = ui.modalEditingPresetId || "";
    state.customPresets = (state.customPresets || []).filter((preset) => preset.id !== previousId && preset.id !== normalized.id);
    state.customPresets.push(normalized);

    if (previousId && previousId !== normalized.id) {
        if (state.presetStates?.[previousId] && !state.presetStates[normalized.id]) {
            state.presetStates[normalized.id] = state.presetStates[previousId];
        }
        delete state.presetStates?.[previousId];
    }
    if (!state.presetStates[normalized.id]) {
        seedBucketFromPreset(state, normalized);
    }

    state.selectedPresetId = normalized.id;
    await persistPresetLibraryNow(node, state);
    state.customPresets = Array.isArray(node.__omnivoiceInstructionLibraryPresets)
        ? node.__omnivoiceInstructionLibraryPresets.map((preset, index) => normalizePreset(preset, `custom_preset_${index + 1}`))
        : state.customPresets;
    applyStateToWidgets(node, state, ui);
    node.__omnivoiceInstructionState = state;
    closePresetEditor(ui);
    node.__omnivoiceInstructionRefresh?.();
}

async function deletePresetEditor(node, ui) {
    if (!ui.modalEditingPresetId) {
        return;
    }
    if (!window.confirm("Delete this preset?")) {
        return;
    }
    const state = node.__omnivoiceInstructionState || syncStateFromWidgets(node);
    const presetId = ui.modalEditingPresetId;
    state.customPresets = (state.customPresets || []).filter((preset) => preset.id !== presetId);
    delete state.presetStates?.[presetId];
    delete node.properties?.omnivoiceInstructionLayout?.layouts?.[presetId];
    if (state.selectedPresetId === presetId) {
        state.selectedPresetId = state.temporaryPreset?.id || OMNIVOICE_PRESET_ID;
    }
    await persistPresetLibraryNow(node, state);
    state.customPresets = Array.isArray(node.__omnivoiceInstructionLibraryPresets)
        ? node.__omnivoiceInstructionLibraryPresets.map((preset, index) => normalizePreset(preset, `custom_preset_${index + 1}`))
        : state.customPresets;
    applyStateToWidgets(node, state, ui);
    node.__omnivoiceInstructionState = state;
    closePresetEditor(ui);
    node.__omnivoiceInstructionRefresh?.();
}

function bindSharedInteractions(node, ui, refresh, panelWidget = null) {
    ui.previewInput.addEventListener("input", () => {
        ui.previewInputDirty = true;
        clearTimeout(ui.previewCommitTimer);
        if (hasTrailingSeparator(ui.previewInput.value)) {
            return;
        }
        ui.previewCommitTimer = setTimeout(() => {
            const state = node.__omnivoiceInstructionState || syncStateFromWidgets(node);
            applyPreviewTextToState(state, ui.previewInput.value);
            ui.previewInputDirty = false;
            applyStateToWidgets(node, state, ui);
            node.__omnivoiceInstructionState = state;
            refresh();
        }, 350);
    });

    ui.previewInput.addEventListener("blur", () => {
        if (!ui.previewInputDirty) {
            return;
        }
        clearTimeout(ui.previewCommitTimer);
        const state = node.__omnivoiceInstructionState || syncStateFromWidgets(node);
        applyPreviewTextToState(state, ui.previewInput.value);
        ui.previewInputDirty = false;
        applyStateToWidgets(node, state, ui);
        node.__omnivoiceInstructionState = state;
        refresh();
    });

    ui.presetSelect.addEventListener("change", () => {
        const state = node.__omnivoiceInstructionState || syncStateFromWidgets(node);
        state.selectedPresetId = ui.presetSelect.value || OMNIVOICE_PRESET_ID;
        const preset = getActivePreset(state);
        getPresetStateBucket(state, preset.id);
        applyStateToWidgets(node, state, ui);
        node.__omnivoiceInstructionState = state;
        refresh();
    });

    ui.editPresetButton.addEventListener("click", () => {
        openPresetEditor(node, ui);
    });

    ui.modalCancelButton.addEventListener("click", () => {
        closePresetEditor(ui);
    });

    ui.modalSaveButton.addEventListener("click", async () => {
        try {
            await savePresetEditor(node, ui);
        } catch (error) {
            console.error("Failed to save OmniVoice preset library:", error);
            window.alert("Could not save preset library.");
        }
    });

    ui.modalDeleteButton.addEventListener("click", async () => {
        try {
            await deletePresetEditor(node, ui);
        } catch (error) {
            console.error("Failed to delete OmniVoice preset from library:", error);
            window.alert("Could not delete preset from library.");
        }
    });

    ui.addSingleColumnButton.addEventListener("click", () => {
        ui.modalDraft.columns.push(createEmptySingleColumn(ui.modalDraft.columns.length));
        renderPresetEditor(ui, node.__omnivoiceInstructionState || syncStateFromWidgets(node));
    });

    ui.addSwitchColumnButton.addEventListener("click", () => {
        ui.modalDraft.columns.push(createEmptySwitchColumn(ui.modalDraft.columns.length));
        renderPresetEditor(ui, node.__omnivoiceInstructionState || syncStateFromWidgets(node));
    });

    ui.columnsEditor.addEventListener("input", (event) => {
        const target = event.target;
        if (!(target instanceof HTMLElement)) {
            return;
        }
        updateDraftFromInput(ui, target);
    });

    ui.columnsEditor.addEventListener("change", (event) => {
        const target = event.target;
        if (!(target instanceof HTMLElement)) {
            return;
        }
        updateDraftFromInput(ui, target);
    });

    ui.columnsEditor.addEventListener("click", (event) => {
        const target = event.target;
        if (!(target instanceof HTMLElement)) {
            return;
        }
        if (!target.dataset.action) {
            return;
        }
        handleDraftButtonAction(ui, target, node.__omnivoiceInstructionState || syncStateFromWidgets(node));
    });

    ui.modalBackdrop.addEventListener("click", (event) => {
        if (event.target === ui.modalBackdrop) {
            closePresetEditor(ui);
        }
    });

    ui.localeEnglishButton.addEventListener("click", () => {
        const state = node.__omnivoiceInstructionState || syncStateFromWidgets(node);
        state.output_language = "English";
        applyStateToWidgets(node, state, ui);
        node.__omnivoiceInstructionState = state;
        refresh();
    });

    ui.localeChineseButton.addEventListener("click", () => {
        const state = node.__omnivoiceInstructionState || syncStateFromWidgets(node);
        state.output_language = "Chinese";
        applyStateToWidgets(node, state, ui);
        node.__omnivoiceInstructionState = state;
        refresh();
    });

    ui.grid.addEventListener("pointerdown", (event) => {
        const chip = event.target instanceof Element ? event.target.closest(".ovib-chip") : null;
        if (!(chip instanceof HTMLElement)) {
            return;
        }

        const state = node.__omnivoiceInstructionState || syncStateFromWidgets(node);
        const preset = getActivePreset(state);
        const columnId = chip.dataset.columnId;
        const modeId = chip.dataset.modeId || "";
        const bucket = getPresetStateBucket(state, preset.id);
        const currentValue = normalizeWidgetValue(bucket.values[columnId]);
        let selectionChangedOnPointerDown = false;

        if (currentValue !== chip.dataset.value || (modeId && bucket.switchModes[columnId] !== modeId)) {
            setPresetColumnValue(state, preset, columnId, chip.dataset.value, false, modeId);
            applyStateToWidgets(node, state, ui);
            node.__omnivoiceInstructionState = state;
            refresh();
            selectionChangedOnPointerDown = true;
            ui.pendingClickToken = `${preset.id}:${columnId}:${modeId}:${chip.dataset.value}`;
        } else {
            ui.pendingClickToken = null;
        }

        beginColumnDrag(ui, event, columnId, {
            columnId,
            modeId,
            value: chip.dataset.value,
        }, selectionChangedOnPointerDown);
        if (ui.dragState) {
            ui.dragState.toggleOffOnPointerEnd = !selectionChangedOnPointerDown;
        }
    });

    ui.grid.addEventListener("click", (event) => {
        const target = event.target instanceof Element ? event.target.closest(".ovib-chip, .ovib-mode-chip") : null;
        if (!(target instanceof HTMLElement)) {
            return;
        }
        if (ui.suppressNextClick) {
            return;
        }

        const state = node.__omnivoiceInstructionState || syncStateFromWidgets(node);
        const preset = getActivePreset(state);

        if (target.classList.contains("ovib-mode-chip")) {
            setPresetMode(state, preset, target.dataset.columnId, target.dataset.modeId);
            applyStateToWidgets(node, state, ui);
            node.__omnivoiceInstructionState = state;
            refresh();
            return;
        }

        const token = `${preset.id}:${target.dataset.columnId}:${target.dataset.modeId || ""}:${target.dataset.value}`;
        if (ui.pendingClickToken === token) {
            ui.pendingClickToken = null;
            return;
        }
        setPresetColumnValue(
            state,
            preset,
            target.dataset.columnId,
            target.dataset.value,
            true,
            target.dataset.modeId || "",
        );
        applyStateToWidgets(node, state, ui);
        node.__omnivoiceInstructionState = state;
        refresh();
    });

    setupColumnDragging(node, ui);

    const resizeObserver = new ResizeObserver(() => {
        drawPath(ui, node.__omnivoiceInstructionState || syncStateFromWidgets(node));
        resizePanel(node, ui, panelWidget);
    });
    resizeObserver.observe(ui.body);

    const handleWindowResize = () => {
        drawPath(ui, node.__omnivoiceInstructionState || syncStateFromWidgets(node));
        resizePanel(node, ui, panelWidget);
    };
    window.addEventListener("resize", handleWindowResize);

    const handleWheel = (event) => {
        if (event.defaultPrevented || ui.dragState?.active) {
            return;
        }
        if (event.target instanceof Element && event.target.closest(".ovib-modal")) {
            return;
        }
        const wheelCallback = window.app?.canvas?._mousewheel_callback;
        if (typeof wheelCallback !== "function") {
            return;
        }
        event.preventDefault();
        event.stopPropagation();
        wheelCallback.call(window.app.canvas, event);
    };
    ui.panel.addEventListener("wheel", handleWheel, { passive: false });

    return {
        resizeObserver,
        handleWindowResize,
        handleWheel,
    };
}

function createBuilder(node) {
    if (!isBuilderNode(node) || typeof node.addDOMWidget !== "function") {
        return false;
    }
    if (node.__omnivoiceInstructionUi) {
        return true;
    }

    const relevantNames = new Set(["gender", "age", "pitch", "style", "accent", "dialect", "output_language", "instruct_text"]);
    for (const widget of node.widgets || []) {
        if (relevantNames.has(widget.name)) {
            hideWidget(widget);
        }
    }

    const ui = createPanelDom();
    ui.layoutAdapter = createNodeLayoutAdapter(node);
    cachePresetLibraryOnNode(node, getCachedPresetLibrary());
    node.__omnivoiceInstructionLibraryReady = false;
    node.__omnivoiceInstructionAppliedPanelHeight = PANEL_WIDGET_MIN_HEIGHT;
    node.__omnivoiceInstructionBasePanelHeight = PANEL_WIDGET_MIN_HEIGHT;
    node.__omnivoiceInstructionLastY = 0;

    const panelWidget = node.addDOMWidget("omnivoice_instruction_builder_panel", "div", ui.panel, {
        serialize: false,
        hideOnZoom: true,
        getMinHeight() {
            return PANEL_WIDGET_MIN_HEIGHT;
        },
        getHeight() {
            return PANEL_WIDGET_MIN_HEIGHT;
        },
    });
    panelWidget.computeSize = (inputWidth) => {
        const width = Array.isArray(inputWidth) ? inputWidth[0] : inputWidth;
        return [Math.max(PANEL_MIN_WIDTH, width || PANEL_MIN_WIDTH), PANEL_WIDGET_MIN_HEIGHT];
    };
    panelWidget.getHeight = () => PANEL_WIDGET_MIN_HEIGHT;
    panelWidget.options = panelWidget.options || {};
    panelWidget.options.minNodeSize = [PANEL_MIN_WIDTH, getRequiredNodeMinHeight(node)];
    panelWidget.draw = function (_ctx, currentNode, _widgetWidth, y) {
        const activeNode = currentNode || node;
        this.last_y = y;
        activeNode.__omnivoiceInstructionLastY = y;
        panelWidget.options.minNodeSize = [
            PANEL_MIN_WIDTH,
            getRequiredNodeMinHeight(activeNode),
        ];
        updateBasePanelHeight(activeNode, panelWidget);
        resizePanel(activeNode, ui, panelWidget);
    };
    setWidgetHeightSafe(panelWidget, PANEL_WIDGET_MIN_HEIGHT);

    const refresh = () => {
        const state = syncStateFromWidgets(node);
        node.__omnivoiceInstructionState = state;
        renderState(node, ui, state);
        applyColumnLayout(ui);
        drawPath(ui, state);
        resizePanel(node, ui, panelWidget);
        persistCurrentLayout(ui, state);
        schedulePresetLibraryPersist(node, state);
        requestAnimationFrame(() => {
            drawPath(ui, node.__omnivoiceInstructionState || state);
        });
    };

    const bound = bindSharedInteractions(node, ui, refresh, panelWidget);

    node.__omnivoiceInstructionUi = ui;
    node.__omnivoiceInstructionPanelWidget = panelWidget;
    node.__omnivoiceInstructionRefresh = refresh;
    node.__omnivoiceInstructionCleanup = () => {
        clearTimeout(ui.previewCommitTimer);
        clearTimeout(node.__omnivoiceInstructionLibraryPersistTimer);
        bound.resizeObserver.disconnect();
        window.removeEventListener("resize", bound.handleWindowResize);
        ui.panel.removeEventListener("wheel", bound.handleWheel);
        if (ui.handlePointerMove) {
            window.removeEventListener("pointermove", ui.handlePointerMove);
        }
        if (ui.handlePointerEnd) {
            window.removeEventListener("pointerup", ui.handlePointerEnd);
            window.removeEventListener("pointercancel", ui.handlePointerEnd);
        }
    };

    if (typeof node.setSize === "function") {
        node.setSize([
            Math.max(Number(node.size?.[0] || 0), PANEL_MIN_WIDTH),
            Math.max(Number(node.size?.[1] || 0), PANEL_MIN_HEIGHT),
        ]);
    }

    updateBasePanelHeight(node, panelWidget, true);
    refresh();
    loadPresetLibraryIntoNode(node).then(() => {
        node.__omnivoiceInstructionRefresh?.();
    });
    return true;
}

function createPrototypeState(initialState = {}) {
    const state = createEmptyBuilderState();
    state.output_language = normalizeWidgetValue(initialState.output_language) || "English";
    const bucket = getPresetStateBucket(state, OMNIVOICE_PRESET_ID);
    bucket.values.gender = normalizeWidgetValue(initialState.gender);
    bucket.values.age = normalizeWidgetValue(initialState.age);
    bucket.values.pitch = normalizeWidgetValue(initialState.pitch);
    bucket.values.style = normalizeWidgetValue(initialState.style);
    bucket.values.language = normalizeWidgetValue(initialState.accent || initialState.dialect);
    bucket.switchModes.language = normalizeWidgetValue(initialState.dialect) ? "dialect" : "accent";
    return state;
}

function createPrototypeController(container, initialState = {}, initialLayout = {}) {
    const ui = createPanelDom();
    ui.layoutAdapter = createMemoryLayoutAdapter(initialLayout, OMNIVOICE_PRESET_ID);

    const prototypeNode = {
        comfyClass: NODE_CLASS,
        widgets: [],
        properties: {},
        graph: {
            setDirtyCanvas() {},
        },
    };
    prototypeNode.__omnivoiceInstructionState = createPrototypeState(initialState);

    const refresh = () => {
        const state = prototypeNode.__omnivoiceInstructionState;
        renderState(prototypeNode, ui, state);
        applyColumnLayout(ui);
        drawPath(ui, state);
        requestAnimationFrame(() => {
            drawPath(ui, prototypeNode.__omnivoiceInstructionState || state);
        });
    };
    prototypeNode.__omnivoiceInstructionRefresh = refresh;

    const bound = bindSharedInteractions(prototypeNode, ui, refresh, null);

    container.style.height = `${PANEL_WIDGET_MIN_HEIGHT}px`;
    container.style.minHeight = `${PANEL_WIDGET_MIN_HEIGHT}px`;
    container.style.maxHeight = `${PANEL_WIDGET_MIN_HEIGHT}px`;
    container.style.display = "block";
    container.innerHTML = "";
    container.appendChild(ui.panel);

    prototypeNode.__omnivoiceInstructionUi = ui;
    refresh();

    return {
        ui,
        state: prototypeNode.__omnivoiceInstructionState,
        refresh,
        destroy() {
            clearTimeout(ui.previewCommitTimer);
            bound.resizeObserver.disconnect();
            window.removeEventListener("resize", bound.handleWindowResize);
            ui.panel.removeEventListener("wheel", bound.handleWheel);
            if (ui.handlePointerMove) {
                window.removeEventListener("pointermove", ui.handlePointerMove);
            }
            if (ui.handlePointerEnd) {
                window.removeEventListener("pointerup", ui.handlePointerEnd);
                window.removeEventListener("pointercancel", ui.handlePointerEnd);
            }
        },
    };
}

export function getOmniVoiceInstructionBuilderDefaults() {
    return {
        nodeWidth: PANEL_MIN_WIDTH,
        nodeHeight: PANEL_MIN_HEIGHT,
        widgetHeight: PANEL_WIDGET_MIN_HEIGHT,
    };
}

export function installOmniVoiceInstructionBuilderExtension(app) {
    app.registerExtension({
        name: "tts-audio-suite.omnivoice-instruction-builder",
        beforeRegisterNodeDef(nodeType, nodeData) {
            if (nodeData.name !== NODE_CLASS) {
                return;
            }

            const originalNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = originalNodeCreated ? originalNodeCreated.apply(this, arguments) : undefined;
                createBuilder(this);
                return result;
            };

            const originalOnConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                const result = originalOnConfigure ? originalOnConfigure.apply(this, arguments) : undefined;
                this.properties = this.properties || {};
                if (info?.properties?.omnivoiceInstructionLayout) {
                    this.properties.omnivoiceInstructionLayout = info.properties.omnivoiceInstructionLayout;
                }
                if (info?.properties?.omnivoiceInstructionPresetState) {
                    this.properties.omnivoiceInstructionPresetState = info.properties.omnivoiceInstructionPresetState;
                }
                createBuilder(this);
                this.__omnivoiceInstructionRefresh?.();
                return result;
            };

            const originalOnResize = nodeType.prototype.onResize;
            nodeType.prototype.onResize = function () {
                const result = originalOnResize ? originalOnResize.apply(this, arguments) : undefined;
                this.__omnivoiceInstructionRefresh?.();
                return result;
            };

            const originalOnSerialize = nodeType.prototype.onSerialize;
            nodeType.prototype.onSerialize = function (info) {
                if (originalOnSerialize) {
                    originalOnSerialize.apply(this, arguments);
                }
                info.properties = info.properties || {};
                if (this.properties?.omnivoiceInstructionLayout) {
                    info.properties.omnivoiceInstructionLayout = this.properties.omnivoiceInstructionLayout;
                }
                if (this.properties?.omnivoiceInstructionPresetState) {
                    info.properties.omnivoiceInstructionPresetState = this.properties.omnivoiceInstructionPresetState;
                }
            };

            const originalOnRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function () {
                this.__omnivoiceInstructionCleanup?.();
                return originalOnRemoved ? originalOnRemoved.apply(this, arguments) : undefined;
            };
        },
        nodeCreated(node) {
            createBuilder(node);
            node.__omnivoiceInstructionRefresh?.();
        },
    });
}

export function mountOmniVoiceInstructionBuilderPrototype(container, options = {}) {
    return createPrototypeController(
        container,
        options.initialState || {},
        options.initialLayout || {},
    );
}
