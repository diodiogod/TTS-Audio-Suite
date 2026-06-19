const NODE_CLASS = "OmniVoiceInstructionBuilderNode";
const PANEL_MIN_WIDTH = 900;
const PANEL_MIN_HEIGHT = 620;
const PANEL_WIDGET_MIN_HEIGHT = 520;
const COLUMN_IDS = ["gender", "age", "pitch", "style", "language"];
const RESTING_OFFSET_LIMIT = 26;
const DRAG_START_THRESHOLD = 4;

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
        title: "OmniVoice Instruction Builder",
        previewLabel: "OUT >",
        emptyPreview: "Select attributes to build an OmniVoice instruct string",
        gender: "Gender",
        age: "Age",
        pitch: "Pitch",
        style: "Style",
        language: "Language",
        accent: "Accent",
        dialect: "Dialect",
        localeEn: "EN",
        localeZh: "中文",
    },
    zh: {
        title: "OmniVoice 指令构建器",
        previewLabel: "输出 >",
        emptyPreview: "选择属性以构建 OmniVoice 指令字符串",
        gender: "性别",
        age: "年龄",
        pitch: "音调",
        style: "风格",
        language: "语言",
        accent: "口音",
        dialect: "方言",
        localeEn: "EN",
        localeZh: "中文",
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

const BUILDER_DEFS = [
    {
        id: "gender",
        title: "Gender",
        column: 0,
        items: [
            { label: "Male", value: "male", title: "male" },
            { label: "Female", value: "female", title: "female" },
        ],
    },
    {
        id: "age",
        title: "Age",
        column: 1,
        items: [
            { label: "Child", value: "child", title: "child" },
            { label: "Teen", value: "teenager", title: "teenager" },
            { label: "Young Adult", value: "young adult", title: "young adult" },
            { label: "Middle-aged", value: "middle-aged", title: "middle-aged" },
            { label: "Elderly", value: "elderly", title: "elderly" },
        ],
    },
    {
        id: "pitch",
        title: "Pitch",
        column: 2,
        items: [
            { label: "Very Low", value: "very low pitch", title: "very low pitch" },
            { label: "Low", value: "low pitch", title: "low pitch" },
            { label: "Moderate", value: "moderate pitch", title: "moderate pitch" },
            { label: "High", value: "high pitch", title: "high pitch" },
            { label: "Very High", value: "very high pitch", title: "very high pitch" },
        ],
    },
    {
        id: "style",
        title: "Style",
        column: 3,
        items: [
            { label: "Whisper", value: "whisper", title: "whisper" },
        ],
    },
];

const ACCENT_ITEMS = [
    { label: "US", value: "american accent", title: "american accent" },
    { label: "UK", value: "british accent", title: "british accent" },
    { label: "AU", value: "australian accent", title: "australian accent" },
    { label: "CA", value: "canadian accent", title: "canadian accent" },
    { label: "IN", value: "indian accent", title: "indian accent" },
    { label: "CN", value: "chinese accent", title: "chinese accent" },
    { label: "KR", value: "korean accent", title: "korean accent" },
    { label: "JP", value: "japanese accent", title: "japanese accent" },
    { label: "PT", value: "portuguese accent", title: "portuguese accent" },
    { label: "RU", value: "russian accent", title: "russian accent" },
];

const DIALECT_ITEMS = [
    { label: "Henan", value: "河南话", title: "河南话" },
    { label: "Shaanxi", value: "陕西话", title: "陕西话" },
    { label: "Sichuan", value: "四川话", title: "四川话" },
    { label: "Guizhou", value: "贵州话", title: "贵州话" },
    { label: "Yunnan", value: "云南话", title: "云南话" },
    { label: "Guilin", value: "桂林话", title: "桂林话" },
    { label: "Jinan", value: "济南话", title: "济南话" },
    { label: "Shijiazhuang", value: "石家庄话", title: "石家庄话" },
    { label: "Gansu", value: "甘肃话", title: "甘肃话" },
    { label: "Ningxia", value: "宁夏话", title: "宁夏话" },
    { label: "Qingdao", value: "青岛话", title: "青岛话" },
    { label: "Northeast", value: "东北话", title: "东北话" },
];

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

function normalizeWidgetValue(value) {
    const text = String(value ?? "").trim();
    return !text || text === "None" ? "" : text;
}

function setCategoryValueOnState(state, category, value, toggle = true) {
    if (!state || !category) {
        return state;
    }
    const currentValue = normalizeWidgetValue(state[category]);
    if (category === "accent") {
        state.dialect = "";
    } else if (category === "dialect") {
        state.accent = "";
    }
    state[category] = toggle && currentValue === value ? "" : normalizeWidgetValue(value);
    return state;
}

function getOutputLocale(state) {
    return normalizeWidgetValue(state?.output_language) === "Chinese" ? "zh" : "en";
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

function getColumnOffset(ui, columnId) {
    return normalizeColumnOffset(ui.columnOffsets?.[columnId]);
}

function setColumnOffset(ui, columnId, offset) {
    ui.columnOffsets[columnId] = normalizeColumnOffset(offset);
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

function buildPreviewFromState(state, columnOrder = COLUMN_IDS) {
    const locale = getOutputLocale(state);
    const accent = normalizeWidgetValue(state.accent);
    let dialect = normalizeWidgetValue(state.dialect);
    if (accent && dialect) {
        dialect = "";
    }

    const useChinese = Boolean(dialect) && !accent;
    const ordered = [];
    for (const columnId of columnOrder) {
        if (columnId === "language") {
            const languageValue = useChinese ? dialect : accent;
            if (languageValue) {
                ordered.push(languageValue);
            }
            continue;
        }
        const value = normalizeWidgetValue(state[columnId]);
        if (value) {
            ordered.push(value);
        }
    }

    if (!ordered.length) {
        return "";
    }

    if (locale === "zh") {
        return ordered.map((item) => EN_TO_ZH[item] || item).join("，");
    }

    return ordered.join(", ");
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
            color: #e3e2e6;
            font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            box-sizing: border-box;
            overflow: visible;
            padding: 4px 2px 0 2px;
        }
        .ovib-shell {
            border: 1px solid rgba(152, 203, 255, 0.12);
            border-radius: 12px;
            overflow: hidden;
            background: #17191d;
            box-shadow: 0 14px 38px rgba(0, 0, 0, 0.34);
        }
        .ovib-topline {
            height: 2px;
            background: linear-gradient(90deg, rgba(152, 203, 255, 0.95) 0%, rgba(0, 163, 255, 0.86) 60%, rgba(111, 251, 190, 0.72) 100%);
        }
        .ovib-header {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 13px 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            background: rgba(23, 25, 29, 0.96);
        }
        .ovib-header-spacer {
            flex: 1 1 auto;
        }
        .ovib-header-icon {
            position: relative;
            width: 24px;
            height: 16px;
            flex: 0 0 auto;
        }
        .ovib-header-icon::before,
        .ovib-header-icon::after {
            content: "";
            position: absolute;
            border-radius: 999px;
            background: linear-gradient(180deg, #98cbff 0%, #68c6ff 100%);
            box-shadow: 0 0 10px rgba(152, 203, 255, 0.5);
        }
        .ovib-header-icon::before {
            left: 1px;
            top: 1px;
            width: 4px;
            height: 10px;
        }
        .ovib-header-icon::after {
            right: 1px;
            top: 5px;
            width: 4px;
            height: 10px;
        }
        .ovib-header-icon span {
            position: absolute;
            inset: 0;
            display: block;
        }
        .ovib-header-icon span::before,
        .ovib-header-icon span::after {
            content: "";
            position: absolute;
            width: 4px;
            height: 4px;
            border-radius: 50%;
            background: #98cbff;
            box-shadow: 0 0 8px rgba(152, 203, 255, 0.58);
        }
        .ovib-header-icon span::before {
            left: 0;
            top: 0;
        }
        .ovib-header-icon span::after {
            right: 0;
            bottom: 0;
        }
        .ovib-header-title {
            margin: 0;
            color: #f1f5f9;
            font-size: 14px;
            font-weight: 600;
            letter-spacing: 0.01em;
        }
        .ovib-locale-switch {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            width: 112px;
            flex: 0 0 auto;
        }
        .ovib-locale-chip {
            min-height: 28px;
            padding: 6px 8px;
            border-radius: 9px;
            border: 1px solid rgba(121, 134, 155, 0.34);
            background: rgba(34, 37, 43, 0.92);
            color: #9fb0c6;
            font-size: 10px;
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
            min-height: 430px;
            padding: 20px 18px 16px 18px;
            background:
                linear-gradient(rgba(255, 255, 255, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.03) 1px, transparent 1px),
                radial-gradient(circle at top left, rgba(152, 203, 255, 0.08) 0%, transparent 34%),
                #17191d;
            background-size: 24px 24px, 24px 24px, auto, auto;
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
            grid-template-columns: 1fr 1.1fr 1.1fr 0.9fr 1.15fr;
            gap: 16px;
            align-items: start;
        }
        .ovib-column {
            display: flex;
            flex-direction: column;
            gap: 10px;
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
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .ovib-subtitle {
            font-size: 10px;
            color: #79869b;
            margin-bottom: 2px;
        }
        .ovib-chip-list {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .ovib-chip {
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 38px;
            width: min(100%, 172px);
            padding: 9px 12px;
            margin: 0 auto;
            border-radius: 10px;
            border: 1px solid rgba(121, 134, 155, 0.38);
            background: rgba(41, 43, 49, 0.94);
            color: #d7dfeb;
            font-size: 11px;
            font-weight: 600;
            text-align: center;
            line-height: 1.2;
            letter-spacing: 0.01em;
            user-select: none;
            box-sizing: border-box;
            cursor: pointer;
            transition: border-color 0.18s ease, background 0.18s ease, color 0.18s ease, box-shadow 0.18s ease, opacity 0.18s ease, transform 0.18s ease;
        }
        .ovib-chip:hover:not(.is-disabled) {
            border-color: rgba(152, 203, 255, 0.44);
            background: rgba(50, 54, 62, 0.96);
            color: #f8fbff;
            transform: translateY(-1px);
        }
        .ovib-chip.is-active {
            border-color: rgba(152, 203, 255, 0.98);
            background: linear-gradient(180deg, rgba(47, 68, 92, 0.78) 0%, rgba(35, 52, 74, 0.68) 100%);
            color: #9ed0ff;
            box-shadow: 0 0 0 1px rgba(152, 203, 255, 0.18) inset, 0 0 16px rgba(64, 167, 255, 0.26);
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
            width: 5px;
            height: 5px;
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
            gap: 12px;
        }
        .ovib-mode-switch {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-bottom: 4px;
        }
        .ovib-mode-chip {
            min-height: 30px;
            padding: 7px 8px;
            border-radius: 9px;
            border: 1px solid rgba(121, 134, 155, 0.34);
            background: rgba(34, 37, 43, 0.92);
            color: #9fb0c6;
            font-size: 10px;
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
            gap: 8px;
            transition: opacity 0.2s ease;
        }
        .ovib-language-group .ovib-chip-list {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 10px;
        }
        .ovib-language-group.is-hidden {
            display: none;
        }
        .ovib-language-group.is-dimmed {
            opacity: 0.24;
        }
        .ovib-preview {
            display: flex;
            align-items: center;
            gap: 10px;
            border-top: 1px solid rgba(255, 255, 255, 0.06);
            background: #101215;
            padding: 12px 16px;
            min-height: 44px;
            box-sizing: border-box;
        }
        .ovib-preview-label {
            color: #8492a7;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            white-space: nowrap;
        }
        .ovib-preview-value {
            color: #98cbff;
            font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
            font-size: 11px;
            line-height: 1.35;
            word-break: break-word;
        }
        .ovib-preview-value.is-empty {
            color: #64748b;
        }
        .ovib-path {
            fill: none;
            stroke: url(#ovibPathGradient);
            stroke-width: 3;
            stroke-linecap: round;
            stroke-linejoin: round;
            filter: drop-shadow(0 0 5px rgba(152, 203, 255, 0.58));
            opacity: 0.92;
        }
    `;
    panel.appendChild(style);
}

function createChip(item, categoryId) {
    const button = createEl("button", "ovib-chip", item.label);
    button.type = "button";
    button.dataset.category = categoryId;
    button.dataset.value = item.value;
    button.dataset.labelEn = item.label;
    button.dataset.labelZh = item.zhLabel || ZH_UI_LABELS[item.value] || item.label;
    button.dataset.titleEn = item.title || item.value;
    button.dataset.titleZh = item.zhTitle || item.zhLabel || ZH_UI_LABELS[item.value] || item.title || item.value;
    button.title = button.dataset.titleEn;
    return button;
}

function createPanelDom() {
    const panel = createEl("div", "omnivoice-instruction-builder-panel");
    ensureStyles(panel);

    const shell = createEl("div", "ovib-shell");
    panel.appendChild(shell);

    shell.appendChild(createEl("div", "ovib-topline"));

    const header = createEl("div", "ovib-header");
    const icon = createEl("div", "ovib-header-icon");
    icon.appendChild(document.createElement("span"));
    header.appendChild(icon);
    const headerTitle = createEl("h1", "ovib-header-title", UI_TEXT.en.title);
    header.appendChild(headerTitle);
    header.appendChild(createEl("div", "ovib-header-spacer"));
    const localeSwitch = createEl("div", "ovib-locale-switch");
    const localeEnglishButton = createEl("button", "ovib-locale-chip is-active", UI_TEXT.en.localeEn);
    localeEnglishButton.type = "button";
    localeEnglishButton.dataset.locale = "English";
    const localeChineseButton = createEl("button", "ovib-locale-chip", UI_TEXT.en.localeZh);
    localeChineseButton.type = "button";
    localeChineseButton.dataset.locale = "Chinese";
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

    const chipButtons = new Map();
    const columns = new Map();
    const columnTitles = new Map();
    for (const def of BUILDER_DEFS) {
        const column = createEl("div", "ovib-column");
        column.dataset.category = def.id;
        columns.set(def.id, column);
        const columnTitle = createEl("div", "ovib-column-title", def.title);
        column.appendChild(columnTitle);
        columnTitles.set(def.id, columnTitle);
        const list = createEl("div", "ovib-chip-list");
        for (const item of def.items) {
            const chip = createChip(item, def.id);
            chipButtons.set(`${def.id}:${item.value}`, chip);
            list.appendChild(chip);
        }
        column.appendChild(list);
        grid.appendChild(column);
    }

    const languageColumn = createEl("div", "ovib-column ovib-language-column");
    languageColumn.dataset.category = "language";
    columns.set("language", languageColumn);
    const languageColumnTitle = createEl("div", "ovib-column-title", UI_TEXT.en.language);
    languageColumn.appendChild(languageColumnTitle);
    columnTitles.set("language", languageColumnTitle);

    const languageStack = createEl("div", "ovib-language-stack");
    const modeSwitch = createEl("div", "ovib-mode-switch");
    const accentModeButton = createEl("button", "ovib-mode-chip", UI_TEXT.en.accent);
    accentModeButton.type = "button";
    accentModeButton.dataset.mode = "accent";
    const dialectModeButton = createEl("button", "ovib-mode-chip", UI_TEXT.en.dialect);
    dialectModeButton.type = "button";
    dialectModeButton.dataset.mode = "dialect";
    modeSwitch.append(accentModeButton, dialectModeButton);
    languageStack.appendChild(modeSwitch);

    const accentGroup = createEl("div", "ovib-language-group");
    accentGroup.dataset.group = "accent";
    const accentSubtitle = createEl("div", "ovib-subtitle", UI_TEXT.en.accent);
    accentGroup.appendChild(accentSubtitle);
    const accentList = createEl("div", "ovib-chip-list");
    for (const item of ACCENT_ITEMS) {
        const chip = createChip(item, "accent");
        chipButtons.set(`accent:${item.value}`, chip);
        accentList.appendChild(chip);
    }
    accentGroup.appendChild(accentList);
    languageStack.appendChild(accentGroup);

    const dialectGroup = createEl("div", "ovib-language-group");
    dialectGroup.dataset.group = "dialect";
    const dialectSubtitle = createEl("div", "ovib-subtitle", UI_TEXT.en.dialect);
    dialectGroup.appendChild(dialectSubtitle);
    const dialectList = createEl("div", "ovib-chip-list");
    for (const item of DIALECT_ITEMS) {
        const chip = createChip(item, "dialect");
        chipButtons.set(`dialect:${item.value}`, chip);
        dialectList.appendChild(chip);
    }
    dialectGroup.appendChild(dialectList);
    languageStack.appendChild(dialectGroup);

    languageColumn.appendChild(languageStack);
    grid.appendChild(languageColumn);
    shell.appendChild(body);

    const preview = createEl("div", "ovib-preview");
    const previewLabel = createEl("span", "ovib-preview-label", UI_TEXT.en.previewLabel);
    preview.appendChild(previewLabel);
    const previewValue = createEl("span", "ovib-preview-value is-empty", UI_TEXT.en.emptyPreview);
    preview.appendChild(previewValue);
    shell.appendChild(preview);

    return {
        panel,
        body,
        grid,
        svg,
        path: svg.querySelector(".ovib-path"),
        headerTitle,
        previewLabel,
        previewValue,
        chipButtons,
        columns,
        columnTitles,
        accentSubtitle,
        dialectSubtitle,
        accentGroup,
        dialectGroup,
        accentModeButton,
        dialectModeButton,
        localeEnglishButton,
        localeChineseButton,
        modeSwitch,
    };
}

function syncStateFromWidgets(node) {
    return {
        gender: normalizeWidgetValue(findWidgetByName(node, "gender")?.value),
        age: normalizeWidgetValue(findWidgetByName(node, "age")?.value),
        pitch: normalizeWidgetValue(findWidgetByName(node, "pitch")?.value),
        style: normalizeWidgetValue(findWidgetByName(node, "style")?.value),
        accent: normalizeWidgetValue(findWidgetByName(node, "accent")?.value),
        dialect: normalizeWidgetValue(findWidgetByName(node, "dialect")?.value),
        output_language: normalizeWidgetValue(findWidgetByName(node, "output_language")?.value) || "English",
    };
}

function applyStateToWidgets(node, state) {
    setWidgetValue(findWidgetByName(node, "gender"), state.gender);
    setWidgetValue(findWidgetByName(node, "age"), state.age);
    setWidgetValue(findWidgetByName(node, "pitch"), state.pitch);
    setWidgetValue(findWidgetByName(node, "style"), state.style);
    setWidgetValue(findWidgetByName(node, "accent"), state.accent);
    setWidgetValue(findWidgetByName(node, "dialect"), state.dialect);
    setWidgetValue(findWidgetByName(node, "output_language"), state.output_language || "English");
}

function applyLocalizedUiText(ui, state) {
    const locale = getOutputLocale(state);
    const text = UI_TEXT[locale];
    ui.headerTitle.textContent = text.title;
    ui.previewLabel.textContent = text.previewLabel;
    ui.columnTitles.get("gender").textContent = text.gender;
    ui.columnTitles.get("age").textContent = text.age;
    ui.columnTitles.get("pitch").textContent = text.pitch;
    ui.columnTitles.get("style").textContent = text.style;
    ui.columnTitles.get("language").textContent = text.language;
    ui.accentSubtitle.textContent = text.accent;
    ui.dialectSubtitle.textContent = text.dialect;
    ui.accentModeButton.textContent = text.accent;
    ui.dialectModeButton.textContent = text.dialect;
    ui.localeEnglishButton.textContent = text.localeEn;
    ui.localeChineseButton.textContent = text.localeZh;
    ui.localeEnglishButton.classList.toggle("is-active", locale === "en");
    ui.localeChineseButton.classList.toggle("is-active", locale === "zh");

    for (const button of ui.chipButtons.values()) {
        button.textContent = locale === "zh" ? button.dataset.labelZh : button.dataset.labelEn;
        button.title = locale === "zh" ? button.dataset.titleZh : button.dataset.titleEn;
    }
}

function renderState(node, ui, state) {
    applyLocalizedUiText(ui, state);
    for (const button of ui.chipButtons.values()) {
        const category = button.dataset.category;
        const active = normalizeWidgetValue(state[category]) === button.dataset.value;
        button.classList.toggle("is-active", active);
        button.classList.remove("is-disabled");
    }

    for (const columnId of COLUMN_IDS) {
        const column = ui.columns.get(columnId);
        if (!column) {
            continue;
        }
        const isDraggable = columnId === "language"
            ? Boolean(normalizeWidgetValue(state.accent) || normalizeWidgetValue(state.dialect))
            : Boolean(normalizeWidgetValue(state[columnId]));
        column.classList.toggle("is-draggable", isDraggable);
        if (!isDraggable && !ui.dragState?.active) {
            setColumnOffset(ui, columnId, { x: 0, y: 0 });
        }
    }

    if (!ui.languageMode) {
        ui.languageMode = "accent";
    }
    const languageMode = ui.languageMode || "accent";
    ui.languageMode = languageMode;
    ui.accentGroup.classList.remove("is-dimmed");
    ui.dialectGroup.classList.remove("is-dimmed");
    ui.accentGroup.classList.toggle("is-hidden", languageMode !== "accent");
    ui.dialectGroup.classList.toggle("is-hidden", languageMode !== "dialect");
    ui.accentModeButton.classList.toggle("is-active", languageMode === "accent");
    ui.dialectModeButton.classList.toggle("is-active", languageMode === "dialect");

    const preview = buildPreviewFromState(state, ui.columnOrder);
    if (preview) {
        ui.previewValue.textContent = preview;
        ui.previewValue.classList.remove("is-empty");
    } else {
        ui.previewValue.textContent = UI_TEXT[getOutputLocale(state)].emptyPreview;
        ui.previewValue.classList.add("is-empty");
    }
}

function drawPath(ui, state) {
    const bodyRect = ui.body.getBoundingClientRect();
    const svgWidth = Math.max(1, Math.round(bodyRect.width));
    const svgHeight = Math.max(1, Math.round(bodyRect.height));
    ui.svg.setAttribute("viewBox", `0 0 ${svgWidth} ${svgHeight}`);
    ui.svg.setAttribute("width", String(svgWidth));
    ui.svg.setAttribute("height", String(svgHeight));
    ui.svg.setAttribute("preserveAspectRatio", "none");
    const selected = [];
    for (const columnId of ui.columnOrder) {
        if (columnId === "language") {
            const languageCategory = normalizeWidgetValue(state.dialect)
                ? "dialect"
                : normalizeWidgetValue(state.accent)
                    ? "accent"
                    : null;
            if (!languageCategory) {
                continue;
            }
            const languageValue = normalizeWidgetValue(state[languageCategory]);
            if (languageValue) {
                const button = ui.chipButtons.get(`${languageCategory}:${languageValue}`);
                if (button && button.getClientRects().length > 0) {
                    selected.push(button);
                } else {
                    selected.push(languageCategory === "dialect" ? ui.dialectModeButton : ui.accentModeButton);
                }
            }
            continue;
        }
        const value = normalizeWidgetValue(state[columnId]);
        if (!value) {
            continue;
        }
        const button = ui.chipButtons.get(`${columnId}:${value}`);
        if (button) {
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
        const handle = Math.max(28, Math.min(64, Math.abs(dx) * 0.35));
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
    }
}

function applyColumnLayout(ui) {
    for (const [columnId, column] of ui.columns.entries()) {
        const slotIndex = ui.columnOrder.indexOf(columnId);
        if (slotIndex >= 0) {
            column.style.order = String(slotIndex);
        }
        const offset = getColumnOffset(ui, columnId);
        column.style.transform = `translate(${offset.x}px, ${offset.y}px)`;
        column.classList.toggle("is-dragging", ui.dragState?.active && ui.dragState.columnId === columnId);
    }
}

function getColumnCenterX(column) {
    const rect = column.getBoundingClientRect();
    return rect.left + rect.width / 2;
}

function clampColumnOffsetY(ui, columnId, desiredY) {
    const column = ui.columns.get(columnId);
    if (!column) {
        return desiredY;
    }
    const bodyRect = ui.body.getBoundingClientRect();
    const rect = column.getBoundingClientRect();
    const currentOffset = getColumnOffset(ui, columnId);
    const naturalTop = rect.top - currentOffset.y;
    const naturalBottom = rect.bottom - currentOffset.y;
    const padding = 8;
    const minY = bodyRect.top + padding - naturalTop;
    const maxY = bodyRect.bottom - padding - naturalBottom;
    return Math.max(minY, Math.min(maxY, desiredY));
}

function queueSuppressNextClick(ui) {
    ui.suppressNextClick = true;
    setTimeout(() => {
        ui.suppressNextClick = false;
    }, 0);
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
    setColumnOffset(ui, sourceId, {
        x: offset.x + (oldCenter - newCenter),
        y: offset.y,
    });
    applyColumnLayout(ui);
    return true;
}

function beginColumnDrag(node, ui, event, columnId, selectionTarget = null, selectionChangedOnPointerDown = false) {
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
    };
    const chip = event.currentTarget;
    chip.setPointerCapture?.(event.pointerId);
    event.preventDefault();
}

function applyDragSelection(node, state, selectionTarget) {
    if (!selectionTarget?.category) {
        return state;
    }
    setCategoryValueOnState(state, selectionTarget.category, selectionTarget.value, false);
    if (node.widgets) {
        applyStateToWidgets(node, state);
    }
    node.__omnivoiceInstructionState = state;
    return state;
}

function setupColumnDragging(node, ui) {
    const handlePointerMove = (event) => {
        const dragState = ui.dragState;
        if (!dragState?.active || event.pointerId !== dragState.pointerId) {
            return;
        }

        const deltaX = event.clientX - dragState.startX;
        const deltaY = event.clientY - dragState.startY;
        if (!dragState.moved && Math.hypot(deltaX, deltaY) < DRAG_START_THRESHOLD) {
            return;
        }

        dragState.moved = true;
        const columnId = dragState.columnId;
        const column = ui.columns.get(columnId);
        if (!column) {
            return;
        }

        const nextOffsetX = dragState.baseOffsetX + deltaX;
        const desiredOffsetY = dragState.baseOffsetY + deltaY;
        const nextOffsetY = clampColumnOffsetY(ui, columnId, desiredOffsetY);
        setColumnOffset(ui, columnId, {
            x: nextOffsetX,
            y: nextOffsetY,
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
        state = applyDragSelection(node, state, dragState.selectionTarget);
        const stillSelected = columnId === "language"
            ? Boolean(normalizeWidgetValue(state.accent) || normalizeWidgetValue(state.dialect))
            : Boolean(normalizeWidgetValue(state[columnId]));
        const currentOffset = getColumnOffset(ui, columnId);
        const clampedOffset = stillSelected
            ? {
                x: Math.max(-RESTING_OFFSET_LIMIT, Math.min(RESTING_OFFSET_LIMIT, currentOffset.x)),
                y: clampColumnOffsetY(ui, columnId, currentOffset.y),
            }
            : { x: 0, y: 0 };
        setColumnOffset(ui, columnId, clampedOffset);
        ui.dragState = null;
        applyColumnLayout(ui);
        drawPath(ui, state);
        if (dragState.moved || dragState.selectionChangedOnPointerDown) {
            queueSuppressNextClick(ui);
        }
        persistLayoutState(node, ui);
        node.__omnivoiceInstructionRefresh?.();
    };

    ui.handlePointerMove = handlePointerMove;
    ui.handlePointerEnd = handlePointerEnd;
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerEnd);
    window.addEventListener("pointercancel", handlePointerEnd);
}

function loadLayoutState(node) {
    const stored = node.properties?.omnivoiceInstructionLayout;
    const order = Array.isArray(stored?.columnOrder)
        ? stored.columnOrder.filter((columnId) => COLUMN_IDS.includes(columnId))
        : null;
    const completeOrder = order && order.length === COLUMN_IDS.length
        ? order
        : [...COLUMN_IDS];
    const offsets = {};
    for (const columnId of COLUMN_IDS) {
        offsets[columnId] = normalizeColumnOffset(stored?.columnOffsets?.[columnId]);
    }
    return {
        columnOrder: completeOrder,
        columnOffsets: offsets,
    };
}

function persistLayoutState(node, ui) {
    node.properties = node.properties || {};
    node.properties.omnivoiceInstructionLayout = {
        columnOrder: [...ui.columnOrder],
        columnOffsets: Object.fromEntries(COLUMN_IDS.map((columnId) => [
            columnId,
            getColumnOffset(ui, columnId),
        ])),
    };
}

function resizePanel(node, ui, panelWidget) {
    requestAnimationFrame(() => {
        setWidgetHeightSafe(panelWidget, PANEL_WIDGET_MIN_HEIGHT);
        if (panelWidget.element) {
            panelWidget.element.style.width = "100%";
            panelWidget.element.style.maxWidth = "100%";
            panelWidget.element.style.height = `${PANEL_WIDGET_MIN_HEIGHT}px`;
            panelWidget.element.style.minHeight = `${PANEL_WIDGET_MIN_HEIGHT}px`;
            panelWidget.element.style.overflow = "visible";
            panelWidget.element.style.display = "block";
            panelWidget.element.style.position = "relative";
            panelWidget.element.style.boxSizing = "border-box";
        }
        if (typeof node.setSize === "function") {
            const targetWidth = Math.max(Number(node.size?.[0] || 0), PANEL_MIN_WIDTH);
            if (Math.abs((node.size?.[0] || 0) - targetWidth) > 1 || Math.abs((node.size?.[1] || 0) - PANEL_MIN_HEIGHT) > 1) {
                node.setSize([targetWidth, PANEL_MIN_HEIGHT]);
            }
        }
        node.graph?.setDirtyCanvas?.(true, true);
        drawPath(ui, node.__omnivoiceInstructionState || {});
    });
}

function createBuilder(node) {
    if (!isBuilderNode(node) || typeof node.addDOMWidget !== "function") {
        return false;
    }
    if (node.__omnivoiceInstructionUi) {
        return true;
    }

    const relevantNames = new Set(["gender", "age", "pitch", "style", "accent", "dialect", "output_language"]);
    for (const widget of node.widgets || []) {
        if (relevantNames.has(widget.name)) {
            hideWidget(widget);
        }
    }

    const ui = createPanelDom();
    const layoutState = loadLayoutState(node);
    ui.columnOrder = layoutState.columnOrder;
    ui.columnOffsets = layoutState.columnOffsets;
    ui.dragState = null;
    ui.suppressNextClick = false;
    applyColumnLayout(ui);
    const panelWidget = node.addDOMWidget("omnivoice_instruction_builder_panel", "div", ui.panel, {
        serialize: false,
        hideOnZoom: false,
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
    setWidgetHeightSafe(panelWidget, PANEL_WIDGET_MIN_HEIGHT);

    const refresh = () => {
        const state = syncStateFromWidgets(node);
        node.__omnivoiceInstructionState = state;
        renderState(node, ui, state);
        applyColumnLayout(ui);
        drawPath(ui, state);
        resizePanel(node, ui, panelWidget);
        persistLayoutState(node, ui);
    };

    ui.accentModeButton.addEventListener("click", () => {
        ui.languageMode = "accent";
        renderState(node, ui, node.__omnivoiceInstructionState || syncStateFromWidgets(node));
        resizePanel(node, ui, panelWidget);
    });
    ui.dialectModeButton.addEventListener("click", () => {
        ui.languageMode = "dialect";
        renderState(node, ui, node.__omnivoiceInstructionState || syncStateFromWidgets(node));
        resizePanel(node, ui, panelWidget);
    });
    ui.localeEnglishButton.addEventListener("click", () => {
        const state = syncStateFromWidgets(node);
        state.output_language = "English";
        applyStateToWidgets(node, state);
        refresh();
    });
    ui.localeChineseButton.addEventListener("click", () => {
        const state = syncStateFromWidgets(node);
        state.output_language = "Chinese";
        applyStateToWidgets(node, state);
        refresh();
    });

    for (const button of ui.chipButtons.values()) {
        button.addEventListener("pointerdown", (event) => {
            const columnId = button.dataset.category === "accent" || button.dataset.category === "dialect"
                ? "language"
                : button.dataset.category;
            const state = syncStateFromWidgets(node);
            let selectionChangedOnPointerDown = false;
            if (normalizeWidgetValue(state[button.dataset.category]) !== button.dataset.value) {
                setCategoryValueOnState(state, button.dataset.category, button.dataset.value, false);
                applyStateToWidgets(node, state);
                node.__omnivoiceInstructionState = state;
                refresh();
                selectionChangedOnPointerDown = true;
            }
            beginColumnDrag(node, ui, event, columnId, {
                category: button.dataset.category,
                value: button.dataset.value,
            }, selectionChangedOnPointerDown);
        });
        button.addEventListener("click", () => {
            if (ui.suppressNextClick) {
                return;
            }
            const state = syncStateFromWidgets(node);
            setCategoryValueOnState(state, button.dataset.category, button.dataset.value, true);
            applyStateToWidgets(node, state);
            refresh();
        });
    }

    setupColumnDragging(node, ui);

    const resizeObserver = new ResizeObserver(() => {
        drawPath(ui, node.__omnivoiceInstructionState || {});
    });
    resizeObserver.observe(ui.body);

    window.addEventListener("resize", () => {
        drawPath(ui, node.__omnivoiceInstructionState || {});
    });

    node.__omnivoiceInstructionUi = ui;
    node.__omnivoiceInstructionPanelWidget = panelWidget;
    node.__omnivoiceInstructionRefresh = refresh;

    if (typeof node.setSize === "function") {
        node.setSize([
            PANEL_MIN_WIDTH,
            PANEL_MIN_HEIGHT,
        ]);
    }

    refresh();
    return true;
}

function createPrototypeState(initialState = {}) {
    return {
        gender: normalizeWidgetValue(initialState.gender),
        age: normalizeWidgetValue(initialState.age),
        pitch: normalizeWidgetValue(initialState.pitch),
        style: normalizeWidgetValue(initialState.style),
        accent: normalizeWidgetValue(initialState.accent),
        dialect: normalizeWidgetValue(initialState.dialect),
        output_language: normalizeWidgetValue(initialState.output_language) || "English",
    };
}

function createPrototypeController(container, initialState = {}, initialLayout = {}) {
    const ui = createPanelDom();
    const state = createPrototypeState(initialState);
    ui.columnOrder = Array.isArray(initialLayout.columnOrder)
        ? initialLayout.columnOrder.filter((columnId) => COLUMN_IDS.includes(columnId))
        : [...COLUMN_IDS];
    if (ui.columnOrder.length !== COLUMN_IDS.length) {
        ui.columnOrder = [...COLUMN_IDS];
    }
    ui.columnOffsets = Object.fromEntries(COLUMN_IDS.map((columnId) => [
        columnId,
        normalizeColumnOffset(initialLayout.columnOffsets?.[columnId]),
    ]));
    ui.dragState = null;
    ui.suppressNextClick = false;

    const prototypeNode = {
        __omnivoiceInstructionState: state,
        properties: {},
        graph: {
            setDirtyCanvas() {},
        },
    };

    const refresh = () => {
        renderState(prototypeNode, ui, state);
        applyColumnLayout(ui);
        drawPath(ui, state);
    };
    prototypeNode.__omnivoiceInstructionRefresh = refresh;

    const setCategoryValue = (category, value) => {
        setCategoryValueOnState(state, category, value, true);
        refresh();
    };

    ui.accentModeButton.addEventListener("click", () => {
        ui.languageMode = "accent";
        renderState(prototypeNode, ui, state);
        drawPath(ui, state);
    });
    ui.dialectModeButton.addEventListener("click", () => {
        ui.languageMode = "dialect";
        renderState(prototypeNode, ui, state);
        drawPath(ui, state);
    });
    ui.localeEnglishButton.addEventListener("click", () => {
        state.output_language = "English";
        refresh();
    });
    ui.localeChineseButton.addEventListener("click", () => {
        state.output_language = "Chinese";
        refresh();
    });

    for (const button of ui.chipButtons.values()) {
        button.addEventListener("pointerdown", (event) => {
            const columnId = button.dataset.category === "accent" || button.dataset.category === "dialect"
                ? "language"
                : button.dataset.category;
            let selectionChangedOnPointerDown = false;
            if (normalizeWidgetValue(state[button.dataset.category]) !== button.dataset.value) {
                setCategoryValueOnState(state, button.dataset.category, button.dataset.value, false);
                refresh();
                selectionChangedOnPointerDown = true;
            }
            beginColumnDrag(prototypeNode, ui, event, columnId, {
                category: button.dataset.category,
                value: button.dataset.value,
            }, selectionChangedOnPointerDown);
        });
        button.addEventListener("click", () => {
            if (ui.suppressNextClick) {
                return;
            }
            setCategoryValue(button.dataset.category, button.dataset.value);
        });
    }

    setupColumnDragging(prototypeNode, ui);
    const resizeObserver = new ResizeObserver(() => {
        drawPath(ui, state);
    });
    resizeObserver.observe(ui.body);
    window.addEventListener("resize", () => {
        drawPath(ui, state);
    });

    container.innerHTML = "";
    container.appendChild(ui.panel);
    refresh();

    return {
        ui,
        state,
        refresh,
        setCategoryValue,
        destroy() {
            resizeObserver.disconnect();
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
                createBuilder(this);
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
