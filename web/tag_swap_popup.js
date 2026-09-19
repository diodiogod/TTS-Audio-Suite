import { isLanguageCode } from "./language-constants.js";
import { TagUtilities } from "./tag-utilities.js";
import { createLanguageGlobe, languageMeta, languageTileCode } from "./language-swap-globe.js";

const COLORS = { character: "#46d6b1", language: "#56a8ff", reference: "#ff8f70", parameter: "#f4c45e" };
const EMOTIONS = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"];
const ENGINE_PARAMETERS = {
    index_tts: ["seed", "temperature", "cfg", "top_p", "top_k", "emotion_alpha", "vector", "emotion", ...EMOTIONS],
    higgs_audio_v3: ["seed", "temperature", "top_p", "top_k", "max_new_tokens"],
    step_audio_editx: ["seed", "temperature"],
    cosyvoice3: ["seed", "speed"],
    omnivoice: ["seed", "num_steps", "guidance_scale", "duration", "t_shift", "layer_penalty_factor", "position_temperature", "class_temperature", "audio_chunk_duration", "audio_chunk_threshold", "speed"],
    dramabox: ["seed", "cfg", "stg_scale", "duration_multiplier", "gen_duration", "ref_duration", "rescale_scale", "negative", "template"],
};
const ALL_PARAMETERS = new Set(Object.values(ENGINE_PARAMETERS).flat());
const VALUES = {
    seed: ["0", "1", "2", "3", "4", "42", "123", "777"],
    temperature: ["0.3", "0.5", "0.7", "0.8", "1", "1.2"],
    cfg: ["0", "0.5", "1", "3", "5", "7", "10"],
    speed: ["0.5", "0.75", "0.9", "1", "1.1", "1.25", "1.5", "2"],
    top_p: ["0.5", "0.7", "0.8", "0.9", "0.95", "1"],
    top_k: ["10", "20", "30", "40", "50", "80"],
    steps: ["10", "20", "30", "40", "50", "75", "100"],
    emotion_alpha: ["0", "0.25", "0.5", "0.75", "1", "1.5", "2"],
    stg_scale: ["0", "0.5", "1", "1.5", "2", "3", "5"],
    duration_multiplier: ["0.75", "1", "1.1", "1.25", "1.5", "2"],
    gen_duration: ["0", "3", "5", "10", "15", "20", "30", "45", "60"],
    ref_duration: ["3", "5", "10", "15", "20", "30"],
    rescale_scale: ["auto", "0", "0.1", "0.2", "0.3", "0.5", "0.75", "1"],
};
const options = values => (values || []).map(value => typeof value === "string"
    ? { value, label: value }
    : value).filter(value => value?.value !== undefined && String(value.value));
const ranges = (text, separator) => {
    const result = [];
    let start = 0;
    for (let index = 0; index <= text.length; index++) if (index === text.length || text[index] === separator) {
        result.push({ start, end: index, text: text.slice(start, index) });
        start = index + 1;
    }
    return result;
};
const replace = (text, start, end, value) => text.slice(0, start) + value + text.slice(end);

function buildBracketSwap({ tag, cursorOffset, characterOptions, languageOptions, engine }) {
    const content = tag.slice(1, -1);
    const offset = Math.max(0, Math.min(content.length, cursorOffset - 1));
    const parts = ranges(content, "|");
    let partIndex = parts.findIndex(part => offset >= part.start && offset <= part.end);
    if (partIndex < 0) partIndex = 0;
    const part = parts[partIndex];
    const engineParameters = ENGINE_PARAMETERS[engine] || ["seed", "temperature"];
    const firstColon = part.text.indexOf(":");
    const firstKey = firstColon < 0 ? "" : part.text.slice(0, firstColon).trim().toLowerCase();
    const firstValue = firstColon < 0 ? "" : part.text.slice(firstColon + 1).trim();
    const numericIndexEmotion = engine === "index_tts" && EMOTIONS.includes(firstKey)
        && /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$/.test(firstValue);
    const parameterOnly = partIndex === 0 && firstColon >= 0
        && (engineParameters.includes(firstKey) || (ALL_PARAMETERS.has(firstKey) && !EMOTIONS.includes(firstKey)) || numericIndexEmotion);
    if (partIndex > 0 || parameterOnly) {
        const colon = part.text.indexOf(":");
        if (colon < 0) return null;
        const key = part.text.slice(0, colon).trim();
        const value = part.text.slice(colon + 1).trim();
        const editKey = offset - part.start <= colon;
        let choices = editKey ? engineParameters : (VALUES[key.toLowerCase()] || [value]);
        if (!editKey && EMOTIONS.includes(key.toLowerCase())) {
            const sign = value.startsWith("-") ? "-" : value.startsWith("+") ? "+" : "";
            choices = ["0", "0.1", "0.2", "0.3", "0.5", "0.75", "1", "1.2"].map(number => `${sign}${number}`);
        }
        choices = [...choices, "__remove__"];
        return {
            title: editKey ? "Swap parameter" : `Swap ${key}`,
            current: editKey ? key : value,
            color: COLORS.parameter,
            items: choices.map(candidate => ({ value: candidate, label: candidate === "__remove__" ? "×" : candidate, remove: candidate === "__remove__" })),
            replace(candidate) {
                if (candidate === "__remove__") {
                    if (partIndex === 0) return "";
                    return `[${content.slice(0, Math.max(0, part.start - 1)) + content.slice(part.end)}]`;
                }
                const next = editKey ? `${candidate}:${value}` : `${key}:${candidate}`;
                return `[${replace(content, part.start, part.end, next)}]`;
            },
        };
    }
    const tokens = ranges(part.text, ":");
    const languages = new Set(options(languageOptions).map(item => String(item.value).toLowerCase()));
    const hasLanguage = tokens.length > 1 && (languages.has(tokens[0].text.toLowerCase()) || isLanguageCode(tokens[0].text));
    let tokenIndex = tokens.findIndex(token => offset >= token.start && offset <= token.end);
    if (tokenIndex < 0) tokenIndex = 0;
    let kind = "character";
    let choices = options(characterOptions);
    if (hasLanguage && tokenIndex === 0) {
        kind = "language";
        choices = options(languageOptions);
    } else if ((hasLanguage && tokenIndex >= 2) || (!hasLanguage && tokenIndex >= 1)) {
        kind = "reference";
        choices = [{ value: "__remove__", label: "×", remove: true }, ...options(characterOptions)];
    }
    const token = tokens[tokenIndex];
    return {
        title: kind === "reference" ? "Swap emotion reference" : `Swap ${kind}`,
        current: token.text.trim(), color: COLORS[kind], items: choices, kind,
        replace(candidate) {
            if (candidate === "__remove__") return `[${content.slice(0, Math.max(0, token.start - 1)) + content.slice(token.end)}]`;
            return `[${replace(content, token.start, token.end, candidate)}]`;
        },
    };
}

function angleSwap(tag, engine) {
    const content = tag.slice(1, -1);
    if (engine === "higgs_audio_v3") {
        const match = content.match(/^\|?(emotion|style|prosody|sfx):([^|>]+)\|?$/);
        if (match && TagUtilities.HIGGS_TAGS[match[1]]) return {
            title: `Swap Higgs ${match[1]}`, current: match[2], color: "#ef5da8",
            items: [...TagUtilities.HIGGS_TAGS[match[1]]].map(value => ({ value, label: value })),
            replace: value => content.startsWith("|") ? `<|${match[1]}:${value}|>` : `<${match[1]}:${value}>`,
        };
    }
    if (engine === "step_audio_editx") {
        const match = content.match(/^(emotion|style|speed):([^:>]+)(:\d+)?$/);
        if (match) {
            const source = match[1] === "emotion" ? TagUtilities.STEP_EMOTIONS : match[1] === "style" ? TagUtilities.STEP_STYLES : TagUtilities.STEP_SPEEDS;
            return { title: `Swap ${match[1]}`, current: match[2], color: "#a879ff", items: [...source].map(value => ({ value, label: value })), replace: value => `<${match[1]}:${value}${match[3] || ""}>` };
        }
        const plain = content.match(/^([^:>]+)(:\d+)?$/);
        if (plain && TagUtilities.STEP_PARALINGUISTIC_TAGS.has(plain[1])) return { title: "Swap paralinguistic tag", current: plain[1], color: "#62d69d", items: [...TagUtilities.STEP_PARALINGUISTIC_TAGS].map(value => ({ value, label: value })), replace: value => `<${value}${plain[2] || ""}>` };
    }
    if (engine === "cosyvoice3") {
        const closing = content.startsWith("/");
        const name = closing ? content.slice(1) : content;
        if (TagUtilities.COSY_WRAPPER_TAGS.has(name)) return { title: "Swap CosyVoice wrapper", current: name, color: "#a879ff", items: [...TagUtilities.COSY_WRAPPER_TAGS].map(value => ({ value, label: value })), replace: value => `<${closing ? "/" : ""}${value}>` };
        if (TagUtilities.COSY_SINGLE_TAGS.has(content)) return { title: "Swap CosyVoice sound", current: content, color: "#62d69d", items: [...TagUtilities.COSY_SINGLE_TAGS].map(value => ({ value, label: value })), replace: value => `<${value}>` };
    }
    if (engine === "omnivoice" && TagUtilities.OMNIVOICE_NON_VERBAL_TAGS.has(content)) return { title: "Swap OmniVoice sound", current: content, color: "#62d69d", items: [...TagUtilities.OMNIVOICE_NON_VERBAL_TAGS].map(value => ({ value, label: value })), replace: value => `<${value}>` };
    return null;
}

function openSwapPopup(args, swap) {
    if (!swap?.items.length) return null;
    if (swap.kind === "language") {
        const available=new Set(swap.items.map(item=>String(item.value).toLowerCase()));
        const hidden=new Set([
            ...(available.has("pt") ? ["pt-pt"] : []),
            ...(available.has("zh") ? ["zh-cn"] : []),
        ]);
        const seen=new Set();
        swap.items=swap.items.filter(item=>{
            const code=String(item.value).toLowerCase();
            const tile=languageTileCode(code);
            if (hidden.has(code)||seen.has(tile)) return false;
            seen.add(tile); return true;
        }).sort((left,right)=>languageTileCode(left.value).localeCompare(languageTileCode(right.value)));
    }
    document.querySelectorAll(".tts-tag-quick-swap").forEach(element => element.remove());
    const popup = document.createElement("div");
    popup.className = "tts-tag-quick-swap";
    const languageSwap = swap.kind === "language";
    const popupWidth = languageSwap ? 620 : 330;
    const popupHeight = languageSwap ? 620 : 280;
    const sizeKey = "tts-language-globe-picker-size";
    let savedSize = null;
    if (languageSwap) {
        try { savedSize = JSON.parse(localStorage.getItem(sizeKey)); } catch { /* use defaults */ }
    }
    const savedSquareSize = Math.min(Number(savedSize?.width) || popupWidth, Number(savedSize?.height) || popupHeight);
    const initialSize = Math.min(innerWidth - 16, innerHeight - 16, Math.max(300, savedSquareSize));
    popup.style.cssText = `position:fixed;z-index:2147483646;width:${languageSwap ? initialSize : popupWidth}px;${languageSwap ? `height:${initialSize}px;min-width:300px;min-height:300px;display:flex;flex-direction:column;` : ""}box-sizing:border-box;padding:10px;overflow:hidden;background:linear-gradient(145deg,rgba(26,31,40,.96),rgba(13,17,23,.97));border:1px solid ${swap.color};border-radius:11px;box-shadow:0 14px 40px #0009,0 0 18px ${swap.color}30;color:#f4f6fb;font-family:system-ui,sans-serif;`;
    const rect = args.anchorRect || { left: 8, bottom: 8 };
    popup.style.left = `${Math.max(8, Math.min(innerWidth - popupWidth - 8, rect.left || 8))}px`;
    popup.style.top = `${Math.max(8, Math.min(innerHeight - popupHeight - 8, (rect.bottom || 8) + 7))}px`;
    const heading = document.createElement("div");
    heading.textContent = languageSwap ? swap.title : `${swap.title} · ${swap.current || "none"}`;
    heading.style.cssText = "position:relative;z-index:1;font-size:12px;font-weight:750;margin-bottom:8px";
    const grid = document.createElement("div");
    grid.style.cssText = languageSwap
        ? "position:relative;z-index:1;flex:1;display:grid;grid-template-columns:repeat(8,minmax(0,1fr));grid-auto-rows:max-content;align-content:start;gap:10px;overflow:auto"
        : "position:relative;z-index:1;display:flex;flex-wrap:wrap;gap:6px;max-height:220px;overflow:auto";
    const globe = languageSwap ? createLanguageGlobe(popup, swap.current) : null;
    let outside;
    let parkedSelectionRange = null;
    let selectionClearFrame = null;
    const clearLiveSelection = () => {
        const selection = window.getSelection();
        if (selection?.rangeCount && !parkedSelectionRange) {
            parkedSelectionRange = selection.getRangeAt(0).cloneRange();
        }
        selection?.removeAllRanges();
        if (selectionClearFrame !== null) cancelAnimationFrame(selectionClearFrame);
        selectionClearFrame = requestAnimationFrame(() => {
            window.getSelection()?.removeAllRanges();
            selectionClearFrame = null;
        });
    };
    const restoreParkedSelection = () => {
        if (selectionClearFrame !== null) {
            cancelAnimationFrame(selectionClearFrame);
            selectionClearFrame = null;
        }
        if (!parkedSelectionRange) return;
        const selection = window.getSelection();
        selection?.removeAllRanges();
        selection?.addRange(parkedSelectionRange);
        parkedSelectionRange = null;
    };
    const close = cancelled => {
        document.removeEventListener("pointerdown", outside, true);
        window.removeEventListener("pointermove", trackHeldPointer, true);
        window.removeEventListener("pointerup", releaseHeldPointer, true);
        window.removeEventListener("pointercancel", releaseHeldPointer, true);
        if (selectionClearFrame !== null) cancelAnimationFrame(selectionClearFrame);
        selectionClearFrame = null;
        parkedSelectionRange = null;
        globe?.destroy();
        popup.remove();
        args.onGestureEnd?.();
        if (cancelled) args.onCancel?.();
    };
    outside = event => { if (!popup.contains(event.target)) close(true); };
    const entries = [];
    let highlighted = null;
    const paint = (entry, hover) => {
        const color = entry.item.remove ? "#e05263" : swap.color;
        entry.button.style.background = (entry.active || hover) ? color : (languageSwap ? "rgba(8,13,20,.20)" : `${color}18`);
        entry.button.style.color = (entry.active || hover) ? "#11151b" : "#edf0f6";
        entry.button.style.transform = hover ? "scale(1.07)" : "scale(1)";
        entry.button.style.boxShadow = hover ? `0 0 15px ${color}a0` : "none";
        entry.button.style.borderColor = color;
        if (languageSwap) entry.button.style.backdropFilter = (entry.active || hover) ? "blur(3px)" : "none";
    };
    const choose = entry => { const result = swap.replace(entry.item.value); close(false); args.onSelect?.(result); };
    for (const item of swap.items) {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = languageSwap ? languageTileCode(item.value) : (item.label || item.value);
        const active = String(item.value).toLowerCase() === String(swap.current).toLowerCase();
        const languageDescription = languageSwap ? languageMeta(item.value).slice(0,2).filter(Boolean).join(", ") : "";
        button.title = item.remove ? "Remove this tag component" : "";
        button.setAttribute("aria-label", item.remove ? "Remove this tag component" : (languageDescription || String(item.label || item.value)));
        button.style.cssText = item.remove
            ? "width:25px;height:25px;padding:0;border-radius:50%;border:1px solid #e05263;font-size:15px;font-weight:800;line-height:1;cursor:pointer;transition:transform .1s,box-shadow .1s,background .1s;"
            : languageSwap
                ? `width:100%;min-width:0;aspect-ratio:1;padding:0;border-radius:5px;border:1px solid ${swap.color}80;background:rgba(8,13,20,.20);font:14px ui-monospace,monospace;letter-spacing:.06em;cursor:pointer;backdrop-filter:none;transition:transform .1s,box-shadow .1s,background .1s,backdrop-filter .1s;`
                : `padding:7px 10px;border-radius:999px;border:1px solid ${swap.color};font-size:10px;cursor:pointer;transition:transform .1s,box-shadow .1s,background .1s;`;
        const entry = { button, item, active };
        entries.push(entry); paint(entry, false);
        if (languageSwap) {
            button.onpointerenter = () => globe.focus(item.value, button);
            button.onfocus = () => globe.focus(item.value, button);
        }
        button.onpointerdown = event => {
            event.preventDefault(); event.stopPropagation();
            choose(entry);
        };
        grid.append(button);
    }
    grid.style.padding = "9px";
    grid.style.margin = "-9px";
    const updateHighlight = event => {
        let nearest = null, distance = Infinity;
        for (const entry of entries) {
            const rect = entry.button.getBoundingClientRect();
            const dx = Math.max(rect.left - event.clientX, 0, event.clientX - rect.right);
            const dy = Math.max(rect.top - event.clientY, 0, event.clientY - rect.bottom);
            const nextDistance = Math.hypot(dx, dy);
            if (nextDistance < distance) { nearest = entry; distance = nextDistance; }
        }
        const next = distance <= 34 ? nearest : null;
        if (next !== highlighted) { if (highlighted) paint(highlighted, false); highlighted = next; if (highlighted) paint(highlighted, true); }
    };
    grid.onpointermove = updateHighlight;
    if (languageSwap) grid.onpointerleave = () => {
        if (highlighted) { paint(highlighted,false); highlighted=null; }
        const selected=entries.find(entry=>entry.active);
        if (selected) globe.focus(selected.item.value,selected.button);
    };
    grid.onpointerdown = event => { if (event.target === grid && highlighted) { event.preventDefault(); choose(highlighted); } };
    const trackHeldPointer = event => {
        if (event.pointerId !== args.holdPointerId) return;
        if (event.cancelable) event.preventDefault();
        const popupRect = popup.getBoundingClientRect();
        const isOverPopup = event.clientX >= popupRect.left && event.clientX <= popupRect.right
            && event.clientY >= popupRect.top && event.clientY <= popupRect.bottom;
        if (isOverPopup) clearLiveSelection();
        else restoreParkedSelection();
        updateHighlight(event);
    };
    const releaseHeldPointer = event => {
        if (event.pointerId !== args.holdPointerId) return;
        window.removeEventListener("pointermove", trackHeldPointer, true);
        window.removeEventListener("pointerup", releaseHeldPointer, true);
        window.removeEventListener("pointercancel", releaseHeldPointer, true);
        if (event.type === "pointerup" && highlighted) choose(highlighted);
    };
    const resizeHandle = languageSwap ? document.createElement("div") : null;
    if (resizeHandle) {
        resizeHandle.setAttribute("aria-label", "Resize language picker");
        resizeHandle.style.cssText = `position:absolute;z-index:4;right:2px;bottom:2px;width:22px;height:22px;cursor:nwse-resize;touch-action:none;background:linear-gradient(135deg,transparent 48%,${swap.color} 50%,${swap.color} 57%,transparent 59%,transparent 68%,${swap.color} 70%,${swap.color} 77%,transparent 79%);opacity:.8`;
        let drag = null;
        resizeHandle.onpointerdown = event => {
            event.preventDefault(); event.stopPropagation();
            const box = popup.getBoundingClientRect();
            drag = { x:event.clientX, y:event.clientY, size:box.width };
            resizeHandle.setPointerCapture(event.pointerId);
        };
        resizeHandle.onpointermove = event => {
            if (!drag || !resizeHandle.hasPointerCapture(event.pointerId)) return;
            event.preventDefault(); event.stopPropagation();
            const dx=event.clientX-drag.x, dy=event.clientY-drag.y;
            const delta=Math.abs(dx)>Math.abs(dy)?dx:dy;
            const box=popup.getBoundingClientRect();
            const maxSize=Math.max(300,Math.min(innerWidth-box.left-8,innerHeight-box.top-8));
            const size=Math.max(300,Math.min(maxSize,drag.size+delta));
            popup.style.width=`${size}px`; popup.style.height=`${size}px`;
        };
        const finishResize = event => {
            if (!drag) return;
            event.preventDefault(); event.stopPropagation();
            const {width}=popup.getBoundingClientRect();
            localStorage.setItem(sizeKey,JSON.stringify({width:Math.round(width),height:Math.round(width)}));
            drag=null;
            if(resizeHandle.hasPointerCapture(event.pointerId)) resizeHandle.releasePointerCapture(event.pointerId);
        };
        resizeHandle.onpointerup=finishResize;
        resizeHandle.onpointercancel=finishResize;
    }
    popup.append(heading, grid, ...(resizeHandle ? [resizeHandle] : []));
    document.body.append(popup);
    if (languageSwap) {
        const activeEntry = entries.find(entry => entry.active) || entries[0];
        requestAnimationFrame(() => globe.focus(activeEntry.item.value, activeEntry.button));
    }
    if (args.holdPointerId !== null && args.holdPointerId !== undefined) {
        window.addEventListener("pointermove", trackHeldPointer, true);
        window.addEventListener("pointerup", releaseHeldPointer, true);
        window.addEventListener("pointercancel", releaseHeldPointer, true);
    }
    setTimeout(() => document.addEventListener("pointerdown", outside, true), 0);
    return popup;
}

export function openTagSwapPopup(args) {
    const swap = args.tag.startsWith("<") ? angleSwap(args.tag,args.engine) : buildBracketSwap(args);
    return openSwapPopup(args,swap);
}

export function openLanguagePicker({ languageOptions, current="", anchorRect, onSelect, onCancel }) {
    const items=options(languageOptions);
    return openSwapPopup({ anchorRect,onSelect,onCancel },{
        title:"Select language", current, color:COLORS.language, kind:"language", items,
        replace:value=>value,
    });
}
