import { INDEX_TTS_EMOTION_VISUALS } from "./emotion_radar_canvas_widget.js";
import { findNumericTagParameter, isNumericTagValue } from "./numeric-tag-parameters.js";

const SHARED_EMOTIONS = { happy: "Happy", angry: "Angry", sad: "Sad", calm: "Calm", scared: "Afraid", confused: "Surprised" };

export const ZONOS3_TAG_SECTIONS = {
    Emotions: { title: "Named Emotion", color: "#ffcc66" },
    "Delivery Style": { color: "#8fcbff" },
    Prosody: { color: "#b5a3ff" },
    Mood: { color: "#64d7ff" },
    "Voice Characteristics": { color: "#8cff9d", hint: "Requires embedding voice mode." },
    "Additional Effects": { color: "#ff9e7a" },
    "Effect Timing": { color: "#f6c95b", hint: "Pace and whisper keep their strength throughout the segment." },
};

export function zonos3EmotionAxis(name, control, index, count) {
    const color = INDEX_TTS_EMOTION_VISUALS.find(emotion => emotion.name === SHARED_EMOTIONS[name]).color;
    return { name, label: control.label, min: control.min, max: control.max, step: control.step, angle: index * 2 * Math.PI / count, color };
}
export function createZonos3VectorProfile(definition, parameters) {
    if (!definition) return null;
    const { tag, axes: names } = definition;
    const axes = names.map((name, index) => zonos3EmotionAxis(name, findNumericTagParameter(parameters, name), index, names.length));
    return {
        tag,
        axes,
        title: "ZONOS3-Mini Emotion Vector",
        presetStorageKey: "tts-zonos3-emotion-vector-presets",
        parse(raw) {
            const content = raw.startsWith(`[${tag}:`) && raw.endsWith("]") ? raw.slice(tag.length + 2, -1) : raw;
            const values = content.split(",").map(value => value.trim());
            if (values.length !== axes.length || !values.every(isNumericTagValue)) return null;
            const signed = values.map(value => /^[+-]/.test(value));
            if (signed.some(Boolean) && !signed.every(Boolean)) return null;
            return {
                values: values.map((value, index) => Math.min(Math.abs(Number(value)), axes[index].max)),
                signs: values.map(value => value.startsWith("-") ? -1 : 1),
                relative: signed.every(Boolean),
            };
        },
        serialize(values, signs, relative) {
            const encoded = values.map((value, index) => {
                const prefix = relative ? (signs[index] < 0 ? "-" : "+") : "";
                return `${prefix}${Number(value.toFixed(2))}`;
            });
            return `[${tag}:${encoded.join(",")}]`;
        },
    };
}
