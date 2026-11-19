export function findNumericTagParameter(parameters, name) {
    const key = name.trim().toLowerCase();
    return parameters.find(parameter => parameter.tag === key || parameter.aliases.includes(key));
}

export function isNumericTagValue(value) {
    return /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$/.test(value);
}

export function validNumericTagValue(parameter, value) {
    if (!isNumericTagValue(value)) return false;
    const number = Number(value);
    const min = parameter.relative && /^[+-]/.test(value) ? -parameter.max : parameter.min;
    return number >= min && number <= parameter.max && (!parameter.integer || Number.isInteger(number));
}

export function upsertTagParameter(content, parameter, value) {
    const names = [parameter.tag, ...parameter.aliases];
    const replacement = `${parameter.tag}:${value}`;
    const parts = content.split("|");
    const index = parts.findLastIndex((part, i) => {
        const colon = part.indexOf(":");
        if (colon < 0 || !names.includes(part.slice(0, colon).trim().toLowerCase())) return false;
        // Preserve leading character references such as [Happy:whisper_ref].
        return i > 0 || parameter.vector || isNumericTagValue(part.slice(colon + 1).trim());
    });
    if (index < 0) parts.push(replacement);
    else parts[index] = replacement;
    return parts.join("|");
}
