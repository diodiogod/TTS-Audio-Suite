export function escapeRegExp(value) {
    return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function buildFindReplaceQuery(query, options = {}) {
    if (!query) {
        return { valid: true, regex: null, flags: "", source: "" };
    }

    const flags = options.matchCase ? "g" : "gi";
    let source = "";

    try {
        if (options.regex) {
            source = options.wholeWord ? `\\b(?:${query})\\b` : query;
        } else {
            source = escapeRegExp(query);
            if (options.wholeWord) {
                source = `\\b${source}\\b`;
            }
        }

        return {
            valid: true,
            regex: new RegExp(source, flags),
            flags,
            source
        };
    } catch (error) {
        return {
            valid: false,
            regex: null,
            flags,
            source,
            error: error instanceof Error ? error.message : String(error)
        };
    }
}

export function findTextMatches(text, query, options = {}, selectionRange = null) {
    const compiled = buildFindReplaceQuery(query, options);
    if (!compiled.valid) {
        return { matches: [], error: compiled.error, compiled };
    }

    if (!query) {
        return { matches: [], error: "", compiled };
    }

    const scopeStart = options.selectionOnly && selectionRange ? selectionRange.start : 0;
    const scopeEnd = options.selectionOnly && selectionRange ? selectionRange.end : text.length;
    const scopedText = text.slice(scopeStart, scopeEnd);
    const matches = [];

    if (!scopedText) {
        return { matches, error: "", compiled };
    }

    compiled.regex.lastIndex = 0;
    let result;
    while ((result = compiled.regex.exec(scopedText)) !== null) {
        const matchedText = result[0];
        if (!matchedText) {
            compiled.regex.lastIndex += 1;
            continue;
        }

        matches.push({
            start: scopeStart + result.index,
            end: scopeStart + result.index + matchedText.length,
            text: matchedText
        });
    }

    return { matches, error: "", compiled };
}

export function getReplacementForMatch(match, replaceText, compiled, options = {}) {
    if (!compiled?.regex || !options.regex) {
        return replaceText;
    }

    const singleFlags = compiled.flags.replace(/g/g, "");
    const singleRegex = new RegExp(compiled.source, singleFlags);
    return match.text.replace(singleRegex, replaceText);
}

export function replaceMatches(text, matches, replaceText, compiled, options = {}) {
    if (!Array.isArray(matches) || matches.length === 0) {
        return { text, replacements: 0 };
    }

    let nextText = text;
    const sortedMatches = [...matches].sort((a, b) => b.start - a.start);
    sortedMatches.forEach((match) => {
        const replacement = getReplacementForMatch(match, replaceText, compiled, options);
        nextText = nextText.slice(0, match.start) + replacement + nextText.slice(match.end);
    });

    return { text: nextText, replacements: sortedMatches.length };
}
