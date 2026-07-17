/**
 * 🏷️ Tag Utilities
 * Utility functions for tag operations, parsing, validation, and modification
 */

export class TagUtilities {
    static STEP_PARALINGUISTIC_TAGS = new Set([
        "Laughter", "Breathing", "Sigh", "Uhm", "Surprise-oh", "Surprise-ah",
        "Surprise-wa", "Confirmation-en", "Question-ei", "Dissatisfaction-hnn"
    ]);

    static STEP_EMOTIONS = new Set([
        "happy", "sad", "angry", "excited", "calm", "fearful", "surprised",
        "disgusted", "confusion", "empathy", "embarrass", "depressed", "coldness", "admiration"
    ]);

    static STEP_STYLES = new Set([
        "whisper", "serious", "child", "older", "girl", "pure", "sister", "sweet",
        "exaggerated", "ethereal", "generous", "recite", "act_coy", "warm", "shy",
        "comfort", "authority", "chat", "radio", "soulful", "gentle", "story", "vivid",
        "program", "news", "advertising", "roar", "murmur", "shout", "deeply", "loudly",
        "arrogant", "friendly"
    ]);

    static STEP_SPEEDS = new Set(["faster", "slower", "more_faster", "more_slower"]);

    static HIGGS_TAGS = {
        emotion: new Set([
            "elation", "amusement", "enthusiasm", "determination", "pride", "contentment", "affection",
            "relief", "contemplation", "confusion", "surprise", "awe", "longing", "arousal", "anger",
            "fear", "disgust", "bitterness", "sadness", "shame", "helplessness"
        ]),
        style: new Set(["singing", "shouting", "whispering"]),
        prosody: new Set([
            "speed_very_slow", "speed_slow", "speed_fast", "speed_very_fast",
            "pitch_low", "pitch_high", "pause", "long_pause", "expressive_high", "expressive_low"
        ]),
        sfx: new Set(["cough", "laughter", "crying", "screaming", "burping", "humming", "sigh", "sniff", "sneeze"]),
    };

    static COSY_SINGLE_TAGS = new Set([
        "breath", "quick_breath", "laughter", "cough", "sigh", "gasp", "noise",
        "hissing", "vocalized-noise", "lipsmack", "mn", "clucking", "accent"
    ]);

    static COSY_WRAPPER_TAGS = new Set(["laughing", "strong"]);

    static OMNIVOICE_NON_VERBAL_TAGS = new Set([
        "laughter", "sigh", "confirmation-en", "question-en", "question-ah",
        "question-oh", "question-ei", "question-yi", "surprise-ah", "surprise-oh",
        "surprise-wa", "surprise-yo", "dissatisfaction-hnn"
    ]);

    static parseExistingTags(text) {
        const tags = [];
        const tagPattern = /\[([^\]]+)\]/g;
        let match;

        while ((match = tagPattern.exec(text)) !== null) {
            const tagContent = match[1];
            const tag = {
                full: `[${tagContent}]`,
                position: match.index,
                character: "",
                language: "",
                parameters: {}
            };

            // Parse character and parameters
            const parts = tagContent.split("|");

            // First part: character or language:character
            const firstPart = parts[0].trim();
            if (firstPart.includes(":") && !firstPart.includes(".")) {
                const [lang, char] = firstPart.split(":", 2);
                tag.language = lang.trim();
                tag.character = char.trim();
            } else {
                tag.character = firstPart;
            }

            // Remaining parts: parameters
            for (let i = 1; i < parts.length; i++) {
                const part = parts[i];
                if (part.includes(":")) {
                    const [paramName, paramValue] = part.split(":", 2);
                    tag.parameters[paramName.trim().toLowerCase()] = paramValue.trim();
                }
            }

            tags.push(tag);
        }

        return tags;
    }

    static validateTagSyntax(text) {
        const tagPattern = /\[([^\]]+)\]/g;
        let match;

        while ((match = tagPattern.exec(text)) !== null) {
            const tagContent = match[1];

            // Check for mismatched brackets
            if ((tagContent.match(/\[/g) || []).length !== (tagContent.match(/\]/g) || []).length) {
                return { valid: false, error: `Mismatched brackets in tag: [${tagContent}]` };
            }

            // Validate parameter syntax if present
            if (tagContent.includes("|")) {
                const parts = tagContent.split("|");
                for (let i = 1; i < parts.length; i++) {
                    if (!parts[i].includes(":")) {
                        return { valid: false, error: `Invalid parameter syntax: ${parts[i]} (expected format: param:value)` };
                    }

                    const [paramName] = parts[i].split(":", 2);
                    if (!paramName.trim()) {
                        return { valid: false, error: `Empty parameter name in ${parts[i]}` };
                    }
                }
            }
        }

        return { valid: true };
    }

    static validateInlineSyntax(text) {
        let cursor = 0;

        while (cursor < text.length) {
            const openIndex = text.indexOf("<", cursor);
            if (openIndex === -1) {
                break;
            }

            const lineEndIndex = text.indexOf("\n", openIndex);
            const searchEnd = lineEndIndex === -1 ? text.length : lineEndIndex;
            const closeIndex = text.indexOf(">", openIndex + 1);
            const nextOpenIndex = text.indexOf("<", openIndex + 1);

            if (closeIndex === -1 || closeIndex >= searchEnd) {
                return {
                    valid: false,
                    error: `Malformed inline tag near: ${text.slice(openIndex, searchEnd).trim() || "<"}`,
                };
            }

            if (nextOpenIndex !== -1 && nextOpenIndex < closeIndex) {
                return {
                    valid: false,
                    error: `Nested or broken inline tag near: ${text.slice(openIndex, closeIndex + 1).trim()}`,
                };
            }

            cursor = closeIndex + 1;
        }

        return { valid: true };
    }

    static getInlineTagMatches(text) {
        const matches = [];
        const tagPattern = /<[^<>\r\n]+>/g;
        let match;
        while ((match = tagPattern.exec(text)) !== null) {
            const full = match[0];
            matches.push({
                full,
                start: match.index,
                end: match.index + full.length,
                analysis: this.analyzeInlineTag(full),
            });
        }
        return matches;
    }

    static analyzeInlineTag(tag) {
        const engines = new Set();
        const conversions = {};
        const content = tag.slice(1, -1);

        const higgsCanonical = content.match(/^\|([a-z_]+):([^|>]+)\|$/);
        if (higgsCanonical) {
            const category = higgsCanonical[1];
            const value = higgsCanonical[2];
            if (this.HIGGS_TAGS[category]?.has(value)) {
                engines.add("higgs_audio_v3");
                this._fillHiggsConversions(category, value, conversions);
                return { engines, kind: "higgs", category, value, conversions };
            }
            return { engines, kind: "unknown", conversions };
        }

        const cosyWrapper = content.match(/^\/?([a-zA-Z_-]+)$/);
        if (cosyWrapper && this.COSY_WRAPPER_TAGS.has(cosyWrapper[1])) {
            engines.add("cosyvoice3");
            return { engines, kind: "cosy_wrapper", value: cosyWrapper[1], conversions };
        }

        if (this.OMNIVOICE_NON_VERBAL_TAGS.has(content)) {
            engines.add("omnivoice");
            this._fillOmniVoiceConversions(content, conversions);
        }

        if (this.COSY_SINGLE_TAGS.has(content)) {
            engines.add("cosyvoice3");
            this._fillCosySingleConversions(content, conversions);
            return { engines, kind: "cosy_single", value: content, conversions };
        }

        const stepParts = content.split("|").map((part) => part.trim()).filter(Boolean);
        if (stepParts.length > 0) {
            const parsedStepParts = [];
            let stepValid = true;
            let hasAnyStepConvertible = false;

            for (const part of stepParts) {
                const parsed = this._analyzeStepOrHiggsAliasPart(part);
                if (!parsed) {
                    stepValid = false;
                    break;
                }
                parsedStepParts.push(parsed);
                if (parsed.stepValid) {
                    engines.add("step_audio_editx");
                }
                if (parsed.higgsAliasValid) {
                    engines.add("higgs_audio_v3");
                }
                if (parsed.conversions && Object.keys(parsed.conversions).length > 0) {
                    hasAnyStepConvertible = true;
                }
            }

            if (stepValid && (engines.size > 0 || hasAnyStepConvertible)) {
                const combinedConversions = this._combineConversions(parsedStepParts);
                return {
                    engines,
                    kind: "step_or_alias",
                    parts: parsedStepParts,
                    conversions: combinedConversions,
                };
            }
        }

        return { engines, kind: "unknown", conversions };
    }

    static validateInlineTags(text, targetEngine) {
        const matches = this.getInlineTagMatches(text);
        const foreignTags = [];
        const unknownTags = [];

        matches.forEach((match) => {
            const engines = match.analysis.engines || new Set();
            if (engines.size === 0) {
                unknownTags.push(match);
                return;
            }
            if (!engines.has(targetEngine)) {
                foreignTags.push(match);
            }
        });

        const convertibleTags = foreignTags.filter((match) => {
            const conversion = match.analysis.conversions?.[targetEngine];
            return typeof conversion === "string" && conversion.length > 0;
        });

        return {
            valid: foreignTags.length === 0 && unknownTags.length === 0,
            foreignTags,
            unknownTags,
            convertibleTags,
        };
    }

    static convertInlineTagsForEngine(text, targetEngine) {
        const matches = this.getInlineTagMatches(text).sort((a, b) => b.start - a.start);
        let nextText = text;
        let converted = 0;
        let skipped = 0;

        matches.forEach((match) => {
            const replacement = match.analysis.conversions?.[targetEngine];
            if (typeof replacement === "string" && replacement.length > 0) {
                nextText = nextText.slice(0, match.start) + replacement + nextText.slice(match.end);
                converted++;
            } else if ((match.analysis.engines || new Set()).size > 0 && !(match.analysis.engines || new Set()).has(targetEngine)) {
                skipped++;
            }
        });

        return { text: nextText, converted, skipped };
    }

    static _analyzeStepOrHiggsAliasPart(part) {
        if (part === "restore") {
            return {
                stepValid: true,
                higgsAliasValid: false,
                conversions: { step_audio_editx: "<restore>" },
            };
        }

        const paraMatch = part.match(/^([A-Za-z][A-Za-z-]*)(?::(\d+))?$/);
        if (paraMatch && this.STEP_PARALINGUISTIC_TAGS.has(paraMatch[1])) {
            const name = paraMatch[1];
            return {
                stepValid: true,
                higgsAliasValid: false,
                conversions: this._getStepParalinguisticConversions(name),
            };
        }

        const valueMatch = part.match(/^([a-z_]+):([^:>]+)(?::(\d+))?$/);
        if (!valueMatch) {
            return null;
        }

        const category = valueMatch[1];
        const value = valueMatch[2];
        const stepValid = this._isValidStepCategoryValue(category, value);
        const higgsAliasValid = !!this.HIGGS_TAGS[category]?.has(value);
        const conversions = {};

        if (stepValid) {
            Object.assign(conversions, this._getStepCategoryConversions(category, value));
        }
        if (higgsAliasValid) {
            this._fillHiggsConversions(category, value, conversions);
        }
        if (category === "restore") {
            conversions.step_audio_editx = `<${part}>`;
        }

        return { stepValid, higgsAliasValid, category, value, conversions };
    }

    static _isValidStepCategoryValue(category, value) {
        if (category === "emotion") return this.STEP_EMOTIONS.has(value);
        if (category === "style") return this.STEP_STYLES.has(value);
        if (category === "speed") return this.STEP_SPEEDS.has(value);
        if (category === "restore") return value.length > 0;
        return false;
    }

    static _combineConversions(parts) {
        const engines = ["step_audio_editx", "higgs_audio_v3", "cosyvoice3", "omnivoice"];
        const result = {};
        engines.forEach((engine) => {
            const snippets = [];
            parts.forEach((part) => {
                const snippet = part.conversions?.[engine];
                if (typeof snippet === "string" && snippet.length > 0) {
                    snippets.push(snippet);
                }
            });
            if (snippets.length > 0) {
                result[engine] = snippets.join("");
            }
        });
        return result;
    }

    static _getStepParalinguisticConversions(name) {
        const conversions = {
            step_audio_editx: `<${name}>`,
        };
        const mapping = {
            Laughter: { higgs: "<|sfx:laughter|>", cosy: "<laughter>", omnivoice: "<laughter>" },
            Sigh: { higgs: "<|sfx:sigh|>", cosy: "<sigh>", omnivoice: "<sigh>" },
            Breathing: { cosy: "<breath>" },
            "Surprise-oh": { omnivoice: "<surprise-oh>" },
            "Surprise-ah": { omnivoice: "<surprise-ah>" },
            "Surprise-wa": { omnivoice: "<surprise-wa>" },
            "Confirmation-en": { omnivoice: "<confirmation-en>" },
            "Question-ei": { omnivoice: "<question-ei>" },
            "Dissatisfaction-hnn": { omnivoice: "<dissatisfaction-hnn>" },
        };
        const mapped = mapping[name];
        if (mapped?.higgs) conversions.higgs_audio_v3 = mapped.higgs;
        if (mapped?.cosy) conversions.cosyvoice3 = mapped.cosy;
        if (mapped?.omnivoice) conversions.omnivoice = mapped.omnivoice;
        return conversions;
    }

    static _getStepCategoryConversions(category, value) {
        const conversions = {
            step_audio_editx: `<${category}:${value}>`,
        };

        if (category === "style") {
            if (value === "whisper") conversions.higgs_audio_v3 = "<|style:whispering|>";
        }

        if (category === "emotion") {
            const higgsMap = {
                happy: "elation",
                angry: "anger",
                sad: "sadness",
                excited: "enthusiasm",
                calm: "contentment",
                fearful: "fear",
                surprised: "surprise",
                disgusted: "disgust",
                confusion: "confusion",
                empathy: "affection",
                embarrass: "shame",
                admiration: "awe",
                depressed: "sadness",
            };
            const mapped = higgsMap[value];
            if (mapped) conversions.higgs_audio_v3 = `<|emotion:${mapped}|>`;
        }

        return conversions;
    }

    static _fillHiggsConversions(category, value, conversions) {
        conversions.higgs_audio_v3 = `<|${category}:${value}|>`;

        if (category === "sfx") {
            const stepMap = {
                laughter: "<Laughter>",
                sigh: "<Sigh>",
            };
            const cosyMap = {
                laughter: "<laughter>",
                cough: "<cough>",
                sigh: "<sigh>",
            };
            const omnivoiceMap = {
                laughter: "<laughter>",
                sigh: "<sigh>",
            };
            if (stepMap[value]) conversions.step_audio_editx = stepMap[value];
            if (cosyMap[value]) conversions.cosyvoice3 = cosyMap[value];
            if (omnivoiceMap[value]) conversions.omnivoice = omnivoiceMap[value];
            return;
        }

        if (category === "style" && value === "whispering") {
            conversions.step_audio_editx = "<style:whisper>";
            return;
        }

        if (category === "emotion") {
            const stepMap = {
                elation: "happy",
                enthusiasm: "excited",
                contentment: "calm",
                anger: "angry",
                sadness: "sad",
                fear: "fearful",
                surprise: "surprised",
                disgust: "disgusted",
                confusion: "confusion",
                affection: "empathy",
                shame: "embarrass",
                awe: "admiration",
            };
            const mapped = stepMap[value];
            if (mapped) conversions.step_audio_editx = `<emotion:${mapped}>`;
        }
    }

    static _fillCosySingleConversions(value, conversions) {
        conversions.cosyvoice3 = `<${value}>`;

        const stepMap = {
            laughter: "<Laughter>",
            sigh: "<Sigh>",
            breath: "<Breathing>",
            quick_breath: "<Breathing>",
        };
        const higgsMap = {
            laughter: "<|sfx:laughter|>",
            cough: "<|sfx:cough|>",
            sigh: "<|sfx:sigh|>",
        };
        const omnivoiceMap = {
            laughter: "<laughter>",
            sigh: "<sigh>",
        };
        if (stepMap[value]) conversions.step_audio_editx = stepMap[value];
        if (higgsMap[value]) conversions.higgs_audio_v3 = higgsMap[value];
        if (omnivoiceMap[value]) conversions.omnivoice = omnivoiceMap[value];
    }

    static _fillOmniVoiceConversions(value, conversions) {
        conversions.omnivoice = `<${value}>`;

        const stepMap = {
            laughter: "<Laughter>",
            sigh: "<Sigh>",
            "confirmation-en": "<Confirmation-en>",
            "question-ei": "<Question-ei>",
            "surprise-ah": "<Surprise-ah>",
            "surprise-oh": "<Surprise-oh>",
            "surprise-wa": "<Surprise-wa>",
            "dissatisfaction-hnn": "<Dissatisfaction-hnn>",
        };
        const higgsMap = {
            laughter: "<|sfx:laughter|>",
            sigh: "<|sfx:sigh|>",
        };
        const cosyMap = {
            laughter: "<laughter>",
            sigh: "<sigh>",
        };

        if (stepMap[value]) conversions.step_audio_editx = stepMap[value];
        if (higgsMap[value]) conversions.higgs_audio_v3 = higgsMap[value];
        if (cosyMap[value]) conversions.cosyvoice3 = cosyMap[value];
    }

    static insertTagAtPosition(text, tag, position, wrapSelection = false, selectionStart = -1, selectionEnd = -1) {
        if (wrapSelection && selectionStart >= 0 && selectionEnd >= 0) {
            const before = text.substring(0, selectionStart);
            const selected = text.substring(selectionStart, selectionEnd);
            const after = text.substring(selectionEnd);
            return `${before}${tag} ${selected}${after}`;
        } else {
            return text.substring(0, position) + tag + " " + text.substring(position);
        }
    }

    static normalizeStandalonePauseTags(text) {
        return text.replace(/\[([^\]]+)\]/g, (fullTag, tagContent) => {
            const parts = tagContent.split("|").map(part => part.trim()).filter(Boolean);
            const pauseParts = parts.filter(part => /^(?:pause|wait|stop):\d+(?:\.\d+)?(?:s|ms)?$/i.test(part));
            if (pauseParts.length === 0 || (pauseParts.length === 1 && parts.length === 1)) {
                return fullTag;
            }

            const regularParts = parts.filter(part => !pauseParts.includes(part));
            const standalonePauses = pauseParts.map(part => `[${part}]`).join(" ");
            const regularTag = regularParts.length > 0 ? ` [${regularParts.join("|")}]` : "";
            return standalonePauses + regularTag;
        });
    }

    static modifyTagContent(text, caretPos, modifyFn) {
        // Universal tag modification logic - used by both parameter and character insertion
        // modifyFn(tagContent) should return new tagContent for inside-tag cases
        // Returns {newText, newCaretPos} or null if no modification

        const selectionStart = caretPos;
        // Check if right after tag, or after tag with one space
        let isRightAfterTag = selectionStart > 0 && text[selectionStart - 1] === "]";
        let spaceAfterTag = false;
        if (!isRightAfterTag && selectionStart > 1 && text[selectionStart - 1] === " " && text[selectionStart - 2] === "]") {
            isRightAfterTag = true;
            spaceAfterTag = true;
        }

        // Check if caret is INSIDE a tag (between [ and ])
        let isInsideTag = false;
        let tagStartInside = -1;
        let tagEndInside = -1;

        if (!isRightAfterTag) {
            // Look for the nearest tag that contains this position
            let bracketDepth = 0;
            for (let i = selectionStart - 1; i >= 0; i--) {
                if (text[i] === "]") {
                    bracketDepth++;
                } else if (text[i] === "[") {
                    if (bracketDepth === 0) {
                        tagStartInside = i;
                        let innerDepth = 1;
                        for (let j = i + 1; j < text.length; j++) {
                            if (text[j] === "[") {
                                innerDepth++;
                            } else if (text[j] === "]") {
                                innerDepth--;
                                if (innerDepth === 0) {
                                    tagEndInside = j;
                                    if (tagEndInside >= selectionStart) {
                                        isInsideTag = true;
                                    }
                                    break;
                                }
                            }
                        }
                        break;
                    } else {
                        bracketDepth--;
                    }
                }
            }
        }

        if (isRightAfterTag || isInsideTag) {
            let tagStart, tagEnd;

            if (isRightAfterTag) {
                tagEnd = spaceAfterTag ? selectionStart - 2 : selectionStart - 1;
                let bracketDepth = 1;
                tagStart = -1;
                for (let i = tagEnd - 1; i >= 0; i--) {
                    if (text[i] === "]") {
                        bracketDepth++;
                    } else if (text[i] === "[") {
                        bracketDepth--;
                        if (bracketDepth === 0) {
                            tagStart = i;
                            break;
                        }
                    }
                }
            } else {
                tagStart = tagStartInside;
                tagEnd = tagEndInside;
            }

            if (tagStart !== -1 && tagStart < tagEnd) {
                let tagContent = text.substring(tagStart + 1, tagEnd);
                tagContent = modifyFn(tagContent);
                const newText = text.substring(0, tagStart + 1) + tagContent + "]" + text.substring(tagEnd + 1);
                const newCaretPos = tagStart + 1 + tagContent.length + 1;
                return { newText, newCaretPos };
            }
        }

        return null;
    }
}
