/**
 * 🏷️ Tag Utilities
 * Utility functions for tag operations, parsing, validation, and modification
 */

export class TagUtilities {
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
