/**
 * 🏷️ Syntax Highlighter
 * Handles syntax highlighting for SRT content, tags, and special characters
 */

export class SyntaxHighlighter {
    static getPlainText(editor) {
        const clone = editor.cloneNode(true);
        const spans = clone.querySelectorAll('span');
        spans.forEach(span => {
            while (span.firstChild) {
                span.parentNode.insertBefore(span.firstChild, span);
            }
            span.parentNode.removeChild(span);
        });
        return clone.textContent;
    }

    static getCaretPos(editor) {
        const selection = window.getSelection();
        if (selection.rangeCount === 0) return 0;

        const range = selection.getRangeAt(0);
        const preRange = range.cloneRange();
        preRange.selectNodeContents(editor);
        preRange.setEnd(range.endContainer, range.endOffset);

        const tempDiv = document.createElement('div');
        tempDiv.appendChild(preRange.cloneContents());

        const spans = tempDiv.querySelectorAll('span');
        spans.forEach(span => {
            while (span.firstChild) {
                span.parentNode.insertBefore(span.firstChild, span);
            }
            span.parentNode.removeChild(span);
        });

        return tempDiv.textContent.length;
    }

    static setCaretPos(editor, pos) {
        const selection = window.getSelection();
        const range = document.createRange();
        let charCount = 0;
        let nodeStack = [editor];
        let node;
        let foundStart = false;

        while (!foundStart && (node = nodeStack.pop())) {
            if (node.nodeType === Node.TEXT_NODE) {
                const nextCharCount = charCount + node.length;
                if (pos <= nextCharCount) {
                    range.setStart(node, pos - charCount);
                    foundStart = true;
                }
                charCount = nextCharCount;
            } else {
                let i = node.childNodes.length;
                while (i--) {
                    nodeStack.push(node.childNodes[i]);
                }
            }
        }

        if (foundStart) {
            range.collapse(true);
            selection.removeAllRanges();
            selection.addRange(range);
        }
    }

    static updateHighlights(editor) {
        const plainText = this.getPlainText(editor);
        const caretPos = this.getCaretPos(editor);
        let html = plainText;

        // Highlight SRT sequence numbers - bright red
        html = html.replace(
            /^(\d+)\s*\n(\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3})/gm,
            '\x00NUM_START\x00$1\x00NUM_END\x00\n$2'
        );

        // Highlight SRT timings - bright orange
        html = html.replace(
            /\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}/g,
            '\x00SRT_START\x00$&\x00SRT_END\x00'
        );

        // Highlight tags - bright cyan
        html = html.replace(
            /(\[[^\]]+\])/g,
            '\x00TAG_START\x00$1\x00TAG_END\x00'
        );

        // Highlight commas - green
        html = html.replace(/,/g, '\x00COMMA_START\x00,\x00COMMA_END\x00');

        // Highlight periods - golden yellow
        html = html.replace(/\./g, '\x00PERIOD_START\x00.\x00PERIOD_END\x00');

        // Highlight punctuation - light salmon
        html = html.replace(/[?!;]/g, '\x00PUNCT_START\x00$&\x00PUNCT_END\x00');

        // Highlight multiple spaces (2 or more) - subtle background
        html = html.replace(/  +/g, '\x00SPACE_START\x00$&\x00SPACE_END\x00');

        // Escape HTML
        html = html
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");

        // Replace placeholders with spans
        html = html
            .replace(/\x00NUM_START\x00(.*?)\x00NUM_END\x00/g, '<span style="color: #ff5555; font-weight: bold;">$1</span>')
            .replace(/\x00SRT_START\x00(.*?)\x00SRT_END\x00/g, '<span style="color: #ffaa00; font-weight: bold;">$1</span>')
            .replace(/\x00TAG_START\x00(.*?)\x00TAG_END\x00/g, '<span style="color: #00ffff; font-weight: bold;">$1</span>')
            .replace(/\x00COMMA_START\x00(.*?)\x00COMMA_END\x00/g, '<span style="color: #66ff66; font-weight: bold;">$1</span>')
            .replace(/\x00PERIOD_START\x00(.*?)\x00PERIOD_END\x00/g, '<span style="color: #ffcc33; font-weight: bold;">$1</span>')
            .replace(/\x00PUNCT_START\x00(.*?)\x00PUNCT_END\x00/g, '<span style="color: #ff9999;">$1</span>')
            .replace(/\x00SPACE_START\x00(.*?)\x00SPACE_END\x00/g, '<span style="background: #2a2a2a; color: #eee;">$1</span>');

        // Update only if changed to avoid flicker
        if (editor.innerHTML !== html) {
            editor.innerHTML = html;
            this.setCaretPos(editor, caretPos);
        }
    }
}
