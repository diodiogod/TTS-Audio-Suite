/**
 * Shared export helpers for the IndexTTS-2 emotion vector controls.
 *
 * Clipboard access is optional in ComfyUI because the UI may be served from
 * an insecure origin or denied browser permissions.  Export must still work
 * in those environments, so the JSON download and selectable preview do not
 * depend on the clipboard API.
 */

const MODAL_ID = "tts-audio-suite-emotion-export-modal";
const TOAST_ID = "tts-audio-suite-emotion-feedback";
const DEFAULT_FILENAME = "index_tts_2_emotion_vectors.json";
const MODAL_Z_INDEX = 100010;
const TOAST_Z_INDEX = MODAL_Z_INDEX + 1;
let toastTimeout = null;

export function showEmotionFeedback(message, level = "success") {
    const existingToast = document.getElementById(TOAST_ID);
    if (existingToast) {
        existingToast.remove();
    }

    const toast = document.createElement("div");
    toast.id = TOAST_ID;
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed;
        left: 50%;
        bottom: 24px;
        z-index: ${TOAST_Z_INDEX};
        transform: translateX(-50%);
        max-width: min(520px, calc(100vw - 40px));
        padding: 9px 14px;
        color: #fff;
        background: ${level === "error" ? "#a33" : "#287a4b"};
        border: 1px solid rgba(255, 255, 255, 0.25);
        border-radius: 5px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
        font: 12px Arial, sans-serif;
        text-align: center;
    `;
    document.body.appendChild(toast);

    if (toastTimeout) {
        clearTimeout(toastTimeout);
    }
    toastTimeout = setTimeout(() => toast.remove(), 2600);
}

function downloadJson(text, filename) {
    const blob = new Blob([text], { type: "application/json;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = filename;
    link.style.display = "none";
    document.body.appendChild(link);
    link.click();
    link.remove();

    // Keep the object URL alive until the browser has started the download.
    setTimeout(() => URL.revokeObjectURL(url), 0);
}

async function copyText(text) {
    if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
        return true;
    }

    const fallback = document.createElement("textarea");
    fallback.value = text;
    fallback.setAttribute("readonly", "");
    fallback.style.position = "fixed";
    fallback.style.left = "-9999px";
    document.body.appendChild(fallback);
    fallback.select();

    let copied = false;
    try {
        copied = document.execCommand("copy");
    } finally {
        fallback.remove();
    }
    return copied;
}

function closeModal(modal) {
    if (modal?.parentNode) {
        modal.parentNode.removeChild(modal);
    }
}

function createButton(label, background, onClick) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.style.cssText = `
        background: ${background};
        color: #fff;
        border: 0;
        border-radius: 4px;
        padding: 7px 14px;
        cursor: pointer;
        font-size: 12px;
    `;
    button.addEventListener("click", onClick);
    return button;
}

function showEmotionConfigModal({ mode, text = "", filename = DEFAULT_FILENAME, onApply }) {
    const isImport = mode === "import";
    const existingModal = document.getElementById(MODAL_ID);
    closeModal(existingModal);

    const modal = document.createElement("div");
    modal.id = MODAL_ID;
    modal.style.cssText = `
        position: fixed;
        inset: 0;
        z-index: ${MODAL_Z_INDEX};
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 20px;
        box-sizing: border-box;
        background: rgba(0, 0, 0, 0.72);
        font-family: Arial, sans-serif;
    `;

    const closeOnEscape = event => {
        if (event.key === "Escape") {
            closeModalAndCleanup();
        }
    };
    const closeModalAndCleanup = () => {
        closeModal(modal);
        document.removeEventListener("keydown", closeOnEscape);
    };

    const dialog = document.createElement("div");
    dialog.setAttribute("role", "dialog");
    dialog.setAttribute("aria-modal", "true");
    dialog.style.cssText = `
        width: min(520px, 100%);
        max-height: min(620px, 100%);
        display: flex;
        flex-direction: column;
        gap: 10px;
        padding: 18px;
        box-sizing: border-box;
        color: #eee;
        background: #242424;
        border: 1px solid #555;
        border-radius: 8px;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.55);
    `;

    const title = document.createElement("h2");
    title.textContent = "IndexTTS-2 Emotion Vectors";
    title.style.cssText = "margin: 0; font-size: 16px;";

    const description = document.createElement("div");
    description.textContent = isImport
        ? "Paste or edit a JSON emotion configuration, then apply it."
        : `Exported as ${filename}. Select or copy the JSON below.`;
    description.style.cssText = "color: #bbb; font-size: 12px;";

    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.readOnly = !isImport;
    textarea.placeholder = isImport
        ? '{\n  "Happy": 0.5,\n  "Angry": 0.0,\n  "Sad": 0.2\n}'
        : "";
    textarea.setAttribute("aria-label", isImport ? "Emotion vector JSON to import" : "Emotion vector JSON");
    textarea.style.cssText = `
        width: 100%;
        min-height: 220px;
        resize: vertical;
        box-sizing: border-box;
        padding: 10px;
        color: #eee;
        background: #151515;
        border: 1px solid #555;
        border-radius: 4px;
        font: 12px/1.45 Consolas, monospace;
    `;

    const status = document.createElement("div");
    status.style.cssText = `min-height: 16px; color: ${isImport ? "#bbb" : "#8fd694"}; font-size: 12px;`;
    status.textContent = isImport
        ? "Paste JSON above, then choose Apply."
        : "Use Download JSON when you are ready to save a file.";

    const actions = document.createElement("div");
    actions.style.cssText = "display: flex; justify-content: flex-end; gap: 8px;";

    if (isImport) {
        const applyButton = createButton("Apply", "#4ecdc4", () => {
            try {
                const result = onApply?.(textarea.value);
                if (!result?.ok) {
                    status.textContent = result?.message || "Import failed. Check the JSON and try again.";
                    status.style.color = "#f0c674";
                    return;
                }
                closeModalAndCleanup();
            } catch (error) {
                console.error("Emotion vector import failed:", error);
                status.textContent = `Import failed: ${error.message}`;
                status.style.color = "#f0c674";
            }
        });
        const cancelButton = createButton("Cancel", "#666", closeModalAndCleanup);
        actions.append(applyButton, cancelButton);
    } else {
        const copyButton = createButton("Copy", "#45b7d1", async () => {
            try {
                const copied = await copyText(text);
                status.textContent = copied ? "Copied to clipboard." : "Copy was not available; select the text manually.";
                status.style.color = copied ? "#8fd694" : "#f0c674";
            } catch (error) {
                console.warn("Emotion vector clipboard copy failed:", error);
                status.textContent = "Copy was blocked; select the text manually.";
                status.style.color = "#f0c674";
            }
        });

        const downloadButton = createButton("Download JSON", "#4ecdc4", () => {
            try {
                downloadJson(text, filename);
                status.textContent = "JSON download started.";
                status.style.color = "#8fd694";
            } catch (error) {
                console.error("Emotion vector JSON download failed:", error);
                status.textContent = "Download failed; select and save the JSON manually.";
                status.style.color = "#f0c674";
            }
        });

        const closeButton = createButton("Close", "#666", closeModalAndCleanup);
        actions.append(copyButton, downloadButton, closeButton);
    }
    dialog.append(title, description, textarea, status, actions);
    modal.appendChild(dialog);
    modal.addEventListener("click", event => {
        if (event.target === modal) {
            closeModalAndCleanup();
        }
    });
    document.body.appendChild(modal);
    document.addEventListener("keydown", closeOnEscape);

    textarea.focus();
    if (!isImport) {
        textarea.select();
    }
}

export function exportEmotionConfiguration(config, filename = DEFAULT_FILENAME) {
    const text = JSON.stringify(config, null, 2);
    showEmotionConfigModal({ mode: "export", text, filename });
    return text;
}

export function importEmotionConfiguration(onApply) {
    showEmotionConfigModal({ mode: "import", onApply });
}
