import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const SETTING_ID = "TTSAudioSuite.Runtime.InstallShared";
let syncing = true;
let currentValue = true;
let desiredValue = true;
let saveQueue = Promise.resolve();
let status;

async function syncCheckbox(value) {
    syncing = true;
    try {
        if (app.extensionManager?.setting) {
            await app.extensionManager.setting.set(SETTING_ID, value);
        } else {
            await app.ui.settings.setSettingValueAsync(SETTING_ID, value);
        }
    } finally { syncing = false; }
}

function notify(message, severity = "info") {
    app.extensionManager.toast.add({ severity, summary: "TTS Audio Suite runtimes", detail: message, life: 12000 });
}

async function loadStatus() {
    await saveQueue;
    const response = await api.fetchApi("/tts-audio-suite/runtime-settings");
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Unable to read runtime settings.");
    status = data;
    currentValue = data.install_shared_runtime;
    desiredValue = currentValue;
    await syncCheckbox(currentValue);
}

app.registerExtension({
    name: "tts-audio-suite.runtime-installation",
    settings: [{
        id: SETTING_ID,
        name: "Install shared runtime automatically",
        type: "boolean",
        defaultValue: true,
        category: ["TTS Audio Suite", "Runtime installation", "Install shared runtime automatically"],
        tooltip: "Recommended. Applies when Manager installs, updates, or repairs the suite. Turning it off keeps any installed runtime. Restarting or running a workflow does not install packages.",
        async onChange(value, oldValue) {
            if (syncing || oldValue === undefined) return;
            desiredValue = value;
            // Preserve the final choice when users toggle again before a save finishes.
            saveQueue = saveQueue.then(async () => {
                try {
                    const response = await api.fetchApi("/tts-audio-suite/runtime-settings", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ install_shared_runtime: value }),
                    });
                    const data = await response.json();
                    if (!response.ok) throw new Error(data.error || "Unable to save runtime settings.");
                    status = data;
                    currentValue = data.install_shared_runtime;
                    if (value === desiredValue) notify(value
                        ? "Enabled for the next Manager install, update, or repair. Restart when Manager requests it."
                        : "Automatic installation disabled. Existing runtimes are kept. Use Runtime installation details for their folder and cleanup instructions.");
                } catch (error) {
                    if (value === desiredValue) {
                        desiredValue = currentValue;
                        await syncCheckbox(currentValue);
                    }
                    notify(error.message, "error");
                }
            });
            await saveQueue;
        },
    }],
    commands: [{
        id: "TTSAudioSuite.Runtime.Details",
        label: "TTS Audio Suite: Runtime installation details",
        async function() {
            try {
                await loadStatus();
                const dialog = document.createElement("dialog");
                dialog.style.cssText = "max-width:800px;max-height:80vh;overflow:auto;padding:24px;background:var(--comfy-menu-bg,#222);color:var(--input-text,#ddd);border:1px solid var(--border-color,#555);border-radius:8px";
                const title = document.createElement("h3");
                title.textContent = "TTS Audio Suite runtime installation";
                const message = document.createElement("pre");
                message.style.whiteSpace = "pre-wrap";
                message.textContent = `Shared Runtime: ${status.runtime_ready ? "ready" : "needs installation or repair"}\n\nSettings file:\n${status.settings_file}\n\nRuntime folder:\n${status.runtime_folder}\n\nTo remove an unused shared runtime: disable automatic installation, close ComfyUI, then delete the runtime folder above. Model weights and voice files are stored separately. Engines set to Shared Runtime need it restored to run.\n\nTo restore support: enable automatic installation and repair/reinstall TTS Audio Suite through Manager.`;
                const close = document.createElement("button");
                close.textContent = "Close";
                close.onclick = () => dialog.close();
                dialog.append(title, message, close);
                dialog.addEventListener("close", () => dialog.remove(), { once: true });
                document.body.append(dialog);
                dialog.showModal();
            } catch (error) { notify(error.message, "error"); }
        },
    }],
    menuCommands: [{ path: ["TTS Audio Suite"], commands: ["TTSAudioSuite.Runtime.Details"] }],
    async setup() {
        try { await loadStatus(); }
        catch (error) { notify(error.message, "error"); }
    },
});
