import { app } from '../../scripts/app.js'

const SETTING_CATEGORY = "TTS Audio Suite";
const SETTING_SECTION_STEP_EDITX = "Step Audio EditX (Inline Tags)";

app.registerExtension({
    name: "TTS_Audio_Suite.Settings",
    settings: [
        {
            id: "TTSAudioSuite.InlineEditTags.Precision",
            name: "Model Precision for Inline Edit Tags",
            type: "combo",
            defaultValue: "auto",
            options: ["auto", "fp32", "fp16", "bf16", "int8", "int4"],
            category: [SETTING_CATEGORY, SETTING_SECTION_STEP_EDITX, "Model Precision"],
            tooltip:
                "Torch dtype for Step Audio EditX model when using inline edit tags (<Laughter>, <style:whisper>, etc.).\n" +
                "Use int8 or int4 for low VRAM systems.\n" +
                "Note: This ONLY affects inline edit tags. When using the Step Audio EditX node directly, use the node's precision parameter instead."
        },
        {
            id: "TTSAudioSuite.InlineEditTags.Device",
            name: "Device for Inline Edit Tags",
            type: "combo",
            defaultValue: "auto",
            options: ["auto", "cuda", "cpu", "xpu"],
            category: [SETTING_CATEGORY, SETTING_SECTION_STEP_EDITX, "Device"],
            tooltip:
                "Device for Step Audio EditX model when using inline edit tags (<Laughter>, <style:whisper>, etc.).\n" +
                "Auto selects best available device (cuda > xpu > cpu).\n" +
                "Note: This ONLY affects inline edit tags. When using the Step Audio EditX node directly, use the node's device parameter instead."
        }
    ],
    async setup() {
        // Send settings to Python backend whenever they change
        const sendSettingsToBackend = async () => {
            try {
                const precision = app.ui.settings.getSettingValue("TTSAudioSuite.InlineEditTags.Precision", "auto");
                const device = app.ui.settings.getSettingValue("TTSAudioSuite.InlineEditTags.Device", "auto");

                const response = await fetch("/api/tts-audio-suite/settings", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        precision: precision,
                        device: device
                    })
                });

                if (!response.ok) {
                    console.error("TTS Audio Suite: Failed to send settings to backend");
                }
            } catch (error) {
                console.error("TTS Audio Suite: Error sending settings to backend:", error);
            }
        };

        // Send settings on initial load (with delay to ensure settings are loaded)
        setTimeout(() => {
            sendSettingsToBackend();
            console.log("TTS Audio Suite: Initial settings sent to backend");
        }, 1000);

        // Send settings before each prompt execution
        app.registerExtension({
            name: "TTS_Audio_Suite.Settings.BeforePrompt",
            async beforeQueuing() {
                await sendSettingsToBackend();
                return null;
            }
        });

        // Listen for setting changes and send to backend
        const originalChangeSettings = app.ui.settings.setSettingValue;
        app.ui.settings.setSettingValue = function(id, value) {
            originalChangeSettings.call(this, id, value);

            // If it's one of our settings, send to backend immediately
            if (id.startsWith("TTSAudioSuite.InlineEditTags.")) {
                sendSettingsToBackend();
            }
        };

        console.log("TTS Audio Suite: Settings registered and synced with backend");
    }
})
