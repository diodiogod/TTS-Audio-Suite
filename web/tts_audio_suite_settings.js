import { app } from '../../scripts/app.js'

const SETTING_CATEGORY = "TTS Audio Suite";
const SETTING_SECTION_STEP_EDITX = "Step Audio EditX (Inline Tags)";
const SETTING_SECTION_RESTORE_VC = "Voice Restoration (<restore> Tags)";

// Shared function to send settings to backend
async function sendSettingsToBackend() {
    try {
        const precision = app.ui.settings.getSettingValue("TTSAudioSuite.InlineEditTags.Precision", "auto");
        const device = app.ui.settings.getSettingValue("TTSAudioSuite.InlineEditTags.Device", "auto");
        const vcEngine = app.ui.settings.getSettingValue("TTSAudioSuite.RestoreTags.VCEngine", "chatterbox_23lang");
        const cosyvoiceVariant = app.ui.settings.getSettingValue("TTSAudioSuite.RestoreTags.CosyVoiceVariant", "RL");

        console.log(`TTS Audio Suite: Sending settings to backend - precision=${precision}, device=${device}, vcEngine=${vcEngine}, cosyvoiceVariant=${cosyvoiceVariant}`);

        const response = await fetch("/api/tts-audio-suite/settings", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                precision: precision,
                device: device,
                vc_engine: vcEngine,
                cosyvoice_variant: cosyvoiceVariant
            })
        });

        if (!response.ok) {
            console.error(`TTS Audio Suite: Failed to send settings to backend (status ${response.status})`);
        } else {
            const result = await response.json();
            console.log("TTS Audio Suite: Backend confirmed settings:", result);
        }
    } catch (error) {
        console.error("TTS Audio Suite: Error sending settings to backend:", error);
    }
}

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
                "Note: This ONLY affects inline edit tags. When using the Step Audio EditX node directly, use the node's precision parameter instead.\n\n" +
                "⚠️ IMPORTANT: Refresh browser (F5) after changing settings for them to take effect."
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
                "Note: This ONLY affects inline edit tags. When using the Step Audio EditX node directly, use the node's precision parameter instead.\n\n" +
                "⚠️ IMPORTANT: Refresh browser (F5) after changing settings for them to take effect."
        },
        {
            id: "TTSAudioSuite.RestoreTags.VCEngine",
            name: "Voice Conversion Engine for <restore> Tags",
            type: "combo",
            defaultValue: "chatterbox_23lang",
            options: ["chatterbox_23lang", "chatterbox", "cosyvoice"],
            category: [SETTING_CATEGORY, SETTING_SECTION_RESTORE_VC, "VC Engine"],
            tooltip:
                "Voice conversion engine used for <restore> tags in post-processing.\n" +
                "ChatterBox 23-Lang (default): Best quality, supports 23 languages\n" +
                "ChatterBox: Original engine, English/German/Norwegian\n" +
                "CosyVoice: New option with native CosyVoice3 VC (24kHz output)\n\n" +
                "⚠️ IMPORTANT: Refresh browser (F5) after changing settings for them to take effect."
        },
        {
            id: "TTSAudioSuite.RestoreTags.CosyVoiceVariant",
            name: "CosyVoice Model Variant for <restore> Tags",
            type: "combo",
            defaultValue: "RL",
            options: ["RL", "standard"],
            category: [SETTING_CATEGORY, SETTING_SECTION_RESTORE_VC, "CosyVoice Variant"],
            tooltip:
                "CosyVoice model variant to use for voice restoration (only applies when VC Engine is CosyVoice).\n" +
                "RL: Fun-CosyVoice3-0.5B-RL (reinforcement learning variant, default)\n" +
                "standard: Fun-CosyVoice3-0.5B (standard variant)\n\n" +
                "💡 Tip: Match this to the variant you use for TTS to avoid model unload/reload during restore.\n\n" +
                "⚠️ IMPORTANT: Refresh browser (F5) after changing settings for them to take effect."
        }
    ],
    async setup() {
        // Send settings on initial load (with delay to ensure settings are loaded)
        setTimeout(() => {
            sendSettingsToBackend();
            console.log("TTS Audio Suite: Initial settings sent to backend");
        }, 1000);

        // Listen for setting changes and send to backend
        const originalChangeSettings = app.ui.settings.setSettingValue;
        app.ui.settings.setSettingValue = function(id, value) {
            originalChangeSettings.call(this, id, value);

            // If it's one of our settings, send to backend immediately
            if (id.startsWith("TTSAudioSuite.InlineEditTags.") || id.startsWith("TTSAudioSuite.RestoreTags.")) {
                sendSettingsToBackend();
            }
        };

        console.log("TTS Audio Suite: Settings registered and synced with backend");
    }
})

// Register separate extension for beforeQueuing hook (must be at top level)
app.registerExtension({
    name: "TTS_Audio_Suite.Settings.BeforePrompt",
    async beforeQueuing() {
        await sendSettingsToBackend();
        return null;
    }
})
