import { app } from "../../scripts/app.js";
import { createEmotionRadarCanvasWidget } from "./emotion_radar_canvas_widget.js";

/**
 * IndexTTS-2 Emotion Radar Chart Integration
 * Adds a canvas-based radar chart widget to 🌈 IndexTTS-2 Emotion Vectors node
 */

// Register the extension
app.registerExtension({
    name: "TTS_Audio_Suite.IndexTTSEmotionRadar",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Only target the 🌈 IndexTTS-2 Emotion Vectors node
        if (nodeData.name === "IndexTTSEmotionOptionsNode") {
            console.log("🎭 Registering IndexTTS-2 Emotion Radar Chart extension");

            // Store original node creation method
            const originalNodeCreated = nodeType.prototype.onNodeCreated;

            // Override node creation to add radar chart
            nodeType.prototype.onNodeCreated = function() {
                // Call original creation first
                if (originalNodeCreated) {
                    originalNodeCreated.apply(this, arguments);
                }

                // Set appropriate node size for the radar chart (compact default)
                this.setSize([340, 580]); // Width, Height to accommodate sliders + radar

                try {
                    // Create and add the radar chart canvas widget
                    const radarWidget = createEmotionRadarCanvasWidget(this);

                    if (!this.widgets) {
                        this.widgets = [];
                    }

                    // Add radar widget at the end
                    this.widgets.push(radarWidget);

                    console.log("🎭 IndexTTS-2 Emotion Radar Chart: Successfully added canvas widget");
                } catch (error) {
                    console.error("🎭 Failed to add emotion radar chart widget:", error);
                }
            };
        }
    }
});

console.log("🎭 IndexTTS-2 Emotion Radar Chart extension loaded");