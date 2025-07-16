// ChatterBox Voice Capture Extension

import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "ChatterBoxVoiceCapture.UI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "ChatterBoxVoiceCaptureDiogod") {
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                // Set larger size
                this.setSize([420, 280]);
                
                // Initialize recording state
                this.isRecording = false;
                this.recordingTimeout = null;
                
                // Hide trigger widget
                setTimeout(() => {
                    const triggerWidget = this.widgets?.find(w => w.name === "voice_trigger");
                    if (triggerWidget) {
                        triggerWidget.type = "hidden";
                        triggerWidget.value = 0;
                        console.log("🔧 ChatterBox: Trigger widget hidden");
                    }
                }, 100);
                
                return result;
            };

            nodeType.prototype.onDrawForeground = function(ctx) {
                const size = this.size;
                const w = size[0];
                const h = size[1];
                
                // Button area
                const buttonX = 20;
                const buttonY = h - 70;
                const buttonW = w - 40;
                const buttonH = 40;
                
                // Draw button
                ctx.fillStyle = this.isRecording ? "#ff4444" : "#44aa44";
                ctx.fillRect(buttonX, buttonY, buttonW, buttonH);
                
                // Button border
                ctx.strokeStyle = this.isRecording ? "#ff0000" : "#00aa00";
                ctx.lineWidth = 2;
                ctx.strokeRect(buttonX, buttonY, buttonW, buttonH);
                
                // Button text
                ctx.fillStyle = "#ffffff";
                ctx.font = "bold 14px Arial";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                
                const text = this.isRecording ? "🔴 RECORDING..." : "🎙️ START RECORDING";
                ctx.fillText(text, w / 2, buttonY + buttonH / 2);
                
                // Store button area for click detection
                this.buttonArea = [buttonX, buttonY, buttonW, buttonH];
            };

            nodeType.prototype.onMouseDown = function(event, localPos) {
                if (!this.buttonArea) return false;
                
                const [x, y, w, h] = this.buttonArea;
                
                // Check if click is within button
                if (localPos[0] >= x && localPos[0] <= x + w &&
                    localPos[1] >= y && localPos[1] <= y + h) {
                    
                    console.log("🎯 ChatterBox: Button clicked!");
                    
                    if (!this.isRecording) {
                        // Start recording
                        this.isRecording = true;
                        console.log("▶️ ChatterBox: Starting recording");
                        
                        // Auto-stop after 10 seconds
                        this.recordingTimeout = setTimeout(() => {
                            this.isRecording = false;
                            console.log("⏹️ ChatterBox: Recording stopped (timeout)");
                            app.graph.setDirtyCanvas(true);
                        }, 10000);
                        
                        // Trigger the Python node
                        const triggerWidget = this.widgets?.find(w => w.name === "voice_trigger");
                        if (triggerWidget) {
                            triggerWidget.value = (triggerWidget.value || 0) + 1;
                            console.log("🔢 ChatterBox: Trigger value:", triggerWidget.value);
                        }
                        
                        // Execute the node
                        app.queuePrompt();
                    } else {
                        // Stop recording
                        this.isRecording = false;
                        if (this.recordingTimeout) {
                            clearTimeout(this.recordingTimeout);
                        }
                        console.log("⏹️ ChatterBox: Recording stopped (manual)");
                    }
                    
                    // Redraw the canvas
                    app.graph.setDirtyCanvas(true);
                    return true;
                }
                
                return false;
            };
            
            console.log("✅ ChatterBox: UI extension registered successfully");
        }
    }
});

console.log("🎙️ ChatterBox: Extension loaded");