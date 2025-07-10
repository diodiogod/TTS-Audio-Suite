import { app } from "../../scripts/app.js";
import { AudioAnalyzerInterface } from "./audio_analyzer_core.js";

// Import CSS - try multiple possible paths
const cssLink = document.createElement('link');
cssLink.rel = 'stylesheet';
cssLink.href = './extensions/ComfyUI_ChatterBox_Voice/audio_analyzer.css';
cssLink.onerror = () => {
    console.warn('Failed to load CSS from extensions path, trying web path');
    cssLink.href = './web/extensions/ComfyUI_ChatterBox_Voice/audio_analyzer.css';
};
document.head.appendChild(cssLink);

console.log('🎵 Audio Analyzer: Extension script loaded');

// ComfyUI Extension Registration
console.log('🎵 Audio Analyzer: Registering extension with ComfyUI');

app.registerExtension({
    name: "chatterbox.audio_analyzer",
    
    async beforeRegisterNodeDef(nodeType, nodeData) {
        // Only log our target node to reduce console spam
        if (nodeData.name === "ChatterBoxAudioAnalyzer" || nodeData.name === "AudioAnalyzerNode") {
            console.log("🔍 Audio Analyzer: Found target node type:", nodeData.name);
            console.log("✅ Audio Analyzer: Registering UI for ChatterBoxAudioAnalyzer");
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                console.log("🎵 Audio Analyzer: Creating node instance");
                const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                // Set larger size for the interface
                this.setSize([800, 600]);
                
                // Create the audio analyzer interface
                try {
                    console.log('🎵 Audio Analyzer: Creating interface for node:', this);
                    const analyzerInterface = new AudioAnalyzerInterface(this);
                    this.audioAnalyzerInterface = analyzerInterface;
                    console.log('✅ Audio Analyzer: Interface created successfully');
                    
                    // Setup execution monitoring
                    app.registerExtension.setupNodeExecution(this);
                } catch (error) {
                    console.error('❌ Audio Analyzer: Failed to create interface:', error);
                }
                
                return result;
            };
            
            // Override onExecuted to handle visualization data
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                console.log("🎵 Audio Analyzer: Node executed with message:", message);
                const result = onExecuted ? onExecuted.apply(this, arguments) : undefined;
                
                // Check if we have visualization data in the execution results
                if (message && message.visualization_data && this.audioAnalyzerInterface) {
                    try {
                        // Parse visualization data if it's a JSON string
                        let vizData = message.visualization_data;
                        if (typeof vizData === 'string') {
                            vizData = JSON.parse(vizData);
                        }
                        
                        setTimeout(() => {
                            console.log('🎵 Audio Analyzer: Parsed visualization data:', vizData);
                            this.audioAnalyzerInterface.updateVisualization(vizData);
                        }, 100);
                    } catch (error) {
                        console.error('❌ Audio Analyzer: Failed to parse visualization data:', error);
                        console.log('Raw visualization data:', message.visualization_data);
                        if (this.audioAnalyzerInterface?.ui) {
                            this.audioAnalyzerInterface.ui.updateStatus('Failed to parse visualization data');
                        }
                    }
                }
                
                return result;
            };
        }
    },
    
    // Fallback: Also try to catch nodes after they're created
    async nodeCreated(node) {
        // Check for our target node but only if no interface exists yet
        if ((node.comfyClass === "AudioAnalyzerNode" || node.comfyClass === "ChatterBoxAudioAnalyzer") && !node.audioAnalyzerInterface) {
            console.log(`🎵 Audio Analyzer: Found Audio Analyzer node with comfyClass: ${node.comfyClass}`);
            console.log("🎵 Audio Analyzer: Creating interface via nodeCreated fallback");
            
            // Wait for node to be fully initialized
            await new Promise(resolve => setTimeout(resolve, 200));
            
            try {
                // Set larger size for the interface
                node.setSize([800, 600]);
                
                // Create the audio analyzer interface
                console.log('🎵 Audio Analyzer: Creating interface for node:', node);
                const analyzerInterface = new AudioAnalyzerInterface(node);
                node.audioAnalyzerInterface = analyzerInterface;
                console.log('✅ Audio Analyzer: Interface created successfully via nodeCreated');
                
                // Monitor for node execution completion
                this.setupNodeExecution(node);
                
            } catch (error) {
                console.error('❌ Audio Analyzer: Failed to create interface:', error);
            }
        }
    },
    
    // Shared execution setup to avoid duplication
    setupNodeExecution(node) {
        const originalOnExecuted = node.onExecuted;
        node.onExecuted = function(message) {
            console.log("🎵 Audio Analyzer: Node executed with message:", message);
            if (originalOnExecuted) {
                originalOnExecuted.call(this, message);
            }
            
            // Check if we have visualization data in the execution results
            if (message && message.visualization_data && this.audioAnalyzerInterface) {
                try {
                    // Parse visualization data if it's a JSON string
                    let vizData = message.visualization_data;
                    if (typeof vizData === 'string') {
                        vizData = JSON.parse(vizData);
                    }
                    
                    setTimeout(() => {
                        console.log('🎵 Audio Analyzer: Parsed visualization data:', vizData);
                        this.audioAnalyzerInterface.updateVisualization(vizData);
                    }, 100);
                } catch (error) {
                    console.error('❌ Audio Analyzer: Failed to parse visualization data:', error);
                    console.log('Raw visualization data:', message.visualization_data);
                    if (this.audioAnalyzerInterface?.ui) {
                        this.audioAnalyzerInterface.ui.updateStatus('Failed to parse visualization data');
                    }
                }
            }
        };
    }
});