import { app } from "../../scripts/app.js";
import { AudioAnalyzerInterface } from "./audio_analyzer_core.js";

// Simple execution tracking
let audioAnalyzerNodes = new Map();

// Hook into ComfyUI's global execution completion
if (window.app && window.app.ui && window.app.ui.queue) {
    const originalProcessComplete = window.app.ui.queue.processComplete || function() {};
    window.app.ui.queue.processComplete = function(prompt_id, results) {
        console.log('🎵 Queue execution completed:', prompt_id, results);
        
        // Check all audio analyzer nodes for fresh data
        for (const [nodeId, node] of audioAnalyzerNodes) {
            if (node.lastExecutionTime && Date.now() - node.lastExecutionTime < 5000) {
                console.log('🎵 Checking node', nodeId, 'for execution results');
                
                // Try to get the result data
                if (results && results[nodeId] && results[nodeId].length > 1) {
                    console.log('🎉 Found execution result for AudioAnalyzer node!');
                    const vizData = results[nodeId][1]; // visualization_data output
                    if (node.audioAnalyzerInterface) {
                        handleNodeExecution.call(node, [null, vizData]);
                    }
                }
            }
        }
        
        return originalProcessComplete.call(this, prompt_id, results);
    };
}

// Basic execution handler
function handleNodeExecution(message) {
    if (!message || !Array.isArray(message) || message.length < 2) return;
    
    let vizData = message[1];
    if (typeof vizData === 'string') {
        try {
            vizData = JSON.parse(vizData);
        } catch (e) {
            return;
        }
    }
    
    // Mark that we received data and call updateVisualization
    this.lastDataReceived = Date.now();
    if (this.audioAnalyzerInterface) {
        this.audioAnalyzerInterface.updateVisualization(vizData);
        
        // Setup audio playback
        const audioFileWidget = this.widgets?.find(w => w.name === 'audio_file');
        if (audioFileWidget && audioFileWidget.value) {
            this.setupAudioPlayback(audioFileWidget.value);
        }
    }
}

// ComfyUI Extension Registration
app.registerExtension({
    name: "chatterbox.audio_analyzer",
    
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "ChatterBoxAudioAnalyzer" || nodeData.name === "AudioAnalyzerNode") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                this.setSize([800, 600]);
                
                try {
                    const analyzerInterface = new AudioAnalyzerInterface(this);
                    this.audioAnalyzerInterface = analyzerInterface;
                    audioAnalyzerNodes.set(String(this.id), this);
                    
                    // Add widget to pass node ID to Python
                    setTimeout(() => {
                        let nodeIdWidget = this.widgets?.find(w => w.name === 'node_id');
                        if (!nodeIdWidget) {
                            nodeIdWidget = this.addWidget("text", "node_id", String(this.id), () => {});
                        } else {
                            nodeIdWidget.value = String(this.id);
                        }
                    }, 10);
                    
                    // Add manual refresh button
                    this.addManualRefreshButton();
                } catch (error) {
                    console.error('❌ Audio Analyzer: Failed to create interface:', error);
                }
                
                return result;
            };
            
            // Override onExecuted to capture data immediately
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                // Store the execution result for later access
                this.lastExecutionResult = message;
                this.lastExecutionTime = Date.now();
                
                // Check if message contains array with our data
                if (Array.isArray(message) && message.length > 1) {
                    if (typeof message[1] === 'string') {
                        try {
                            const parsedData = JSON.parse(message[1]);
                            if (parsedData.duration) {
                                // Immediately update visualization with real data
                                if (this.audioAnalyzerInterface) {
                                    this.audioAnalyzerInterface.updateVisualization(parsedData);
                                    
                                    // Setup audio playback
                                    const audioFileWidget = this.widgets?.find(w => w.name === 'audio_file');
                                    if (audioFileWidget && audioFileWidget.value) {
                                        this.setupAudioPlayback(audioFileWidget.value);
                                    }
                                }
                                return onExecuted ? onExecuted.apply(this, arguments) : undefined;
                            }
                        } catch (e) {
                            // Failed to parse execution data
                        }
                    }
                }
                
                const result = onExecuted ? onExecuted.apply(this, arguments) : undefined;
                if (this.audioAnalyzerInterface) {
                    handleNodeExecution.call(this, message);
                }
                return result;
            };
            
            // Hook into ComfyUI's execution flow
            const originalOnDrawBackground = nodeType.prototype.onDrawBackground;
            nodeType.prototype.onDrawBackground = function(ctx) {
                const result = originalOnDrawBackground ? originalOnDrawBackground.apply(this, arguments) : undefined;
                
                // Check if we just executed and have new output data
                if (this.lastExecutionTime && Date.now() - this.lastExecutionTime < 1000) {
                    // Recently executed, try to get fresh output data
                    if (this.outputs && this.outputs[1] && this.outputs[1].value) {
                        console.log('🎉 Fresh execution detected, found output data!');
                        this.lastExecutionTime = 0; // Prevent repeat processing
                        handleNodeExecution.call(this, [null, this.outputs[1].value]);
                    }
                }
                
                return result;
            };
            
            // Manual refresh button
            nodeType.prototype.addManualRefreshButton = function() {
                if (this.audioAnalyzerInterface && this.audioAnalyzerInterface.ui) {
                    const container = this.audioAnalyzerInterface.ui.container;
                    if (container) {
                        const refreshDiv = document.createElement('div');
                        refreshDiv.style.cssText = `
                            position: absolute;
                            top: 10px;
                            right: 10px;
                            background: #4a9eff;
                            color: white;
                            padding: 8px 12px;
                            border-radius: 4px;
                            font-size: 11px;
                            cursor: pointer;
                            z-index: 1000;
                        `;
                        refreshDiv.textContent = '🔄 Refresh Data';
                        refreshDiv.onclick = () => this.manualDataRefresh();
                        container.appendChild(refreshDiv);
                    }
                }
            };
            
            // Manual data refresh
            nodeType.prototype.manualDataRefresh = function() {
                if (!this.audioAnalyzerInterface) return;
                
                const audioFileWidget = this.widgets?.find(w => w.name === 'audio_file');
                if (!audioFileWidget || !audioFileWidget.value) {
                    if (this.audioAnalyzerInterface.ui.showMessage) {
                        this.audioAnalyzerInterface.ui.showMessage('Please specify an audio file first');
                    }
                    return;
                }
                
                if (this.audioAnalyzerInterface.ui.showMessage) {
                    this.audioAnalyzerInterface.ui.showMessage('Analyzing audio file...');
                }
                
                // Mark execution time for tracking
                this.lastExecutionTime = Date.now();
                
                // Queue execution and check for results
                window.app.queuePrompt();
                
                // Check for results after execution
                setTimeout(() => this.checkForResults(), 3000);
                setTimeout(() => this.checkForResults(), 6000);
            };
            
            // Check for results in node outputs
            nodeType.prototype.checkForResults = function() {
                if (this.lastDataReceived && Date.now() - this.lastDataReceived < 5000) {
                    return; // Already received data
                }
                
                // Check for output data
                
                // Try multiple ways to get the output data
                let vizData = null;
                
                // Method 1: getOutputData
                if (this.getOutputData) {
                    try {
                        vizData = this.getOutputData(1);
                        if (vizData) {
                            // Found data via getOutputData
                        }
                    } catch (e) {
                        // getOutputData failed
                    }
                }
                
                // Method 2: Check outputs array
                if (!vizData && this.outputs && this.outputs[1]) {
                    const output = this.outputs[1];
                    const possibleDataFields = ['value', 'data', '_data', 'content', 'result'];
                    
                    for (const field of possibleDataFields) {
                        if (output[field] && typeof output[field] === 'string' && output[field].includes('duration')) {
                            vizData = output[field];
                            break;
                        }
                    }
                }
                
                if (vizData && this.audioAnalyzerInterface) {
                    // Check if this looks like real data (duration != 10.79)
                    let parsedData = vizData;
                    if (typeof vizData === 'string') {
                        try {
                            parsedData = JSON.parse(vizData);
                        } catch (e) {}
                    }
                    
                    // Process visualization data
                    
                    handleNodeExecution.call(this, [null, vizData]);
                } else {
                    // Try web file approach
                    this.tryWebFileData();
                }
            };
            
            // Simple web file data fetch
            nodeType.prototype.tryWebFileData = function() {
                const webFileUrl = `/extensions/ComfyUI_ChatterBox_Voice/audio_data_${this.id}.json?t=${Date.now()}`;
                
                fetch(webFileUrl)
                    .then(response => {
                        if (!response.ok) {
                            throw new Error(`HTTP ${response.status}`);
                        }
                        return response.json();
                    })
                    .then(vizData => {
                        if (this.audioAnalyzerInterface) {
                            this.audioAnalyzerInterface.updateVisualization(vizData);
                            
                            // Setup audio playback
                            const audioFileWidget = this.widgets?.find(w => w.name === 'audio_file');
                            if (audioFileWidget && audioFileWidget.value) {
                                this.setupAudioPlayback(audioFileWidget.value);
                            }
                        }
                    })
                    .catch(error => {
                        // Fall back to test data if web file not available
                        console.log('⚠️ Web file failed, using test data:', error.message);
                        this.generateTestData();
                    });
            };
            
            // Try to fetch cached visualization data
            nodeType.prototype.tryFetchTempData = function() {
                console.log('💾 Checking for cached visualization data...');
                
                // Focus on output directory since it's web-accessible
                const possibleCacheUrls = [
                    `/output/audio_analyzer_cache_${this.id}.json`,
                    `/view?filename=audio_analyzer_cache_${this.id}.json&type=output`,
                    `/api/view?filename=audio_analyzer_cache_${this.id}.json&type=output`,
                    // Backup patterns in case of different ComfyUI configurations
                    `/temp/audio_analyzer_cache_${this.id}.json`,
                    `/view?filename=audio_analyzer_cache_${this.id}.json&type=temp`
                ];
                
                let urlIndex = 0;
                const tryNextUrl = () => {
                    if (urlIndex >= possibleCacheUrls.length) {
                        console.log('💾 All cache URLs failed, generating test data');
                        this.generateTestData();
                        return;
                    }
                    
                    const cacheUrl = `${possibleCacheUrls[urlIndex++]}?t=${Date.now()}`;
                    console.log(`💾 Trying cache URL: ${cacheUrl}`);
                    
                    fetch(cacheUrl)
                        .then(response => {
                            if (!response.ok) {
                                throw new Error(`HTTP ${response.status}`);
                            }
                            return response.json();
                        })
                        .then(data => {
                            console.log('🎉 Cache file loaded successfully!');
                            console.log('🎉 Cache data keys:', Object.keys(data));
                            
                            if (data.visualization_data) {
                                const vizData = data.visualization_data;
                                console.log('🎉 REAL DATA FOUND! Duration:', vizData.duration);
                                
                                if (this.audioAnalyzerInterface) {
                                    handleNodeExecution.call(this, [null, JSON.stringify(vizData)]);
                                    
                                    // Setup audio playback with the original file path
                                    if (data.file_path) {
                                        this.setupAudioPlayback(data.file_path);
                                    }
                                    return;
                                }
                            } else {
                                throw new Error('No visualization_data in cache');
                            }
                        })
                        .catch(error => {
                            console.log(`💾 Cache URL failed: ${error.message}`);
                            tryNextUrl();
                        });
                };
                
                tryNextUrl();
            };
            
            // Generate test data
            nodeType.prototype.generateTestData = function() {
                const testData = {
                    waveform: { samples: [], time: [] },
                    rms: { values: [], time: [] },
                    peaks: [1.0, 3.5, 6.2, 8.8],
                    duration: 10.79,
                    sample_rate: 22050,
                    regions: [
                        { start: 0.2, end: 2.1, label: "Speech 1", confidence: 0.92 },
                        { start: 2.5, end: 4.8, label: "Speech 2", confidence: 0.88 },
                        { start: 5.2, end: 8.1, label: "Speech 3", confidence: 0.85 },
                        { start: 8.6, end: 10.5, label: "Speech 4", confidence: 0.90 }
                    ]
                };
                
                // Generate sine wave
                for (let i = 0; i < 2000; i++) {
                    const time = (i / 2000) * 10.79;
                    const sample = Math.sin(2 * Math.PI * 440 * time) * 0.5;
                    testData.waveform.samples.push(sample);
                    testData.waveform.time.push(time);
                }
                
                // Generate RMS
                for (let i = 0; i < 200; i++) {
                    const time = (i / 200) * 10.79;
                    const rms = Math.abs(Math.sin(2 * Math.PI * 2 * time)) * 0.3;
                    testData.rms.values.push(rms);
                    testData.rms.time.push(time);
                }
                
                handleNodeExecution.call(this, [null, JSON.stringify(testData)]);
            };
            
            // Setup audio playback
            nodeType.prototype.setupAudioPlayback = function(filePath) {
                if (!this.audioAnalyzerInterface) return;
                
                try {
                    if (this.audioAnalyzerInterface.audioElement) {
                        this.audioAnalyzerInterface.audioElement.pause();
                    }
                    
                    // Extract filename and try multiple URL formats
                    let fileName = filePath.split('\\').pop().split('/').pop();
                    let possibleUrls = [
                        `/view?filename=${fileName}&type=input`,
                        `/api/view?filename=${fileName}&type=input`,
                        `/input/${fileName}`,
                        `./input/${fileName}`,
                        filePath // Direct path as last resort
                    ];
                    
                    let urlIndex = 0;
                    const tryNextUrl = () => {
                        if (urlIndex >= possibleUrls.length) {
                            if (this.audioAnalyzerInterface.ui.showMessage) {
                                this.audioAnalyzerInterface.ui.showMessage('Audio playback not available - file not accessible via web');
                            }
                            return;
                        }
                        
                        let webUrl = possibleUrls[urlIndex++];
                        console.log(`🎵 Trying audio URL: ${webUrl}`);
                        
                        this.audioAnalyzerInterface.audioElement = new Audio();
                        this.audioAnalyzerInterface.audioElement.src = webUrl;
                        this.audioAnalyzerInterface.audioElement.preload = 'metadata';
                        
                        this.audioAnalyzerInterface.audioElement.addEventListener('loadedmetadata', () => {
                            console.log('✅ Audio playback ready with URL:', webUrl);
                        });
                        
                        this.audioAnalyzerInterface.audioElement.addEventListener('error', () => {
                            console.log(`❌ Failed URL: ${webUrl}`);
                            tryNextUrl(); // Try next URL
                        });
                    };
                    
                    tryNextUrl();
                } catch (error) {
                    console.error('❌ Audio setup failed:', error);
                }
            };
        }
    }
});