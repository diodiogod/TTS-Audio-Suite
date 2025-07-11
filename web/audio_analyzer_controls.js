/**
 * Audio Analyzer Controls Module
 * Handles creation and management of all UI controls
 */
export class AudioAnalyzerControls {
    constructor(core) {
        this.core = core;
    }
    
    createPlaybackControls() {
        const playbackContainer = document.createElement('div');
        playbackContainer.style.cssText = `
            display: flex;
            gap: 8px;
            align-items: center;
            padding: 2px 0;
        `;
        
        // Play button
        this.core.ui.playButton = document.createElement('button');
        this.core.ui.playButton.textContent = '▶️ Play';
        this.core.ui.playButton.style.cssText = `
            padding: 4px 8px;
            background: #4a9eff;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
        `;
        this.core.ui.playButton.onclick = () => this.core.togglePlayback();
        
        // Stop button
        this.core.ui.stopButton = document.createElement('button');
        this.core.ui.stopButton.textContent = '⏹️ Stop';
        this.core.ui.stopButton.style.cssText = `
            padding: 4px 8px;
            background: #666;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
        `;
        this.core.ui.stopButton.onclick = () => this.core.stopPlayback();
        
        // Time display
        this.core.ui.timeDisplay = document.createElement('span');
        this.core.ui.timeDisplay.textContent = '00:00.000';
        this.core.ui.timeDisplay.style.cssText = `
            font-family: monospace;
            color: #fff;
            font-size: 11px;
            margin-left: 8px;
        `;
        
        // Selection display
        this.core.ui.selectionDisplay = document.createElement('span');
        this.core.ui.selectionDisplay.textContent = 'No selection';
        this.core.ui.selectionDisplay.style.cssText = `
            font-family: monospace;
            color: #ffff00;
            font-size: 11px;
            margin-left: 16px;
        `;
        
        playbackContainer.appendChild(this.core.ui.playButton);
        playbackContainer.appendChild(this.core.ui.stopButton);
        playbackContainer.appendChild(this.core.ui.timeDisplay);
        playbackContainer.appendChild(this.core.ui.selectionDisplay);
        
        return playbackContainer;
    }
    
    createMainControls() {
        // Single row with all main action buttons
        const mainControls = document.createElement('div');
        mainControls.style.cssText = `
            display: flex;
            gap: 6px;
            align-items: center;
            padding: 2px 0;
            flex-wrap: wrap;
        `;
        
        // Button style template
        const buttonStyle = `
            padding: 3px 6px;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 10px;
            color: white;
        `;
        
        // Analyze button
        const analyzeButton = document.createElement('button');
        analyzeButton.textContent = '🔍 Analyze';
        analyzeButton.style.cssText = buttonStyle + 'background: #28a745;';
        analyzeButton.onclick = () => this.core.onParametersChanged();
        
        // Clear selection button
        const clearButton = document.createElement('button');
        clearButton.textContent = '🗑️ Clear';
        clearButton.style.cssText = buttonStyle + 'background: #dc3545;';
        clearButton.onclick = () => this.core.clearSelection();
        
        // Add region button
        const addRegionButton = document.createElement('button');
        addRegionButton.textContent = '➕ Add Region';
        addRegionButton.style.cssText = buttonStyle + 'background: #17a2b8;';
        addRegionButton.onclick = () => this.core.addSelectedRegion();
        
        // Clear all regions button
        const clearAllButton = document.createElement('button');
        clearAllButton.textContent = '🗑️ Clear All';
        clearAllButton.style.cssText = buttonStyle + 'background: #6c757d;';
        clearAllButton.onclick = () => this.core.clearAllRegions();
        
        // Export timing button
        const exportButton = document.createElement('button');
        exportButton.textContent = '📋 Export Timings';
        exportButton.style.cssText = buttonStyle + 'background: #fd7e14;';
        exportButton.onclick = () => this.core.exportTiming();
        
        mainControls.appendChild(analyzeButton);
        mainControls.appendChild(clearButton);
        mainControls.appendChild(addRegionButton);
        mainControls.appendChild(clearAllButton);
        mainControls.appendChild(exportButton);
        
        return mainControls;
    }
    
    createZoomControls() {
        const zoomControls = document.createElement('div');
        zoomControls.style.cssText = `
            display: flex;
            gap: 8px;
            align-items: center;
            padding: 2px 0;
            margin-left: auto;
        `;
        
        // Zoom in button
        const zoomInButton = document.createElement('button');
        zoomInButton.textContent = '🔍+';
        zoomInButton.style.cssText = `
            padding: 4px 8px;
            background: #6f42c1;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
        `;
        zoomInButton.onclick = () => this.core.zoomIn();
        
        // Zoom out button
        const zoomOutButton = document.createElement('button');
        zoomOutButton.textContent = '🔍-';
        zoomOutButton.style.cssText = `
            padding: 4px 8px;
            background: #6f42c1;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
        `;
        zoomOutButton.onclick = () => this.core.zoomOut();
        
        // Reset zoom button
        const resetZoomButton = document.createElement('button');
        resetZoomButton.textContent = '🔄 Reset';
        resetZoomButton.style.cssText = `
            padding: 4px 8px;
            background: #6f42c1;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
        `;
        resetZoomButton.onclick = () => this.core.resetZoom();
        
        zoomControls.appendChild(zoomInButton);
        zoomControls.appendChild(zoomOutButton);
        zoomControls.appendChild(resetZoomButton);
        
        return zoomControls;
    }
    
    createStatusDisplays() {
        const statusContainer = document.createElement('div');
        
        // Status display
        this.core.ui.statusDisplay = document.createElement('div');
        this.core.ui.statusDisplay.style.cssText = `
            font-size: 11px;
            color: #888;
            padding: 2px 0;
            border-top: 1px solid #333;
            margin-top: 4px;
        `;
        this.core.ui.statusDisplay.textContent = 'Ready to analyze audio';
        
        // Message display
        this.core.ui.messageDisplay = document.createElement('div');
        this.core.ui.messageDisplay.style.cssText = `
            font-size: 11px;
            color: #4a9eff;
            padding: 2px 0;
            min-height: 14px;
        `;
        
        statusContainer.appendChild(this.core.ui.statusDisplay);
        statusContainer.appendChild(this.core.ui.messageDisplay);
        
        return statusContainer;
    }
    
    setupCanvasSize() {
        const rect = this.core.canvas.getBoundingClientRect();
        this.core.canvas.width = rect.width * devicePixelRatio;
        this.core.canvas.height = rect.height * devicePixelRatio;
        this.core.ctx.scale(devicePixelRatio, devicePixelRatio);
    }
    
    setupCanvasResize() {
        const resizeObserver = new ResizeObserver(() => {
            this.core.resizeCanvas();
        });
        resizeObserver.observe(this.core.canvas);
    }
    
    setupDragAndDrop() {
        // Add drag and drop functionality
        this.core.canvas.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.core.canvas.style.opacity = '0.7';
        });
        
        this.core.canvas.addEventListener('dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.core.canvas.style.opacity = '1';
        });
        
        this.core.canvas.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.core.canvas.style.opacity = '1';
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                if (file.type.startsWith('audio/')) {
                    
                    // Update the audio_file widget with the file name
                    const audioFileWidget = this.core.node.widgets.find(w => w.name === 'audio_file');
                    if (audioFileWidget) {
                        // For now, just set the name - user will need to provide the actual path
                        audioFileWidget.value = file.name;
                        
                        // Show message that user needs to provide the actual file path
                        this.showMessage(`File dropped: ${file.name}. Please enter the full file path in the audio_file widget.`);
                        
                        // Update the widget display
                        if (audioFileWidget.callback) {
                            audioFileWidget.callback(file.name);
                        }
                    }
                } else {
                    this.showMessage('Please drop an audio file');
                }
            }
        });
    }
    
    // Update time display
    updateTimeDisplay() {
        if (this.core.ui.timeDisplay) {
            this.core.ui.timeDisplay.textContent = this.core.formatTime(this.core.currentTime);
        }
    }
    
    // Update selection display
    updateSelectionDisplay() {
        if (this.core.ui.selectionDisplay) {
            if (this.core.selectedStart !== null && this.core.selectedEnd !== null) {
                const duration = this.core.selectedEnd - this.core.selectedStart;
                this.core.ui.selectionDisplay.textContent = 
                    `Selected: ${this.core.formatTime(this.core.selectedStart)} - ${this.core.formatTime(this.core.selectedEnd)} (${this.core.formatTime(duration)})`;
            } else {
                this.core.ui.selectionDisplay.textContent = 'No selection';
            }
        }
    }
    
    // Show message
    showMessage(message) {
        if (this.core.ui.messageDisplay) {
            this.core.ui.messageDisplay.textContent = message;
            this.core.ui.messageDisplay.style.color = '#4a9eff';
            
            // Clear message after 3 seconds
            setTimeout(() => {
                if (this.core.ui.messageDisplay) {
                    this.core.ui.messageDisplay.textContent = '';
                }
            }, 3000);
        }
    }
    
    // Update status
    updateStatus(status) {
        if (this.core.ui.statusDisplay) {
            this.core.ui.statusDisplay.textContent = status;
        }
    }
}