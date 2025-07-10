/**
 * Audio Analyzer UI Module
 * Handles all DOM creation and UI management
 */
export class AudioAnalyzerUI {
    constructor(core) {
        this.core = core;
        this.container = null;
        this.canvas = null;
        this.playButton = null;
        this.stopButton = null;
        this.timeDisplay = null;
        this.selectionDisplay = null;
        this.statusDisplay = null;
        this.messageDisplay = null;
        this.controls = null;
        this.analysisControls = null;
        this.regionControls = null;
        this.exportControls = null;
        this.zoomControls = null;
    }
    
    createInterface() {
        // Create Audio Analyzer UI interface
        
        // Remove existing interface
        const existingInterface = this.core.node.widgets?.find(w => w.name === 'audio_analyzer_interface');
        if (existingInterface) {
            const existingContainer = existingInterface.element;
            if (existingContainer && existingContainer.parentNode) {
                existingContainer.parentNode.removeChild(existingContainer);
            }
        }
        
        // Create main container
        this.container = document.createElement('div');
        this.container.className = 'audio-analyzer-container';
        this.container.style.cssText = `
            width: 100%;
            height: 420px;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 4px;
            overflow: hidden;
            position: relative;
            font-family: Arial, sans-serif;
            font-size: 12px;
            color: #ffffff;
        `;
        
        // Create canvas
        this.canvas = document.createElement('canvas');
        this.canvas.className = 'audio-analyzer-canvas';
        this.canvas.style.cssText = `
            width: 100%;
            height: 300px;
            background: #1a1a1a;
            display: block;
            cursor: crosshair;
        `;
        
        // Get canvas context
        this.core.canvas = this.canvas;
        this.core.ctx = this.canvas.getContext('2d');
        
        // Create controls container
        this.controls = document.createElement('div');
        this.controls.className = 'audio-analyzer-controls';
        this.controls.style.cssText = `
            height: 80px;
            background: #2a2a2a;
            border-top: 1px solid #333;
            padding: 8px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        `;
        
        // Create playback controls
        this.createPlaybackControls();
        
        // Create main action controls (consolidated)
        this.createMainControls();
        
        // Create zoom controls
        this.createZoomControls();
        
        // Create status displays
        this.createStatusDisplays();
        
        // Assemble interface
        this.container.appendChild(this.canvas);
        this.container.appendChild(this.controls);
        
        // Add container to node using ComfyUI's DOM widget system
        try {
            if (typeof this.core.node.addDOMWidget === 'function') {
                const widget = this.core.node.addDOMWidget('audio_analyzer_interface', 'div', this.container, {
                    serialize: false,
                    hideOnZoom: false
                });
                this.widget = widget;
            } else {
                // Alternative method: create custom widget
                const widget = {
                    type: 'div',
                    name: 'audio_analyzer_interface',
                    element: this.container,
                    value: null,
                    serialize: false,
                    hideOnZoom: false,
                    draw: function(ctx, node, widget_width, widget_height) {
                        // This will be called for custom drawing
                    }
                };
                
                this.core.node.widgets = this.core.node.widgets || [];
                this.core.node.widgets.push(widget);
                this.widget = widget;
            }
        } catch (error) {
            console.error('Failed to add DOM widget:', error);
        }
        
        // Setup initial canvas size
        this.setupCanvasSize();
        
        // Force UI to appear by adding to DOM
        this.ensureUIVisible();
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
        this.playButton = document.createElement('button');
        this.playButton.textContent = '▶️ Play';
        this.playButton.style.cssText = `
            padding: 4px 8px;
            background: #4a9eff;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
        `;
        this.playButton.onclick = () => this.core.togglePlayback();
        
        // Stop button
        this.stopButton = document.createElement('button');
        this.stopButton.textContent = '⏹️ Stop';
        this.stopButton.style.cssText = `
            padding: 4px 8px;
            background: #666;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
        `;
        this.stopButton.onclick = () => this.core.stopPlayback();
        
        // Time display
        this.timeDisplay = document.createElement('span');
        this.timeDisplay.textContent = '00:00.000';
        this.timeDisplay.style.cssText = `
            font-family: monospace;
            color: #fff;
            font-size: 11px;
            margin-left: 8px;
        `;
        
        // Selection display
        this.selectionDisplay = document.createElement('span');
        this.selectionDisplay.textContent = 'No selection';
        this.selectionDisplay.style.cssText = `
            font-family: monospace;
            color: #ffff00;
            font-size: 11px;
            margin-left: 16px;
        `;
        
        playbackContainer.appendChild(this.playButton);
        playbackContainer.appendChild(this.stopButton);
        playbackContainer.appendChild(this.timeDisplay);
        playbackContainer.appendChild(this.selectionDisplay);
        
        this.controls.appendChild(playbackContainer);
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
        
        this.controls.appendChild(mainControls);
    }
    
    createZoomControls() {
        this.zoomControls = document.createElement('div');
        this.zoomControls.style.cssText = `
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
        
        this.zoomControls.appendChild(zoomInButton);
        this.zoomControls.appendChild(zoomOutButton);
        this.zoomControls.appendChild(resetZoomButton);
        
        // Position zoom controls at the end of the first row
        this.controls.firstChild.appendChild(this.zoomControls);
    }
    
    createStatusDisplays() {
        // Status display
        this.statusDisplay = document.createElement('div');
        this.statusDisplay.style.cssText = `
            font-size: 11px;
            color: #888;
            padding: 2px 0;
            border-top: 1px solid #333;
            margin-top: 4px;
        `;
        this.statusDisplay.textContent = 'Ready to analyze audio';
        
        // Message display
        this.messageDisplay = document.createElement('div');
        this.messageDisplay.style.cssText = `
            font-size: 11px;
            color: #4a9eff;
            padding: 2px 0;
            min-height: 14px;
        `;
        
        this.controls.appendChild(this.statusDisplay);
        this.controls.appendChild(this.messageDisplay);
    }
    
    setupCanvasSize() {
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width * devicePixelRatio;
        this.canvas.height = rect.height * devicePixelRatio;
        this.core.ctx.scale(devicePixelRatio, devicePixelRatio);
    }
    
    setupCanvasResize() {
        const resizeObserver = new ResizeObserver(() => {
            this.core.resizeCanvas();
        });
        resizeObserver.observe(this.canvas);
    }
    
    setupDragAndDrop() {
        // Add drag and drop functionality
        this.canvas.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.canvas.style.opacity = '0.7';
        });
        
        this.canvas.addEventListener('dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.canvas.style.opacity = '1';
        });
        
        this.canvas.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.canvas.style.opacity = '1';
            
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
        if (this.timeDisplay) {
            this.timeDisplay.textContent = this.core.formatTime(this.core.currentTime);
        }
    }
    
    // Update selection display
    updateSelectionDisplay() {
        if (this.selectionDisplay) {
            if (this.core.selectedStart !== null && this.core.selectedEnd !== null) {
                const duration = this.core.selectedEnd - this.core.selectedStart;
                this.selectionDisplay.textContent = 
                    `Selected: ${this.core.formatTime(this.core.selectedStart)} - ${this.core.formatTime(this.core.selectedEnd)} (${this.core.formatTime(duration)})`;
            } else {
                this.selectionDisplay.textContent = 'No selection';
            }
        }
    }
    
    // Show message
    showMessage(message) {
        if (this.messageDisplay) {
            this.messageDisplay.textContent = message;
            this.messageDisplay.style.color = '#4a9eff';
            
            // Clear message after 3 seconds
            setTimeout(() => {
                if (this.messageDisplay) {
                    this.messageDisplay.textContent = '';
                }
            }, 3000);
        }
    }
    
    // Update status
    updateStatus(status) {
        if (this.statusDisplay) {
            this.statusDisplay.textContent = status;
        }
    }
    
    ensureUIVisible() {
        // Ensure UI is visible
        
        // Simple approach: let ComfyUI's DOM widget system handle positioning
        // Don't manually append to DOM - let the widget system do it
        if (this.widget && this.widget.element) {
            // The widget system should handle this automatically
            return;
        }
        
        // Only use fallback if widget system failed completely
        setTimeout(() => {
            if (!document.body.contains(this.container)) {
                this.container.style.position = 'relative';
                this.container.style.width = '100%';
                this.container.style.maxWidth = '780px';
                this.container.style.zIndex = '9999';
                document.body.appendChild(this.container);
            }
        }, 200);
    }
}