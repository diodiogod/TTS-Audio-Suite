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
        console.log('Creating Audio Analyzer UI interface');
        
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
            height: 400px;
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
            height: 100px;
            background: #2a2a2a;
            border-top: 1px solid #333;
            padding: 8px;
            display: flex;
            flex-direction: column;
            gap: 6px;
        `;
        
        // Create playback controls
        this.createPlaybackControls();
        
        // Create analysis controls
        this.createAnalysisControls();
        
        // Create region controls
        this.createRegionControls();
        
        // Create export controls
        this.createExportControls();
        
        // Create zoom controls
        this.createZoomControls();
        
        // Create status displays
        this.createStatusDisplays();
        
        // Assemble interface
        this.container.appendChild(this.canvas);
        this.container.appendChild(this.controls);
        
        // Add container to node using ComfyUI's DOM widget system
        console.log('Adding DOM widget to node, available methods:', Object.keys(this.core.node));
        
        try {
            // Try ComfyUI's addDOMWidget method
            if (typeof this.core.node.addDOMWidget === 'function') {
                const widget = this.core.node.addDOMWidget('audio_analyzer_interface', 'div', this.container, {
                    serialize: false,
                    hideOnZoom: false
                });
                console.log('DOM widget added successfully:', widget);
                this.widget = widget;
            } else {
                console.warn('addDOMWidget not available, trying alternative method');
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
                console.log('Custom widget created');
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
    
    createAnalysisControls() {
        this.analysisControls = document.createElement('div');
        this.analysisControls.style.cssText = `
            display: flex;
            gap: 8px;
            align-items: center;
            padding: 2px 0;
        `;
        
        // Analyze button
        const analyzeButton = document.createElement('button');
        analyzeButton.textContent = '🔍 Analyze';
        analyzeButton.style.cssText = `
            padding: 4px 8px;
            background: #28a745;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
        `;
        analyzeButton.onclick = () => this.core.onParametersChanged();
        
        // Clear button
        const clearButton = document.createElement('button');
        clearButton.textContent = '🗑️ Clear';
        clearButton.style.cssText = `
            padding: 4px 8px;
            background: #dc3545;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
        `;
        clearButton.onclick = () => this.core.clearSelection();
        
        this.analysisControls.appendChild(analyzeButton);
        this.analysisControls.appendChild(clearButton);
        
        this.controls.appendChild(this.analysisControls);
    }
    
    createRegionControls() {
        this.regionControls = document.createElement('div');
        this.regionControls.style.cssText = `
            display: flex;
            gap: 8px;
            align-items: center;
            padding: 2px 0;
        `;
        
        // Add region button
        const addRegionButton = document.createElement('button');
        addRegionButton.textContent = '➕ Add Region';
        addRegionButton.style.cssText = `
            padding: 4px 8px;
            background: #17a2b8;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
        `;
        addRegionButton.onclick = () => this.core.addSelectedRegion();
        
        // Clear all regions button
        const clearRegionsButton = document.createElement('button');
        clearRegionsButton.textContent = '🗑️ Clear All';
        clearRegionsButton.style.cssText = `
            padding: 4px 8px;
            background: #6c757d;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
        `;
        clearRegionsButton.onclick = () => this.core.clearAllRegions();
        
        this.regionControls.appendChild(addRegionButton);
        this.regionControls.appendChild(clearRegionsButton);
        
        this.controls.appendChild(this.regionControls);
    }
    
    createExportControls() {
        this.exportControls = document.createElement('div');
        this.exportControls.style.cssText = `
            display: flex;
            gap: 8px;
            align-items: center;
            padding: 2px 0;
        `;
        
        // Export timing button
        const exportButton = document.createElement('button');
        exportButton.textContent = '📋 Export Timing';
        exportButton.style.cssText = `
            padding: 4px 8px;
            background: #fd7e14;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 11px;
        `;
        exportButton.onclick = () => this.core.exportTiming();
        
        this.exportControls.appendChild(exportButton);
        
        this.controls.appendChild(this.exportControls);
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
                    // Create a temporary URL for the file
                    const url = URL.createObjectURL(file);
                    
                    // Update the audio_file widget
                    const audioFileWidget = this.core.node.widgets.find(w => w.name === 'audio_file');
                    if (audioFileWidget) {
                        audioFileWidget.value = file.name;
                        
                        // Trigger analysis with the file
                        this.core.onAudioFileSelected(url);
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
        console.log('Ensuring UI is visible');
        
        // Try multiple methods to make the UI visible
        setTimeout(() => {
            // Method 1: Find the node element and append directly
            const nodeElement = document.querySelector(`[data-id="${this.core.node.id}"]`);
            if (nodeElement && !nodeElement.contains(this.container)) {
                console.log('Appending container to node element');
                nodeElement.appendChild(this.container);
            }
            
            // Method 2: Try to find ComfyUI's widget container
            const widgetContainer = nodeElement?.querySelector('.comfy-widget-container');
            if (widgetContainer && !widgetContainer.contains(this.container)) {
                console.log('Appending container to widget container');
                widgetContainer.appendChild(this.container);
            }
            
            // Method 3: Add to node's DOM element if it exists
            if (this.core.node.element && !this.core.node.element.contains(this.container)) {
                console.log('Appending container to node DOM element');
                this.core.node.element.appendChild(this.container);
            }
            
            // Method 4: Force visibility with absolute positioning as fallback
            if (!document.body.contains(this.container)) {
                console.log('Using fallback: appending to body');
                this.container.style.position = 'absolute';
                this.container.style.top = '100px';
                this.container.style.left = '100px';
                this.container.style.zIndex = '9999';
                document.body.appendChild(this.container);
            }
        }, 200);
    }
}